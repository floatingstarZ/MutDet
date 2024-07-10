# Copyright (c) OpenMMLab. All rights reserved.
import math
import warnings
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import (build_activation_layer, build_conv_layer,
                      build_norm_layer, xavier_init)
from mmcv.cnn.bricks.registry import (TRANSFORMER_LAYER,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import (BaseTransformerLayer,
                                         TransformerLayerSequence,
                                         build_transformer_layer_sequence)
from torch.nn.init import normal_

from mmrotate.models.utils.builder import ROTATED_TRANSFORMER
from mmdet.models.utils import Transformer
from mmdet.models.utils.transformer import inverse_sigmoid
from AlignRotate.models.utils.transformer import TwoWayTransformer

try:
    from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention

except ImportError:
    warnings.warn(
        '`MultiScaleDeformableAttention` in MMCV has been moved to '
        '`mmcv.ops.multi_scale_deform_attn`, please update your MMCV')
    from mmcv.cnn.bricks.transformer import MultiScaleDeformableAttention

def obb2poly_tr(rboxes):
    """Convert oriented bounding boxes to polygons.

    Args:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]

    Returns:
        polys (torch.Tensor): [x0,y0,x1,y1,x2,y2,x3,y3]
    """
    x = rboxes[..., 0]
    y = rboxes[..., 1]
    w = rboxes[..., 2]
    h = rboxes[..., 3]
    a = rboxes[..., 4]
    cosa = torch.cos(a)
    sina = torch.sin(a)
    wx, wy = w / 2 * cosa, w / 2 * sina
    hx, hy = -h / 2 * sina, h / 2 * cosa
    p1x, p1y = x - wx - hx, y - wy - hy
    p2x, p2y = x + wx - hx, y + wy - hy
    p3x, p3y = x + wx + hx, y + wy + hy
    p4x, p4y = x - wx + hx, y - wy + hy
    return torch.stack([p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y], dim=2)

def bbox_cxcywh_to_xyxy_tr(bbox):
    """Convert bbox coordinates from (cx, cy, w, h) to (x1, y1, x2, y2).

    Args:
        bbox (Tensor): Shape (n, 4) for bboxes.

    Returns:
        Tensor: Converted bboxes.
    """
    cx, cy, w, h = bbox.split((1, 1, 1, 1), dim=-1)
    bbox_new = [(cx - 0.5 * w), (cy - 0.5 * h), (cx + 0.5 * w), (cy - 0.5 * h),
                (cx - 0.5 * w), (cy + 0.5 * h), (cx + 0.5 * w), (cy + 0.5 * h)]
    return torch.cat(bbox_new, dim=-1)

@ROTATED_TRANSFORMER.register_module()
class DNARSRotatedDeformableDetrTransformerTemplateMatching(Transformer):
    """Implements the DeformableDETR transformer.

    Args:
        as_two_stage (bool): Generate query from encoder features.
            Default: False.
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
        two_stage_num_proposals (int): Number of proposals when set
            `as_two_stage` as True. Default: 300.
    """

    def __init__(self,
                 as_two_stage=False,
                 num_feature_levels=5,
                 two_stage_num_proposals=300,
                 # ----------- template_matching ----------
                 with_template_matching=False,
                 depth=2,
                 num_heads=8,
                 #
                 **kwargs):
        super(DNARSRotatedDeformableDetrTransformerTemplateMatching, self).__init__(**kwargs)
        self.as_two_stage = as_two_stage
        self.num_feature_levels = num_feature_levels
        self.two_stage_num_proposals = two_stage_num_proposals
        self.embed_dims = self.encoder.embed_dims
        self.init_layers()
        # ----------- template_matching ----------
        self.with_template_matching = with_template_matching
        self.feature_enhance = TwoWayTransformer(depth=depth,
                                                 embedding_dim=self.embed_dims,
                                                 num_heads=num_heads,
                                                 mlp_dim=2048)
        self.patch_enc_output = nn.Linear(self.embed_dims, self.embed_dims)
        self.patch_enc_output_norm = nn.LayerNorm(self.embed_dims)
        self.patch_enc_decode = nn.Sequential(
            nn.Linear(self.embed_dims, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, self.embed_dims)
        )

    def init_layers(self):
        """Initialize layers of the DeformableDetrTransformer."""
        self.level_embeds = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims))

        if self.as_two_stage:
            self.enc_output = nn.Linear(self.embed_dims, self.embed_dims)
            self.enc_output_norm = nn.LayerNorm(self.embed_dims)
            # self.pos_trans = nn.Linear(self.embed_dims * 2,
            #                            self.embed_dims * 2)
            # self.pos_trans_norm = nn.LayerNorm(self.embed_dims * 2)
        else:
            self.reference_points = nn.Linear(self.embed_dims, 2)

        self.query_embed = nn.Embedding(self.two_stage_num_proposals, self.embed_dims)

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()
        if not self.as_two_stage:
            xavier_init(self.reference_points, distribution='uniform', bias=0.)
        normal_(self.level_embeds)
        nn.init.normal_(self.query_embed.weight.data)

    def gen_encoder_output_proposals(self, memory, patch_memory, memory_padding_mask,
                                     spatial_shapes):
        """Generate proposals from encoded memory.

        Args:
            memory (Tensor) : The output of encoder,
                has shape (bs, num_key, embed_dim).  num_key is
                equal the number of points on feature map from
                all level.
            memory_padding_mask (Tensor): Padding mask for memory.
                has shape (bs, num_key).
            spatial_shapes (Tensor): The shape of all feature maps.
                has shape (num_level, 2).

        Returns:
            tuple: A tuple of feature map and bbox prediction.

                - output_memory (Tensor): The input of decoder,  \
                    has shape (bs, num_key, embed_dim).  num_key is \
                    equal the number of points on feature map from \
                    all levels.
                - output_proposals (Tensor): The normalized proposal \
                    after a inverse sigmoid, has shape \
                    (bs, num_keys, 4).
        """

        N, S, C = memory.shape
        proposals = []
        _cur = 0
        for lvl, (H, W) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H * W)].view(
                N, H, W, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(
                torch.linspace(
                    0, H - 1, H, dtype=torch.float32, device=memory.device),
                torch.linspace(
                    0, W - 1, W, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W.unsqueeze(-1),
                               valid_H.unsqueeze(-1)], 1).view(N, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0**lvl)
            proposal = torch.cat((grid, wh), -1).view(N, -1, 4)
            proposals.append(proposal)
            _cur += (H * W)
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) &
                                  (output_proposals < 0.99)).all(
            -1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(
            memory_padding_mask.unsqueeze(-1), 10000)
        output_proposals = output_proposals.masked_fill(
            ~output_proposals_valid, 10000)

        # 检测分支的memory
        output_memory = memory
        output_memory = output_memory.masked_fill(
            memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid,
                                                  float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))

        # template_matching的memory
        patch_output_memory = patch_memory
        patch_output_memory = patch_output_memory.masked_fill(
            memory_padding_mask.unsqueeze(-1), float(0))
        patch_output_memory = patch_output_memory.masked_fill(~output_proposals_valid,
                                                  float(0))
        patch_output_memory = self.patch_enc_output_norm(self.patch_enc_output(patch_output_memory))

        return output_memory, patch_output_memory, output_proposals

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        """Get the reference points used in decoder.

        Args:
            spatial_shapes (Tensor): The shape of all
                feature maps, has shape (num_level, 2).
            valid_ratios (Tensor): The radios of valid
                points on the feature map, has shape
                (bs, num_levels, 2)
            device (obj:`device`): The device where
                reference_points should be.

        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """
        reference_points_list = []
        for lvl, (H, W) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=torch.float32, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (
                    valid_ratios[:, None, lvl, 1] * H)
            ref_x = ref_x.reshape(-1)[None] / (
                    valid_ratios[:, None, lvl, 0] * W)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def get_valid_ratio(self, mask):
        """Get the valid radios of feature maps of all  level."""
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def get_proposal_pos_embed(self,
                               proposals,
                               num_pos_feats=128,
                               temperature=10000):
        """Get the position embedding of proposal."""
        scale = 2 * math.pi
        dim_t = torch.arange(
            num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature**(2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()),
                          dim=4).flatten(2)
        return pos

    def forward(self,
                mlvl_feats,
                mlvl_masks,
                query_embed,
                mlvl_pos_embeds,
                dn_label_query,
                dn_bbox_query,
                bbox_coder=None,
                reg_branches=None,
                cls_branches=None,
                angle_braches=None,
                angle_coder=None,
                attn_masks=None,
                # -------- patch特征
                patch_feats=None,
                **kwargs):
        assert self.as_two_stage and query_embed is None, \
            'as_two_stage must be True for DINO'

        feat_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask, pos_embed) in enumerate(
                zip(mlvl_feats, mlvl_masks, mlvl_pos_embeds)):
            bs, c, h, w = feat.shape
            # pos_embed.shape = [2, 256, 128, 128]
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            # [bs, w*h, c]
            feat = feat.flatten(2).transpose(1, 2)
            # [bs, w*h]
            mask = mask.flatten(1)
            # [bs, w*h]
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embeds[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            feat_flatten.append(feat)
            mask_flatten.append(mask)
        feat_flatten = torch.cat(feat_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feat_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack(
            [self.get_valid_ratio(m) for m in mlvl_masks], 1)
        # multi-scale reference points
        # ----------------- 获得多尺度的Reference points, B N(多尺度图像特征数目，也就是key的数目) 4(Scale) 2(x,y)
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios,
                                      device=feat.device)

        feat_flatten = feat_flatten.permute(1, 0, 2)  # (H*W, bs, embed_dims)
        lvl_pos_embed_flatten = lvl_pos_embed_flatten.permute(
            1, 0, 2)  # (H*W, bs, embed_dims)
        # ----------------- Encoder过程，多尺度特征融合
        memory = self.encoder(
            query=feat_flatten,
            key=None,
            value=None,
            query_pos=lvl_pos_embed_flatten,
            query_key_padding_mask=mask_flatten,
            spatial_shapes=spatial_shapes,
            reference_points=reference_points,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            **kwargs)
        # ---------- memory: B N(num Key) 256
        memory = memory.permute(1, 0, 2)
        bs, _, c = memory.shape
        # ---------- output_memory:    B N(num Key) 256，将memory经过了一层变换
        # ---------- output_proposals: B N(num Key) 4，  Proposal还是和Anchor一样，是生成的
        output_memory, output_patch_memory, output_proposals = \
            self.gen_encoder_output_proposals(memory, memory, mask_flatten, spatial_shapes)
        ########################################## 常规query 生成 ###################################
        # ---------- enc_outputs_class:          B N(num Key) 15，   将memory经过了一层变换
        # ---------- enc_outputs_angle_cls:      B N(num Key) 180，  ARS_CSL的角度编码输出
        # ---------- enc_outputs_coord_unact:    B N(num Key) 4，    坐标输出
        enc_outputs_class = cls_branches[self.decoder.num_layers](
            output_memory)
        enc_outputs_angle_cls = angle_braches[self.decoder.num_layers](
            output_memory)
        enc_outputs_coord_unact= \
            reg_branches[self.decoder.num_layers](output_memory) + output_proposals
        # ---------- 选取topk个，enc_outputs_class实际上只有第一个值有意义（Proposal前背景分类）
        # 因此只使用了第0维进行筛选。最后对angle进行解码，获得topk_angle
        topk = self.two_stage_num_proposals
        topk_proposals = torch.topk(
            enc_outputs_class[..., 0], topk, dim=1)[1]
        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact, 1,
            topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
        topk_coords_unact = topk_coords_unact.detach()
        topk_angle_cls = torch.gather(
            enc_outputs_angle_cls, 1,
            topk_proposals.unsqueeze(-1).repeat(1, 1, enc_outputs_angle_cls.shape[-1])
        )
        topk_angle = angle_coder.decode(topk_angle_cls.detach()).detach()
        # ---------- 将query与dnquery（denoise）联合起来，准备decoder工作
        query = self.query_embed.weight[:, None, :].repeat(1, bs, 1).transpose(0, 1)
        if dn_label_query is not None:
            query = torch.cat([dn_label_query, query], dim=1)
        if dn_bbox_query is not None:
            reference_points = torch.cat([dn_bbox_query[..., :4], topk_coords_unact], dim=1)
            reference_angle = torch.cat([dn_bbox_query[..., 4], topk_angle], dim=1)
        else:
            reference_points = topk_coords_unact
            reference_angle = topk_angle
        reference_points = reference_points.sigmoid()
        init_reference_out = reference_points
        init_reference_angle_out = reference_angle
        # ------------- decoder过程
        # INPUT：
        #   reference_points: B N(Query) 4
        #   reference_angle:  B N(Query)    这两个最为主要，记录了坐标与角度
        # Output
        # inter_stater:     N(Decoder Layer)   N(Query) B 256
        # inter_references: N(Decoder Layer+1) B N(Query) 4，
        #           第一个元素为初始的reference point，最后一个维度为4是因为只记录了坐标，而没有角度

        query = query.permute(1, 0, 2)
        memory = memory.permute(1, 0, 2)
        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=memory,
            attn_masks=attn_masks,
            key_padding_mask=mask_flatten,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reg_branches=reg_branches,
            bbox_coder=bbox_coder,
            reference_angle=reference_angle,
            angle_braches=angle_braches,
            angle_coder=angle_coder,
            **kwargs)

        inter_references_out = inter_references

        ########################################## patch-based query 生成 ###################################
        # ------- 对Image特征与patch特征进行交互
        # output_patch_memory: B M 256
        # output_patch_feats:  B N 256
        all_patch_topk_coords_unact = []
        all_patch_topk_angle = []
        all_output_patch_feats = []
        all_output_patch_memories = []

        all_patch_enc_outputs_sims = []
        all_patch_enc_outputs_coord_unact = []
        all_patch_enc_outputs_class = []
        all_patch_enc_outputs_angle_cls = []
        all_patch_enc_outputs_embeds = []

        for i, patch_feat in enumerate(patch_feats):
            out_single_patch_feats, out_single_patch_memory = \
                self.feature_enhance(patch_feat[None, ...], output_patch_memory[i:i+1, ...])
            # B, N, M
            enc_outputs_embed = self.patch_enc_decode(out_single_patch_memory)
            enc_outputs_embed = torch.nn.functional.normalize(enc_outputs_embed, dim=-1)
            norm_patch_feats = torch.nn.functional.normalize(out_single_patch_feats, dim=-1)
            patch_enc_outputs_sims = \
                torch.matmul(enc_outputs_embed, norm_patch_feats.transpose(2, 1))
            # B, N
            patch_enc_outputs_max_sim= torch.max(patch_enc_outputs_sims, dim=-1)[0] # 实例相似度分数
            patch_enc_outputs_class = cls_branches[self.decoder.num_layers](        # 似物性分数
                out_single_patch_memory)
            patch_enc_outputs_score = patch_enc_outputs_max_sim * patch_enc_outputs_class[..., 0] # 总体分数

            patch_enc_outputs_angle_cls = angle_braches[self.decoder.num_layers](
                out_single_patch_memory)
            patch_enc_outputs_coord_unact= \
                reg_branches[self.decoder.num_layers](out_single_patch_memory) + output_proposals[i:i+1, ...]
            # ---------- 选取topk个，enc_outputs_class实际上只有第一个值有意义（Proposal前背景分类）
            # 因此只使用了第0维进行筛选。最后对angle进行解码，获得topk_angle
            topk = self.two_stage_num_proposals
            patch_topk_proposals = torch.topk(
                patch_enc_outputs_score, topk, dim=1)[1]
            patch_topk_coords_unact = torch.gather(
                patch_enc_outputs_coord_unact, 1,
                patch_topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
            patch_topk_coords_unact = patch_topk_coords_unact.detach()
            patch_topk_angle_cls = torch.gather(
                patch_enc_outputs_angle_cls, 1,
                patch_topk_proposals.unsqueeze(-1).repeat(1, 1, patch_enc_outputs_angle_cls.shape[-1])
            )
            patch_topk_angle = angle_coder.decode(patch_topk_angle_cls.detach()).detach()
            all_patch_topk_coords_unact.append(patch_topk_coords_unact)
            all_patch_topk_angle.append(patch_topk_angle)
            all_output_patch_memories.append(out_single_patch_memory)
            all_output_patch_feats.append(out_single_patch_feats[0]) # 去除batch
            all_patch_enc_outputs_embeds.append(enc_outputs_embed)

            all_patch_enc_outputs_sims.append(patch_enc_outputs_sims)  # 去除batch
            all_patch_enc_outputs_coord_unact.append(patch_enc_outputs_coord_unact)
            all_patch_enc_outputs_angle_cls.append(patch_enc_outputs_angle_cls)
            all_patch_enc_outputs_class.append(patch_enc_outputs_class)


        patch_topk_coords_unact = torch.cat(all_patch_topk_coords_unact)
        patch_topk_angle = torch.cat(all_patch_topk_angle)
        output_patch_memory = torch.cat(all_output_patch_memories)

        output_patch_feats = all_output_patch_feats
        patch_enc_outputs_sims = all_patch_enc_outputs_sims  # torch.cat(all_patch_enc_outputs_sims)
        patch_enc_outputs_embeds = torch.cat(all_patch_enc_outputs_embeds)

        patch_enc_outputs_coord_unact = torch.cat(all_patch_enc_outputs_coord_unact)
        patch_enc_outputs_angle_cls = torch.cat(all_patch_enc_outputs_angle_cls)
        patch_enc_outputs_class = torch.cat(all_patch_enc_outputs_class)

        # ---------- 将query与dnquery（denoise）联合起来，准备decoder工作
        patch_query = self.query_embed.weight[:, None, :].repeat(1, bs, 1).transpose(0, 1)
        if dn_label_query is not None:
            patch_query = torch.cat([dn_label_query, patch_query], dim=1)
        if dn_bbox_query is not None:
            patch_reference_points = torch.cat([dn_bbox_query[..., :4], patch_topk_coords_unact], dim=1)
            patch_reference_angle = torch.cat([dn_bbox_query[..., 4], patch_topk_angle], dim=1)
        else:
            patch_reference_points = patch_topk_coords_unact
            patch_reference_angle = patch_topk_angle
        patch_reference_points = patch_reference_points.sigmoid()
        patch_init_reference_out = patch_reference_points
        patch_init_reference_angle_out = patch_reference_angle

        # ------------- decoder过程
        # INPUT：
        #   reference_points: B N(Query) 4
        #   reference_angle:  B N(Query)    这两个最为主要，记录了坐标与角度
        # Output
        # inter_stater:     N(Decoder Layer)   N(Query) B 256
        # inter_references: N(Decoder Layer+1) B N(Query) 4，
        #           第一个元素为初始的reference point，最后一个维度为4是因为只记录了坐标，而没有角度

        patch_query = patch_query.permute(1, 0, 2)
        patch_memory = output_patch_memory.permute(1, 0, 2)
        patch_inter_states, patch_inter_references = self.decoder(
            query=patch_query,
            key=None,
            value=patch_memory,
            attn_masks=attn_masks,
            key_padding_mask=mask_flatten,
            reference_points=patch_reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reg_branches=reg_branches,
            bbox_coder=bbox_coder,
            reference_angle=patch_reference_angle,
            angle_braches=angle_braches,
            angle_coder=angle_coder,
            **kwargs)

        patch_inter_references_out = patch_inter_references

        # patch_enc_outputs_sims与output_patch_feats是list
        return inter_states, init_reference_out, init_reference_angle_out,\
               inter_references_out, enc_outputs_class, \
               enc_outputs_coord_unact, enc_outputs_angle_cls, \
               patch_inter_states, patch_init_reference_out, patch_init_reference_angle_out, \
               patch_inter_references_out, patch_enc_outputs_class, \
               patch_enc_outputs_coord_unact, patch_enc_outputs_angle_cls, \
               patch_enc_outputs_embeds, output_patch_feats

