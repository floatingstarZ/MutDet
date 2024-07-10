# Copyright (c) OpenMMLab. All rights reserved.
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear, bias_init_with_prob, constant_init
from mmcv.runner import force_fp32, ModuleList

from mmdet.core import multi_apply, reduce_mean
from mmdet.core.bbox.transforms import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh
from mmdet.models.utils.transformer import inverse_sigmoid
from mmrotate.models.utils.rotated_transformer import obb2poly_tr
import numpy as np
from mmrotate.models.dense_heads.ars_detr_head import ARSDeformableDETRHead
from mmrotate.models.dense_heads.dn_ars_detr_head import DNARSDeformableDETRHead
from mmrotate.models.builder import ROTATED_HEADS
from mmdet.models.builder import build_loss

"""
相比于DetRegv2：
1. 增加了类别平衡损失以及Focal Loss
2. 类别信息获取
    2.1 dn_generator的输入为0,1 label
    2.2 匹配的loss_weight设置为0，不依据类别进行匹配
    2.3 cls_target在loss计算的时候重新设置为二分类目标

"""

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


@ROTATED_HEADS.register_module()
class DETReg_DNARSDeformableDETRHeadv7(DNARSDeformableDETRHead):
    """
    参考：DETReg-main/models/deformable_detr.py
    """

    def __init__(self,
                 *args,
                 object_embedding_loss=True,
                 obj_embedding_head='head',
                 embed_loss=dict(
                     type='L1Loss',
                     loss_weight=1.0
                 ),
                 erase_gt_label=False,

                 **kwargs):
        super(DETReg_DNARSDeformableDETRHeadv7, self).__init__(
            *args, **kwargs)
        hidden_dim = 256
        self.object_embedding_loss = object_embedding_loss
        assert self.object_embedding_loss
        if obj_embedding_head == 'intermediate':
            last_channel_size = 2*hidden_dim
        elif obj_embedding_head == 'head':
            last_channel_size = hidden_dim   # hidden_dim//2
        else:
            raise Exception(obj_embedding_head)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # -------- feature_embed 分支不参与encoder
        num_pred = self.transformer.decoder.num_layers

        self.feature_embed = MLP(hidden_dim, hidden_dim, last_channel_size, 2)

        self.obj_embed_dim = last_channel_size
        self.embed_loss = build_loss(embed_loss)
        self.embed_loss_type = embed_loss['type']
        self.erase_gt_label = erase_gt_label


    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      ####################
                      obj_embeddings=None,
                      ####################
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        assert proposal_cfg is None, '"proposal_cfg" must be None'
        if self.erase_gt_label:
            # ------- 擦除label的类别信息
            gt_labels = [l.new_zeros(l.shape) for l in gt_labels]
        # ------- 擦除label的类别信息，获得的ps_labels用来送入dn_generator生成query
        ps_labels = gt_labels # [l.new_zeros(l.shape) for l in gt_labels]
        dn_label_query, dn_bbox_query, attn_mask, dn_meta = \
            self.dn_generator(gt_bboxes, ps_labels,
                              self.label_embedding, img_metas)
        outs_main, outs_patch, patch_enc_outputs_embeds, output_patch_feats = self(x, obj_embeddings, img_metas, dn_label_query, dn_bbox_query, attn_mask)
        output_patch_feats = [torch.nn.functional.normalize(x, dim=-1) for x in output_patch_feats]

        # ---------------- loss main
        if gt_labels is None:
            loss_inputs = outs_main + (gt_bboxes, img_metas, dn_meta)
        else:
            # -------- 引入ps_gt_labels只是为了二分匹配
            loss_inputs = outs_main + (gt_bboxes, gt_labels, img_metas, dn_meta)
        losses = self.loss(*loss_inputs, obj_embed_list=output_patch_feats, gt_bboxes_ignore=gt_bboxes_ignore)

        # ---------------- loss main
        if gt_labels is None:
            patch_loss_inputs = outs_patch + (gt_bboxes, img_metas, dn_meta)
        else:
            # -------- 引入ps_gt_labels只是为了二分匹配
            patch_loss_inputs = outs_patch + (gt_bboxes, gt_labels, img_metas, dn_meta)
        patch_losses = self.loss(*patch_loss_inputs, obj_embed_list=output_patch_feats, gt_bboxes_ignore=gt_bboxes_ignore,
                                 with_encoder_embed_loss=True, enc_pred_feats=patch_enc_outputs_embeds)
        for k, v in patch_losses.items():
            losses['p_' + k] = v

        return losses


    def forward(self,
                mlvl_feats,
                obj_embeddings,
                img_metas,
                dn_label_query=None,
                dn_bbox_query=None,
                attn_mask=None):
        # print(dn_label_query.shape)
        # print(dn_bbox_query.shape)
        batch_size = mlvl_feats[0].size(0)
        input_img_h, input_img_w = img_metas[0]['batch_input_shape']
        img_masks = mlvl_feats[0].new_ones(
            (batch_size, input_img_h, input_img_w))
        for img_id in range(batch_size):
            img_h, img_w, _ = img_metas[img_id]['img_shape']
            img_masks[img_id, :img_h, :img_w] = 0
        mlvl_masks = []
        mlvl_positional_encodings = []
        spatial_shapes = []
        for feat in mlvl_feats:
            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            mlvl_masks.append(
                F.interpolate(img_masks[None],
                              size=feat.shape[-2:]).to(torch.bool).squeeze(0))
            mlvl_positional_encodings.append(
                self.positional_encoding(mlvl_masks[-1]))
        query_embeds = None
        """
        patch_xxx是patch分支输出的结果，包括预测分数等
        patch_enc_outputs_sims是与各个patch的相似度分数
        output_patch_feats是经过融合后的patch特征
        """
        hs, init_reference, init_reference_angle, inter_references, \
        enc_outputs_class, enc_outputs_coord, enc_outputs_angle_cls, \
        patch_hs, patch_init_reference, patch_init_reference_angle, patch_inter_references, \
        patch_enc_outputs_class, patch_enc_outputs_coord, patch_enc_outputs_angle_cls, \
        patch_enc_outputs_embeds, output_patch_feats \
            = self.transformer(
            mlvl_feats,
            mlvl_masks,
            query_embeds,
            mlvl_positional_encodings,
            dn_label_query,
            dn_bbox_query,
            bbox_coder = self.bbox_coder,
            reg_branches=self.reg_branches if self.with_box_refine else None,  # noqa:E501
            cls_branches=self.cls_branches if self.as_two_stage else None,  # noqa:E501
            angle_braches=self.angle_branches if self.as_two_stage else None,
            angle_coder=self.angle_coder,
            attn_masks=attn_mask,
            img_metas=img_metas,
            patch_feats=obj_embeddings
        )

        outs_main = self.decode_forward(hs, init_reference, init_reference_angle, inter_references,
                                        enc_outputs_class, enc_outputs_coord, enc_outputs_angle_cls,
                                        img_metas, dn_label_query, dn_bbox_query, attn_mask)

        outs_patch = self.decode_forward(patch_hs, patch_init_reference, patch_init_reference_angle, patch_inter_references,
                                        patch_enc_outputs_class, patch_enc_outputs_coord, patch_enc_outputs_angle_cls,
                                        img_metas, dn_label_query, dn_bbox_query, attn_mask)
        return outs_main, outs_patch, patch_enc_outputs_embeds, output_patch_feats


    def decode_forward(self, hs, init_reference, init_reference_angle, inter_references,
                       enc_outputs_class, enc_outputs_coord, enc_outputs_angle_cls,
                       img_metas,
                       dn_label_query=None,
                       dn_bbox_query=None,
                       attn_mask=None
                       ):
        hs = hs.permute(0, 2, 1, 3)

        if dn_label_query is not None and dn_label_query.size(1) == 0:
            # NOTE: If there is no target in the image, the parameters of
            # label_embedding won't be used in producing loss, which raises
            # RuntimeError when using distributed mode.
            hs[0] += self.label_embedding.weight[0, 0] * 0.0

        outputs_classes = []
        outputs_coords = []
        outputs_angles = []
        outputs_angle_clses = []
        outputs_anlge = init_reference_angle
        outputs_angles.append(outputs_anlge)
        #
        output_pred_features = []
        #
        for lvl in range(hs.shape[0]):
            reference = inter_references[lvl]
            reference = inverse_sigmoid(reference)
            ######
            pred_feat = self.feature_embed(hs[lvl])
            output_pred_features.append(pred_feat)
            ######
            outputs_class = self.cls_branches[lvl](hs[lvl])
            outputs_anlge_cls = self.angle_branches[lvl](hs[lvl])
            outputs_anlge = self.angle_coders[lvl].decode(outputs_anlge_cls)
            tmp = self.reg_branches[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp[..., :4] += reference
            elif reference.shape[-1] == 8:
                tmp = obb2poly_tr(tmp)
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            #------------------------
            # outputs_coord = []
            # for i in range(batch_size):
            #     outputs_coord.append(poly2obb(tmp[i].sigmoid(), 'oc'))
            # outputs_coord = torch.stack(outputs_coord)
            #------------------------
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
            outputs_angles.append(outputs_anlge)
            outputs_angle_clses.append(outputs_anlge_cls)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)
        outputs_anlges = torch.stack(outputs_angles)
        outputs_angle_clses = torch.stack(outputs_angle_clses)
        #################
        # --------------尽管这里计算了所有Layer的Feature embed，但是只有最后一层特征参与了loss计算
        output_pred_features = torch.stack(output_pred_features)
        #################
        return outputs_classes, outputs_coords, outputs_anlges, outputs_angle_clses,\
            enc_outputs_class, enc_outputs_coord.sigmoid(), enc_outputs_angle_cls, \
               output_pred_features


    @staticmethod
    def extract_dn_outputs(all_cls_scores, all_bbox_preds, all_angle_preds,
                           all_angle_cls_preds, all_pred_features, dn_meta):
        """
        分离DN的Score与正常query的Score
        最终all_pred_features不参与训练
        """

        if dn_meta is not None:
            denoising_cls_scores = all_cls_scores[:, :, :dn_meta['pad_size'], :]
            denoising_bbox_preds = all_bbox_preds[:, :, :dn_meta['pad_size'], :]
            denoising_angle_preds = all_angle_preds[:, :, :dn_meta['pad_size']]
            denoising_angle_cls_preds = all_angle_cls_preds[:, :, :dn_meta['pad_size'], :]
            denoising_pred_features = all_pred_features[:, :, :dn_meta['pad_size'], :]

            matching_cls_scores = all_cls_scores[:, :, dn_meta['pad_size']:, :]
            matching_bbox_preds = all_bbox_preds[:, :, dn_meta['pad_size']:, :]
            matching_angle_preds = all_angle_preds[:, :, dn_meta['pad_size']:]
            matching_angle_cls_preds = all_angle_cls_preds[:, :, dn_meta['pad_size']:, :]
            matching_pred_features = all_pred_features[:, :, dn_meta['pad_size']:, :]

        else:
            denoising_cls_scores = None
            denoising_bbox_preds = None
            denoising_angle_preds = None
            denoising_angle_cls_preds = None
            matching_cls_scores = all_cls_scores
            matching_bbox_preds = all_bbox_preds
            matching_angle_preds = all_angle_preds
            matching_angle_cls_preds = all_angle_cls_preds
            matching_pred_features = all_pred_features
        return (matching_cls_scores, matching_bbox_preds, matching_angle_preds, matching_angle_cls_preds,
                matching_pred_features,
                denoising_cls_scores, denoising_bbox_preds, denoising_angle_preds, denoising_angle_cls_preds,
                denoising_pred_features)

    @force_fp32(apply_to=('all_cls_scores', 'all_bbox_preds', 'all_angle_preds'))
    def loss(self,
             all_cls_scores,
             all_bbox_preds,
             all_angle_preds,
             all_angle_cls_preds,
             enc_cls_scores,
             enc_bbox_preds,
             enc_angle_cls_preds,
             #################
             all_pred_features,
             #################
             gt_bboxes_list,
             gt_labels_list,
             img_metas,
             dn_meta=None,
             gt_bboxes_ignore=None,
             #######################
             obj_embed_list=None,
             with_encoder_embed_loss=False,
             enc_pred_feats=None
             #######################
             ):
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'
        loss_dict = dict()

        # extract denoising and matching part of outputs
        all_cls_scores, all_bbox_preds, all_angle_preds, all_angle_cls_preds, all_pred_features, \
        dn_cls_scores, dn_bbox_preds, dn_angle_preds, dn_angle_cls_preds, dn_pred_features  = \
            self.extract_dn_outputs(all_cls_scores, all_bbox_preds, all_angle_preds,
                                    all_angle_cls_preds, all_pred_features, dn_meta)

        num_dec_layers = len(all_cls_scores)
        all_gt_rbboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]

        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]
        img_metas_list = [img_metas for _ in range(num_dec_layers)]
        # ------------------- 加入loss_single，对每一层都计算损失，但最终使用的只有最后一层
        all_obj_embed_list = [obj_embed_list for _ in range(num_dec_layers)]
        losses_cls, losses_bbox, losses_iou, losses_angle, losses_obj_embed, pos_sims, neg_sims = multi_apply(
            self.loss_single_with_embed,
            all_cls_scores, all_bbox_preds, all_angle_preds[:-1], all_angle_cls_preds,
            all_pred_features,
            self.angle_coders[:-1], all_gt_rbboxes_list, all_gt_labels_list, img_metas_list,
            all_gt_bboxes_ignore_list, all_obj_embed_list)

        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            binary_labels_list = [
                torch.zeros_like(gt_labels_list[i])
                for i in range(len(img_metas))
            ]
            if not with_encoder_embed_loss:
                # 在检测模式下，Encoder输出的是似物性分数
                # Encoder只需要估计似物性分数，并给所有潜在物体打出尽可能高的分数
                # 不需要进行对比损失来特化Proposal的选择性
                enc_losses_cls, enc_losses_bbox, enc_losses_iou, enc_losses_angle = \
                    self.loss_single(enc_cls_scores, enc_bbox_preds,
                                     torch.zeros((enc_angle_cls_preds.shape[0], enc_angle_cls_preds.shape[1]),
                                                 dtype=torch.float32).to(enc_angle_cls_preds.device),
                                     enc_angle_cls_preds, self.angle_coders[-1],
                                     gt_bboxes_list, binary_labels_list,
                                     img_metas, gt_bboxes_ignore)
                loss_dict['enc_loss_cls'] = enc_losses_cls
                loss_dict['enc_loss_bbox'] = enc_losses_bbox
                loss_dict['enc_loss_piou'] = enc_losses_iou
                loss_dict['enc_loss_angle'] = enc_losses_angle
            else:
                # 在template matching模式下，Encoder输出的则是相似性分数
                # Encoder需要根据与patch特征的相似性进行筛选，强调选择性
                # 因此需要使用embedding损失进行监督
                # enc_cls_scores -> 依然是似物性分数
                # enc_cls_feats -> 特征
                enc_losses_cls, enc_losses_bbox, enc_losses_iou, enc_losses_angle, enc_losses_embed, enc_pos_sims, enc_neg_sims = \
                    self.loss_single_with_embed(enc_cls_scores, enc_bbox_preds,
                                     torch.zeros((enc_angle_cls_preds.shape[0], enc_angle_cls_preds.shape[1]),
                                                 dtype=torch.float32).to(enc_angle_cls_preds.device),
                                     enc_angle_cls_preds,
                                    enc_pred_feats,
                                                self.angle_coders[-1],
                                     gt_bboxes_list, binary_labels_list,
                                     img_metas, gt_bboxes_ignore, obj_embed_list)
                loss_dict['enc_loss_cls'] = enc_losses_cls
                loss_dict['enc_loss_bbox'] = enc_losses_bbox
                loss_dict['enc_loss_piou'] = enc_losses_iou
                loss_dict['enc_loss_angle'] = enc_losses_angle
                loss_dict['enc_loss_embed'] = enc_losses_embed
                loss_dict['enc_pos_sims'] = enc_pos_sims.detach()
                loss_dict['enc_neg_sims'] = enc_neg_sims.detach()



        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_piou'] = losses_iou[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]
        loss_dict['loss_angle'] = losses_angle[-1]
        loss_dict['loss_embed'] = losses_obj_embed[-1]
        loss_dict['pos_sim'] = pos_sims[-1]
        loss_dict['neg_sim'] = neg_sims[-1]

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i, loss_iou_i, loss_angle_i, loss_embed_i in zip(losses_cls[:-1],
                                                                                   losses_bbox[:-1],
                                                                                   losses_iou[:-1],
                                                                                   losses_angle[:-1],
                                                                                   losses_obj_embed[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_iou'] = loss_iou_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            loss_dict[f'd{num_dec_layer}.loss_angle'] = loss_angle_i
            # # ----------- 只有最后一层计算loss_embed
            # loss_dict[f'd{num_dec_layer}.loss_embed'] = loss_embed_i

            num_dec_layer += 1

        if dn_cls_scores is not None:
            # calculate denoising loss from all decoder layers
            dn_meta = [dn_meta for _ in img_metas]
            dn_losses_cls, dn_losses_bbox, dn_losses_iou, dn_losses_angle, dn_losses_obj_embed, dn_pos_sims, dn_neg_sims = self.loss_dn(
                dn_cls_scores, dn_bbox_preds, dn_angle_preds, dn_angle_cls_preds, dn_pred_features,
                gt_bboxes_list, gt_labels_list, obj_embed_list, img_metas, dn_meta)
            # collate denoising loss
            loss_dict['dn_loss_cls'] = dn_losses_cls[-1]
            loss_dict['dn_loss_bbox'] = dn_losses_bbox[-1]
            loss_dict['dn_loss_iou'] = dn_losses_iou[-1]
            loss_dict['dn_loss_angle'] = dn_losses_angle[-1]
            loss_dict['dn_loss_embed'] = dn_losses_obj_embed[-1]
            loss_dict['dn_pos_sims'] = dn_pos_sims[-1]
            loss_dict['dn_neg_sims'] = dn_neg_sims[-1]


            num_dec_layer = 0
            for loss_cls_i, loss_bbox_i, loss_iou_i, loss_angle_i, loss_embed_i in zip(
                    dn_losses_cls[:-1], dn_losses_bbox[:-1],
                    dn_losses_iou[:-1], dn_losses_angle[:-1], dn_losses_obj_embed):
                loss_dict[f'd{num_dec_layer}.dn_loss_cls'] = loss_cls_i
                loss_dict[f'd{num_dec_layer}.dn_loss_bbox'] = loss_bbox_i
                loss_dict[f'd{num_dec_layer}.dn_loss_iou'] = loss_iou_i
                loss_dict[f'd{num_dec_layer}.dn_loss_angle'] = loss_angle_i
                # # ----------- 只有最后一层计算loss_embed
                # loss_dict[f'd{num_dec_layer}.dn_loss_embed'] = loss_embed_i
                num_dec_layer += 1

        return loss_dict

    def loss_single_with_embed(self,
                    cls_scores,
                    bbox_preds,
                    angle_preds,
                    angle_cls,
                    ######
                    pred_features,
                    ######
                    angle_coder,
                    gt_bboxes_list,
                    gt_labels_list,
                    img_metas,
                    gt_bboxes_ignore_list=None,
                    ##########
                    obj_embed_list=None
                    ##########
                    ):
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        angle_cls_list = [angle_cls[i] for i in range(num_imgs)]
        angle_preds_list = [angle_preds[i] for i in range(num_imgs)]
        # ------------ 同样的带入obj_embed_list
        cls_reg_targets = self.get_targets_with_embed(cls_scores_list, bbox_preds_list,
                                                      angle_preds_list, angle_cls_list,
                                                      angle_coder, gt_bboxes_list, gt_labels_list,
                                                      img_metas, gt_bboxes_ignore_list, obj_embed_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         angle_list, angle_weights_list, embed_targets_list, embed_target_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)
        angle_targets = torch.cat(angle_list, 0)
        angle_weights = torch.cat(angle_weights_list, 0)
        embed_targets = torch.cat(embed_targets_list, 0)
        embed_weights = torch.cat(embed_target_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
                         num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # DETR regress the relative position of boxes (cxcywh) in the image,
        # thus the learning target is normalized by the image size. So here
        # we need to re-scale them for calculating IoU loss
        bbox_preds = bbox_preds.reshape(-1, 4)
        rbboxes_gt = bbox_targets
        rbboxes = bbox_preds
        loss_iou = self.loss_iou(
            bbox_cxcywh_to_xyxy(rbboxes),
            bbox_cxcywh_to_xyxy(rbboxes_gt),
            weight=bbox_weights,
            avg_factor=num_total_pos)
        # # regression L1 loss
        loss_bbox = self.loss_bbox(
            rbboxes, rbboxes_gt, bbox_weights, avg_factor=num_total_pos)

        angle_cls = angle_cls.reshape(-1, self.coding_len)
        loss_angle = self.loss_angle(
            angle_cls, angle_targets, angle_weights, avg_factor=num_total_pos * self.coding_len
        )
        # ----------- 计算embedding loss
        pos_inds = embed_weights > 1e-6
        pos_embed_target = embed_targets[pos_inds]
        pos_pred_feats = pred_features.reshape(-1, self.obj_embed_dim)[pos_inds]
        # 使用MSELoss，因为都是单位球特征

        temperature = 0.2
        q = torch.nn.functional.normalize(pos_pred_feats, dim=1)
        k = torch.nn.functional.normalize(pos_embed_target, dim=1)

        logits_12 = torch.einsum('nc,mc->nm', [q, k]) / temperature
        logits_21 = torch.einsum('nc,mc->nm', [k, q]) / temperature

        # 这里已经进行了一一匹配
        labels = torch.arange(len(pos_pred_feats), dtype=torch.long).cuda()

        loss = 2 * temperature * nn.CrossEntropyLoss()(logits_12, labels) + \
               2 * temperature * nn.CrossEntropyLoss()(logits_21, labels)
        loss_obj_embed = loss

        sims = (logits_21 + logits_12) * temperature / 2
        pos_sims = torch.mean(torch.diag(sims)).detach()
        neg_sims = torch.mean(sims - torch.diag(sims).diag()).detach()

        return loss_cls, loss_bbox, loss_iou, loss_angle, loss_obj_embed, pos_sims, neg_sims

    def get_targets_with_embed(self,
                    cls_scores_list,
                    bbox_preds_list,
                    angle_preds_list,
                    angle_cls_list,
                    angle_coder,
                    gt_bboxes_list,
                    gt_labels_list,
                    img_metas,
                    gt_bboxes_ignore_list=None,
                    obj_embed_list=None):
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]
        angle_coder = [angle_coder for _ in range(num_imgs)]

        # -------------- 带入obj_embed_list
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         angle_list, angle_weights_list, pos_inds_list, neg_inds_list,
         embed_targets_list, embed_target_weights_list) = multi_apply(
            self._get_target_single_with_embed,
            cls_scores_list, bbox_preds_list, angle_preds_list, angle_cls_list, angle_coder,
            gt_bboxes_list, gt_labels_list, img_metas, gt_bboxes_ignore_list, obj_embed_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))

        return (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
                angle_list, angle_weights_list, embed_targets_list, embed_target_weights_list,
                num_total_pos, num_total_neg)


    def _get_target_single_with_embed(self,
                           cls_score,
                           bbox_pred,
                           angle_pred,
                           angle_cls,
                           angle_coder,
                           gt_bboxes,
                           gt_labels,
                           img_meta,
                           gt_bboxes_ignore=None,
                           ######################
                           obj_embeddings=None
                           ################
                        ):
        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        aspect_ratios = None
        if angle_coder.window == 'aspect_ratio':
            aspect_ratios = torch.stack((gt_bboxes[:, 2] / gt_bboxes[:, 3], gt_bboxes[:, 3] / gt_bboxes[:, 2]), dim=1)
            aspect_ratios = torch.max(aspect_ratios, dim=1).values.view(-1, 1)
            gt_angles = angle_coder.encode(gt_bboxes[:, -1].view(-1, 1), aspect_ratios)
        else:
            gt_angles = angle_coder.encode(gt_bboxes[:, -1].view(-1, 1))

        assign_result = self.assigner.assign(bbox_cxcywh_to_xyxy(bbox_pred), cls_score, angle_cls,
                                             bbox_cxcywh_to_xyxy(gt_bboxes[:, :-1]), gt_labels, gt_angles,
                                             img_meta, gt_bboxes_ignore,
                                             aspect_ratios if self.aspect_ratio_weighting else None)
        sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_bboxes.new_full((num_bboxes,),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0

        angles = torch.zeros_like(angle_cls)
        angle_weights = torch.zeros_like(angle_cls)
        angle_weights[pos_inds] = 1.0
        if self.aspect_ratio_weighting:
            angle_weights[pos_inds] *= 2 * aspect_ratios[sampling_result.pos_assigned_gt_inds] / \
                                       (aspect_ratios[sampling_result.pos_assigned_gt_inds] + 1)

        img_h, img_w, _ = img_meta['img_shape']

        factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                       img_h]).unsqueeze(0)
        pos_gt_bboxes_normalized = sampling_result.pos_gt_bboxes[:, :-1] / factor
        pos_gt_bboxes_targets = pos_gt_bboxes_normalized
        bbox_targets[pos_inds] = pos_gt_bboxes_targets
        angles[pos_inds] = gt_angles[sampling_result.pos_assigned_gt_inds]

        # ------------------------ 设置embed的target
        embed_targets = bbox_targets.new_zeros([num_bboxes, self.obj_embed_dim])
        embed_targets[pos_inds] = obj_embeddings[sampling_result.pos_assigned_gt_inds]
        embed_target_weights = gt_bboxes.new_zeros(num_bboxes)
        embed_target_weights[pos_inds] = 1.0


        return (labels, label_weights, bbox_targets, bbox_weights, angles, angle_weights,
                pos_inds, neg_inds, embed_targets, embed_target_weights)

    def loss_dn(self, dn_cls_scores, dn_bbox_preds, dn_angle_preds, dn_angle_cls_preds, dn_pred_features,
                gt_bboxes_list, gt_labels_list, obj_embed_list, img_metas, dn_meta):
        num_dec_layers = len(dn_cls_scores)
        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_obj_embed_list = [obj_embed_list for _ in range(num_dec_layers)]
        img_metas_list = [img_metas for _ in range(num_dec_layers)]
        dn_meta_list = [dn_meta for _ in range(num_dec_layers)]
        return multi_apply(self.loss_dn_single, dn_cls_scores, dn_bbox_preds, dn_angle_preds[:-1],
                           dn_angle_cls_preds, dn_pred_features, self.angle_coders[:-1], all_gt_bboxes_list,
                           all_gt_labels_list, all_obj_embed_list, img_metas_list, dn_meta_list)
    def loss_dn_single(self, dn_cls_scores, dn_bbox_preds, dn_angle_preds, dn_angle_cls_preds, dn_pred_features,
                       angle_coder, gt_bboxes_list, gt_labels_list, obj_embed_list, img_metas, dn_meta):
        num_imgs = dn_cls_scores.size(0)
        bbox_preds_list = [dn_bbox_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_dn_target(bbox_preds_list, angle_coder, gt_bboxes_list,
                                             gt_labels_list, obj_embed_list, img_metas,
                                             dn_meta)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         angles_list, angle_weights_list, embed_targets_list, embed_targets_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)
        angle_targets = torch.cat(angles_list, 0)
        angle_weights = torch.cat(angle_weights_list, 0)
        embed_targets = torch.cat(embed_targets_list, 0)
        embed_weights = torch.cat(embed_targets_weights_list, 0)

        # classification loss
        cls_scores = dn_cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = \
            num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        if len(cls_scores) > 0:
            loss_cls = self.loss_cls(
                cls_scores, labels, label_weights, avg_factor=cls_avg_factor)
        else:
            loss_cls = torch.zeros(  # TODO: How to better return zero loss
                1,
                dtype=cls_scores.dtype,
                device=cls_scores.device)

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # construct factors used for rescale bboxes
        factors = []
        for img_meta, bbox_pred in zip(img_metas, dn_bbox_preds):
            img_h, img_w, _ = img_meta['img_shape']
            factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                           img_h]).unsqueeze(0).repeat(
                                               bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors, 0)

        # DETR regress the relative position of boxes (cxcywh) in the image,
        # thus the learning target is normalized by the image size. So here
        # we need to re-scale them for calculating IoU loss
        bbox_preds = dn_bbox_preds.reshape(-1, 4)
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.loss_iou(
            bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos)

        # regression L1 loss
        loss_bbox = self.loss_bbox(
            bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos)

        angle_cls = dn_angle_cls_preds.reshape(-1, self.coding_len)
        loss_angle = self.loss_angle(
            angle_cls, angle_targets, angle_weights, avg_factor=num_total_pos * self.coding_len
        )
        # ----------- 计算embedding loss
        pos_inds = embed_weights > 1e-6
        pos_embed_target = embed_targets[pos_inds]
        pos_pred_feats = dn_pred_features.reshape(-1, self.obj_embed_dim)[pos_inds]
        # 使用MSELoss，则进行归一化，因为都是单位球特征
        if self.embed_loss_type == 'MSELoss':
            pos_pred_feats = torch.nn.functional.normalize(pos_pred_feats, dim=1)

        temperature = 0.2
        n_feats = sum([len(g) for g in gt_bboxes_list])
        img_feats = pos_embed_target[:n_feats]
        query_feats = pos_pred_feats[:n_feats]

        q = torch.nn.functional.normalize(query_feats, dim=1)
        k = torch.nn.functional.normalize(img_feats, dim=1)

        logits_12 = torch.einsum('nc,mc->nm', [q, k]) / temperature
        logits_21 = torch.einsum('nc,mc->nm', [k, q]) / temperature

        labels = torch.arange(n_feats, dtype=torch.long).cuda()

        loss = 2 * temperature * nn.CrossEntropyLoss()(logits_12, labels) + \
               2 * temperature * nn.CrossEntropyLoss()(logits_21, labels)
        loss_obj_embed = loss

        sims = (logits_21 + logits_12) * temperature / 2
        pos_sims = torch.mean(torch.diag(sims)).detach()
        neg_sims = torch.mean(sims - torch.diag(sims).diag()).detach()

        return loss_cls, loss_bbox, loss_iou, loss_angle, loss_obj_embed, \
               pos_sims, neg_sims

    def get_dn_target(self, dn_bbox_preds_list, angle_coder, gt_bboxes_list, gt_labels_list, obj_embed_list,
                      img_metas, dn_meta):
        angle_coder = [angle_coder for _ in range(len(dn_bbox_preds_list))]
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         angles_list, angle_weights_list, embed_targets_list, embed_targets_weights_list, pos_inds_list, neg_inds_list) = \
            multi_apply(self._get_dn_target_single,
                                      dn_bbox_preds_list, angle_coder, gt_bboxes_list,
                                      gt_labels_list, obj_embed_list, img_metas, dn_meta)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
                angles_list, angle_weights_list,
                embed_targets_list, embed_targets_weights_list,
                num_total_pos, num_total_neg)

    def _get_dn_target_single(self, dn_bbox_pred, angle_coder, gt_bboxes, gt_labels, obj_embeddings,
                              img_meta, dn_meta):
        num_groups = dn_meta['num_dn_group']
        pad_size = dn_meta['pad_size']
        assert pad_size % num_groups == 0
        single_pad = pad_size // num_groups
        num_bboxes = dn_bbox_pred.size(0)

        if len(gt_labels) > 0:
            t = torch.range(0, len(gt_labels) - 1).long().cuda()
            t = t.unsqueeze(0).repeat(num_groups, 1)
            pos_assigned_gt_inds = t.flatten()
            pos_inds = (torch.tensor(range(num_groups)) *
                        single_pad).long().cuda().unsqueeze(1) + t
            pos_inds = pos_inds.flatten()
        else:
            pos_inds = pos_assigned_gt_inds = torch.tensor([]).long().cuda()
        neg_inds = pos_inds + single_pad // 2

        # label targets
        labels = gt_bboxes.new_full((num_bboxes, ),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(dn_bbox_pred)
        bbox_weights = torch.zeros_like(dn_bbox_pred)
        bbox_weights[pos_inds] = 1.0
        img_h, img_w, _ = img_meta['img_shape']

        factor = dn_bbox_pred.new_tensor([img_w, img_h, img_w,
                                          img_h]).unsqueeze(0)
        gt_bboxes_normalized = gt_bboxes[..., :4] / factor
        bbox_targets[pos_inds] = gt_bboxes_normalized.repeat([num_groups, 1])

        angles = gt_bboxes.new_full((num_bboxes, self.coding_len), 0)
        angle_weights = torch.zeros_like(angles)
        angle_weights[pos_inds] = 1.0
        if angle_coder.window == 'aspect_ratio':
            aspect_ratios = torch.stack((gt_bboxes[pos_assigned_gt_inds, 2] / gt_bboxes[pos_assigned_gt_inds, 3],
                                         gt_bboxes[pos_assigned_gt_inds, 3] / gt_bboxes[pos_assigned_gt_inds, 2]), dim=1)
            aspect_ratios = torch.max(aspect_ratios, dim=1).values.view(-1, 1)
            gt_angle_clses = angle_coder.encode(gt_bboxes[pos_assigned_gt_inds, -1].view(-1, 1), aspect_ratios)
        else:
            gt_angle_clses = angle_coder.encode(gt_bboxes[pos_assigned_gt_inds, -1].view(-1, 1))
        angles[pos_inds] = gt_angle_clses
        if self.aspect_ratio_weighting:
            angle_weights[pos_inds] *= 2 * aspect_ratios / (aspect_ratios + 1)

        # ------------------------ 设置embed的target
        embed_targets = bbox_targets.new_zeros([num_bboxes, self.obj_embed_dim])
        embed_targets[pos_inds] = obj_embeddings[pos_assigned_gt_inds]
        embed_target_weights = gt_bboxes.new_zeros(num_bboxes)
        embed_target_weights[pos_inds] = 1.0


        return (labels, label_weights, bbox_targets, bbox_weights, angles, angle_weights,
                embed_targets, embed_target_weights,
                pos_inds, neg_inds)


