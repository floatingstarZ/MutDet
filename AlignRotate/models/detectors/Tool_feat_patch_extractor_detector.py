# Copyright 2023 Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch

from mmdet.models import DETECTORS, BaseDetector, build_detector
from mmrotate.models.builder import ROTATED_DETECTORS
from mmrotate.models.detectors.base import RotatedBaseDetector
from mmrotate.models.detectors.oriented_rcnn import OrientedRCNN
from mmcv.ops.roi_align_rotated import RoIAlignRotated
from mmrotate.core.bbox import rbbox2roi
import numpy as np
import torch
from collections import OrderedDict
from pathlib import Path
from commonlibs.common_tools import *

def to_array(x):
    if type(x) == torch.Tensor:
        return x.detach().cpu().numpy()
    elif type(x) == np.array:
        return x
    elif type(x) == list:
        return [to_array(i) for i in x]
    elif type(x) == tuple:
        return [to_array(i) for i in x]
    elif type(x) == dict:
        return {k: to_array(v) for k, v in x.items()}
    elif type(x) == OrderedDict:
        d = OrderedDict()
        for k, v in x.items():
            d[k] = to_array(v)
        return d
    else:
        return x

@DETECTORS.register_module()
class ToolFeatPatchExtractorDetector(OrientedRCNN):
    def __init__(self,
                 out_dir,
                 bbox_roi_extractor=dict(
                     out_size=(224, 224),  # (224, 224),
                     spatial_scale=1.0,
                     sampling_ratio=2,
                     clockwise=True
                 ),
                 roi_scale_factor=1.5,
                 *args,
                 **kwargs):
        super(ToolFeatPatchExtractorDetector, self).__init__(*args, **kwargs)
        self.roi_align = RoIAlignRotated(**bbox_roi_extractor)
        # -对roi进行适当放大，获得一些Context信息
        self.roi_scale_factor = roi_scale_factor
        # -保留roi的宽高比，使得Crop的图像中，物体的宽高比信息得以保留
        # -保留ROI的宽高比通过以下方式实现
        # 1. 取ROI的w,h的最大值，并将短边设置为最大值，获得正方形ROI
        # 2. 由于部分物体宽高比过于悬殊，这种正方形ROI很容易引入过多的Context信息，
        #    扰乱特征的学习（可以使用padding解决该问题，但是也比较麻烦。）
        self.roi_keep_as_ratio = False
        self.pooling = torch.nn.Sequential(
            torch.nn.AvgPool2d((7, 7)),
            torch.nn.Flatten(1)
        )
        ############
        self.out_dir = out_dir
        mkdir(self.out_dir)


    def crop(self, img, rois):
        """

        :param img: Tensor, (B,C,H,W)
        :param rois: Tensor, (N, 6)
        :return: patches: Tensor, (N, C, H_roi, W_roi)
        """
        # -------- scale --------
        h_scale_factor, w_scale_factor = self.roi_scale_factor, self.roi_scale_factor
        new_rois = rois.clone()
        new_rois[:, 3] = w_scale_factor * new_rois[:, 3]
        new_rois[:, 4] = h_scale_factor * new_rois[:, 4]
        rois = new_rois
        # -------- keep ratio --------
        if self.roi_keep_as_ratio:
            new_rois = rois.clone()
            max_wh = torch.max(rois[:, 3:5], dim=1)[0]
            new_rois[:, 3] = max_wh
            new_rois[:, 4] = max_wh
            rois = new_rois
        patches = self.roi_align(img, rois)

        return patches

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        device = gt_bboxes[0].device
        x = torch.Tensor(img).to(device)
        rois = rbbox2roi(gt_bboxes)
        img_patches = self.crop(x, rois)
        # 使用最后的特征 -> Pooling，获得图像特征
        patch_feats = self.backbone(img_patches)[-1]
        patch_feats = self.pooling(patch_feats)

        filename = Path(img_metas[0]['filename']).stem
        out_data = dict(
            img_metas=img_metas,
            rois=to_array(rois),
            # img_patches=to_array(img_patches),
            patch_feats=to_array(patch_feats),
            gt_bboxex=to_array(gt_bboxes),
            gt_labels=to_array(gt_labels)
        )
        out_data_pth = self.out_dir + '/' + filename + '.pkl'
        pklsave(out_data, out_data_pth)

        losses = dict()

        loss_out = sum(
            x.view(-1)[0]
            for x in self.parameters()) * 0. + gt_bboxes[0].sum() * 0.
        losses['loss_out'] = loss_out
        return losses




