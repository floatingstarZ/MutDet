# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import numpy as np
import torch.nn as nn

from mmrotate.models.builder import ROTATED_DETECTORS
#from mmdet.core import choose_best_match_batch, gt_mask_bp_obbs_list, choose_best_Rroi_batch
"""
DetReg复现
关键点：
0. 如何获取patches
    ./datasets/selfdet.py的L96~L98是获取patch的过程，先抠图，然后进行变换：
    transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.ToTensor(),
            # transforms.RandomHorizontalFlip(),  HorizontalFlip may cause the pretext too difficult, so we remove it
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    引入了一些数据增强？为啥获取embedding的过程还要数据增强，可以验证下
    
1. 如何获取embedding
    ./
    ./engine.py的L48使用swav_model进行特征提取，获得embedding
        ./models/backbone.py使用build_swav_backbone_old，提取完特征直接avg_pooling获得embedding
        ./models/swar_resnet50.py中可以找到swav的resnet，可以发现还包含了projection与l2norm

2. 其他超参数
    max_prop = 30, 最大采样数目应该不用管，SAM生成的质量高，而且使用的数据集物体也多
    f_emb = 512
    
"""

from mmcv.ops.roi_align_rotated import RoIAlignRotated
from mmrotate.core.bbox import rbbox2roi
from mmrotate.models.detectors import RotatedDETR, RotatedSingleStageDetector
from copy import deepcopy
from ctlib.os import *
import torch.nn.functional as F

@ROTATED_DETECTORS.register_module()
class DETRegRotatedDETRFast(RotatedSingleStageDetector):
    """
    通过预先读取特征来加快预训练速度
    """

    def __init__(self,
                 max_gt_num=1000,
                 *args,
                 **kwargs):
        super(DETRegRotatedDETRFast, self).__init__(*args, **kwargs)
        self.max_gt_num = max_gt_num

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      pca_feats,
                      gt_bboxes_ignore=None):
        # -------------强制三者相等，避免后边出现bug
        for i, boxes, labels, embeds, img_meta in zip(range(len(img_metas)),
                                                      gt_bboxes, gt_labels, pca_feats, img_metas):
            n_box, n_label, n_emb = len(boxes), len(labels), len(embeds)
            if not (n_box == n_label == n_emb):
                print('#' * 100)
                print(f'Image: {str(img_meta)} has not the same number of box, embed, and label: \n '
                      f'n_box: {n_box}, n_label: {n_label}, n_emb: {n_emb} \n '
                      f'The number has been fixed')
                print('#' * 100)

                n_min = min([n_box, n_label, n_emb])
                gt_bboxes[i] = boxes[:n_min]
                gt_labels[i] = labels[:n_min]
                pca_feats[i] = embeds[:n_min]
            if len(boxes) > self.max_gt_num:
                gt_bboxes[i] = boxes[:self.max_gt_num]
                gt_labels[i] = labels[:self.max_gt_num]
                pca_feats[i] = embeds[:self.max_gt_num]
                print('Out of Number GT |||||' * 4)

        # from mmdet base detector
        batch_input_shape = tuple(img[0].size()[-2:])
        for img_meta in img_metas:
            img_meta['batch_input_shape'] = batch_input_shape
        x = self.extract_feat(img)
        cost_matrix = np.asarray(x[0].cpu().detach())
        contain_nan = (True in np.isnan(cost_matrix))
        if contain_nan:
            a = 1
            print('Find!!!')
            for i in range(len(img_metas)):
                print('The image is', img_metas[i]['file_name'])
        obj_emb_list = pca_feats
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, obj_emb_list, gt_bboxes_ignore)
        return losses
