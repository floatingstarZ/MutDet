# -*- coding: utf-8 -*-
_base_ = [
    './base_settings_DIOR.py',
    './base_ars_detr.py'
]
load_from = './checkpoints/ARS_DETR_Pretrained_Models/E4_MulDet_epoch_36.pth'

num_classes = 20
model = dict(
    bbox_head=dict(
        num_classes=num_classes,
        transformer=dict(
            encoder=dict(
                num_layers=6),
            decoder=dict(
                num_layers=6)
        ),
    ),
)
