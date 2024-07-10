# -*- coding: utf-8 -*-
_base_ = [
    './base_settings.py',
    './base_ars_detr_E2D2.py'
]

log_config = dict(
    interval=50
)

custom_imports = dict(
    imports=[
        'AlignRotate.models.detectors.DetReg_ars_detr_Fast',
        'AlignRotate.models.dense_heads.DetReg_dn_ars_detr_headv7',
        'AlignRotate.models.utils.dn_ars_rotated_transformer_template_matching',

        'AlignRotate.datasets.pipelines.loading',
        'AlignRotate.datasets.pipelines.formatting',
    ], allow_failed_imports=False)
frozen_parameters = ['swav_model',
                     'backbone']

"""
与DETReg_base的不同：
1. Backbone使用ImageNet预训练权重
    因为ImageNet预训练能更有效的检测遥感物体
2. 使用ImageNet权重模型来提取特征
    同上理由
3. 使用Layer4+Pooling+PCA+Normalize作为Patch特征，而不是Layer2+Pooling
    获得更多语义特征
4. L1损失按照单位球改为L2损失以契合特征空间

DetRegv2相比于DetReg的不同：
1. DN分支也使用了对齐损失
2. 只有最后一层参与reconstruction任务（参考论文）

"""
angle_version = 'le90'
dataset_type = 'CustomDOTADataset'
classes = [f'cluster_{i+1}' for i in range(256)]
num_classes = len(classes)
data_root = 'data/DOTA_800_600/'

model = dict(
    type='DETRegRotatedDETRFast',
    backbone=dict(
        frozen_stages=4),
    bbox_head=dict(
        type='DETReg_DNARSDeformableDETRHeadv7',
        obj_embedding_head='head',
        embed_loss=dict(
            type='MSELoss',
            loss_weight=1.0
        ),
        num_classes=num_classes,
        transformer=dict(
            type='DNARSRotatedDeformableDetrTransformerTemplateMatching',
            two_stage_num_proposals=300,
            with_template_matching=True,
            depth=3,
            num_heads=8,
            encoder=dict(
                num_layers=2),
            decoder=dict(
                num_layers=2)
        ),
    ),
)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    # ----------- 载入特征
    dict(type='LoadAnnotationsWithEx', with_bbox=True, with_ex=True, ex_keys=['pca_feats']),
    dict(type='RResize', img_scale=(1024, 1024)),
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version=angle_version),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundleWithEx', ex_keys=['pca_feats']),
    # ----------- 收集特征
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'pca_feats'])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'train/SAM_Cluster_Labels_10_31/',
        ex_ann_folder=data_root + 'train/SAM_PS_Labels_PCA_feats_256_10_30',
        pipeline=train_pipeline,
    ),
)
# do not evaluate during pre-training
evaluation = dict(interval=65535)














