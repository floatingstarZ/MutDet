num_query = 900
max_per_img = 500
pretrained = \
    'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/' \
    'swin_tiny_patch4_window7_224.pth'  # noqa
model = dict(
    type='ARSDETR',
    backbone=dict(
        type='SwinTransformerNew',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(1, 2, 3),
        with_cp=False,
        convert_weights=True,
        frozen_stages=1,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)
    ),
    neck=dict(
        type='ChannelMapper',
        in_channels=[192, 384, 768],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
    bbox_head=dict(
        type='DNARSDeformableDETRHead',
        num_query=num_query,
        num_classes=15,
        in_channels=2048,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=True,
        angle_coder=dict(
            type='CSLCoder',
            angle_version='le90',
            omega=1,
            window='aspect_ratio',
            radius=6,
            normalize=True),
        transformer=dict(
            type='DNARSRotatedDeformableDetrTransformer',
            two_stage_num_proposals=num_query,
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=2,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention', embed_dims=256,
                        dropout=0.0), # 0.1 -> 0.0
                    feedforward_channels=2048, # 1024 -> 2048
                    ffn_dropout=0.0, # 0.1 -> 0.0
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            decoder=dict(
                type='DNARSDeformableDetrTransformerDecoder',
                num_layers=2,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.0), # 0.1 -> 0.0
                        dict(
                            type='RotatedMultiScaleDeformableAttention',
                            embed_dims=256,
                            dropout=0.0) # 0.1 -> 0.0
                    ],
                    feedforward_channels=2048, # 1024 -> 2048
                    ffn_dropout=0.0, # 0.1 -> 0.0
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))
        ),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=128,
            normalize=True,
            offset=-0.5),
        bbox_coder=dict(
            type='DeltaXYWHAOBBoxCoder',
            angle_range='le90',
            norm_factor=None,
            edge_swap=True,
            proj_xy=True,
            target_means=(0.0, 0.0, 0.0, 0.0, 0.0),
            target_stds=(1, 1, 1, 1, 1)),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=2.0),
        loss_iou=dict(type='GIoULoss', loss_weight=5.0),
        reg_decoded_bbox=True,
        loss_angle=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=2.0),
        rotate_deform_attn=True,
        aspect_ratio_weighting=True,
        dn_cfg=dict(
            type='DnQueryGenerator',
            noise_scale=dict(label=0.5, box=0.4, angle=0.02),
            group_cfg=dict(dynamic=True, num_groups=None,
                           num_dn_queries=100))
    ),
    train_cfg=dict(
        assigner=dict(
            type='ARS_HungarianAssigner',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBoxL1Cost', weight=2.0, box_format='xywh'),
            iou_cost=dict(type='IoUCost', iou_mode='giou', weight=5.0),
            angle_cost=dict(type='CrossEntropyLossCost', weight=2.0))),
    test_cfg=dict(max_per_img=max_per_img),
)