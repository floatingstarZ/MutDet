import torch
from copy import deepcopy
from collections import OrderedDict
org_ckpt = torch.load('/opt/data/nfs/huangziyue/mmdet_checkpoints/ARS_DETR_Pretrained_Models/E3_AlignDet_epoch_36.pth', map_location='cpu')
new_ckpt = deepcopy(org_ckpt)
new_ckpt['state_dict'] = OrderedDict()
for k, v in org_ckpt['state_dict'].items():
    if 'target_backbone' in k:
        print('Pass:', k)
        continue
    if 'online' not in k:
        print('Pass:', k)
        continue

    new_k = deepcopy(k).replace('online_backbone.', '')
    new_ckpt['state_dict'][new_k] = v

torch.save(new_ckpt, '/opt/data/nfs/huangziyue/mmdet_checkpoints/ARS_DETR_Pretrained_Models/E3_AlignDet_epoch_36_CVT.pth')
