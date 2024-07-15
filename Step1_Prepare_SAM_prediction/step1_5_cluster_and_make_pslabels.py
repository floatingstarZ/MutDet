import numpy as np
import os
# print(os.path.abspath(os.path.dirname(__file__))) #输出绝对路径
# 获取当前文件的绝对路径
Project_root = os.path.abspath(os.path.dirname(__file__))
# 去除当前文件的其他路径，保留项目根目录
backback_id = Project_root.find('M_Tools')
Project_root = Project_root[:backback_id-1]
import sys
sys.path.append(Project_root)

from ctlib.os import *
import os
import torch
import matplotlib.pyplot as plt
import random
from matplotlib import colors  # 注意！为了调整“色盘”，需要导入colors
from sklearn.manifold import TSNE
from tqdm import tqdm
import colorsys
from pathlib import Path

import faiss
import time
import numpy as np
from PIL import Image
from PIL import ImageFile
from scipy.sparse import csr_matrix, find
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from copy import deepcopy
from ctlib.rbox import *
from mmrotate.core.bbox import rbbox_overlaps
import torch.nn.functional as F

def read_ann_file(ann_pth):
    with open(ann_pth, 'r') as f:
        lines = list(f.readlines())
        lines = [l.strip().split(' ') for l in lines]
    if len(lines) == 0:
        return [], []
    gt_polys = []
    # 外接矩形框
    for l in lines:
        poly = [float(coord) for coord in l[:8]]
        poly = np.array(poly)
        gt_polys.append(poly)
    gt_polys = np.array(gt_polys)
    gt_rbboxes = poly2obb(gt_polys)
    return gt_polys, gt_rbboxes

# ------------------- load data and merge ---------------------

root = '/data/space2/huangziyue/DIOR_R_dota'
proposal_dir = root + '/train_val/SAM_PS_Labels'
pred_obj_embeds_dir = root + '/SAM_temp_output/11_17_Tool_DIOR_R_trainval_Feats_IQ_Scores'

out_cluster_label_dir = root + '/train_val/SAM_Cluster_Labels_13_24'
out_obj_embeds_dir = root + '/train_val/SAM_Obj_Embeds_13_24'
mkdir(out_cluster_label_dir)
mkdir(out_obj_embeds_dir)

obj_embed_names = [Path(f).stem for f in sorted(list(os.listdir(pred_obj_embeds_dir)))]
proposal_names = [Path(f).stem for f in sorted(list(os.listdir(proposal_dir)))]

###### 获取PCA信息
# ------------------- PCA -> Norm -> Sim(Ctrs) -> Norm -> get logits ---------------------
meta_pca_info = pklload('/data/space2/huangziyue/meta_pca_info.pkl')
A = meta_pca_info['pca_A']
b = meta_pca_info['pca_b']
ctrs = meta_pca_info['ctrs']
ctrs = np.array(ctrs)
row_sums = np.linalg.norm(ctrs, axis=1)
ctrs = ctrs / row_sums[:, np.newaxis]

###### 检查GT数量与标注数量的一致性，发现主要是这些标注中没有GT
print(f'n_obj_embed_file: {len(obj_embed_names)}, n_proposal_file: {len(proposal_names)}')
print(f'Differs: ', set(proposal_names) - set(obj_embed_names))
if len(set(proposal_names) - set(obj_embed_names)) > 0:
    raise Exception('You must remove the differs')

out_data = dict()
for f_name in tqdm(proposal_names):
    # if f_name != 'P1998__800__0___0':
    #     continue
    gt_ann_pth = proposal_dir + '/' + f_name + '.txt'
    obj_embed_pth = pred_obj_embeds_dir + '/' + f_name + '.pkl'
    out_ann_pth = out_cluster_label_dir + '/' + f_name + '.txt'
    out_obj_embed_pth = out_obj_embeds_dir + '/' + f_name + '.pkl'

    gt_polys, gt_rbboxes = read_ann_file(gt_ann_pth)
    if len(gt_polys) == 0:
        print(f'\nEmpty GT: {f_name}')
        out_data = dict(pca_feats=np.zeros([0, 256]))
        pklsave(out_data, out_obj_embed_pth, msg=False)
        continue

    if f_name not in obj_embed_names:
        print(f'\nMissing Log File: {f_name}')
        raise Exception('Missing Log File')

    log_data = pklload(obj_embed_pth, msg=False)

    gt_bs = deepcopy(log_data['gt_bboxex'][0])
    scale_factor = log_data['img_metas'][0]['scale_factor']
    gt_bs[:, :4] = gt_bs[:, :4] / scale_factor.reshape(1, -1)
    feats = log_data['patch_feats']
    if len(gt_bs) != len(gt_rbboxes):
        # 如果ROI数量少于GT数量的话，那就计算IoU为GT匹配上随机一个特征
        print(f'\nMismatch GT num, {f_name}: n_proposal {len(gt_rbboxes)}; n_obj_embeds: {len(gt_bs)}')
        ious = rbbox_overlaps(torch.Tensor(gt_rbboxes), torch.Tensor(gt_bs))
        m_ious, m_inds = torch.max(ious, dim=1)
        match_feats = feats[m_inds.numpy()]
        print(f'\nAdd feats, from {len(feats)} to {len(match_feats)}')
    else:
        match_feats = feats
    # 对特征进行PCA和归一化
    pca_feats = match_feats @ A.T + b
    row_sums = np.linalg.norm(pca_feats, axis=1)
    pca_feats = pca_feats / row_sums[:, np.newaxis]
    cos_sims = pca_feats @ ctrs.T
    labels = np.argmax(cos_sims, axis=1)

    with open(out_ann_pth, 'wt+') as f:
        for poly, label in zip(gt_polys, labels):
            poly = poly.clip(min=0)
            s = ''
            for p in poly.tolist():
                s += '%.1f ' % float(p)
            s += f'cluster_{int(label)+1} 0\n'
            f.write(s)
    
    out_data = dict(pca_feats=pca_feats)
    pklsave(out_data, out_obj_embed_pth, msg=False)
