from segment_anything import build_sam
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from tqdm import tqdm
import time
import json
import pickle as pkl
import traceback
from pathlib import Path
from copy import deepcopy

def mkdir(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
        print('Make dir: %s' % dir_path)


def pklsave(obj, file_path, msg=True):
    with open(file_path, 'wb+') as f:
        pkl.dump(obj, f)
        if msg:
            print('SAVE OBJ: %s' % file_path)

def jsonsave(obj, file_path, msg=True):
    with open(file_path, 'wt+') as f:
        json.dump(obj, f)
        if msg:
            print('SAVE JSON: %s' % file_path)

def pklload(file_path, msg=True):
    with open(file_path, 'rb') as f:
        if msg:
            print('LOAD OBJ: %s' % file_path)
        try:
            return pkl.load(f)
        except Exception:
            traceback.print_exc()

def splicing(images,
             row, column,
             resize_h=None,
             resize_w=None, gap=1, RGB=True):
    """
    将多个图片拼接成一个图片，按照row行，col列。
    图片在拼接前会统一成(height, width)的大小，如果为None的话，则不进行缩放
    :param images: images, cv2 type
    :param row: 图片行数
    :param column: 图片列数
    :param resize_h: 统一缩放高度
    :param resize_w:
    :param gap: 间隔的粗细
    :param RGB: 输入图像是RGB还是Gray
    :return: array
    """
    if row * column != len(images):
        raise Exception('Img number(%d) is not match with row * column(%d)'
                        % (len(images), row * column))
    if resize_h and resize_w:
        images = [cv2.resize(img, (resize_w, resize_h)) for img in images]
        max_height = resize_h
        max_width = resize_w
    else:
        max_height = max([img.shape[0] for img in images])
        max_width = max([img.shape[1] for img in images])

    # 创建空图像
    if RGB:
        target = np.zeros(((max_height + gap) * row, (max_width + gap) * column, 3), np.uint8)
    else:
        target = np.zeros(((max_height + gap) * row, (max_width + gap) * column), np.uint8)
    target.fill(200)
    # splicing images
    for i in range(row):
        for j in range(column):
            img = images[i * column + j].copy()
            h, w, c = img.shape
            target[i * (max_height + gap): i * (max_height + gap) + h,
            j * (max_width + gap): j * (gap + max_width) + w] = \
                img.copy()
    return target

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

def show_anns(anns, color_list, org_img):
    if len(anns) == 0:
        img = np.zeros((org_img.shape[0], org_img.shape[1], 3))
        img = (img * 255).astype(np.uint8)
        return img
    sorted_anns = sorted(anns, key=(lambda x: int(x['area'])), reverse=True)
    m_base = sorted_anns[0]['segmentation']
    img = np.zeros((m_base.shape[0], m_base.shape[1], 3))

    for a_id, ann in enumerate(sorted_anns):
        m = ann['segmentation']
        for i in range(3):
            img[:, :, i][m] = color_list[a_id][i]
    img = (img * 255).astype(np.uint8)
    return img

def mask2poly_single(binary_mask):
    binary_mask = binary_mask.astype(np.uint8)
    contours, hierarchy = cv2.findContours(binary_mask,
                                           cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)
    max_contour = max(contours, key=len)
    rect = cv2.minAreaRect(max_contour)
    poly = cv2.boxPoints(rect).flatten()
    return poly

def mask2poly(binary_mask_list):
    polys = map(mask2poly_single, binary_mask_list)
    return list(polys)

def mask2polys(anns):
    if len(anns) == 0:
        return []
    sorted_anns = sorted(anns, key=(lambda x: int(x['area'])), reverse=True)
    b_mask_list = [a['segmentation'] for a in sorted_anns]
    polys = mask2poly(b_mask_list)
    return polys

def zipdata(data):
    out_data = deepcopy(data)
    for mask in out_data['masks']:
        seg = mask['segmentation']
        H, W = seg.shape
        f_seg = seg.reshape(H*W)
        nonzero_inds = f_seg.nonzero()
        mask['segmentation'] = nonzero_inds
        mask['seg_shape'] = (H, W)
    return out_data

def unzipdata(data):
    out_data = deepcopy(data)
    for mask in out_data['masks']:
        seg = mask['segmentation']
        (H, W) = mask['seg_shape']
        bi_mask = np.zeros(H*W)
        bi_mask[seg] = 1
        bi_mask = bi_mask > 0
        bi_mask = bi_mask.reshape(H, W)
        mask['segmentation'] = bi_mask
    return out_data

"""
sam_vit_l_0b3195.pth
"""
sam_checkpoint = '../checkpoints/sam_vit_h_4b8939.pth'
device = "cuda"
model_type = "vit_h"
print('Meta info:')
print(sam_checkpoint, model_type)
print('#' * 100)
"""
DIOR:
/gpfsdata/home/huangziyue/data/DIOR/JPEGImages-trainval
-> ./SegAnything_RS_results/DIOR_trainval

"""
result_dir = '../M_Seg_RS_images/SegAnything_RS_results/DOTA_800_600_train'
poly_ann_dir = '/gpfsdata/home/huangziyue/data/DOTA_800_600/train/SAM_PS_Labels'

mkdir(poly_ann_dir)

result_file_list = sorted(list(os.listdir(result_dir)))

color_list = [np.random.random((1, 3)).tolist()[0]
              for i in range(1000)]
count = 0
for result_file in tqdm(result_file_list):
    count += 1
    # if count > 100:
    #     break

    result_pth = result_dir + '/' + result_file
    img_stem = Path(result_pth).stem
    result = pklload(result_pth, msg=False)
    result = unzipdata(result)

    polys = mask2polys(result['masks'])
    ann_file_pth = poly_ann_dir + '/%s.txt' % img_stem
    with open(ann_file_pth, 'wt+') as f:
        for poly in polys:
            poly = poly.clip(min=0)
            s = ''
            for p in poly.tolist():
                s += '%.1f ' % float(p)
            s += 'SAMObject 0\n'
            f.write(s)





