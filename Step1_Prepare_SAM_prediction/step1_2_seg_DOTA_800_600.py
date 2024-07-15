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
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    m_base = sorted_anns[0]['segmentation']
    img = np.zeros((m_base.shape[0], m_base.shape[1], 3))

    for a_id, ann in enumerate(sorted_anns):
        m = ann['segmentation']
        for i in range(3):
            img[:, :, i][m] = color_list[a_id][i]
    img = (img * 255).astype(np.uint8)
    return img

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
/gpfsdata/home/huangziyue/data/DOTA_800_600/train
-> ./SegAnything_RS_results/DOTA_800_600

"""
input_dir = '/gpfsdata/home/huangziyue/data/DOTA_800_600/train/images'
out_dir = './SegAnything_RS_results'
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
    print('Make %s' % out_dir)
out_dir = out_dir + '/DOTA_800_600_train'
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
    print('Make %s' % out_dir)

img_file_list = sorted(list(os.listdir(input_dir)))
# from 0 to 2, total 3
PartID = 2
file_parts = pklload('./SegAnything_RS_results/DOTA_800_600_train_partition.pkl')
result_dir = out_dir
result_file_list = sorted(list(os.listdir(result_dir)))
result_stems = [Path(f).stem for f in result_file_list]
file_list = file_parts[PartID]
stems = [Path(f).stem for f in file_list]
has_down = [s for s in stems if s in result_stems]
print('Part %d: %d / %d' % (PartID, len(has_down), len(file_list)))
file_list = [f for f in file_list if Path(f).stem not in result_stems]

pklsave(file_list, './file_list_tmp.pkl')

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device='cuda')
mask_generator = SamAutomaticMaskGenerator(sam)

for img_file in tqdm(file_list):
    img_pth = input_dir + '/' + img_file
    img_name = Path(img_pth).stem
    image = cv2.imread(input_dir + '/' + img_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    masks = mask_generator.generate(image)

    results = dict(
        input_dir=input_dir,
        img_pth=img_pth,
        img_name=img_name,
        masks=masks
    )
    zip_results = zipdata(results)
    pklsave(zip_results, out_dir + '/' + img_name + '.pkl', msg=False)

