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

input_dir = '/gpfsdata/home/huangziyue/data/DOTA_800_600/train/images'
img_file_list = sorted(list(os.listdir(input_dir)))
result_dir = './SegAnything_RS_results/DOTA_800_600_train'
result_file_list = sorted(list(os.listdir(result_dir)))
nPart = 3
Size = len(img_file_list) // nPart

file_parts = []
for i in range(nPart):
    s = Size * i
    if i == (nPart - 1):
        end = len(img_file_list)
    else:
        end = Size * (i + 1)
    file_parts.append(img_file_list[s: end])
# pklsave(file_parts, './SegAnything_RS_results/DOTA_800_600_train_partition.pkl')
# check process
result_stems = [Path(f).stem for f in result_file_list]

for i, file_list in enumerate(file_parts):
    stems = [Path(f).stem for f in file_list]
    has_down = [s for s in stems if s in result_stems]
    print('Part %d: %d / %d' % (i, len(has_down), len(file_list)))


