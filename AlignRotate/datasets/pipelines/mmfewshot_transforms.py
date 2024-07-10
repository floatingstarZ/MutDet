# Copyright (c) OpenMMLab. All rights reserved.
import copy
import math
from typing import Dict, List, Tuple

import mmcv
import numpy as np
import torch

from mmdet.datasets import PIPELINES
from mmrotate.core.bbox.transforms import obb2xyxy

@PIPELINES.register_module()
class GenerateOutSideMask:
    """Resize support image and generate a mask.

    Args:
        target_size (tuple[int, int]): Crop and resize to target size.
            Default: (224, 224).
    """

    def __init__(self, target_size: Tuple[int] = (224, 224)) -> None:
        self.target_size = target_size

    def _resize_bboxes(self, results: Dict) -> None:
        """Resize bounding boxes with ``results['scale_factor']``."""
        # ---- from mmrotate
        for key in results.get('bbox_fields', []):
            bboxes = results[key]
            orig_shape = bboxes.shape
            bboxes = bboxes.reshape((-1, 5))
            w_scale, h_scale, _, _ = results['scale_factor']
            bboxes[:, 0] *= w_scale
            bboxes[:, 1] *= h_scale
            bboxes[:, 2:4] *= np.sqrt(w_scale * h_scale)
            results[key] = bboxes.reshape(orig_shape)

    def _resize_img(self, results: Dict) -> None:
        """Resize images with ``results['scale']``."""
        for key in results.get('img_fields', ['img']):
            img, w_scale, h_scale = mmcv.imresize(
                results[key],
                self.target_size,
                return_scale=True,
                backend='cv2')
            results[key] = img
            scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                    dtype=np.float32)
            results['img_shape'] = img.shape
            # in case that there is no padding
            results['pad_shape'] = img.shape
            results['scale_factor'] = scale_factor

    def _generate_mask(self, results: Dict) -> Dict:
        mask = np.zeros(self.target_size, dtype=np.float32)
        gt_bboxes = results['gt_bboxes'][0]
        out_hbb = obb2xyxy(torch.Tensor(gt_bboxes)[None, ...], version='le90')[0].numpy()
        mask[int(out_hbb[1]):int(out_hbb[3]),
             int(out_hbb[0]):int(out_hbb[2])] = 1
        results['img'] = np.concatenate(
            [results['img'], np.expand_dims(mask, axis=2)], axis=2)
        results['img_shape'] = results['img'].shape
        return results

    def __call__(self, results: Dict) -> Dict:
        """Call function to flip bounding boxes, masks, semantic segmentation
        maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized images with additional dimension of bbox mask.
        """
        self._resize_img(results)
        self._resize_bboxes(results)
        self._generate_mask(results)

        return results

    def __repr__(self) -> str:
        return self.__class__.__name__ + \
               f'(num_context_pixels={self.num_context_pixels},' \
               f' target_size={self.target_size})'

@PIPELINES.register_module()
class GenerateCropImage:
    """Crop the instance

    Args:
        target_size (tuple[int, int]): Crop and resize to target size.
            Default: (224, 224).
    """

    def __init__(self, target_size: Tuple[int] = (224, 224)) -> None:
        self.target_size = target_size

    def _resize_img(self, results: Dict) -> None:
        """Resize images with ``results['scale']``."""
        for key in results.get('img_fields', ['img']):
            img, w_scale, h_scale = mmcv.imresize(
                results[key],
                self.target_size,
                return_scale=True,
                backend='cv2')
            results[key] = img
            scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                    dtype=np.float32)
            results['img_shape'] = img.shape
            # in case that there is no padding
            results['pad_shape'] = img.shape
            results['scale_factor'] = scale_factor

    def _generate_mask(self, results: Dict) -> Dict:
        # ---- mask全0，失去作用
        mask = np.zeros(self.target_size, dtype=np.float32)
        results['img'] = np.concatenate(
            [results['img'], np.expand_dims(mask, axis=2)], axis=2)
        results['img_shape'] = results['img'].shape
        return results

    def _crop_instance(self, results: Dict) -> Dict:
        # ---- mask全0，失去作用
        for key in results.get('img_fields', ['img']):
            gt_bboxes = results['gt_bboxes'][0]
            out_hbb = obb2xyxy(torch.Tensor(gt_bboxes)[None, ...], version='le90')[0].numpy()
            x1, y1, x2, y2 = int(out_hbb[0]), int(out_hbb[1]), \
                             int(out_hbb[2]), int(out_hbb[3])
            w, h = x2 - x1, y2 - y1
            img_h, img_w, _ = results[key].shape
            x1 = min(max(x1, 0), img_h)
            x2 = min(max(x2, 0), img_h)
            y1 = min(max(y1, 0), img_w)
            y2 = min(max(y2, 0), img_w)

            if w <= 0 or h <= 0:
                x1, y1, x2, y2 = 0, 0, img_h, img_w
            crop_img = results[key][y1:y2, x1:x2, :]
            # print(results[key].shape, x1, y1, x2, y2, crop_img.shape)
            results[key] = crop_img
            results['img_shape'] = crop_img.shape
            # in case that there is no padding
            results['pad_shape'] = crop_img.shape

    def __call__(self, results: Dict) -> Dict:
        """Call function to flip bounding boxes, masks, semantic segmentation
        maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized images with additional dimension of bbox mask.
        """
        self._crop_instance(results)
        self._resize_img(results)
        # self._generate_mask(results)

        return results

    def __repr__(self) -> str:
        return self.__class__.__name__ + \
               f'(num_context_pixels={self.num_context_pixels},' \
               f' target_size={self.target_size})'