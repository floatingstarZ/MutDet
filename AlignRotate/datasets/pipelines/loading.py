# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmcv
import numpy as np
import pycocotools.mask as maskUtils

from mmdet.core import BitmapMasks, PolygonMasks
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines.loading import LoadAnnotations

try:
    from panopticapi.utils import rgb2id
except ImportError:
    rgb2id = None

@PIPELINES.register_module()
class LoadAnnotationsWithEx(LoadAnnotations):

    def __init__(self,
                 with_ex,
                 ex_keys,
                 *args,
                 **kwargs):
        self.with_ex = with_ex
        self.ex_keys = ex_keys
        super(LoadAnnotationsWithEx, self).__init__(*args, **kwargs)

    def _load_ex_anns(self, results):
        for k in self.ex_keys:
            results[k] = results['ann_info'][k].copy()

        return results

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded bounding box, label, mask and
                semantic segmentation annotations.
        """

        if self.with_bbox:
            results = self._load_bboxes(results)
            if results is None:
                return None
        if self.with_label:
            results = self._load_labels(results)
        if self.with_mask:
            results = self._load_masks(results)
        if self.with_seg:
            results = self._load_semantic_seg(results)
        if self.with_ex:
            results = self._load_ex_anns(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(with_bbox={self.with_bbox}, '
        repr_str += f'with_label={self.with_label}, '
        repr_str += f'with_mask={self.with_mask}, '
        repr_str += f'with_seg={self.with_seg}, '
        repr_str += f'poly2mask={self.poly2mask}, '
        repr_str += f'poly2mask={self.file_client_args})'
        repr_str += f'with_ex={self.with_ex}, '
        return repr_str

