# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import platform
import shutil
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from typing import (Any, Callable, Dict, List, Optional, Tuple, Union,
                    no_type_check)
import mmcv
from mmcv.runner.base_runner import BaseRunner
from mmcv.runner.epoch_based_runner import EpochBasedRunner
from mmcv.runner.builder import RUNNERS
from mmcv.runner.checkpoint import save_checkpoint
from mmcv.runner.utils import get_host_info
import torch.nn as nn

@RUNNERS.register_module()
class MetaRemoveRunner(EpochBasedRunner):

    def __init__(self,
                 frozen_parameters=None,
                 *args, **kwargs,) -> None:
        super(MetaRemoveRunner, self).__init__(*args, **kwargs)

        trainable_params = 0
        all_param = 0
        # freeze parameters by prefix
        if frozen_parameters is not None:
            self.logger.info(f'Frozen parameters: {frozen_parameters}')
            for name, param in self.model.named_parameters():
                for frozen_prefix in frozen_parameters:
                    if frozen_prefix in name:
                        param.requires_grad = False
                if param.requires_grad:
                    self.logger.info(f'Training parameters: {name}')
                    trainable_params += param.numel()
                else:
                    self.logger.info(f'Frozen parameters: {name}')
                all_param += param.numel()
            self.logger.info(
                f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
            )
        else:
            self.logger.info(
                f"Pass Frozen"
            )

    def resume(self,
               checkpoint: str,
               resume_optimizer: bool = True,
               map_location: Union[str, Callable] = 'default') -> None:
        """Resume model from checkpoint.

        Args:
            filename (str): Accept local filepath, URL, ``torchvision://xxx``,
                ``open-mmlab://xxx``.
            resume_optimizer (bool): Whether to resume optimizer state.
                Defaults to True.
            resume_param_scheduler (bool): Whether to resume param scheduler
                state. Defaults to True.
            map_location (str or callable):A string or a callable function to
                specifying how to remap storage locations.
                Defaults to 'default'.
        """
        if map_location == 'default':
            if torch.cuda.is_available():
                device_id = torch.cuda.current_device()
                checkpoint = self.load_checkpoint(
                    checkpoint,
                    map_location=lambda storage, loc: storage.cuda(device_id))
            else:
                checkpoint = self.load_checkpoint(checkpoint)
        else:
            checkpoint = self.load_checkpoint(
                checkpoint, map_location=map_location)
        last_epoch = checkpoint['meta']['epoch']
        last_iter = checkpoint['meta']['iter']
        self.logger.info('SSEpochBasedRunner '
                         f'resumed epoch {last_epoch}, iter{last_iter}')
        self.logger.info('SSEpochBasedRunner '
                         'All Meta Data Removed')
        self.logger.info('SSEpochBasedRunner '
                         f'Start from epoch {self._epoch}, iter {self._iter}')

        print('MetaRemoveRunner '
                         f'resumed epoch {last_epoch}, iter{last_iter}')
        print('MetaRemoveRunner '
                         'All Meta Data Removed')
        print('MetaRemoveRunner '
                         f'Start from epoch {self._epoch}, iter {self._iter}')

        self.logger.info(f'resumed epoch: {self._epoch}, iter: {self._iter}')

