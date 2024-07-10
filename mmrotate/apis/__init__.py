# Copyright (c) OpenMMLab. All rights reserved.
from .train import train_detector

__all__ = ['train_detector']

###################
from .meta_remove_runner import MetaRemoveRunner
__all__.extend(['MetaRemoveRunner'])




