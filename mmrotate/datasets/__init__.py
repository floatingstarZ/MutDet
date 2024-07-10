# Copyright (c) OpenMMLab. All rights reserved.
from .builder import build_dataset  # noqa: F401, F403
from .dota import DOTADataset  # noqa: F401, F403
from .pipelines import *  # noqa: F401, F403
from .sar import SARDataset  # noqa: F401, F403
from .hrsc import HRSCDataset
from .dota15 import DOTA15Dataset
from .sku import SKUDataset
from .dior_r import DIOR_RDataset
from .OHD_SJTU import OHD_SJTUDataset
from .dota_custom import CustomDOTADataset
__all__ = ['SARDataset', 'DOTADataset', 'build_dataset', 'HRSCDataset',
           'DOTA15Dataset', 'SKUDataset', 'DIOR_RDataset', 'OHD_SJTUDataset',
           'CustomDOTADataset']
