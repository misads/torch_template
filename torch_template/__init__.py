"""
    PyTorch Template

    Version: 0.0.1

    File Structure:
        .
        ├── network
        │     ├── base_model.py     :Define models, losses and parameter updating
        │     ├── norm.py           :Normalizations
        │     └── weights_init.py   :weights init
        │
        ├── dataloader/             :Define Dataloaders
        ├── model_zoo               :Commonly used models
        └── utils
              └── torch_utils.py    :PyTorch utils

    Author: xuhaoyu@tju.edu.cn

    License: MIT

"""

from .utils import torch_utils
from .dataloader.tta import OverlapTTA

from .model_zoo import *






