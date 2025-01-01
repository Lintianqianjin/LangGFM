import pandas as pd
import shutil, os
import os.path as osp
import torch
import warnings

from ogb.nodeproppred import PygNodePropPredDataset

class CustomPygNodePropPredDataset(PygNodePropPredDataset):

    def __init__(self, name, root = 'dataset', transform=None, pre_transform=None, meta_dict = None):

       with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)  # suppress torch.load warning of `weight_only`
            super().__init__(name=name, root=root, transform=transform, pre_transform=pre_transform, meta_dict=meta_dict)