import pandas as pd
import shutil, os
import os.path as osp
import torch
import warnings

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.serialization")

from ogb.nodeproppred import PygNodePropPredDataset
from ogb.linkproppred import PygLinkPropPredDataset


class CustomPygNodePropPredDataset(PygNodePropPredDataset):
    def __init__(self, name, root = 'dataset', transform=None, pre_transform=None, meta_dict = None):

       with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)  # suppress torch.load warning of `weight_only`
            super().__init__(name=name, root=root, transform=transform, pre_transform=pre_transform, meta_dict=meta_dict)


class CustomPygLinkPropPredDataset(PygLinkPropPredDataset):

    def __init__(self, name, root = 'dataset', transform=None, pre_transform=None, meta_dict = None):

       with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)  # suppress torch.load warning of `weight_only`
            super().__init__(name=name, root=root, transform=transform, pre_transform=pre_transform, meta_dict=meta_dict)
        
    
    def get_edge_split(self, split_type=None):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)  # suppress torch.load warning of `weight_only`
            res = super().get_edge_split(split_type=split_type)
        return res
