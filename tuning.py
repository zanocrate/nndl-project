from src.training import *
# # # # # ALSO IMPORTS THE FOLLOWING
# from torch.utils.data import DataLoader, Dataset
# from .models.voxnet import VoxNet
# from functools import partial
# import numpy as np
# import os
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# and the train_one_epoch function

from src.dataset import ModelNetDataset

import torch.optim as optim

from ray import tune


def trainable(config):

    metadata_path = '/home/ubuntu/nndl-project/data/modelnet10/metadata.parquet'
    num_workers = 1


    training_set = ModelNetDataset(metadata_path,split='train')
    validation_set = ModelNetDataset(metadata_path,split='test')
    
    # Create data loaders for our datasets; shuffle for training, not for validation
    training_loader = DataLoader(training_set, batch_size=config['batch_size'], shuffle=True, num_workers=num_workers)
    validation_loader = DataLoader(validation_set, batch_size=config['batch_size'], shuffle=False, num_workers=num_workers)

    # initialize model
    model = VoxNet()

    # initialize optimizer
    optimizer = optim.Adam(model.parameters(),lr = config['lr'])

    train_one_epoch()