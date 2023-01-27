# import ModelNetDataset class
# also imports torch, numpy as np and torch
from src.dataset import *
from torch.utils.data import DataLoader
import pandas as pd

# metadata for the ModelNet10 dataset
metadata_path = '/home/ubuntu/nndlproject/data/modelnet10/metadata.csv'
# resolution of the NxNxN grid
N=30

# Create datasets for training & validation
training_set = ModelNetDataset(metadata_path,N,'train')
validation_set = ModelNetDataset(metadata_path,N,'test')


BATCH_SIZE=30
# Create data loaders for our datasets; shuffle for training, not for validation
training_loader = DataLoader(training_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
validation_loader = DataLoader(validation_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# get int to label table
metadata = pd.read_csv(metadata_path)
int_to_label = dict(metadata.groupby(['label','label_str']).groups.keys())


# Report split sizes
print('Training set has {} instances'.format(len(training_set)))
print('Validation set has {} instances'.format(len(validation_set)))