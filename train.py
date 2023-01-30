# import ModelNetDataset class
# also imports torch, numpy as np and torch
from src.dataset import *
from torch.utils.data import DataLoader
import pandas as pd
from src.models.voxnet import * # imports VoxNet
from torch.nn.functional import one_hot

# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# metadata for the ModelNet10 dataset
metadata_path = '/home/ubuntu/nndl-project/data/modelnet10/metadata.csv'
# resolution of the NxNxN grid
N=30
# path for the orientation classes
orientation_classes_path = '/home/ubuntu/nndl-project/data/modelnet10/orientation_classes.csv'

# Create datasets for training & validation
training_set = ModelNetDataset(metadata_path,N,'train',orientation_classes_path=orientation_classes_path)
validation_set = ModelNetDataset(metadata_path,N,'test',orientation_classes_path=orientation_classes_path)


BATCH_SIZE=32
# Create data loaders for our datasets; shuffle for training, not for validation
training_loader = DataLoader(training_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
validation_loader = DataLoader(validation_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# get int to label table
metadata = pd.read_csv(metadata_path)
int_to_label = dict(metadata.groupby(['label','label_str']).groups.keys())


# Report split sizes
print('Training set has {} instances'.format(len(training_set)))
print('Validation set has {} instances'.format(len(validation_set)))

# initialize model

model = VoxNet()

# LOSS FUNCTION
GAMMA = 0.5 # the relative weight of the class loss vs orientation loss: L = (1-GAMMA)*L_C + GAMMA*L_O

loss_c_fn = torch.nn.CrossEntropyLoss()
loss_o_fn = torch.nn.CrossEntropyLoss()

# OPTIMIZER

optimizer = torch.optim.Adam(model.parameters(),lr=0.001)


# one epoch

def train_one_epoch(epoch_index, tb_writer,log_every=10):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + orientation_class + label
        voxels, orientation,label = data

        o_c_onehot = one_hot(orientation,num_classes=40) # 4 classes for each of the 10 objects
        label_onehot = one_hot(label,num_classes=10)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        label_pred, o_c_pred = model(voxels.float())

        # Compute the loss and its gradients
        loss_c = loss_c_fn(label_pred, label_onehot.float())
        loss_o = loss_o_fn(o_c_pred,o_c_onehot.float())

        total_loss = (1-GAMMA)*loss_c + GAMMA*loss_o
        total_loss.backward()


        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += total_loss.item()
        if i % log_every == 0:
            last_loss = running_loss / log_every # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss


# EPOCHS LOOP


timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/vox_trainer_{}'.format(timestamp)) # save logs in runs/ directory
epoch_number = 0

EPOCHS = 5

best_vloss = 1_000_000.

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch_number, writer)

    # We don't need gradients on to do reporting
    model.train(False)

    running_vloss = 0.0
    for i, vdata in enumerate(validation_loader):
        vvoxels,vorientation,vlabel = vdata
        v_c, v_o = model(vvoxels.float())
        o_c_onehot = one_hot(vorientation,num_classes=40) # 4 classes for each of the 10 objects
        label_onehot = one_hot(vlabel,num_classes=10)
        loss_c = loss_c_fn(v_c, label_onehot.float())
        loss_o = loss_o_fn(v_o,o_c_onehot.float())

        vtotal_loss = (1-GAMMA)*loss_c + GAMMA*loss_o

        running_vloss += vtotal_loss

    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch_number + 1)
    writer.flush()

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = 'model_{}_{}'.format(timestamp, epoch_number)
        torch.save(model.state_dict(), model_path)

    epoch_number += 1