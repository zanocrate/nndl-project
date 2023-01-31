metadata_path = '/home/ubuntu/nndl-project/data/modelnet10/metadata.csv'
orientation_classes_path = '/home/ubuntu/nndl-project/data/modelnet10/orientation_classes.csv'


########################################## SETTINGS

################## MODEL 

# resolution of the NxNxN grid
N=30

################## TRAINING


BATCH_SIZE=512
# NUMBER OF EPOCHS (= sweeps of the dataset)
EPOCHS = 50

# the relative weight of the class loss vs orientation loss: L = (1-GAMMA)*L_C + GAMMA*L_O
GAMMA = 0.5 

# LR for ADAM
OPTIMIZER_LR = 0.001
GAMMA_LR = 0.9 # for scheduler decay

# number of orientation classes and classes
N_ORIENTATION_CLASSES=40 # 10 for each category
N_CLASSES=10 # 10 different objects

# number of parallel processes used in data laoding
NUM_WORKERS_DATA = 4
# number of batches between logging of records
BATCHES_INTERVAL = 10

# LOAD PATH: if not None, load params_dict from path
LOAD_MODEL_PATH = None
# SAVE MODEL PATH
SAVE_MODEL_PATH = '/home/ubuntu/nndl-project/trained_models/'

# import ModelNetDataset class
# also imports torch, numpy as np and torch
from src.dataset import *
from torch.utils.data import DataLoader
import pandas as pd
from src.models.voxnet import * # imports VoxNet
from torch.nn.functional import one_hot
from torch.optim import lr_scheduler

# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# Create datasets for training & validation
training_set = ModelNetDataset(metadata_path,N,'train','npy')
validation_set = ModelNetDataset(metadata_path,N,'test','npy')


# Create data loaders for our datasets; shuffle for training, not for validation
training_loader = DataLoader(training_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS_DATA)
validation_loader = DataLoader(validation_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS_DATA)


# Report split sizes
print('Training set has {} instances'.format(len(training_set)))
print('Validation set has {} instances'.format(len(validation_set)))

# initialize model

model = VoxNet()
if LOAD_MODEL_PATH is not None: 
    print('Loading state...')
    print(LOAD_MODEL_PATH)
    model.load_state_dict(torch.load(LOAD_MODEL_PATH))


# LOSS FUNCTION

loss_c_fn = torch.nn.CrossEntropyLoss()
loss_o_fn = torch.nn.CrossEntropyLoss()

# OPTIMIZER

optimizer = torch.optim.Adam(model.parameters(),lr=OPTIMIZER_LR)
# learning rate scheduler, update LR every epoch
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=GAMMA_LR)


# one epoch

def train_one_epoch(epoch_index, tb_writer,log_every=BATCHES_INTERVAL):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):

        # Every data instance is (voxel grid, o_y, y) ordinal encoded
        voxels, o_y, y = data

        # we need one hot encoding for cross entropy loss
        o_y = one_hot(o_y,num_classes=N_ORIENTATION_CLASSES).float()
        y = one_hot(y,num_classes=N_CLASSES).float()

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        y_pred, o_y_pred = model(voxels.float())

        # Compute the loss and its gradients
        loss_c = loss_c_fn(y_pred, y)
        loss_o = loss_o_fn(o_y_pred,o_y)

        total_loss = (1-GAMMA)*loss_c + GAMMA*loss_o
        total_loss.backward()


        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += total_loss.item()

        if i % log_every == (log_every-1):
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

best_vloss = 1_000_000.

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch_number, writer)

    # We don't need gradients on to do reporting
    model.train(False)

    running_vloss = 0.0

    # VALIDATION
    with torch.no_grad():
        for i, v_data in enumerate(validation_loader):
            # get prediction and target
            v_voxels,v_o_y,v_y = v_data
            v_y_pred, v_o_y_pred = model(v_voxels.float())

            # one hot encoding of y
            v_o_y = one_hot(v_o_y,num_classes=N_ORIENTATION_CLASSES).float()
            v_y = one_hot(v_y,num_classes=N_CLASSES).float()

            loss_c = loss_c_fn(v_y_pred, v_y)
            loss_o = loss_o_fn(v_o_y_pred,v_o_y)

            vtotal_loss = (1-GAMMA)*loss_c + GAMMA*loss_o

            running_vloss += vtotal_loss

        avg_vloss = running_vloss / (i + 1)


    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars('Training/Validation Loss, Learning Rate',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss , 'Learning Rate' : scheduler.get_lr()},
                    epoch_number + 1)
    writer.flush()

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = SAVE_MODEL_PATH+'model_{}_{}'.format(timestamp, epoch_number)
        torch.save(model.state_dict(), model_path)

    epoch_number += 1
    # UPDATE LR
    scheduler.step()