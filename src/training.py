from torch.utils.data import DataLoader, Dataset
from .models.voxnet import VoxNet
from functools import partial
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler


def train_one_epoch(
        training_loader,
        optimizer, 
        loss_c_fn, # loss for class
        loss_o_fn, # loss for orientation
        device : str = 'cpu'
                    ):


    for i, data in enumerate(training_loader):
        
        # Every data instance is (voxel grid, o_y, y) ordinal encoded
        voxels, o_y, y = data
        voxels = voxels.to(device)
        # we need one hot encoding for cross entropy loss
        o_y = F.one_hot(o_y,num_classes=N_ORIENTATION_CLASSES).float().to(device)
        y = F.one_hot(y,num_classes=N_CLASSES).float().to(device)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        y_pred, o_y_pred = model(voxels.float())

        # Compute the loss and its gradients
        loss_c = loss_c_fn(y_pred, y)
        loss_o = loss_o_fn(o_y_pred,o_y)

        total_loss = (1-config['gamma'])*loss_c + config['gamma']*loss_o
        total_loss.backward()
        
        # Adjust learning weights
        optimizer.step()

        # print statistics
        running_loss += total_loss.item()
        epoch_steps += 1
        if i % log_every == log_every-1:  # print every 2000 mini-batches
            print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,
                                            running_loss / epoch_steps))
            running_loss = 0.0


def train(
    training_set,
    validation_set,
    config,
    num_workers : int = 1, # for dataloader processes
    load_model_path = None,
    n_epochs : int = 10,
    log_every : int = 20
    ):

    """
    config must have 

    batch_size
    lr
    gamma
    """

    # Create data loaders for our datasets; shuffle for training, not for validation
    training_loader = DataLoader(training_set, batch_size=config['batch_size'], shuffle=True, num_workers=num_workers)
    validation_loader = DataLoader(validation_set, batch_size=config['batch_size'], shuffle=False, num_workers=num_workers)

    # ONLY WORKS FOR NPY DATASETS
    N_ORIENTATION_CLASSES=training_set.metadata.loc['npy']['orientation_class_id'].unique().size
    N_CLASSES=training_set.metadata.loc['npy']['label_int'].unique().size

    model = VoxNet()
    if load_model_path is not None: 
        print('Loading state...')
        print(load_model_path)
        model.load_state_dict(torch.load(load_model_path))

    # device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model.to(device)

    
    # LOSS FUNCTION

    loss_c_fn = torch.nn.CrossEntropyLoss()
    loss_o_fn = torch.nn.CrossEntropyLoss()



    # OPTIMIZER
    optimizer = torch.optim.Adam(model.parameters(),lr=config['lr'])

    # # learning rate scheduler, update LR every epoch
    # scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=GAMMA_LR)


    for epoch in range(n_epochs):
        running_loss = 0.0
        epoch_steps = 0

        ######################## TRAINING

        for i, data in enumerate(training_loader):
            # Every data instance is (voxel grid, o_y, y) ordinal encoded
            voxels, o_y, y = data
            voxels = voxels.to(device)
            # we need one hot encoding for cross entropy loss
            o_y = F.one_hot(o_y,num_classes=N_ORIENTATION_CLASSES).float().to(device)
            y = F.one_hot(y,num_classes=N_CLASSES).float().to(device)

            # Zero your gradients for every batch!
            optimizer.zero_grad()

            # Make predictions for this batch
            y_pred, o_y_pred = model(voxels.float())

            # Compute the loss and its gradients
            loss_c = loss_c_fn(y_pred, y)
            loss_o = loss_o_fn(o_y_pred,o_y)

            total_loss = (1-config['gamma'])*loss_c + config['gamma']*loss_o
            total_loss.backward()
            
            # Adjust learning weights
            optimizer.step()

            # print statistics
            running_loss += total_loss.item()
            epoch_steps += 1
            if i % log_every == log_every-1:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,
                                                running_loss / epoch_steps))
                running_loss = 0.0

        ######################## VALIDATION

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        with torch.no_grad():
            for i, v_data in enumerate(validation_loader):
                # get prediction and target
                v_voxels,v_o_y,v_y = v_data
                v_voxels = v_voxels.to(device)
                v_y_pred, v_o_y_pred = model(v_voxels.float())
                v_y_pred, v_o_y_pred = v_y_pred.to(device), v_o_y_pred.to(device)

                # one hot encoding of y from dataset
                v_o_y = F.one_hot(v_o_y,num_classes=N_ORIENTATION_CLASSES).float().to(device)
                v_y = F.one_hot(v_y,num_classes=N_CLASSES).float().to(device)

                # compute loss
                loss_c = loss_c_fn(v_y_pred, v_y)
                loss_o = loss_o_fn(v_o_y_pred,v_o_y)
                vtotal_loss = (1-config['gamma'])*loss_c + config['gamma']*loss_o

                # extract prediction from NN
                true_orientation=v_o_y.argmax(1)
                predicted_orientation=v_o_y_pred.argmax(1)
                correct_orientation_prediction = true_orientation == predicted_orientation

                true_label=v_y.argmax(1)
                predicted_label=v_y_pred.argmax(1)
                correct_label_prediction = true_label == predicted_label

                # HOW TO COMPUTE ACCURACY? 
                # consider correct only those with correct label AND orientation

                correct_combined = correct_label_prediction*correct_orientation_prediction
                accuracy = correct_combined.sum() / correct_combined.size()



                val_loss += vtotal_loss.numpy()
                val_steps+=1




        ################ CHECKPOINT

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=(val_loss / val_steps), accuracy=accuracy)

    print("Finished Training")
