# main function at the end

from torch.utils.data import DataLoader, Dataset
from src.models.voxnet import VoxNet
from src.dataset import ModelNetDataset
from functools import partial
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from ray import tune
from ray.air import session
from ray.air.checkpoint import Checkpoint
from ray.tune.schedulers import ASHAScheduler



def train_one_epoch(
        model,
        training_loader,
        optimizer, 
        loss_c_fn, # loss for class
        loss_o_fn, # loss for orientation
        gamma,     # relative weights of the losses
        log_every : int = 10, # batches
        device : str = 'cpu'
                    ):

    """
    Args:
    ------
        model : nn.Module to train. Expected to return a tuple (orientation,class) already one hot encoded.
        training_loader : torch.utils.data.DataLoader pointing to the training subset
        optimizer : torch.optim optimizer
        loss_c_fn : torch.nn loss for the class output
        loss_o_fn :               for the orientation output
        gamma : float, relative weights for the losses
        log_every : int, number of batches between logging to tune
        device : str, device to cast the data to. must be the same as model

    Returns:
    -------
        last_loss : the training loss on the last log_every subset of batches
    """

    running_loss = 0
    last_loss = 0.

    for i, data in enumerate(training_loader):
        
        # Every data instance is (voxel grid, o_y, y) already hot encoded
        voxels, o_y, y = data

        # cast them all to float and transfer to device
        voxels = voxels.float().to(device)
        # o_y and y are already one hot encoded! but loss wants integer
        o_y, y = o_y.float().argmax(1).to(device), y.float().argmax(1).to(device)
        
        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        o_y_pred, y_pred = model(voxels)

        # Compute the loss and its gradients
        loss_c = loss_c_fn(y_pred, y)
        loss_o = loss_o_fn(o_y_pred,o_y)

        total_loss = (1-gamma)*loss_c + gamma*loss_o
        total_loss.backward()
        
        # Adjust learning weights
        optimizer.step()

        # Print statistics
        running_loss += total_loss.item()

        if i % log_every == log_every-1:  # print every 2000 mini-batches
            last_loss = running_loss / log_every # return the loss over the last log_every batches
            running_loss = 0.0 # reset the running_loss

    return last_loss




def trainable_modelnet10_voxels(
    config
    ):

    """
    config must have 

    batch_size
    lr
    gamma
    """

    n_epochs = 20

    load_model_path = None
    log_every = 100 # number of batches between logs in console


    #Datasets

    metadata_path = '/home/ubuntu/nndl-project/data/modelnet10/metadata.parquet'
    num_workers = 1

    training_set = ModelNetDataset(metadata_path,split='train')
    validation_set = ModelNetDataset(metadata_path,split='test')

    # Create data loaders for our datasets; shuffle for training, not for validation
    training_loader = DataLoader(training_set, batch_size=config['batch_size'], shuffle=True, num_workers=num_workers)
    validation_loader = DataLoader(validation_set, batch_size=config['batch_size'], shuffle=False, num_workers=num_workers)

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
    # no parameters to tune here
    loss_c_fn = torch.nn.CrossEntropyLoss()
    loss_o_fn = torch.nn.CrossEntropyLoss()



    # OPTIMIZER
    # tune learning rate
    optimizer = torch.optim.Adam(model.parameters(),lr=config['lr'])

    # # learning rate scheduler, update LR every epoch
    # scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=GAMMA_LR)


    for epoch in range(n_epochs):

        ######################## TRAINING
       
        last_training_batch_loss = train_one_epoch(model,training_loader,optimizer,loss_c_fn,loss_o_fn,config['gamma'],log_every,device)

        ######################## VALIDATION

        # Validation loss
        val_loss = 0.0
        val_steps = 0

        # Accuracy
        correct = 0
        total = 0

        with torch.no_grad():
            for i, v_data in enumerate(validation_loader):

                # get prediction and target
                v_voxels,v_o_y,v_y = v_data
                v_voxels = v_voxels.float().to(device)

                # we move these to INTEGER label because the loss accepts it; 
                # but in dataset we encode it beforehand! waste of time
                # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< FIX THIS IN DATASET SRC!
                v_y, v_o_y = v_y.float().to(device).argmax(1), v_o_y.float().to(device).argmax(1)
                v_o_y_pred, v_y_pred = model(v_voxels)
                v_o_y_pred, v_y_pred = v_o_y_pred.to(device), v_y_pred.to(device)


                # compute loss
                loss_c = loss_c_fn(v_y_pred, v_y)
                loss_o = loss_o_fn(v_o_y_pred,v_o_y)
                vtotal_loss = (1-config['gamma'])*loss_c + config['gamma']*loss_o

                # extract prediction from NN
                true_orientation=v_o_y
                predicted_orientation=v_o_y_pred.argmax(1)
                correct_orientation_prediction = true_orientation == predicted_orientation

                true_label=v_y
                predicted_label=v_y_pred.argmax(1)
                correct_label_prediction = true_label == predicted_label


                # HOW TO COMPUTE ACCURACY? 
                # consider correct only those with correct label AND orientation

                correct_combined = correct_label_prediction*correct_orientation_prediction
                correct += correct_combined.sum() 
                total += correct_combined.size()


                val_loss += vtotal_loss.numpy()
                val_steps+=1




        ################ CHECKPOINT

        # Here we save a checkpoint. It is automatically registered with
        # Ray Tune and can be accessed through `session.get_checkpoint()`
        # API in future iterations.
        os.makedirs("voxnet", exist_ok=True)
        torch.save(
            (model.state_dict(), optimizer.state_dict()), "voxnet/checkpoint.pt")
        checkpoint = Checkpoint.from_directory("voxnet")
        session.report({"val_loss": (val_loss / val_steps),
                       "accuracy": (correct/total),
                       "train_loss" : last_training_batch_loss}
                       , checkpoint=checkpoint)

    print("Finished Training")



if __name__ == '__main__':

    num_samples = 10 

    config = {
                "batch_size": tune.choice([8, 16, 32, 128, 512]),
                "lr": tune.loguniform(1e-4, 1e-1),
                "gamma": tune.quniform(0,1,0.2) # will sample from [0,0.2,0.4,0.6,0.8,1]
            }

    scheduler = ASHAScheduler(
    max_t=15, # isnt this the same as setting it in the trainable function? idk
    grace_period=1,
    reduction_factor=2)

        
    tuner = tune.Tuner(
        trainable_modelnet10_voxels,
        tune_config=tune.TuneConfig(
            metric="val_loss",
            mode="min",
            scheduler=scheduler,
            num_samples=num_samples,
        ),
        param_space=config,
    )

    results = tuner.fit()

    best_result = results.get_best_result("loss", "min")

    print("Best trial config: {}".format(best_result.config))
    print("Best trial final validation loss: {}".format(
        best_result.metrics["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_result.metrics["accuracy"]))

    # save results

    df = results.get_dataframe()

    df.to_csv('/home/ubuntu/nndl-project/runs/ray/results.csv')