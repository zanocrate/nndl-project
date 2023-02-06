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
import ray
from ray import tune
from ray.air import session
from ray.air.checkpoint import Checkpoint
from ray.tune.schedulers import ASHAScheduler


def train_loop_per_worker(config : dict, training_set = None, validation_set = None, n_epochs : int = 10):
    """
    Trainable loop to pass to tuner. 
    Expects training_set and validation_set Dataset objects, and a number of epochs.
    tune.with_parameters().

    Config keys:

    batch_size : int
    lr : float
    gamma : float

    """

    # define loader
    training_loader = DataLoader(training_set,batch_size=config['batch_size'],shuffle=True)
    validation_loader = DataLoader(validation_set,batch_size=config['batch_size'])
    
    # choose device
    # QUESTION: DOES THIS INTERFERE WITH THE RESOURCES RAY ALLOCATES TO THE TRAINING LOOP? HOW DO THEY INTERACT?
    device = "cpu"

    # init the net
    model = VoxNet().to(device)

    # loss functions
    # no parameters to tune here
    loss_c_fn = torch.nn.CrossEntropyLoss()
    loss_o_fn = torch.nn.CrossEntropyLoss()

    # optimizer
    # tune learning rate
    optimizer = torch.optim.Adam(model.parameters(),lr=config['lr'])    

    ############################################################### EPOCHS LOOP

    for epoch in range(n_epochs):

        ############################################# TRAINING

        for i, data in enumerate(training_loader):

            # Every data instance is (voxel grid, o_y, y) already hot encoded
            voxels, o_y, y = data

            # cast them all to float and transfer to device
            voxels = voxels.float().to(device)
            # o_y and y are already one hot encoded! but loss wants integer id
            o_y, y = o_y.float().argmax(1).to(device), y.float().argmax(1).to(device)
            
            # Zero your gradients for every batch!
            optimizer.zero_grad()

            # Make predictions for this batch
            o_y_pred, y_pred = model(voxels)

            # Compute the loss and its gradients
            loss_c = loss_c_fn(y_pred, y)
            loss_o = loss_o_fn(o_y_pred,o_y)

            total_loss = (1-config['gamma'])*loss_c + config['gamma']*loss_o
            total_loss.backward()
            
            # Adjust learning weights
            optimizer.step()

        # last training batch loss
        last_training_loss = total_loss.item()

        ############################################# VALIDATION

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
                correct += correct_combined.sum().item()
                total += len(correct_combined)

                val_loss += vtotal_loss.numpy()
                val_steps+=1

        ############################################# REPORT AND CHECKPOINT

                # Here we save a checkpoint. It is automatically registered with
        # Ray Tune and can be accessed through `session.get_checkpoint()`
        # API in future iterations.
        os.makedirs("voxnet", exist_ok=True)
        torch.save(
            (model.state_dict(), optimizer.state_dict()), "voxnet/checkpoint.pt")
        checkpoint = Checkpoint.from_directory("voxnet")
        session.report({"val_loss": (val_loss / val_steps),
                       "accuracy": (correct/total),
                       "train_loss" : last_training_loss}
                       , checkpoint=checkpoint)




if __name__ == '__main__':

    # BEFORE RUNNING THIS, INITIALIZE RAY CLUSTER WITH 
    # ray start --head 
    # on the head node

    # init ray to head
    ray.init('localhost:6379')
    
    # number of trials i guess
    num_samples = 20

    config = {
                "batch_size": tune.choice([8, 16, 32, 128, 512]),
                "lr": tune.loguniform(1e-5, 1e-1),
                "gamma": tune.choice([0.5])
            }

    scheduler = ASHAScheduler(
    # max_t=10, # isnt this the same as setting it in the trainable function? idk
    grace_period=3,
    reduction_factor=2)

    # initialize datasets to pass
    metadata_path = '/dataNfs/modelnet10/metadata.parquet'
    training_set = ModelNetDataset(metadata_path,split='train')
    validation_set = ModelNetDataset(metadata_path,split='test')

    # max epochs per loop
    n_epochs=10
        
    tuner = tune.Tuner(
        # wrap the training loop in this
        tune.with_parameters(
            train_loop_per_worker,
            # parameters to pass to the training loop
            training_set = training_set,
            validation_set = validation_set,
            n_epochs=n_epochs),
        # tune configurations
        tune_config=tune.TuneConfig(
            metric="val_loss",
            mode="min",
            scheduler=scheduler,
            num_samples=num_samples,
            # max_concurrent_trials=8 # maybe this is the number of processes it spawns?
        ),
        param_space=config,
    )

    results = tuner.fit()

    best_result = results.get_best_result("val_loss", "min")

    print("Best trial config: {}".format(best_result.config))
    print("Best trial final validation loss: {}".format(
        best_result.metrics["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_result.metrics["accuracy"]))

    # save results

    df = results.get_dataframe()

    df.to_csv('/home/ubuntu/nndlproject/ray_results.csv')
