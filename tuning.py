from src.training import * # imports train function
from src.dataset import *

metadata_path = '/home/ubuntu/nndl-project/data/modelnet10/metadata.csv'
N=30
num_workers=2
n_epochs = 5 # per grid search
num_samples = 10 # number of runs?

training_set = ModelNetDataset(metadata_path,N,'train','npy')
validation_set = ModelNetDataset(metadata_path,N,'test','npy')

# here we define the search spaces
config = {
    "gamma": tune.sample_from(lambda _: 2**np.random.randint(2, 9)),
    "lr": tune.loguniform(1e-4, 1e-1),
    "batch_size": tune.uniform(0.,1.)
}




scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=n_epochs,
        grace_period=1,
        reduction_factor=2)

reporter = CLIReporter(
    # parameter_columns=["l1", "l2", "lr", "batch_size"],
    metric_columns=["loss", "accuracy", "training_iteration"])


result = tune.run(

    # fill arguments for trainable function
    partial(
        train, 
        training_set=training_set,
        validation_set=validation_set,
        num_workers=num_workers,
        n_epochs=n_epochs,
        load_model_path=None,
        log_every=10
        ),
    # resources_per_trial={"cpu": 8, "gpu": gpus_per_trial},
    config=config,
    num_samples=num_samples,
    scheduler=scheduler,
    progress_reporter=reporter,
    checkpoint_at_end=True)

best_trial = result.get_best_trial("loss", "min", "last")
print("Best trial config: {}".format(best_trial.config))
print("Best trial final validation loss: {}".format(
    best_trial.last_result["loss"]))
print("Best trial final validation accuracy: {}".format(
    best_trial.last_result["accuracy"]))

best_trained_model = VoxNet()
device = "cpu"

best_trained_model.to(device)

best_checkpoint_dir = best_trial.checkpoint.value
model_state, optimizer_state = torch.load(os.path.join(
    best_checkpoint_dir, "checkpoint"))
best_trained_model.load_state_dict(model_state)

