"""
Author: Talip Ucar
email: ucabtuc@gmail.com

Description: Wrapper function for training routine.
"""

import copy
import time

import mlflow
import yaml

import eval
from src.model import SubTab
from utils.arguments import get_arguments, get_config, print_config_summary
from utils.load_data import Loader
from utils.utils import set_dirs, run_with_profiler, update_config_with_model_dims


def train(config, data_loader, save_weights=True):
    """Utility function for training and saving the model.
    Args:
        config (dict): Dictionary containing options and arguments.
        data_loader (IterableDataset): Pytorch data loader.
        save_weights (bool): Saves model if True.

    """
    # Instantiate model
    model = SubTab(config)
    # Start the clock to measure the training time
    start = time.process_time()
    # Fit the model to the data
    model.fit(data_loader)
    # Total time spent on training
    training_time = time.process_time() - start
    # Report the training time
    print(f"Training time:  {training_time // 60} minutes, {training_time % 60} seconds")
    # Save the model for future use
    _ = model.save_weights() if save_weights else None

    # Save the config file to keep a record of the settings
    with open(model._results_path + "/config.yml", 'w') as config_file:
        yaml.dump(config, config_file, default_flow_style=False)
    print("Done with training...")

    # Track results
    if config["mlflow"]:
        # Log config with mlflow
        mlflow.log_artifacts("./config", "config")
        # Log model and results with mlflow
        mlflow.log_artifacts(model._results_path + "/training/" + config["model_mode"] + "/plots", "training_results")
        # log model
        # mlflow.pytorch.log_model(model, "models")


def main(config):
    """Main wrapper function for training routine.

    Args:
        config (dict): Dictionary containing options and arguments.

    """
    # Set directories (or create if they don't exist)
    set_dirs(config)
    # Get data loader for first dataset.
    ds_loader = Loader(config, dataset_name=config["dataset"])
    # Add the number of features in a dataset as the first dimension of the model
    config = update_config_with_model_dims(ds_loader, config)
    # Start training and save model weights at the end
    train(config, ds_loader, save_weights=True)


if __name__ == "__main__":
    # Get parser / command line arguments
    args = get_arguments()
    # Get configuration file
    config = get_config(args)
    # Overwrite the parent folder name for saving results
    config["framework"] = config["dataset"]
    # Get a copy of autoencoder dimensions
    dims = copy.deepcopy(config["dims"])
    # Summarize config and arguments on the screen as a sanity check
    print_config_summary(config, args)
    
    #----- If True, start of MLFlow for experiment tracking:
    if config["mlflow"]:
        # Experiment name
        experiment_name = "Give_Your_Experiment_A_Name"
        # Set the experiment
        mlflow.set_experiment(experiment_name=experiment_name + "_" + str(args.experiment))
        # Start a new mlflow run
        with mlflow.start_run():
            # Run the main with or without profiler
            run_with_profiler(main, config) if config["profile"] else main(config)
    else:
        #----- Run Training - with or without profiler
        run_with_profiler(main, config) if config["profile"] else main(config)
    
        #----- Moving to evaluation stage
        # Reset the autoencoder dimension since it was changed in train.py
        config["dims"] = dims
        # Disable adding noise since we are in evaluation mode
        config["add_noise"] = False
        # Turn off valiation
        config["validate"] = False
        # Get all of available training set for evaluation (i.e. no need for validation set)
        config["training_data_ratio"] = 1.0
        # Run Evaluation
        eval.main(config)
