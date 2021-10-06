"""
Author: Talip Ucar
email: ucabtuc@gmail.com

Description: Utility functions.
"""

import cProfile
import os
import pstats
import random as python_random
import sys

import numpy as np
import torch
import yaml
from numpy.random import seed
from sklearn import manifold
from texttable import Texttable


def set_seed(options):
    """Sets seed to ensure reproducibility"""
    seed(options["seed"])
    np.random.seed(options["seed"])
    python_random.seed(options["seed"])
    torch.manual_seed(options["seed"])


def create_dir(dir_path):
    """Creates a directory if it does not exist"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def set_dirs(config):
    """It sets up directory that will be used to load processed_data and src as well as saving results.
    Directory structure example:
        results > framework (e.g. SubTab) > training  -------> model_mode > model
                                          > evaluation                    > plots
                                                                          > loss
    Args:
        config (dict): Dictionary that defines options to use

    """
    # Set main results directory using database name. Exp:  processed_data/dpp19
    paths = config["paths"]
    # results
    results_dir = make_dir(paths["results"], "")
    # results > framework
    results_dir = make_dir(results_dir, config["framework"])
    # results > framework > training
    training_dir = make_dir(results_dir, "training")
    # results > framework > evaluation
    evaluation_dir = make_dir(results_dir, "evaluation")
    # results > framework > evaluation > clusters
    clusters_dir = make_dir(evaluation_dir, "clusters")
    # results > framework > evaluation > reconstruction
    recons_dir = make_dir(evaluation_dir, "reconstructions")
    # results > framework > training > model_mode = ae
    model_mode_dir = make_dir(training_dir, config["model_mode"])
    # results > framework > training > model_mode > model
    training_model_dir = make_dir(model_mode_dir, "model")
    # results > framework > training > model_mode > plots
    training_plot_dir = make_dir(model_mode_dir, "plots")
    # results > framework > training > model_mode > loss
    training_loss_dir = make_dir(model_mode_dir, "loss")
    # Print a message.
    print("Directories are set.")


def make_dir(directory_path, new_folder_name):
    """Creates an expected directory if it does not exist"""
    directory_path = os.path.join(directory_path, new_folder_name)
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    return directory_path


def get_runtime_and_model_config(args):
    """Returns runtime and model/dataset specific config file"""
    try:
        with open("./config/runtime.yaml", "r") as file:
            config = yaml.safe_load(file)
    except Exception as e:
        sys.exit("Error reading runtime config file")
    # Define the data specific config file
    config["model_config"] = args.dataset
    # Copy dataset names to config to use later
    config["dataset"] = args.dataset
    # Update the config by adding the data specific config to runtime config
    config = update_config_with_model(config)
    return config


def update_config_with_model(config):
    """Updates options with given configuration for a particular model"""
    model_config = config["model_config"]
    try:
        with open("./config/" + model_config + ".yaml", "r") as file:
            model_config = yaml.safe_load(file)
    except Exception as e:
        sys.exit("Error reading model config file")
    config.update(model_config)
    return config


def get_runtime_and_model_config_with_dataset_name(dataset):
    """Gets runtime and model yaml file by using dataset name"""
    try:
        with open("./config/runtime.yaml", "r") as file:
            config = yaml.safe_load(file)
    except Exception as e:
        sys.exit("Error reading runtime config file")
    # Define the data specific config file
    config["model_config"] = dataset
    # Copy dataset names to config to use later
    config["dataset"] = dataset
    # Update the config by adding the data specific config to runtime config
    config = update_config_with_model(config)
    return config


def update_config_with_model_dims(data_loader, config):
    """Updates options by adding the dimension of input features as the dimension of first hidden layer of the model"""
    # Get the first batch (data is in dictionary format)
    x, y = next(iter(data_loader.train_loader))
    # Get the features and turn them into numpy.
    xi = x.cpu().numpy()
    # Get the number of features
    dim = xi.shape[-1]
    # Update the dims of model architecture by adding the number of features as the first dimension
    config["dims"].insert(0, dim)
    return config


def run_with_profiler(main_fn, config):
    """Runs function with profile to see how much time each step takes."""
    profiler = cProfile.Profile()
    profiler.enable()
    # Run the main
    main_fn(config)
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('ncalls')
    stats.print_stats()


def tsne(latent):
    """Reduces dimensionality of embeddings to 2, and returns it"""
    mds = manifold.TSNE(n_components=2, init='pca', random_state=0)
    return mds.fit_transform(latent)


def print_config(args):
    """Prints out options and arguments"""
    # Yaml config is a dictionary while parser arguments is an object. Use vars() only on parser arguments.
    if type(args) is not dict:
        args = vars(args)
    # Sort keys
    keys = sorted(args.keys())
    # Initialize table
    table = Texttable()
    # Add rows to the table under two columns ("Parameter", "Value").
    table.add_rows([["Parameter", "Value"]] + [[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    # Print the table.
    print(table.draw())
