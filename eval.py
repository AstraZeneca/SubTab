"""
Author: Talip Ucar
email: ucabtuc@gmail.com

Description: Wrapper function for evaluation routine.
"""

import mlflow
import torch as th
import torch.utils.data
from tqdm import tqdm

from src.model import SubTab
from utils.arguments import get_arguments, get_config
from utils.arguments import print_config_summary
from utils.eval_utils import linear_model_eval, plot_clusters, append_tensors_to_lists, concatenate_lists, aggregate
from utils.load_data import Loader
from utils.utils import set_dirs, run_with_profiler, update_config_with_model_dims

torch.manual_seed(1)


def eval(data_loader, config):
    """Wrapper function for evaluation.

    Args:
        data_loader (IterableDataset): Pytorch data loader.
        config (dict): Dictionary containing options and arguments.

    """
    # Instantiate Autoencoder model
    model = SubTab(config)
    # Load the model
    model.load_models()
    # Evaluate Autoencoder
    with th.no_grad():
        # Get the joint embeddings and class labels of training set
        z_train, y_train = evalulate_models(data_loader, model, config, plot_suffix="training", mode="train")
        
        # Train and evaluate logistig regression using the joint embeddings of training and test set
        evalulate_models(data_loader, model, config, plot_suffix="test", mode="test", z_train=z_train, y_train=y_train)
        
        # End of the run
        print(f"Evaluation results are saved under ./results/{config['framework']}/evaluation/\n")
        print(f"{100 * '='}\n")

        # If mlflow==True, track results
        if config["mlflow"]:
            # Log model and results with mlflow
            mlflow.log_artifacts(model._results_path + "/evaluation/" + "/clusters", "evaluation")


def evalulate_models(data_loader, model, config, plot_suffix="_Test", mode='train', z_train=None, y_train=None):
    """Evaluates representations using linear model, and visualisation of clusters using t-SNE and PCA on embeddings.

    Args:
        data_loader (IterableDataset): Pytorch data loader.
        model (object): Class that contains the encoder and associated methods
        config (dict): Dictionary containing options and arguments.
        plot_suffix (str): Suffix to be used when saving plots
        mode (str): Defines whether to evaluate the model on training set, or test set.
        z_train (ndarray): Optional numpy array holding latent representations of training set
        y_train (list): Optional list holding labels of training set

    Returns:
        (tuple): tuple containing:
            z_train (numpy.ndarray): Numpy array holding latent representations of data set
            y_train (list): List holding labels of data set

    """
    # A small function to print a line break on the command line.
    break_line = lambda sym: f"{100 * sym}\n{100 * sym}\n"
    
    # Print whether we are evaluating training set, or test set
    decription = break_line('#') + f"Getting the joint embeddings of {plot_suffix} set...\n" + \
                 break_line('=') + f"Dataset used: {config['dataset']}\n" + break_line('=')
    
    # Print the message         
    print(decription)
    
    # Get the model
    encoder = model.encoder
    # Move the model to the device
    encoder.to(config["device"])
    # Set the model to evaluation mode
    encoder.eval()

    # Choose either training, or test data loader    
    data_loader_tr_or_te = data_loader.train_loader if mode == 'train' else data_loader.test_loader

    # Attach progress bar to data_loader to check it during training. "leave=True" gives a new line per epoch
    train_tqdm = tqdm(enumerate(data_loader_tr_or_te), total=len(data_loader_tr_or_te), leave=True)

    # Create empty lists to hold data for representations, and class labels
    z_l, clabels_l = [], []

    # Go through batches
    for i, (x, label) in train_tqdm:

        # Generate subsets
        x_tilde_list = model.subset_generator(x)

        latent_list = []

        # Extract embeddings (i.e. latent) for each subset
        for xi in x_tilde_list:
            # Turn xi to tensor, and move it to the device
            Xbatch = model._tensor(xi)
            # Extract latent
            _, latent, _ = encoder(Xbatch)
            # Collect latent
            latent_list.append(latent)

            
        # Aggregation of latent representations
        latent = aggregate(latent_list, config)
            
        # Append tensors to the corresponding lists as numpy arrays
        z_l, clabels_l = append_tensors_to_lists([z_l, clabels_l],
                                                 [latent, label.int()])

    # Turn list of numpy arrays to a single numpy array for representations.
    z = concatenate_lists([z_l])
    # Turn list of numpy arrays to a single numpy array for class labels.
    clabels = concatenate_lists([clabels_l])

    # Visualise clusters
    plot_clusters(config, z, clabels, plot_suffix="_inLatentSpace_" + plot_suffix)

    if mode == 'test':
        # Title of the section to print 
        print(20 * "*" + " Running evaluation using Logistic Regression trained on the joint embeddings" \
                       + " of training set and tested on that of test set" + 20 * "*")
        # Description of the task (Classification scores using Logistic Regression) to print on the command line
        description = "Sweeping C parameter. Smaller C values specify stronger regularization:"
        # Evaluate the embeddings
        linear_model_eval(config, z_train, y_train, z_test=z, y_test=clabels, description=description)

    else:
        # Return z_train = z, and y_train = clabels
        return z, clabels


def main(config):
    """Main function for evaluation

    Args:
        config (dict): Dictionary containing options and arguments.

    """
    # Set directories (or create if they don't exist)
    set_dirs(config)
    # Get data loader for first dataset.
    ds_loader = Loader(config, dataset_name=config["dataset"], drop_last=False)
    # Add the number of features in a dataset as the first dimension of the model
    config = update_config_with_model_dims(ds_loader, config)
    # Start evaluation
    eval(ds_loader, config)


if __name__ == "__main__":
    # Get parser / command line arguments
    args = get_arguments()
    # Get configuration file
    config = get_config(args)
    # Overwrite the parent folder name for saving results
    config["framework"] = config["dataset"]
    # Turn off valiation
    config["validate"] = False
    # Get all of available training set for evaluation (i.e. no need for validation set)
    config["training_data_ratio"] = 1.0
    # Turn off noise when evaluating the performance
    config["add_noise"] = False
    # Summarize config and arguments on the screen as a sanity check
    print_config_summary(config, args)
    # --If True, start of MLFlow for experiment tracking:
    if config["mlflow"]:
        # Experiment name
        experiment_name = "Give_Your_Experiment_A_Name"
        mlflow.set_experiment(experiment_name=experiment_name + "_" + str(args.experiment))
        # Start a new mlflow run
        with mlflow.start_run():
            # Run the main with or without profiler
            run_with_profiler(main, config) if config["profile"] else main(config)
    else:
        # Run the main with or without profiler
        run_with_profiler(main, config) if config["profile"] else main(config)
