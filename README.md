# SubTab: 
##### Author: Talip Ucar (ucabtuc@gmail.com)

The official implementation of the paper, **![SubTab: Subsetting Features of Tabular Data for Self-Supervised Representation Learning](#TODO: Link to the paper)**.

# Table of Contents:

1. [Model](#model)
2. [Paper](#paper)
3. [Environment](#environment)
4. [Data](#data)
5. [Configuration](#configuration)
6. [Training and Evaluation](#training-and-evaluation)
7. [Adding New Datasets](#adding-new-datasets)
8. [Results](#results)
9. [Experiment tracking](#experiment-tracking)
10. [Citing the paper](#citing-the-paper)
11. [Citing this repo](#citing-this-repo)


# Model

![SubTab](./assets/SubTab_transparent_bg.gif)


# Paper

![SubTab](TODO: Link to the paper)


# Environment
We used Python 3.7 for our experiments. The environment can be set up by following three steps:
1. Install pipenv using pip
2. Install required packages 
3. Activate virtual environment

You can run following commands to set up the environment:
```
pip install pipenv             # To install pipenv if you don't have it already
pipenv install --skip-lock     # To install required packages. 
pipenv shell                   # To activate virtual env
```

If the second step results in issues, you can install packages in Pipfile individually by using pip i.e. "pip install package_name". 

# Data
MNIST dataset is already provided to demo the framework. For your own dataset, follow the instructions in [Adding New Datasets](#adding-new-datasets).

# Configuration
There are two types of configuration files:
```
1. runtime.yaml
2. mnist.yaml
```

1. ```runtime.yaml``` is a high-level configuration file used by all datasets to:

   - define the random seed
   - turn on/off mlflow (Default: False)
   - turn on/off python profiler (Default: False)
   - set data directory
   - set results directory

2. Second configuration file is dataset-specific and is used to configure the architecture of the model, loss functions, and so on. For example, we set up a configuration file for MNIST dataset with the same name. Please note that the name of the configuration file should be same as the dataset with all letters in lowercase. Moreover, we can have configuration files for other datasets such as **tcga.yaml** and **income.yaml** for tcga and income datasets respectively.



# Training and Evaluation
You can train the model using:
```
python train.py 
```

```train.py``` will also run evaluation at the end of the training. You can also run evaluation separately by using:

```
python eval.py 
```

# Adding New Datasets:

For each new dataset:

1- Provide a ```_load_dataset_name()``` function, e.g. ```_load_tcga()``` for tcga dataset, or ```_load_income()``` for income dataset. 
- Place this function as a separate ```elif``` condition within ```_load_data()``` method of ```TabularDataset()``` class in ```utils/load_data.py```
- The function should return (x_train, y_train, x_test, y_test)

2- Create a new config file with the same name as dataset name e.g. ```tcga.yaml``` for tcga dataset, or ```income.yaml``` for income dataset. 
You can also duplicate one of the existing configuration files, and re-name it.
- Place it under ```config/``` directory.

3- Provide data folder with pre-processed training and test set. Or you can do train-test split and pre-processing in your custom ```_load_dataset_name()``` function.
- Place the dataset folder anywhere, and define the data path in ```/config/runtime.yaml```. For example, if the path to tcga dataset is ```/home/user_abc/data/tcga/```, you only need to include ```/home/user_abc/data/``` in ```/config/runtime.yaml```. The code will fill in ```tcga``` folder name from the name of the dataset.


# Results

Results at the end of training is saved under ```./results``` directory. Results directory structure is as following:

<pre>
results
    |-dataset name
            |-evaluation
                |-clusters (for plotting t-SNE and PCA plots of embeddings)
                |-reconstructions (not used)
            |-training
                |-model_mode (e.g. ae for autoencoder)   
                     |-model
                     |-plots
                     |-loss
</pre>

You can save results of evaluations under "evaluation" folder. 


# Experiment tracking
MLFlow is used to track experiments. It is turned off by default, but can be turned on by changing option in 
runtime config file in ```./config/runtime.yaml```


# Citing the paper:

#TODO

# Citing this repo
If you use SubTab framework in your own studies, and work, please cite it by using the following:

```
@Misc{talip_ucar_2021_SubTab,
  author =   {Talip Ucar},
  title =    {{SubTab: Subsetting Features of Tabular Data for Self-Supervised Representation Learning}},
  howpublished = {\url{https://github.com/AstraZeneca/SubTab}},
  month        = June,
  year = {since 2021}
}
```
