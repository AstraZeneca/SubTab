# SubTab: 
##### Author: Talip Ucar (ucabtuc@gmail.com)

The official implementation of the paper, 

[SubTab: Subsetting Features of Tabular Data for Self-Supervised Representation Learning](https://arxiv.org/abs/2110.04361)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/subtab-subsetting-features-of-tabular-data/unsupervised-mnist-on-mnist)](https://paperswithcode.com/sota/unsupervised-mnist-on-mnist?p=subtab-subsetting-features-of-tabular-data)

:large_orange_diamond: **Note:** The extended version of SubTab with codes and pre-processed data for Adult Income and BlogFeedback datasets can be found at: https://github.com/talipucar/SubTab_extended

# Table of Contents:

1. [Model](#model)
2. [Environment](#environment)
3. [Data](#data)
4. [Configuration](#configuration)
5. [Training and Evaluation](#training-and-evaluation)
6. [Adding New Datasets](#adding-new-datasets)
7. [Results](#results)
8. [Experiment tracking](#experiment-tracking)
9. [Citing the paper](#citing-the-paper)
10. [Citing this repo](#citing-this-repo)


NeurIPS 2021 slides        |  NeurIPS 2021 poster
:-------------------------:|:-------------------------:
[![NeurIPS 2021 slides](./assets/presentation_cover.png)](./assets/NeurIPS_2021_slides.pdf)  |  [![NeurIPS 2021 poster](./assets/poster_cover.png)](./assets/NeurIPS_2021_poster.pdf)


# Model

![SubTab](./assets/SubTab.gif)

<details>
  <summary>Click for a slower version of the animation</summary>

![SubTab](./assets/SubTab_slow.gif)

</details>


# Environment
We used Python 3.7 for our experiments. The environment can be set up by following three steps:

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

2. Second configuration file is dataset-specific and is used to configure the architecture of the model, loss functions, and so on. 

   - For example, we set up a configuration file for MNIST dataset with the same name. 
   Please note that the name of the configuration file should be same as name of the dataset with all letters in lowercase. 
   - We can have configuration files for other datasets such as **tcga.yaml** and **income.yaml** for tcga and income datasets respectively.



# Training and Evaluation
You can train and evaluate the model by using:

```
python train.py # For training. 
python eval.py  # For evaluation
```

   - ```train.py``` will also run evaluation at the end of the training. 
   - You can also run evaluation separately by using ```eval.py```.
   - For a list of arguments, please see ```./utils/arguments.py```
     - Use ```-h``` argument to get help when running scripts.
     - Use ```-d dataset_name``` to run scripts on new datasets 

# Adding New Datasets

For each new dataset, you can use the following steps:

1. Provide a ```_load_dataset_name()``` function, similar to [MNIST load function](https://github.com/AstraZeneca/SubTab/blob/070b2ef73fceb0531d2b1d1fc32f7eda4fe5c966/utils/load_data.py#L174-L190)

   - For example, you can add ```_load_tcga()``` for tcga dataset, or ```_load_income()``` for income dataset. 
   - The function should return (x_train, y_train, x_test, y_test)

2. Add a separate ```elif``` condition in [this section](https://github.com/AstraZeneca/SubTab/blob/070b2ef73fceb0531d2b1d1fc32f7eda4fe5c966/utils/load_data.py#L110-L112) within ```_load_data()``` method of ```TabularDataset()``` class in ```utils/load_data.py```

3. Create a new config file with the same name as dataset name.
   - For example, ```tcga.yaml``` for tcga dataset, or ```income.yaml``` for income dataset.
   - You can also duplicate one of the existing configuration files (e.g. mnist.yaml), and re-name it.

   - Make sure that the new config file is under ```config/``` directory.

4. Provide data folder with pre-processed training and test set, and place it under ```./data/``` directory. 
You can also do train-test split and pre-processing within your custom ```_load_dataset_name()``` function.

5. (Optional) If you want to place the new dataset under a different directory than the local "./data/", then:
   - Place the dataset folder anywhere, and define the root directory to it in [this line](https://github.com/AstraZeneca/SubTab/blob/070b2ef73fceb0531d2b1d1fc32f7eda4fe5c966/config/runtime.yaml#L5)
of ```/config/runtime.yaml```. 

   - For example, if the path to tcga dataset is ```/home/.../data/tcga/```, 
   you only need to include ```/home/.../data/``` in ```runtime.yaml```. 
   The code will fill in ```tcga``` folder name from the name given in the command line argument
   (e.g. ```-d dataset_name```. In this case, dataset_name would be tcga).

# Structure of the repo
<pre>
- train.py
- eval.py

- src
    |-model.py
    
- config
    |-runtime.yaml
    |-mnist.yaml
    
- utils
    |-load_data.py
    |-arguments.py
    |-model_utils.py
    |-loss_functions.py
    ...
    
- data
    |-mnist
    ...
    
- results
    |
    ...
</pre>

# Results

Results at the end of training is saved under ```./results``` directory. Results directory structure is as following:

<pre>
- results
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
MLFlow is used to track experiments. It is turned off by default, but can be turned on by changing option [on this line](https://github.com/AstraZeneca/SubTab/blob/070b2ef73fceb0531d2b1d1fc32f7eda4fe5c966/config/runtime.yaml#L2) in 
runtime config file in ```./config/runtime.yaml```


# Citing the paper

```
@article{ucar2021subtab,
  title={SubTab: Subsetting Features of Tabular Data for Self-Supervised Representation Learning},
  author={Ucar, Talip and Hajiramezanali, Ehsan and Edwards, Lindsay},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  year={2021}
}
```

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
