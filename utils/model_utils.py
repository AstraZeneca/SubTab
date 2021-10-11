"""
Author: Talip Ucar
email: ucabtuc@gmail.com

Description: Library of models and related support functions.
"""

import copy

import torch.nn.functional as F
from torch import nn


class AEWrapper(nn.Module):
    """
    Autoencoder wrapper class
    """

    def __init__(self, options):
        """

        Args:
            options (dict): Configuration dictionary.
        """
        super(AEWrapper, self).__init__()
        self.options = options
        self.encoder = ShallowEncoder(options) if options["shallow_architecture"] else Encoder(options)
        self.decoder = ShallowDecoder(options) if options["shallow_architecture"] else Decoder(options)
        
        # Get the last dimension of encoder. This will also be used as the dimension of projection
        output_dim = self.options["dims"][-1]
        # Two-Layer Projection Network
        # First linear layer, which will be followed with non-linear activation function in the forward()
        self.linear_layer1 = nn.Linear(output_dim, output_dim)
        # Last linear layer for final projection
        self.linear_layer2 = nn.Linear(output_dim, output_dim)

    def forward(self, x):
        # Forward pass on Encoder
        latent = self.encoder(x)
        # Forward pass on Projection
        # Apply linear layer followed by non-linear activation to decouple final output, z, from representation layer h.
        z = F.leaky_relu(self.linear_layer1(latent))
        # Apply final linear layer
        z = self.linear_layer2(z)
        # Do L2 normalization
        z = F.normalize(z, p=self.options["p_norm"], dim=1) if self.options["normalize"] else z
        # Forward pass on decoder
        x_recon = self.decoder(latent)
        # Return 
        return z, latent, x_recon


class Encoder(nn.Module):
    def __init__(self, options):
        """Encoder model

        Args:
            options (dict): Configuration dictionary.
        """
        super(Encoder, self).__init__()
        # Deepcopy options to avoid overwriting the original
        self.options = copy.deepcopy(options)
        # Compute the shrunk size of input dimension
        n_column_subset = int(self.options["dims"][0] / self.options["n_subsets"])
        # Ratio of overlapping features between subsets
        overlap = self.options["overlap"]
        # Number of overlapping features between subsets
        n_overlap = int(overlap * n_column_subset)
        # Overwrie the input dimension
        self.options["dims"][0] = n_column_subset + n_overlap
        # Forward pass on hidden layers
        self.hidden_layers = HiddenLayers(self.options)
        # Compute the latent i.e. bottleneck in Autoencoder
        self.latent = nn.Linear(self.options["dims"][-2], self.options["dims"][-1])

    def forward(self, h):
        # Forward pass on hidden layers
        h = self.hidden_layers(h)
        # Compute the mean i.e. bottleneck in Autoencoder
        latent = self.latent(h)
        return latent


class Decoder(nn.Module):
    def __init__(self, options):
        """Decoder model

        Args:
            options (dict): Configuration dictionary.
        """
        super(Decoder, self).__init__()
        # Deepcopy options to avoid overwriting the original
        self.options = copy.deepcopy(options)
        # If recontruct_subset is True, output dimension is same as input dimension of Encoder. Otherwise, 
        # output dimension is same as original feature dimension of tabular data
        if self.options["reconstruction"] and self.options["reconstruct_subset"]:
            # Compute the shrunk size of input dimension
            n_column_subset = int(self.options["dims"][0] / self.options["n_subsets"])
            # Overwrie the input dimension
            self.options["dims"][0] = n_column_subset
        # Revert the order of hidden units so that we can build a Decoder, which is the symmetric of Encoder
        self.options["dims"] = self.options["dims"][::-1]
        # Add hidden layers
        self.hidden_layers = HiddenLayers(self.options)
        # Compute logits and probabilities
        self.logits = nn.Linear(self.options["dims"][-2], self.options["dims"][-1])

    def forward(self, h):
        # Forward pass on hidden layers
        h = self.hidden_layers(h)
        # Compute logits
        logits = self.logits(h)
        return logits

    
class ShallowEncoder(nn.Module):
    def __init__(self, options):
        """Encoder model

        Args:
            options (dict): Configuration dictionary.
        """
        super(ShallowEncoder, self).__init__()
        # Deepcopy options to avoid overwriting the original
        self.options = copy.deepcopy(options)  
        # Compute the shrunk size of input dimension
        n_column_subset = int(self.options["dims"][0]/self.options["n_subsets"])
        # Ratio of overlapping features between subsets
        overlap = self.options["overlap"]
        # Number of overlapping features between subsets
        n_overlap = int(overlap*n_column_subset)
        # Overwrie the input dimension
        self.options["dims"][0] = n_column_subset + n_overlap
        # Forward pass on hidden layers
        self.hidden_layers = HiddenLayers(self.options)

    def forward(self, h):
        # Forward pass on hidden layers
        h = self.hidden_layers(h)
        return h
    
    
class ShallowDecoder(nn.Module):
    def __init__(self, options):
        """Decoder model

        Args:
            options (dict): Configuration dictionary.
        """
        super(ShallowDecoder, self).__init__()
        # Get configuration that contains architecture and hyper-parameters
        self.options = copy.deepcopy(options)
        # Input dimension of predictor == latent dimension
        input_dim, output_dim = self.options["dims"][-1],  self.options["dims"][0]
        # First linear layer with shape (bottleneck dimension, output channel size of last conv layer in CNNEncoder)
        self.first_layer = nn.Linear(input_dim, output_dim)

    def forward(self, z):
        logits = self.first_layer(z)
        return logits
    
    
class HiddenLayers(nn.Module):
    def __init__(self, options):
        """Class to add hidden layers to networks

        Args:
            options (dict): Configuration dictionary.
        """
        super(HiddenLayers, self).__init__()
        self.layers = nn.ModuleList()
        dims = options["dims"]

        for i in range(1, len(dims) - 1):
            self.layers.append(nn.Linear(dims[i - 1], dims[i]))
            if options["isBatchNorm"]:
                self.layers.append(nn.BatchNorm1d(dims[i]))

            self.layers.append(nn.LeakyReLU(inplace=False))
            if options["isDropout"]:
                self.layers.append(nn.Dropout(options["dropout_rate"]))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
