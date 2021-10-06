"""
Author: Talip Ucar
email: ucabtuc@gmail.com

Description: Library of loss functions.
"""

import numpy as np
import torch as th
import torch.nn.functional as F


def getMSEloss(recon, target):
    """

    Args:
        recon (torch.FloatTensor):
        target (torch.FloatTensor):

    """
    dims = list(target.size())
    bs = dims[0]
    loss = th.sum(th.square(recon - target)) / bs
    return loss


def getBCELoss(prediction, label):
    """

    Args:
        prediction (torch.FloatTensor):
        label (torch.FloatTensor):

    """
    dims = list(prediction.size())
    bs = dims[0]
    return F.binary_cross_entropy(prediction, label, reduction='sum') / bs


class JointLoss(th.nn.Module):
    """
    Modifed from: https://github.com/sthalles/SimCLR/blob/master/loss/nt_xent.py
    When computing loss, we are using a 2Nx2N similarity matrix, in which positve samples are on the diagonal of four
    quadrants while negatives are all the other samples as shown below in 8x8 array, where we assume batch_size=4.
                                        P . . . P . . .
                                        . P . . . P . .
                                        . . P . . . P .
                                        . . . P . . . P
                                        P . . . P . . .
                                        . P . . . P . .
                                        . . P . . . P .
                                        . . . P . . . P
    """

    def __init__(self, options):
        super(JointLoss, self).__init__()
        # Assign options to self
        self.options = options
        # Batch size
        self.batch_size = options["batch_size"]
        # Temperature to use scale logits
        self.temperature = options["tau"]
        # Device to use: GPU or CPU
        self.device = options["device"]
        # initialize softmax
        self.softmax = th.nn.Softmax(dim=-1)
        # Mask to use to get negative samples from similarity matrix
        self.mask_for_neg_samples = self._get_mask_for_neg_samples().type(th.bool)
        # Function to generate similarity matrix: Cosine, or Dot product
        self.similarity_fn = self._cosine_simililarity if options["cosine_similarity"] else self._dot_simililarity
        # Loss function
        self.criterion = th.nn.CrossEntropyLoss(reduction="sum")

    def _get_mask_for_neg_samples(self):
        # Diagonal 2Nx2N identity matrix, which consists of four (NxN) quadrants
        diagonal = np.eye(2 * self.batch_size)
        # Diagonal 2Nx2N matrix with 1st quadrant being identity matrix
        q1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        # Diagonal 2Nx2N matrix with 3rd quadrant being identity matrix
        q3 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        # Generate mask with diagonals of all four quadrants being 1.
        mask = th.from_numpy((diagonal + q1 + q3))
        # Reverse the mask: 1s become 0, 0s become 1. This mask will be used to select negative samples
        mask = (1 - mask).type(th.bool)
        # Transfer the mask to the device and return
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        # Reshape x: (2N, C) -> (2N, 1, C)
        x = x.unsqueeze(1)
        # Reshape y: (2N, C) -> (1, C, 2N)
        y = y.T.unsqueeze(0)
        # Similarity shape: (2N, 2N)
        similarity = th.tensordot(x, y, dims=2)
        return similarity

    def _cosine_simililarity(self, x, y):
        similarity = th.nn.CosineSimilarity(dim=-1)
        # Reshape x: (2N, C) -> (2N, 1, C)
        x = x.unsqueeze(1)
        # Reshape y: (2N, C) -> (1, C, 2N)
        y = y.unsqueeze(0)
        # Similarity shape: (2N, 2N)
        return similarity(x, y)

    def XNegloss(self, representation):
        # Compute similarity matrix
        similarity = self.similarity_fn(representation, representation)
        # Get similarity scores for the positive samples from the diagonal of the first quadrant in 2Nx2N matrix
        l_pos = th.diag(similarity, self.batch_size)
        # Get similarity scores for the positive samples from the diagonal of the third quadrant in 2Nx2N matrix
        r_pos = th.diag(similarity, -self.batch_size)
        # Concatenate all positive samples as a 2nx1 column vector
        positives = th.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)
        # Get similarity scores for the negative samples (samples outside diagonals in 4 quadrants in 2Nx2N matrix)
        negatives = similarity[self.mask_for_neg_samples].view(2 * self.batch_size, -1)
        # Concatenate positive samples as the first column to negative samples array
        logits = th.cat((positives, negatives), dim=1)
        # Normalize logits via temperature
        logits /= self.temperature
        # Labels are all zeros since all positive samples are the 0th column in logits array.
        # So we will select positive samples as numerator in NTXentLoss
        labels = th.zeros(2 * self.batch_size).to(self.device).long()
        # Compute total loss
        loss = self.criterion(logits, labels)
        # Loss per sample
        closs = loss / (2 * self.batch_size)
        # Return contrastive loss
        return closs

    def forward(self, representation, xrecon, xorig):
        """

        Args:
            representation (torch.FloatTensor):
            xrecon (torch.FloatTensor):
            xorig (torch.FloatTensor):

        """

        # recontruction loss
        recon_loss = getMSEloss(xrecon, xorig) if self.options["reconstruction"] else getBCELoss(xrecon, xorig)

        # Initialize contrastive and distance losses with recon_loss as placeholder
        closs, zrecon_loss = recon_loss, recon_loss

        # Start with default loss i.e. reconstruction loss
        loss = recon_loss

        if self.options["contrastive_loss"]:
            closs = self.XNegloss(representation)
            loss = loss + closs

        if self.options["distance_loss"]:
            # recontruction loss for z
            zi, zj = th.split(representation, self.batch_size)
            zrecon_loss = getMSEloss(zi, zj)
            loss = loss + zrecon_loss

        # Return
        return loss, closs, recon_loss, zrecon_loss
