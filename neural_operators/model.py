"""Overall neural operator model."""

import operator
from functools import reduce

import torch.nn as nn


class OperatorNet(nn.Module):
    """Object representing the operator surrogate."""

    # TODO:
    # - properties of the dataset it was trained on
    # - properties of the training procedure
    # - properties of the network structure

    def __init__(self, kernel, name=None):
        super(OperatorNet, self).__init__()

        self.kernel = kernel
        self._name = name
        self.param_count = None

    def param_count(self):
        """Count the total number of model parameters."""
        if self.param_count is None:
            c = 0
            for p in self.parameters():
                c += reduce(operator.mul, list(p.size()))
            self.param_count = c

        return self.param_count

    def name(self):
        """Return a string name of the model."""
        if self._name is not None:
            return "{}".format(self._name)
        return "None"

    def forward(self, x):
        """Evaluate the network on an input tensor."""
        x = self.kernel(x)
        return x.squeeze()

    def __str__(self):
        if self._name is not None:
            return "{}".format(self._name)
        return "None"

    def __repr__(self):

        return
