"""Utility functions for mathematical operations."""

from functools import partial

import torch


def complex_mul1d(a, b):
    """Performs complex multiplication between 1d input tensors.

    Args:
    a: tensor of shape (batch, in_channel, x)
    b: tensor of shape (in_channel, out_channel, x)

    return: tensor of form (batch, out_channel, x)
    """
    op = partial(torch.einsum, "bix,iox->box")
    return torch.stack([
        op(a[..., 0], b[..., 0]) - op(a[..., 1], b[..., 1]),
        op(a[..., 1], b[..., 0]) + op(a[..., 0], b[..., 1])
    ],
                       dim=-1)


def complex_mul2d(a, b):
    """Performs complex multiplication between 2d input tensors.

    Args:
    a: tensor of shape (batch, in_channel, x, y)
    b: tensor of shape (in_channel, out_channel, x, y)

    return: tensor of form (batch, out_channel, x, y)
    """
    op = partial(torch.einsum, "bixy,ioxy->boxy")
    return torch.stack([
        op(a[..., 0], b[..., 0]) - op(a[..., 1], b[..., 1]),
        op(a[..., 1], b[..., 0]) + op(a[..., 0], b[..., 1])
    ],
                       dim=-1)


def complex_mul3d(a, b):
    """Performs complex multiplication between 3d input tensors.

    Args:
    a: tensor of shape (batch, in_channel, x, y, z)
    b: tensor of shape (in_channel, out_channel, x, y, z)

    return: tensor of form (batch, out_channel, x, y, z)
    """
    op = partial(torch.einsum, "bixyz,ioxyz->boxyz")
    return torch.stack([
        op(a[..., 0], b[..., 0]) - op(a[..., 1], b[..., 1]),
        op(a[..., 1], b[..., 0]) + op(a[..., 0], b[..., 1])
    ],
                       dim=-1)


class UnitGaussianNormalizer():
    """Applies pointwise Gaussian normalization to an input tensor."""
    def __init__(self, x, eps=0.00001):
        super(UnitGaussianNormalizer, self).__init__()
        """Args:
            x: tensor of shape (ntrain X n) or (ntrain X T X n) or
                (ntrain X n X T)
        """
        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps  # n
            mean = self.mean
        else:
            if len(self.mean.shape) == len(sample_idx[0].shape):
                std = self.std[sample_idx] + self.eps  # (batch X n)
                mean = self.mean[sample_idx]
            if len(self.mean.shape) > len(sample_idx[0].shape):
                std = self.std[:, sample_idx] + self.eps  # (T X batch X n)
                mean = self.mean[:, sample_idx]

        # x in shape of (batch X n) or (T X batch X n)
        x = (x * std) + mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
