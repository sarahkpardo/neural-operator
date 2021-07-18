from pathlib import Path

import h5py
from einops import rearrange, reduce, repeat
import numpy as np
from scipy.io import loadmat
import torch


def read_data(
        path,
        property_names,
        as_torch=False,
        to_cuda=False,
):

    with h5py.File(path) as f:
        properties = dict(
            zip(property_names, [f[p][:].squeeze() for p in property_names]))

        X = f['input'][...].astype(np.float32)
        Y = f['output'][...].astype(np.float32)

    if as_torch:
        X = torch.from_numpy(X)
        Y = torch.from_numpy(Y)

    if to_cuda:
        X = X.cuda()
        Y = Y.cuda()

    return X, Y, properties


def subsample(array, ss_rates):
    """Subsample a tensor with a given rate for each dimension."""
    obj = [
        slice(0, array.shape[dim], ss_rates[dim]) for dim in range(array.ndim)
    ]

    return array[tuple(obj)]


def to_dataloader(X,
                  Y,
                  num_samples,
                  batch_size=20,
                  pin_memory=True,
                  shuffle=True,
                  name=None):
    """Generate a PyTorch DataLoader from input and target tensors."""

    num_samples = num_samples - (num_samples % batch_size)

    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X[:num_samples, ...],
                                       Y[:num_samples, ...]),
        batch_size=batch_size,
        pin_memory=pin_memory,
        shuffle=shuffle,
        name=name,
    )

    return dataloader


class FunctionDataset(object):
    """
    """
    def __init__(self):
        super(FunctionDataset, self).__init__()

        self.X = None
        self.Y = None
        self.properties = None

    def read_data(
            self,
            path,
            property_names,
            backend='torch',
            cuda=False,
    ):

        X_list = []
        Y_list = []

        for file in path.iterdir():
            with h5py.File(file) as f:
                if self.properties is None:
                    self.properties = dict(
                        zip(property_names,
                            [f[p][:].squeeze() for p in property_names]))

                X_ = f['input'][...].astype(np.float32)
                Y_ = f['output'][...].astype(np.float32)

            X_list.append(X_)
            Y_list.append(Y_)

        self.X = rearrange(X_list, 'x b -> b x')
        self.Y = rearrange(Y_list, 'x b -> b x')

        if backend == 'torch':
            self.X = torch.from_numpy(self.X)
            self.Y = torch.from_numpy(self.Y)

        if backend == 'tf':
            self.X = tf.from_numpy(self.X)
            self.Y = tf.from_numpy(self.Y)

        if cuda:
            self.X = self.X.cuda()
            self.Y = self.Y.cuda()

        return

    def add_domain_channel(
            self,
            X,
            lower,
            upper,
    ):

        raise NotImplementedError

    def gen_dataset(
            self,
            ntrain,
            ntest,
            domain,
            ss_rate=None,
            batch_size=20,
    ):

        raise NotImplementedError


class FunctionDataset1D(FunctionDataset):
    """
    """
    def __init__(self, ):
        super(FunctionDataset, self).__init__()

        self.properties = None
        self.X = None
        self.Y = None

    def add_domain_channel(
            X,
            lower,
            upper,
    ):
        """
        Add 1D domain coordinates as a channel to input tensor X.
        """
        b, x = [*X.shape]

        domain = torch.tensor(
            repeat(np.linspace(lower, upper, x), 'x -> b x', b=b))

        X_aug = rearrange([X_, domain], 'c b x -> b x c')

        return X_aug

    def gen_dataset(
            self,
            ntrain,
            ntest,
            domain,
            ss_rate=None,
            batch_size=20,
    ):
        """
        Generate a DataLoader of data for input to a model.

        Specifies input data properties for a given network
        training/testing run:
            subsampling,
            test/train split,
            batch size,

        """
        # resample
        X_ss = resample(self.X, [1, ss_rate])
        Y_ss = resample(self.Y, [1, ss_rate])

        # add domain channel
        X_aug = add_domain_channel(X_ss, *domain)

        # make test/train DataLoaders
        train_DL = to_dataloader(X_aug, Y_ss, ntrain, name='train_data')
        test_DL = to_dataloader(X_aug,
                                Y_ss,
                                ntest,
                                shuffle=False,
                                name='test_data')

        return train_DL, test_DL


class FunctionDataset2D(FunctionDataset):
    """
    """
    def __init__(
            self,
            filename,
            ndim,
            ss_rate=None,
    ):
        super(FunctionDataset, self).__init__()

        self.attributes = attributes
        self.samples = None
        self.ss_rate = None
        self.ndim = None

        self.x_data = None
        self.y_data = None

    def add_domain_channel(
            X,
            lower,
            upper,
    ):
        """
        Add 2D domain coordinates as channels to input tensor X.
        """
        b, x, y = [*X.shape]

        x_ax = torch.tensor(
            repeat(np.linspace(lower, upper, x), 'w -> b h w', h=y, b=b))
        y_ax = torch.tensor(
            repeat(np.linspace(lower, upper, y), 'h -> b h w', w=x, b=b))

        X_aug = rearrange([X, x_ax, y_ax], 'c b x y -> b x y c')

        return X_aug

    def gen_dataset(
            self,
            ntrain,
            ntest,
            domain,
            ss_rate=None,
            batch_size=20,
    ):
        """
        Generate a DataLoader of data for input to a model.

        Specifies input data properties for a given network
        training/testing run:
            subsampling,
            test/train split,
            batch size,

        """
        # resample
        X_ss = resample(self.X, [1, ss_rate, ss_rate])
        Y_ss = resample(self.Y, [1, ss_rate, ss_rate])

        # add domain channel
        X_aug = add_domain_channel(X_ss, *domain)

        # make test/train DataLoaders
        train_DL = to_dataloader(X_aug, Y_ss, ntrain, name='train_data')
        test_DL = to_dataloader(X_aug,
                                Y_ss,
                                ntest,
                                shuffle=False,
                                name='test_data')

        return train_DL, test_DL
