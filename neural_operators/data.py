from timeit import default_timer

import h5py
from einops import rearrange, reduce, repeat
import numpy as np
import scipy.io
from scipy.io import loadmat
import tensorflow as tf
import torch


def read_data(
        path,
        property_names,
        as_torch=False,
        to_cuda=False,
):

    with h5py.File(path) as f:
        properties = dict(
            zip(property_names,
                [f[p][:].squeeze() for p in property_names]))

        X = f['input'][...].astype(np.float32)
        Y = f['output'][...].astype(np.float32)

    if as_torch:
        X = torch.from_numpy(X)
        Y = torch.from_numpy(Y)

    if to_cuda:
        X = X.cuda()
        Y = Y.cuda()

    return X, Y, properties


def subsample(array,
              ss_rates
              ):
    """Subsample a tensor with a given rate for each dimension."""
    obj = [
        slice(0,
              array.shape[dim],
              ss_rates[dim]) for dim in range(array.ndim)
    ]

    return array[tuple(obj)]


def to_dataloader(X,
                  Y,
                  num_samples,
                  batch_size=20,
                  pin_memory=True,
                  shuffle=True,
                  name=None
                  ):
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
        super(FunctionDataset,
              self).__init__()

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

        self.X = rearrange(X_list,
                           'x b -> b x')
        self.Y = rearrange(Y_list,
                           'x b -> b x')

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
        super(FunctionDataset,
              self).__init__()

        self.properties = None
        self.X = None
        self.Y = None

    def add_domain_channel(
            self,
            X,
            lower,
            upper,
    ):
        """
        Add 1D domain coordinates as a channel to input tensor X.
        """
        b, x = [*X.shape]

        domain = torch.tensor(
            repeat(np.linspace(lower,
                               upper,
                               x),
                   'x -> b x',
                   b=b))

        X_aug = rearrange([X, domain],
                          'c b x -> b x c')

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
        X_ss = subsample(self.X,
                         [1, ss_rate])
        Y_ss = subsample(self.Y,
                         [1, ss_rate])

        # add domain channel
        X_aug = self.add_domain_channel(X_ss,
                                        *domain)

        # make test/train DataLoaders
        train_DL = to_dataloader(X_aug,
                                 Y_ss,
                                 ntrain,
                                 name='train_data')
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
            properties,
            filename=None,
            ndim=None,
            ss_rate=None,
    ):
        super(FunctionDataset,
              self).__init__()

        self.properties = properties
        self.samples = None
        self.ss_rate = None
        self.ndim = None

        self.x_data = None
        self.y_data = None

    def add_domain_channel(
            self,
            X,
            lower,
            upper,
    ):
        """
        Add 2D domain coordinates as channels to input tensor X.
        """
        b, x, y = [*X.shape]

        x_ax = torch.tensor(
            repeat(np.linspace(lower,
                               upper,
                               x),
                   'w -> b h w',
                   h=y,
                   b=b))
        y_ax = torch.tensor(
            repeat(np.linspace(lower,
                               upper,
                               y),
                   'h -> b h w',
                   w=x,
                   b=b))

        X_aug = rearrange([X, x_ax, y_ax],
                          'c b x y -> b x y c')

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
        X_ss = subsample(self.X,
                         [1, ss_rate, ss_rate])
        Y_ss = subsample(self.Y,
                         [1, ss_rate, ss_rate])

        # add domain channel
        X_aug = self.add_domain_channel(X_ss,
                                        *domain)

        # make test/train DataLoaders
        train_DL = to_dataloader(X_aug,
                                 Y_ss,
                                 ntrain,
                                 name='train_data')
        test_DL = to_dataloader(X_aug,
                                Y_ss,
                                ntest,
                                shuffle=False,
                                name='test_data')

        return train_DL, test_DL


class MatReader(object):
    def __init__(self,
                 file_path,
                 to_torch=True,
                 to_cuda=False,
                 to_float=True
                 ):
        super(MatReader,
              self).__init__()

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):
        try:
            self.data = scipy.io.loadmat(self.file_path)
            self.old_mat = True
        except:
            self.data = h5py.File(self.file_path)
            self.old_mat = False

    def load_file(self,
                  file_path
                  ):
        self.file_path = file_path
        self._load_file()

    def read_field(self,
                   field
                   ):
        x = self.data[field]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x,
                             axes=range(len(x.shape) - 1,
                                        -1,
                                        -1))

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)

            if self.to_cuda:
                x = x.cuda()

        return x

    def set_cuda(self,
                 to_cuda
                 ):
        self.to_cuda = to_cuda

    def set_torch(self,
                  to_torch
                  ):
        self.to_torch = to_torch

    def set_float(self,
                  to_float
                  ):
        self.to_float = to_float


def mat_to_tensor1d(TRAIN_PATH,
                    TEST_PATH,
                    ss_rate,
                    x_field,
                    y_field,
                    vsamples=None,
                    normalize=False
                    ):
    """Converts .mat file contents to torch tensors.
        Args:
            :param TRAIN_PATH: list of .mat file names to concatenate into
                               training tensors
            :param TEST_PATH: list of .mat file names to concatenate into test
                              tensors
            :param ss_rate: [train subsampling rate, test rate]
            :param x_field: names of input field in the .mat file
            :param y_field: names of output field in the .mat file
            :param vsamples: [ntest, ntrain]
            :param normalize: Apply normalization to the tensors; default False
    """
    reader = MatReader(TRAIN_PATH)
    x_data = reader.read_field(x_field)
    y_data = reader.read_field(y_field)
    dimension = len(x_data.shape) - 1

    mat_info = {
        "input signal vector samples": x_data.shape[0],
        "output signal vector samples": y_data.shape[0],
        "input signal entry samples": x_data.shape[1],
        "output signal entry samples": y_data.shape[1],
        "signal dimension": dimension
    }

    ntrain = vsamples[0]
    ntest = vsamples[1]

    full_res = x_data.shape[1]
    tr_ss = ss_rate[0]
    tst_ss = ss_rate[1]
    tr_esamples = int(((full_res - 1) / tr_ss) + 1)

    x_train = x_data[ntrain:, ::tr_ss][:, :tr_esamples]
    y_train = y_data[ntrain:, ::tr_ss][:, :tr_esamples]

    if TRAIN_PATH != TEST_PATH:
        # using separate files for test/train data
        test_reader = MatReader(TEST_PATH)
        x_test = test_reader.read_field(x_field)
        y_test = test_reader.read_field(y_field)

        full_res = x_test.shape[1]
        tst_esamples = int(((full_res - 1) / tst_ss) + 1)

        x_test = x_test[ntest:, ::tst_ss][:, :tst_esamples]
        y_test = y_test[ntest:, ::tst_ss][:, :tst_esamples]

    else:
        full_res = x_data.shape[1]
        tst_esamples = int(((full_res - 1) / tst_ss) + 1)

        # same file; use last (ntest) samples
        x_test = x_data[-ntest:, ::tst_ss][:, :tst_esamples]
        y_test = y_data[-ntest:, ::tst_ss][:, :tst_esamples]

    ds_info = {
        "training dataset": TRAIN_PATH,
        "test dataset": TEST_PATH,
        "input train samples": x_train.shape[0],
        "output train samples": y_train.shape[0],
        "input train resolution": x_train.shape[1],
        "output train resolution": y_train.shape[1],
        "input test samples": x_test.shape[0],
        "output test samples": y_test.shape[0],
        "input test resolution": x_test.shape[1],
        "output test resolution": y_test.shape[1]
    }

    t2 = default_timer()

    return x_train, y_train, x_test, y_test, mat_info, ds_info


def mat_to_tensor2d(TRAIN_PATH,
                    TEST_PATH,
                    ss_rate,
                    x_field,
                    y_field,
                    vsamples=None,
                    normalize=False
                    ):
    """Converts .mat file contents to torch tensors.
        Args:
            :param TRAIN_PATH: list of .mat file names to concatenate into
                               training tensors
            :param TEST_PATH: list of .mat file names to concatenate into test
                              tensors
            :param ss_rate: [train subsampling rate, test rate]
            :param x_field: names of input field in the .mat file
            :param y_field: names of output field in the .mat file
            :param vsamples: [ntest, ntrain]
            :param normalize: Apply normalization to the tensors; default False
    """
    reader = MatReader(TRAIN_PATH)
    x_data = reader.read_field(x_field)
    y_data = reader.read_field(y_field)
    dimension = len(x_data.shape) - 1

    mat_info = {
        "input signal vector samples": x_data.shape[0],
        "output signal vector samples": y_data.shape[0],
        "input signal entry samples": x_data.shape[1],
        "output signal entry samples": y_data.shape[1],
        "signal dimension": dimension
    }
    print(mat_info)

    ntrain = vsamples[0]
    ntest = vsamples[1]

    full_res = x_data.shape[1]
    tr_ss = ss_rate[0]
    tst_ss = ss_rate[1]
    tr_esamples = int(((full_res - 1) / tr_ss) + 1)

    x_train = x_data[:ntrain, ::tr_ss, ::tr_ss][:, :tr_esamples, :tr_esamples]
    y_train = y_data[:ntrain, ::tr_ss, ::tr_ss][:, :tr_esamples, :tr_esamples]

    if TRAIN_PATH != TEST_PATH:
        # using separate files for test/train data
        test_reader = MatReader(TEST_PATH)
        x_test = test_reader.read_field(x_field)
        y_test = test_reader.read_field(y_field)

        full_res = x_test.shape[1]
        tst_esamples = int(((full_res - 1) / tst_ss) + 1)

        x_test = x_test[:ntest,::tst_ss,::tst_ss][:,:tst_esamples,:tst_esamples]
        y_test = y_test[:ntest,::tst_ss,::tst_ss][:,:tst_esamples,:tst_esamples]

    else:
        full_res = x_data.shape[1]
        tst_esamples = int(((full_res - 1) / tst_ss) + 1)

        # same file; use last (ntest) samples
        x_test = x_data[-ntest:,::tst_ss,::tst_ss][:,:tst_esamples,:tst_esamples]
        y_test = y_data[-ntest:,::tst_ss,::tst_ss][:,:tst_esamples,:tst_esamples]

    ds_info = {
        "training dataset": TRAIN_PATH,
        "test dataset": TEST_PATH,
        "input train samples": x_train.shape[0],
        "output train samples": y_train.shape[0],
        "input train resolution": x_train.shape[1],
        "output train resolution": y_train.shape[1],
        "input test samples": x_test.shape[0],
        "output test samples": y_test.shape[0],
        "input test resolution": x_test.shape[1],
        "output test resolution": y_test.shape[1]
    }
    print(mat_info)

    t2 = default_timer()

    return x_train, y_train, x_test, y_test, mat_info, ds_info
