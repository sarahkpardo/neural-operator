"""Kernel modules for neural operator network."""

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from neural_operators.math_utils import complex_mul1d, complex_mul2d


class SpectralConv1d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 modes,
                 trainable=True):
        super(SpectralConv1d, self).__init__()

        """Applies FFT, linear transform, and inverse FFT.

        Args:
            in_channels: number of input channels
            out_channels: number of output channels
            modes: number of Fourier modes to multiply
            train: make weights trainable parameters
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.trainable = trainable
        self.scale = (1 / (in_channels * out_channels))

        self.weights = nn.Parameter(self.scale *
                                    torch.rand(self.in_channels,
                                               self.out_channels,
                                               self.modes,
                                               dtype=torch.cfloat))
        self.weights.requires_grad = trainable

    def forward(self, x):
        batchsize = x.shape[0]

        # Compute Fourier coeffcients up to factor of e^(-constant)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = complex_mul1d(x_ft[:, :, :self.modes],
                               self.weights)

        # Return to physical space
        x = torch.fft.irfft(out_ft,n=x.size(-1))

        return x


class FourierBlock1d(torch.nn.Module):
    """Fourier kernel block implementing
                    u' = (W + K)(u),
       where W is defined by a convolution and K is defined by
       a spectral convolution in Fourier space.
    """
    def __init__(self, width, modes, nonlinearity, trainable=True):
        super(FourierBlock1d, self).__init__()
        """
        Args:
            modes: Number of Fourier modes to multiply (at most floor(N/2) + 1)
            width: Dimension of lifting transform performed by first layer
            nonlinearity: Component-wise nonlinear activation
            trainablw: Whether or not the weights are trainable parameters;
                   default = True
        """
        self.width = width
        self.modes = modes
        self.trainable = trainable
        self.nonlinearity = nonlinearity

        # call signature SpectralConv2d(in_dim, out_dim, modes_x)
        self.R = SpectralConv1d(self.width, self.width, self.modes,
                                self.trainable)

        # call signature ConvNd(in_dim, out_dim, kernel_size)
        self.w = nn.Conv1d(self.width, self.width, 1)
        self.w.weight.requires_grad = trainable
        self.w.bias.requires_grad = trainable

    def forward(self, x):

        # x = 'b c x'
        Rx = self.R(x)

        # x = 'b c x'
        Wx = self.w(x)

        if self.nonlinearity is None:
            return Rx + Wx

        return self.nonlinearity(Rx + Wx)


class FourierKernel1d(nn.Module):
    """The Fourier kernel containing a specified number of Fourier blocks.

    1. Lift the input to the desired channel dimension by self.fc0.
    2. Apply specified number of iterations of the operator
                            u' = (W + K)(u)
    3. Project from the channel space to the output space by self.fc1
        and self.fc2.
    """
    def __init__(self, n_blocks, modes, width, nonlinearity, trainable):
        """
        Args:
            blocks: Number of kernel integral transform blocks
            modes: Number of Fourier modes to multiply (at most floor(N/2) + 1)
            width: Dimension of lifting transform performed by first layer
            nonlinearity: Component-wise nonlinear activation
        """
        super(FourierKernel1d, self).__init__()

        self.n_blocks = n_blocks
        self.modes = modes
        self.width = width
        self.nonlin = nonlinearity
        self.trainable = trainable

        # input channel is 2: (a(x), x)
        self.fc0 = nn.Linear(2, self.width)

        self.blocks = nn.ModuleList()

        for b in range(self.n_blocks):

            if b == self.n_blocks - 1:
                self.blocks.append(
                    FourierBlock1d(self.width, self.modes, None, trainable))
            else:
                self.blocks.append(
                    FourierBlock1d(self.width, self.modes, self.nonlin,
                                   trainable))

        # return to input space
        self.fc1 = nn.Linear(self.width, self.width * self.n_blocks)
        self.fc2 = nn.Linear(self.width * self.n_blocks, 1)

    def forward(self, x):
        """Apply kernel transform to input data.
        Args:
            x: the solution of the coefficient function and locations
                    (a(x), x); shape (batchsize, x=s, c=2)

            return: solution u(x); shape (batchsize, x=s, c=1)
        """

        x = self.fc0(x)

        x = einops.rearrange(x, 'b x c -> b c x')

        for block in self.blocks:
            x = block(x)

        x = einops.rearrange(x, 'b c x -> b x c')

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x


class SpectralConv2d(nn.Module):
    """Applies FFT, linear transform, and inverse FFT.

        Args:
            in_channels: number of input channels
            out_channels: number of output channels
            modes: list [x, y] of modes per dimension; number of Fourier modes
            to multiply (at most floor(N/2) + 1)
            train: Whether or not the weights are trainable parameters;
                   default = True
    """
    def __init__(self, in_channels, out_channels, modes, trainable=True):
        super(SpectralConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.trainable = trainable
        self.scale = (1 / (in_channels * out_channels))

        self.weights1 = nn.Parameter(self.scale * torch.rand(
            in_channels, out_channels, self.modes[0], self.modes[1], 2))
        self.weights2 = nn.Parameter(self.scale * torch.rand(
            in_channels, out_channels, self.modes[0], self.modes[1], 2))
        self.weights1.requires_grad = trainable
        self.weights2.requires_grad = trainable

    def forward(self, x):
        batchsize = x.shape[0]

        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x, 2,
                              norm="forward",
                              #normalized=True, onesided=True
                              )

        # Apply transform consisting of truncated Fourier modes
        out_ft = torch.zeros(batchsize,
                             self.in_channels,
                             x.size(-2),
                             x.size(-1) // 2 + 1,
                             2,
                             device=x.device)

        out_ft[:, :, :self.modes[0], :self.modes[1]] = \
            complex_mul2d(x_ft[:, :, :self.modes[0], :self.modes[1]],
                          self.weights1)

        out_ft[:, :, -self.modes[0]:, :self.modes[1]] = \
            complex_mul2d(x_ft[:, :, -self.modes[0]:, :self.modes[1]],
                          self.weights2)

        # Return to physical space
        x = torch.fft.irfft(out_ft,
                        2,
                        norm="forward",
                        #normalized=True,
                        #onesided=True,
                        #signal_sizes=(x.size(-2), x.size(-1))
                        )

        return x


class FourierBlock2d(torch.nn.Module):
    """Fourier kernel block implementing
                    u' = (W + K)(u),
       where W is defined by a convolution and K is defined by
       a spectral convolution in Fourier space.
    """
    def __init__(self, width, modes, nonlinearity, trainable=True):
        super(FourierBlock2d, self).__init__()
        """
        Args:
            modes: Number of Fourier modes to multiply (at most floor(N/2) + 1)
            width: Dimension of lifting transform performed by first layer
            nonlinearity: Component-wise nonlinear activation
            internal: True if block is not final output block
            train: Whether or not the weights are trainable parameters;
                   default = True
        """

        self.width = width
        self.modes = modes
        self.trainable = trainable
        self.nonlinearity = nonlinearity

        # SpectralConv2d(in_dim, out_dim, modes_x, modes_y)
        self.R = SpectralConv2d(self.width, self.width, self.modes,
                                self.trainable)

        # ConvNd(in_dim, out_dim, kernel_size)
        self.w = nn.Conv1d(self.width, self.width, 1)
        self.w.weight.requires_grad = self.trainable
        self.w.bias.requires_grad = self.trainable

    def forward(self, x):
        dimx, dimy = x.shape[2], x.shape[3]

        Rx = self.R(x)

        Wx = einops.rearrange(x, 'b c x y -> b c (x y)')
        Wx = self.w(Wx)
        Wx = einops.rearrange(Wx, 'b c (x y) -> b c x y', x=dimx, y=dimy)

        if self.nonlinearity is None:
            return Rx + Wx

        # (x1 + x2) = 'b c x y'
        return self.nonlinearity(Rx + Wx)


class FourierKernel2d(nn.Module):
    """The Fourier kernel containing a specified number of Fourier blocks.

    1. Lift the input to the desired channel dimension by self.fc0.
    2. Apply specified number of iterations of the operator
                            u' = (W + K)(u)
    3. Project from the channel space to the output space by self.fc1
        and self.fc2.
    """
    def __init__(self, n_blocks, modes, width, nonlinearity, trainable=True):
        """
        Args:
            blocks: Number of kernel integral transform blocks
            modes: Number of Fourier modes to multiply (at most floor(N/2) + 1)
            width: Dimension of lifting transform performed by first layer
            nonlinearity: Component-wise nonlinear activation
        """
        super(FourierKernel2d, self).__init__()

        self.n_blocks = n_blocks
        self.modes = modes
        self.width = width
        self.nonlin = nonlinearity
        self.trainable = trainable

        # lift to dimension = self.width
        # input channel is 3: (a(x, y), x, y)
        self.fc0 = nn.Linear(3, self.width)

        self.blocks = nn.ModuleList()

        for b in range(self.n_blocks):

            if b == self.n_blocks - 1:
                self.blocks.append(
                    FourierBlock2d(self.width,
                                   self.modes,
                                   None,
                                   trainable=self.trainable))
            else:
                self.blocks.append(
                    FourierBlock2d(self.width,
                                   self.modes,
                                   self.nonlin,
                                   trainable=self.trainable))

        # return to input space
        self.fc1 = nn.Linear(self.width, self.width * self.n_blocks)
        self.fc2 = nn.Linear(self.width * self.n_blocks, 1)

    def forward(self, x):
        """Apply kernel transform to input data.
        Args:
            x: the solution of the coefficient function and locations
                    (a(x, y), x, y); shape (batchsize, x=s, y=s, c=3)

            return: solution u(x, y); shape (batchsize, x=s, y=s, c=1)
        """
        x = self.fc0(x)

        x = einops.rearrange(x, 'b x y c -> b c x y')

        for block in self.blocks:
            x = block(x)

        x = einops.rearrange(x, 'b c x y -> b x y c')

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x
