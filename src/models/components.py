import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.utils import make_grid

class ResidualBlock(nn.Module):
    """
    Simple Residual Block for the Encoder and Decoder Networks as per the original VQVAE paper, which
    is implemented as:
        ReLU -> 3 x 3 Conv2d -> ReLU -> 1 x 1 Conv2d

    References:
        - Van Den Oord, A., & Vinyals, O. (2017). Neural discrete representation learning. 
            Advances in neural information processing systems, 30.

    Args:
        in_channels (int): Number of input channels.
        num_hiddens (int): Number of hidden channels for the residual block output.
        num_residual_hiddens (int): Number of hidden channels for the residual layers.

    Returns:
        torch.Tensor: Output tensor.

    Shape:
        - Input: (B, in_channels, H, W)
        - Output: (B, num_hiddens, H, W)
        - H and W depend on the input size.

    Examples:
        >>> x = torch.randn(1, 3, 32, 32)
        >>> res_block = ResidualBlock(in_channels=3, num_hiddens=64, num_residual_hiddens=32)
        >>> out = res_block(x)
        >>> out.shape
        torch.Size([1, 64, 32, 32])
    """
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(ResidualBlock, self).__init__()

        self._block = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=num_residual_hiddens,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=num_residual_hiddens,
                out_channels=num_hiddens,
                kernel_size=1,
                stride=1,
                bias=False),
        )

    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    """
    Stacked Residual Blocks for the Encoder and Decoder Networks as per the original VQVAE paper, which
    is implemented as:
        ResidualBlock -> ResidualBlock -> ... -> ResidualBlock

    References:
        - Van Den Oord, A., & Vinyals, O. (2017). Neural discrete representation learning. 
            Advances in neural information processing systems, 30.

    Args:
        in_channels (int): Number of input channels.
        num_hiddens (int): Number of hidden channels for the residual block output.
        num_residual_hiddens (int): Number of hidden channels for the residual layers.

    Returns:
        torch.Tensor: Output tensor.

    Shape:
        - Input: (B, in_channels, H, W)
        - Output: (B, num_hiddens, H, W)
        - H and W depend on the input size.

    Examples:
        >>> x = torch.randn(1, 3, 32, 32)
        >>> res_stack = ResidualStack(in_channels=3, num_hiddens=64, num_residual_hiddens=32)
        >>> out = res_stack(x)
        >>> out.shape
        torch.Size([1, 64, 32, 32])
    """
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens, num_residual_blocks):
        super(ResidualStack, self).__init__()
        self._layers = nn.ModuleList(
            [
                ResidualBlock(
                    in_channels=in_channels if i == 0 else num_hiddens,
                    num_hiddens=num_hiddens,
                    num_residual_hiddens=num_residual_hiddens,
                )
                for i in range(num_residual_blocks)
            ]
        )

    def forward(self, x):
        for layer in self._layers:
            x = layer(x)
        return F.relu(x)

class Encoder(nn.Module):
    """
    Encoder Network for the Vector Quantization Variational Autoencoder (VQVAE) as per the original paper, which
    is implemented as:
        Conv2d -> ReLU -> ResidualStack -> Flatten -> Linear -> Linear

    References:
        - Van Den Oord, A., & Vinyals, O. (2017). Neural discrete representation learning. 
            Advances in neural information processing systems, 30.

    Args:
        in_channels (int): Number of input channels.
        num_hiddens (int): Number of hidden channels for the residual block output.
        num_residual_hiddens (int): Number of hidden channels for the residual layers.
        num_residual_blocks (int): Number of residual blocks.

    Returns:
        torch.Tensor: Output tensor.

    Shape:
        - Input: (B, in_channels, H, W)
        - Output: (B, num_hiddens, H, W)
        - H and W depend on the input size.
    """
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens, num_residual_blocks):
        super(Encoder, self).__init__()

        self._conv_1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_hiddens//2,
            kernel_size=4, stride=2, padding=1)

        self._conv_2 = nn.Conv2d(
            in_channels=num_hiddens//2,
            out_channels=num_hiddens,
            kernel_size=4, stride=2, padding=1)

        self._conv_3 = nn.Conv2d(
            in_channels=num_hiddens,
            out_channels=num_hiddens,
            kernel_size=3, stride=1, padding=1)

        self._residual_stack = ResidualStack(
            in_channels=num_hiddens,
            num_hiddens=num_hiddens,
            num_residual_hiddens=num_residual_hiddens,
            num_residual_blocks=num_residual_blocks)

    def forward(self, x):
        x = F.relu(self._conv_1(x))
        x = F.relu(self._conv_2(x))
        x = self._conv_3(x)

        return self._residual_stack(x)


class Decoder(nn.Module):
    """
    Decoder Network for the Vector Quantization Variational Autoencoder (VQVAE) as per the original paper, which
    is implemented as:
        Linear -> Unflatten -> ResidualStack -> Conv2d -> Sigmoid

    References:
        - Van Den Oord, A., & Vinyals, O. (2017). Neural discrete representation learning. 
            Advances in neural information processing systems, 30.

    Args:
        in_channels (int): Number of input channels.
        num_hiddens (int): Number of hidden channels for the residual block output.
        num_residual_hiddens (int): Number of hidden channels for the residual layers.
        num_residual_blocks (int): Number of residual blocks.

    Returns:
        torch.Tensor: Output tensor.

    Shape:
        - Input: (B, in_channels, H, W)
        - Output: (B, num_hiddens, H, W)
        - H and W depend on the input size.
    """
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens, num_residual_blocks):
        super(Decoder, self).__init__()

        self._residual_stack = ResidualStack(
            in_channels=num_hiddens,
            num_hiddens=num_hiddens,
            num_residual_hiddens=num_residual_hiddens,
            num_residual_blocks=num_residual_blocks)

        self._conv_trans_1 = nn.ConvTranspose2d(
            in_channels=num_hiddens,
            out_channels=num_hiddens//2,
            kernel_size=4, stride=2, padding=1)

        self._conv_trans_2 = nn.ConvTranspose2d(
            in_channels=num_hiddens//2,
            out_channels=in_channels,
            kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = self._residual_stack(x)
        x = F.relu(self._conv_trans_1(x))

        return self._conv_trans_2(x)



