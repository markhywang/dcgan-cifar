"""
Discriminator and Generate implementation from DCGAN paper

The original DCGAN paper can be found here: https://arxiv.org/abs/1511.06434
"""

# Imports
import torch
from torch import nn

# Hyperparameters
LEAKY_SLOPE = 0.2
KERNEL_SIZE = 4
STRIDE = 2
PADDING = 1


def init_weights(model: nn.Module) -> None:
    """
    Manually initializes weights according to the DCGAN paper
    """
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


class Discriminator(nn.Module):
    """
    Implementation for Discriminator model based on DCGAN paper
    Note that the input shape is: N x channels_img x 64 x 64
    """
    def __init__(self, channels_img: int, features_d : int) -> None:
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Conv2d( # input shape -> 32 x 32
                channels_img,
                features_d,
                kernel_size=KERNEL_SIZE,
                stride=STRIDE,
                padding=PADDING
            ),
            nn.LeakyReLU(LEAKY_SLOPE),
            self._convolution_block(features_d, features_d * 2), # 32 x 32 -> 16 x 16
            self._convolution_block(features_d * 2, features_d * 4), # 16 x 16 -> 8 x 8
            self._convolution_block(features_d * 4, features_d * 8),  # 8 x 8 -> 4 x 4
            nn.Conv2d( # 4 x 4 -> 1 x 1
                features_d * 8,
                1, # output dimension
                kernel_size=KERNEL_SIZE,
                stride=STRIDE,
                padding=0 # In final convolutional layer, don't add padding
            ),
            nn.Sigmoid() # 1 x 1 -> logit value from [0, 1]
        )
    
    def _convolution_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """
        Convolutional Neural Network (CNN) block for Discriminator model
        """
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                KERNEL_SIZE,
                STRIDE,
                PADDING,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(LEAKY_SLOPE)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.discriminator(x)


class Generator(nn.Module):
    """
    Implementation for Discriminator model based on DCGAN paper
    Note that the input shape is: N x latent_dim x 1 x 1
    """
    def __init__(self, latent_dim: int, channels_img: int, features_gen: int) -> None:
        super(Generator, self).__init__()
        self.generator = nn.Sequential(
            self._convolution_block( # input shape -> N x features_gen*16 x 4 x 4
                latent_dim,
                features_gen * 16,
                stride=1,
                padding=0
            ),
            self._convolution_block( # N x features_gen*16 x 4 x 4 -> 8 x 8
                features_gen * 16,
                features_gen * 8
            ),
            self._convolution_block( # 8 x 8 -> 16 x 16
                features_gen * 8,
                features_gen * 4
            ),
            self._convolution_block( # 16 x 16 -> 32 x 32
                features_gen * 4,
                features_gen * 2
            ),
            nn.ConvTranspose2d( # 32 x 32 -> 64 x 64
                features_gen * 2,
                channels_img,
                kernel_size=KERNEL_SIZE,
                stride=STRIDE,
                padding=PADDING,
                bias=False
            ),
            nn.Tanh() # 64 x 64 -> 64 x 64 normalized floats in interval [-1, 1]
        )
    
    def _convolution_block(self, in_channels: int, out_channels: int, 
                           stride: int = STRIDE, padding: int = PADDING) -> nn.Sequential:
        """
        Convolutional Neural Network (CNN) block for Generator model
        """
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                KERNEL_SIZE,
                stride,
                padding,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.generator(x)
