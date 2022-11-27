"""AlexNet implementation on Pytorch."""

import torch
import torch.nn as nn


class AlexNet(nn.Module):
    """
    Neural net that mimics AlexNet from scratch.

    Attributes
    ----------
    ConvolutedBlock
        Block conformed by 5 Conv2d layers.
        RElU is used for activation layers, and we perform Batch normalization after every convolution layer.
    DenseBlock
        Block composed by three linear layers.
        RElU is used for activation layers and dropout=0.5 is applied for every layer.

    Notes
    -----
    - The architecture follows the exact same structure than AlexNet (see: https://en.wikipedia.org/wiki/AlexNet).
    - Some output channels and output features have been reduced in size due to performance improvements.
    - We don't apply a sigmoid activation in the last layer. This is because we are using
    torch.nn.functional.binary_cross_entropy_with_logits as the current loss which proves to be more stable than applying a sigmoid layer + binary_cross_entropy.

    Parameters
    ----------
    nn : torch.nn.Module
        Base class for all neural network modules. All pytorch models should be a subclass this class.
    """

    def __init__(self):
        """Init method."""
        super(AlexNet, self).__init__()
        self.ConvolutedBlock = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=0
            ),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),
            nn.Conv2d(
                in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),
            nn.Conv2d(
                in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256, out_channels=96, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),
        )

        self.DenseBlock = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=3456, out_features=512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=512, out_features=248),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=248, out_features=1),
        )

    def forward(self, x: torch.tensor):
        """Forward pass."""
        x = self.ConvolutedBlock(x)
        x = torch.flatten(x, 1)
        x = self.DenseBlock(x)

        return x
