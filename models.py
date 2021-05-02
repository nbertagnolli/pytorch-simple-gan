from typing import Optional
import math
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, input_length: int):
        super(Generator, self).__init__()
        self.dense_layer = nn.Linear(int(input_length), int(input_length))
        self.activation = nn.Sigmoid()

    def forward(self, x):
        return self.activation(self.dense_layer(x))


class Discriminator(nn.Module):
    def __init__(self, input_length: int):
        super(Discriminator, self).__init__()
        self.dense = nn.Linear(int(input_length), 1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        return self.activation(self.dense(x))


class DCGenerator(nn.Module):
    def __init__(self, input_length: int, n_channels: int,  num_base_filters: Optional[int]):
        super(DCGenerator, self).__init__()

        # Calculates the total number of layers
        number_of_layers = int(math.log(self.img_cols, 2) - 3)

        if self.num_base_filters is None:
            num_base_filters = 32 * 2 ** number_of_layers

        # Create the list to hold all sequential layers
        self.layers_list = []

        # Add the initial layer
        self.layers_list.append(nn.Linear(input_length, num_base_filters * 8 * 8))
        self.layers_list.append(nn.ReLU())

        # Add a scaled number of layers
        self.layers_list.append(nn.BatchNorm2d(128))
        self.layers_list.append(nn.Upsample(scale_factor=2))
        self.layers_list.append(nn.Conv2d(128, 128, 3, stride=1, padding=1))
        self.layers_list.append(nn.BatchNorm2d(128, 0.8))
        self.layers_list.append(nn.LeakyReLU(0.2, inplace=True))
        self.layers_list.append(nn.Upsample(scale_factor=2))
        self.layers_list.append(nn.Conv2d(128, 64, 3, stride=1, padding=1))
        self.layers_list.append(nn.BatchNorm2d(64, 0.8))
        self.layers_list.append(nn.LeakyReLU(0.2, inplace=True))
        self.layers_list.append(nn.Conv2d(64, n_channels, 3, stride=1, padding=1))
        self.layers_list.append(nn.Tanh())

        self.layers = nn.ModuleList(self.layers_list)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x


class DCDiscriminator(nn.Module):
    def __init__(self, image_size: int, input_channels: int):
        super(DCDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, stride=2, padding=1),
            nn.ELU(),
            nn.Dropout2d(0.2),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ELU(),
            nn.BatchNorm2d(64, 0.8),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ELU(),
            nn.BatchNorm2d(128, 0.8),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ELU(),
            nn.Linear(256 * 8 * 8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
