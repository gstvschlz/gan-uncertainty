import torch.nn as nn

import var


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.ngpu = var.NGPU
        self.main = nn.Sequential(

            # input is Z, going into a convolution
            nn.ConvTranspose2d(var.NZ, var.NGF * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(var.NGF * 8),
            nn.ReLU(True),
            # state size. (var.NGF*8) x 4 x 4
            nn.ConvTranspose2d(var.NGF * 8, var.NGF * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(var.NGF * 4),
            nn.ReLU(True),
            # state size. (var.NGF*4) x 8 x 8
            nn.ConvTranspose2d(var.NGF * 4, var.NGF * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(var.NGF * 2),
            nn.ReLU(True),
            # state size. (var.NGF*2) x 16 x 16
            nn.ConvTranspose2d(var.NGF * 2, var.NGF, 4, 2, 1, bias=False),
            nn.BatchNorm2d(var.NGF),
            nn.ReLU(True),
            # state size. (var.NGF) x 32 x 32
            nn.ConvTranspose2d(var.NGF, var.NUM_CHANNELS, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (var.NUM_CHANNELS) x 64 x 64
        )

    def forward(self, input):
        """
        The forward pass function for the Generator.

        Arguments:
            input: Takes as argument the input tensor type vector that will be fed to the neural network.
        Returns:
            Returns the output which will be the generated images.
        """
        return self.main(input)
