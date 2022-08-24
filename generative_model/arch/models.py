import torch.nn as nn

class GeneratorModel(nn.Module):
    """
    Generator model for WGAN-GP.

    Given a vector of random values (latent inputs) as input, this network
    generates data with the same structure as the training data.
    The goal is to train the generator to generate data that "fools" the
    Critic.
    """

    def __init__(self, dim_in, dim: int = 128):
        super(GeneratorModel, self).__init__()

        def genblock(dim_in, dim_out):
            
            block = nn.Sequential(nn.ConvTranspose2d(in_channels=dim_in,
                                                     out_channels=dim_out,
                                                     kernel_size=5,
                                                     stride=2,
                                                     padding=2,
                                                     output_padding=1,
                                                     bias=False),
                                  nn.BatchNorm2d(dim_out),
                                  nn.ReLU()
                                  )
            return block

        def genimg(dim_in):
            block = nn.Sequential(nn.ConvTranspose2d(in_channels=dim_in,
                                                     out_channels=1,
                                                     kernel_size=5,
                                                     stride=2,
                                                     padding=2,
                                                     output_padding=1,
                                                     ),
                                  nn.Tanh()
                                  )
            return block

        self.prepare = nn.Sequential(nn.Linear(dim_in, dim*8*4*4, bias=False),
                                     nn.BatchNorm1d(dim*8*4*4),
                                     nn.ReLU())
        self.generate = nn.Sequential(genblock(dim*8, dim*16),
					 genblock(dim*16, dim*8),
					   genblock(dim*8, dim*4),
					    genblock(dim*4, dim*2),
                                      genimg(dim*2))

    def forward(self, x):
        """Forward pass function."""
        x = self.prepare(x)
        x = x.view(x.size(0), -1, 4, 4)
        x = self.generate(x)
        return x

class CriticModel(nn.Module):
    """
    Critic model for WGAN-GP.

    Given batches of data containing observations from both the training data,
    and generated data from the generator, this network attempts to classify
    the observations as "real" or "generated".
    The goal is to train the discriminator to distinguish between real and
    generated data.
    """

    def __init__(self, dim_in, dim=128):
        super(CriticModel, self).__init__()

        def critic_block(dim_in, dim_out):
            block = nn.Sequential(nn.Conv2d(in_channels=dim_in,
                                            out_channels=dim_out,
                                            kernel_size=5,
                                            stride=2,
                                            padding=2),
                                  nn.InstanceNorm2d(dim_out, affine=True),
                                  nn.LeakyReLU(0.2))
            return block

        self.analyze = nn.Sequential(nn.Conv2d(in_channels=dim_in,
                                               out_channels=dim,
                                               kernel_size=5,
                                               stride=2,
                                               padding=2),
                                     nn.LeakyReLU(0.2),
                                     critic_block(dim, dim*2),
                                     critic_block(dim*2, dim*4),
                                     critic_block(dim*4, dim*8),
                                     nn.Conv2d(in_channels=dim*8,
                                               out_channels=1,
                                               kernel_size=4))

    def forward(self, x):
        """Forward pass function."""
        x = self.analyze(x)
        x = x.view(-1)
        return x
