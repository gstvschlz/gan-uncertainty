import random

import torch

SEED = random.randint(1, 10000)
NUM_WORKERS = 1
BATCH_SIZE = 16
IMAGE_SIZE = 64
NUM_CHANNELS = 3
NZ = 64
NGF = 64
NDF = 64
NUM_EPOCHS = 10
LEARNING_RATE = 0.0002
BETA1 = 0.5
NGPU = 0
DEVICE = torch.device("cuda:0" if (torch.cuda.is_available() and NGPU > 0) else "cpu")
