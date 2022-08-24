# Unconditional sampling from Generator network

import os
import re
import cv2
import torch
import random
import shutil
import argparse
import pygeostat as gs
import numpy as np
import torch.nn as nn

from tqdm import tqdm
from graphics.plot import *
from torch.autograd import Variable

n = 5
MAX = 10

def get_args():
    parser = argparse.ArgumentParser(description="Easily samples images from the Generator network!")

    parser.add_argument("--model_path", default="Epoch_155.ckpt", type=str, help="Generator model path")
    parser.add_argument("--num_samples", default=100, type=int, help="Number of samples to be generated")
    
    args = parser.parse_args()
    return args


def seeds(initial_seed: int) -> None:
    os.environ['PYTHONHASHSEED'] = str(initial_seed)

    # Torch RNG
    torch.manual_seed(initial_seed)
    torch.cuda.manual_seed(initial_seed)
    torch.cuda.manual_seed_all(initial_seed)

    # Python RNG
    np.random.seed(initial_seed)
    random.seed(initial_seed)


def load_model(latent_space:int , state_dictionary: dict) -> nn.Module:
    model = GeneratorModel(latent_size)
    model.load_state_dict(state_dictionary["Generator"])
    return model


class Unconditional(object):
    def __init__(self, dimension, generator, latent_size, latent_dist, use_cuda=False):
        self.use_cuda = use_cuda

        self.dimension = dimension
        self.generator = generator

        if self.use_cuda:
            self.generator = self.generator.cuda()

        self.latent_size = latent_size
        self.latent_dist = latent_dist

    def sample_batch(self, batch, m):
        config = [batch, self.latent_size]
        zhat = None
        if self.use_cuda:
            zhat = Variable(torch.FloatTensor(*config).cuda(), requires_grad=True)
        else:
            zhat = Variable(torch.FloatTensor(*config), requires_grad=True)
        zhat.retain_grad()
        if self.latent_dist == "normal":
            zhat.data = zhat.data.normal_(0, 1)
        elif self.latent_dist == "uniform":
            zhat.data = zhat.data.uniform_(-1, 1)
        return zhat

    def create_unconditional_simulations(self, batch, m):
        zhat = self.sample_batch(batch, m)
        if self.use_cuda:
            zhat = zhat.cuda()
        try:
            xhat = self.generator(zhat)
            return xhat.data.cpu().numpy()
        except RuntimeError as reError:
            print(reError)


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


def create_out_file(samples, n:int):
    for image in tqdm(samples,
                    desc="Saving samples from GAN, please wait...",
                    total=len(samples), colour='blue'):
        resized_image = cv2.resize(image, (150, 150))
        binarized_image = np.where(resized_image > 0.5, 1, 0)
        numpy_tensor = binarized_image.squeeze().ravel()
        try:
            # Loads file and save with new image
            dataset = np.loadtxt(f"data/generated.out")
            new_TI = np.column_stack((dataset, numpy_tensor))
            np.savetxt(fname = "data/generated.out", X = new_TI, 
                        newline = os.linesep,
                        header=f"150 150 1\n1\nfacies\n")
            new_TI = np.column_stack((dataset, numpy_tensor))
        
        except FileNotFoundError:
                np.savetxt(fname = f"data/generated.out",
                        X=numpy_tensor,
                        newline = os.linesep,
                        header=f"150 150 1\n1\nfacies\n")
    
    # Saves 
    numpy_array = np.loadtxt("data/generated.out")
    header_name = [f'ti_{idx}' for idx in range(numpy_array.reshape(n*MAX, 150, 150).shape[0])]
    header_name = f'\n'.join(header_name)

    # Save again with header
    np.savetxt(fname="data/generated.out",
                X=numpy_array,
                header=f"gan\n{numpy_array.reshape(n*MAX, 150, 150).shape[1]}\n{header_name}", comments="", fmt="%1d")


def __create_folders():
    os.makedirs("data/out_files", exist_ok=True)
    os.makedirs("data/parfiles", exist_ok=True)
    os.makedirs("data/simulations", exist_ok=True)
    os.makedirs("data/results", exist_ok=True)
    os.makedirs("data/results", exist_ok=True)


def create_ti_files(samples):
    for idx, im in enumerate(samples):
        im = cv2.resize(im, (150, 150))
        image_resized = np.where(im > 0.5, 1, 0)

        # save as .out
        np.savetxt(f"data/out_files/ti_{idx}.out", image_resized.reshape(-1), header="150 150 1\n1\nfacies", fmt="%1d",comments='')


def _execute_script(index: int):
    os.chdir("data/")
    script = f"""echo 'parfiles/snesim_gan_{index}.par' | wine snesim.exe"""
    os.system("bash -c '%s'" % script)


def simulate(samples_array, n):
    seed_initial: int = 69069
    n_reals = n*MAX
    seeds_list = gs.rseed_list(nseeds=n_reals, seed=seed_initial)

    for idx, image in enumerate(samples_array):
        image = cv2.resize(image, (150, 150))
        image = (image > 0.5)

        with open("data/gan.par", 'r') as parfile:
            par = parfile.read()
            par_replaced = par.replace(f"_0.out", f"_{idx}.out")
            par_ready = re.sub("\d{5,6}", f"{seeds_list[idx]}", par_replaced)
            
            with open(f"data/parfiles/snesim_gan_{idx}.par", 'w+') as file:
                file.write(par_ready)
        _execute_script(idx)
        os.chdir("..")


def plots():
    print("[INFO] Loading simulations", end="\r")
    realizations = concatenate_out_files("data/simulations/")

    ti_dict = dict()
    for idx, realization in enumerate(realizations):
        ti_dict[f'ti_{idx+1}'] = np.array(realization.reshape(-1))

    dataframe = pd.DataFrame(ti_dict)
    plot_uncertainty(dataframe)
    plot_realizations_grid(realizations)

    file = read_conditional_samples("../snesim/data/snesim.out")["D"]
    realizations = file[:, 0].reshape(50, 150, 150)
    
    print("[INFO] Loading simulations", end="\r")
    snesim_gen_realizations = concatenate_out_files("data/simulations/")
    np.save("data/realizations.npy", snesim_gen_realizations, allow_pickle=True)

    proportions_comparison(realizations,
                       snesim_gen_realizations)

    histplots(realizations, dataframe)

    mds_plots(snesim_realizations_path="../snesim/data/realizations.npy",
            gan_realizations_path="data/realizations.npy")


if __name__ == "__main__":
    seeds(69069)

    # Create folders and get parameters
    __create_folders()
    param = get_args()

    latent_size = 100
    state_dict = torch.load(param.model_path, map_location='cpu')

    generator = load_model(latent_size, state_dict)

    uncond = Unconditional(1, generator, latent_size, "normal")

    print("[INFO] Sampling images from GAN")
    all_samples = []

    import gc
    for i in range(n):
        samples = uncond.create_unconditional_simulations(MAX, [i, i, i])
        all_samples.append(samples)
        gc.collect()
    samples = np.array(all_samples).reshape(n*MAX, 1, 128, 128)
    samples_arr = np.where(np.concatenate(samples, 0)*0.5+0.5 >= 0.5, 1.0, 0.0)

    print("[INFO] Images sampled!")
    create_out_file(samples_arr, n)

    # Performs SNESIM simulations
    # Prepare each TI
    create_ti_files(samples_arr)

    # Simulated
    simulate(samples_arr, n)
    
    # Starts plotting
    plots()

    # Remove unnecessary files after execution
    shutil.rmtree("data/parfiles")
    shutil.rmtree("data/out_files")  
