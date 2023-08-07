import argparse
import logging
import os
import re
from typing import Dict, Union

import cv2
import numpy as np
import pygeostat as gs
import torch
import torch.nn as nn
from skimage.metrics import structural_similarity as compare_ssim
from torch.autograd import Variable

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeneratorModel(nn.Module):
    def __init__(self, dim_in: int, dim: int = 128):
        super(GeneratorModel, self).__init__()

        def genblock(dim_in: int, dim_out: int) -> nn.Sequential:
            return nn.Sequential(
                nn.ConvTranspose2d(
                    dim_in,
                    dim_out,
                    5,
                    2,
                    2,
                    1,
                    bias=False,
                ),
                nn.BatchNorm2d(dim_out),
                nn.ReLU(),
            )

        def genimg(dim_in: int) -> nn.Sequential:
            return nn.Sequential(
                nn.ConvTranspose2d(
                    dim_in,
                    1,
                    5,
                    2,
                    2,
                    1,
                ),
                nn.Tanh(),
            )

        self.prepare = nn.Sequential(
            nn.Linear(dim_in, dim * 8 * 4 * 4, bias=False),
            nn.BatchNorm1d(dim * 8 * 4 * 4),
            nn.ReLU(),
        )
        self.generate = nn.Sequential(
            genblock(dim * 8, dim * 16),
            genblock(dim * 16, dim * 8),
            genblock(dim * 8, dim * 4),
            genblock(dim * 4, dim * 2),
            genblock(dim * 2, dim),
            genimg(dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.prepare(x)
        x = x.view(x.size(0), -1, 4, 4)
        return self.generate(x)


def load_model(state_dictionary: dict, latent_space: int = 100) -> nn.Module:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = GeneratorModel(latent_space).to(device)
    model.load_state_dict(state_dictionary["Generator"])
    return model


class Unconditional:
    def __init__(self, dimension: int, generator: nn.Module, latent_size: int):
        self.dimension = dimension
        self.generator = generator
        self.latent_size = latent_size

    def sample_batch(self, batch: int) -> torch.Tensor:
        zhat = Variable(torch.randn(batch, self.latent_size), requires_grad=True)
        zhat.retain_grad()
        return zhat.cuda()

    def create_unconditional_simulations(self, batch: int) -> torch.Tensor:
        zhat = self.sample_batch(batch)
        try:
            xhat = self.generator(zhat)
            return xhat.detach()
        except RuntimeError as reError:
            print(reError)


def read_conditional_samples(filename: str, nanval: int = -997799) -> np.ndarray:
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"File '{filename}' does not exist")

    with open(filename, mode="r", encoding="utf8") as file:
        title = file.readline().strip("\n")
        n_cols = int(file.readline())
        header = [file.readline().strip("\n") for _ in range(n_cols)]

    try:
        D = np.genfromtxt(filename, skip_header=2 + n_cols)[
            :, 0
        ]  # Get the first column only
    except Exception as exc:
        raise RuntimeError(f"Could not read data from '{filename}'") from exc

    D[D == nanval] = np.nan

    return D


def compile_samples(directory: str, nanval: int = -997799) -> Dict[str, np.ndarray]:
    files = [f for f in os.listdir(directory) if f.endswith(".out")]
    samples = {}

    for file in files:
        filepath = os.path.join(directory, file)
        samples[file] = read_conditional_samples(filepath, nanval)

    return samples


def create_ti_files(samples, output_directory):
    for idx, im in enumerate(samples):
        if im.dtype not in [np.uint8, np.float32, np.float64]:
            im = (im * 255).astype(np.uint8)
        im = cv2.resize(im, (150, 150))
        image_resized = np.where(im > 0.5, 1, 0)

        print(f'Saving TI #{idx}')
        np.savetxt(
            f"{output_directory}/ti_{idx}.out",
            image_resized.reshape(-1),
            header="150 150 1\n1\nfacies",
            fmt="%1d",
            comments="",
        )


def create_parfile(index: int, image, seed, par_file, output_directory):
    if image.dtype not in [np.uint8, np.float32, np.float64]:
        image = image.astype(np.uint8)

    # Resize the image
    image = cv2.resize(image, (150, 150))
    image = image > 0.5

    with open(par_file, "r") as parfile:
        par = parfile.read()
        par_replaced = par.replace(f"_0.out", f"_{index}.out")
        par_ready = re.sub("\d{5,6}", f"{seed}", par_replaced)
        output_path = f"{output_directory}/windows_{index}.par"
        with open(output_path, "w+") as file:
            file.write(par_ready)


def _execute_script(index: int):
    os.chdir("/home/user/gan-uncertainty/generative_model/data")
    script = f"echo 'parfiles/windows_{index}.par' | wine snesim.exe"
    os.system("bash -c '%s'" % script)


def simulate(samples_array):
    seed_initial: int = 69069
    n_reals = 100
    seeds_list = gs.rseed_list(nseeds=n_reals, seed=seed_initial)
    os.makedirs("/home/user/gan-uncertainty/generative_model/data/parfiles",
                exist_ok=True)
    os.makedirs("/home/user/gan-uncertainty/generative_model/data/simulations", exist_ok=True)
    for idx, image in enumerate(samples_array):
        create_parfile(index=idx, image=image, seed=seeds_list[idx],
                       par_file="/home/user/gan-uncertainty/generative_model/data/gan.par",
                       output_directory="/home/user/gan-uncertainty/generative_model/data/parfiles")

    for idx in range(len(samples_array)):
        _execute_script(idx)


def main(args):
    num_samples = args.num_samples
    latent_size = args.latent_size
    dimension = args.dimension
    model_path = args.model_path
    target_image_path = args.target_image_path
    output_file = args.output_file
    simulations_dir = args.simulations_dir
    snesim_simulations_file = args.snesim_simulations_file

    state_dict = torch.load(model_path)
    gen_dict = {k.replace("module.", ""): v for k, v in state_dict["Generator"].items()}
    state_dict["Generator"] = gen_dict
    generator = load_model(state_dict, latent_size)
    uncond = Unconditional(dimension, generator, latent_size)

    all_samples = torch.zeros((num_samples, 250, 250)).cuda()

    sample_count = 0

    ti = cv2.imread(target_image_path, cv2.IMREAD_GRAYSCALE)
    ti = torch.from_numpy((ti < 127).astype(int)).cuda()
    ti = (
        torch.nn.functional.interpolate(ti.unsqueeze(0).unsqueeze(0), size=(250, 250))
        .squeeze(0)
        .squeeze(0)
    )

    while sample_count < num_samples:
        sample = uncond.create_unconditional_simulations(10)
        for real in sample:
            real = torch.nn.functional.interpolate(
                real.unsqueeze(0), size=(250, 250)
            ).squeeze(0)
            similarity_score = compare_ssim(ti.cpu().numpy(), real.cpu().numpy())
            if similarity_score >= 0.9:
                all_samples[sample_count, :, :] = real
                sample_count += 1
            if sample_count == num_samples:
                break
    samples_arr = (all_samples.cpu().numpy() >= 0.5).astype(int)
    np.save(output_file, samples_arr, allow_pickle=True)

    # Create each TI file
    os.makedirs(args.output_dir, exist_ok=True)
    create_ti_files(samples_arr, args.output_dir)
    simulate(samples_arr)
    samples = compile_samples(simulations_dir)
    np.save(snesim_simulations_file, samples, allow_pickle=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script for running image generation and similarity comparison"
    )
    parser.add_argument(
        "--num_samples", type=int, default=100, help="Number of samples to generate"
    )
    parser.add_argument(
        "--latent_size",
        type=int,
        default=100,
        help="Latent size for the Generator Model",
    )
    parser.add_argument(
        "--dimension", type=int, default=1, help="Dimension for Unconditional sampling"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the saved Generator model",
    )
    parser.add_argument(
        "--target_image_path", type=str, help="Path to the target image"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="generator_samples",
        help="Output file to save the generated samples",
    )
    parser.add_argument(
        "--simulations_dir",
        type=str,
        default="/home/user/gan-uncertainty/generative_model/data/simulations",
        help="Directory of the simulation files",
    )
    parser.add_argument(
        "--snesim_simulations_file",
        type=str,
        default="/home/user/gan-uncertainty/generative_model/data/snesim_simulations",
        help="File to save the snesim simulations",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/user/gan-uncertainty/generative_model/data/out_files",
        help="Directory to save the output files",
    )
    parser.add_argument(
        "--par_file",
        type=str,
        default="/home/user/gan-uncertainty/generative_model/data/gan.par",
        help="Path to the parameter file",
    )
    args = parser.parse_args()

    main(args)
