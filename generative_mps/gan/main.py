import os
from random import random
import shutil
import time

import cv2
import numpy as np
import tensorboardX
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import yaml
from PIL import Image
from imageio import imsave
from skimage.util import view_as_windows
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.wgan_gp import CriticModel, GeneratorModel

DEVICE = torch.device("cuda:0")


def gradient_penalty(x, y, f):
    shape = [x.size(0)] + [1] * (x.dim() - 1)
    #alpha = torch.rand(shape).to(DEVICE, non_blocking=True)
    alpha = torch.rand(shape)
    z = x + alpha * (y - x)
    z = Variable(z, requires_grad=True)
    #z = z.to(DEVICE, non_blocking=True)
    o = f(z)
    #g = grad(o, z, grad_outputs=torch.ones(o.size()).to(DEVICE, non_blocking=True),
    #         create_graph=True)[0].view(z.size(0), -1)
    g = grad(o,z, grad_outputs=torch.ones(o.size()), create_graph=True)[0].view(z.size(0), -1)
    gp = ((g.norm(p=2, dim=1)) ** 2).mean()
    return gp


def save_checkpoint(state, save_path, is_best=True, max_keep=None):
    # save checkpoint
    torch.save(state, save_path)

    # deal with max_keep
    save_dir = os.path.dirname(save_path)
    list_path = os.path.join(save_dir, 'latest_checkpoint')

    save_path = os.path.basename(save_path)
    if os.path.exists(list_path):
        with open(list_path) as f:
            ckpt_list = f.readlines()
            ckpt_list = [save_path + '\n'] + ckpt_list
    else:
        ckpt_list = [save_path + '\n']

    if max_keep is not None:
        for ckpt in ckpt_list[max_keep:]:
            ckpt = os.path.join(save_dir, ckpt[:-1])
            if os.path.exists(ckpt):
                os.remove(ckpt)
        ckpt_list[max_keep:] = []

    with open(list_path, 'w') as f:
        f.writelines(ckpt_list)

    # copy best
    if is_best:
        shutil.copyfile(save_path, os.path.join(save_dir, 'best_model.ckpt'))


def load_checkpoint(ckpt_dir_or_file, map_location=None, load_best=False):
    if os.path.isdir(ckpt_dir_or_file):
        if load_best:
            ckpt_path = os.path.join(ckpt_dir_or_file, 'best_model.ckpt')
        else:
            with open(os.path.join(ckpt_dir_or_file, 'latest_checkpoint')) as f:
                ckpt_path = os.path.join(ckpt_dir_or_file, f.readline()[:-1])
    else:
        ckpt_path = ckpt_dir_or_file
    ckpt = torch.load(ckpt_path, map_location=map_location)
    print('[INFO] Loading checkpoint from %s succeed!' % ckpt_path)
    return ckpt


def config():
    with open(r"C:\Users\algocompretto\PycharmProjects\gan-for-mps\generative_mps\data\parameters.yaml", 'r') as stream:
        try:
            parsed_yaml = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(f"Exception occurred when opening .yaml config file! "
                  f"Exception: {exc}")
    return parsed_yaml


"""### Creating dataset of sliding patches on Strebelle TI"""

"""Sliding through Strebelle TI and saving as image."""


def generate_windows(args_dict: dict):
    """
    Generate sliding windows using scikit-image function `view_as_windows`.

    Parameters
    ----------
    args_dict : dict
        Parsed arguments as dictionary.
    Returns
    -------
    windows : np.ndarray
        Array with batch size containing saved images.
    """
    ti32 = cv2.imread(args_dict["training_image"], cv2.COLOR_BGR2GRAY)

    _, ti = cv2.threshold(ti32, 127, 255, cv2.THRESH_BINARY)

    patch_array = view_as_windows(ti, (150, 150))
    return patch_array


def save_generated_images(windowed_images, args_dict: dict) -> None:
    """
    Save each patch of Strebelle image to output directory.

    Parameters
    ----------
    windowed_images : np.ndarray
        Array of images.
    args_dict : dict
        User defined parameters.
    """
    for i, batch_ti in tqdm(enumerate(windowed_images),
                            desc="Saving each image from batch dimension.",
                            total=windowed_images.shape[0],
                            ncols=100):
        for j, t in enumerate(batch_ti):
            ti_resized = cv2.resize(t, (128, 128))
            imsave(f"{args_dict['output_dir']}/strebelle_{i}_{j}_{time.time()}.png", ti_resized)
            #list_of_transf = [transforms.RandomVerticalFlip(),
            #                transforms.RandomHorizontalFlip()]
            #for flip in list_of_transf:
            #    im = flip(ti_resized)


if __name__ == "__main__":
    # Get parameters
    param = config()

    # Create necessary folders
    os.makedirs(param["output_dir"], exist_ok=True)

    if not os.path.exists(param["training_image"]):
        raise FileNotFoundError("Could not find Strebelle training image" +
                                " in path!")

    print("Generating sliding windows, please wait...")
    windows = generate_windows(args_dict=param)

    print("Saving all sliding windows...")
    save_generated_images(windows, param)

"""
WGAN with Gradient Penalty training module.

This module trains the WGAN model according to user-specified parameters. If
wanted, you can specify each parameter accordingly.

Example
-------
Usage example::

    $ python3 wgan.py --n_epochs 10_000 --cuda True --batch_size 64

Notes
-----
    Make sure you have CUDA device before setting ``CUDA`` flag as True.
"""


def watch_for_checkpoints(args_dict, critic, generator, critic_opt, gen_opt):
    """
    Scan directories to see if there are checkpoints saved.

    Parameters
    ----------
    args_dict : argparse.args
        Parameters defined for model training.
    critic : nn.Module
        The Critic model to be loaded if any checkpoint is found.
    generator : nn.Module
        The Critic model to be loaded if any checkpoint is found
    critic_opt : torch.optim.Optimizer
        PyTorch optimizer for the Critic model.
    gen_opt : torch.optim.Optimizer
        PyTorch optimizer for the Generator model.

    Returns
    -------
    start_epoch : int
        The epoch to resume the training, if no checkpoint is found, then
        starts at epoch zero.
    """
    checkpoint = args_dict["checkpoint"]
    save_dir = args_dict["sample_images"]

    # Check if path exists
    if not isinstance(checkpoint, (list, tuple)):
        paths = [checkpoint]
        for path in paths:
            if not os.path.isdir(path):
                os.makedirs(path)
    if not isinstance(save_dir, (list, tuple)):
        paths = [save_dir]
        for path in paths:
            if not os.path.isdir(path):
                os.makedirs(path)
    try:
        # Loads checkpoint and changes state dictionary
        ckpt = load_checkpoint(checkpoint)
        start_epoch = ckpt['epoch']
        critic.load_state_dict(ckpt['D'])
        generator.load_state_dict(ckpt['Generator'])
        critic_opt.load_state_dict(ckpt['d_optimizer'])
        gen_opt.load_state_dict(ckpt['g_optimizer'])
    except FileNotFoundError:
        print('[*] No checkpoint!')
        start_epoch = 0

    return start_epoch


def train(args_dict) -> None:
    """
    Train both Generative and Critic model.

    Parameters
    ----------
    args_dict : argparse.args
        Parameters defined for model training.
    """
    #critic = CriticModel(args_dict["num_channels"]).to(DEVICE, non_blocking=True)
    #generator = GeneratorModel(args_dict["latent_dim"]).to(DEVICE, non_blocking=True)
    critic = CriticModel(args["num_channels"])
    generator = GeneratorModel(args["latent_dim"])

    # Instantiates optimizers
    G_opt = torch.optim.Adam(
        generator.parameters(), lr=args_dict["learning_rate"], betas=(0.5, 0.999))

    C_opt = torch.optim.Adam(
        critic.parameters(), lr=args_dict["learning_rate"], betas=(0.5, 0.999))

    start_epoch = watch_for_checkpoints(args_dict, critic,
                                        generator, C_opt, G_opt)

    # Loading Dataset
    transformation_funcs = transforms.Compose([
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=1),
        transforms.Normalize([0.5], [0.5])
    ])

    writer = tensorboardX.SummaryWriter(".logs/wgan-gp")

    data = torchvision.datasets.ImageFolder(
        args_dict["images_path"], transform=transformation_funcs)
    dataloader = DataLoader(data, batch_size=args_dict["batch_size"],
                            shuffle=True, num_workers=args_dict["num_workers"],
                            pin_memory=True)

    # Starting training loop
    for epoch in tqdm(range(start_epoch, args_dict["n_epochs"]),
                      desc="Training progress",
                      total=args_dict["n_epochs"] - start_epoch, position=0,
                      ncols=100, leave=True, colour='green'):
        start_time = time.time()
        critic_loss = []
        gen_loss = []
        generator.train()

        # Sampling from dataloader
        for i, (images, _) in enumerate(dataloader):
            step = epoch * len(dataloader) + i + 1
            images = Variable(images)
            batch = images.size(0)

            if args_dict["cuda"]:
                images = images.to(DEVICE, non_blocking=True)

            # Creates random noise
            z = Variable(torch.randn(batch, args_dict["latent_dim"]))

            if args_dict["cuda"]:
                z = z.to(DEVICE, non_blocking=True)

            # Sends random noise to Generator and gets Critic output
            generated = generator(z)
            real_criticized = critic(images)
            fake_criticized = critic(generated)

            # Compare distance between critic output
            em_distance = real_criticized.mean() - fake_criticized.mean()
            grad_penalty = gradient_penalty(images.data, generated.data, critic)

            # Calculates critic loss
            CriticLoss = -em_distance + grad_penalty * 10

            # Append to list for logging purpose
            critic_loss.append(CriticLoss.item())
            critic.zero_grad()
            CriticLoss.backward()
            C_opt.step()

            # Logs to tensorboard
            writer.add_scalar('Critic/em_dist', em_distance.data.cpu().numpy(),
                              global_step=step)
            writer.add_scalar('Critic/gradient_penalty', grad_penalty.data
                              .cpu().numpy(),
                              global_step=step)
            writer.add_scalar('Critic/critic_loss', CriticLoss.data
                              .cpu().numpy(),
                              global_step=step)

            if step % args_dict["n_critic"] == 0:
                # Random latent noise
                z = Variable(torch.randn(batch, args_dict["latent_dim"]))

                if args_dict["cuda"]:
                    z = z.to(DEVICE, non_blocking=True)

                # Generate new images from this latent vector
                generated = generator(z)

                fake_criticized = critic(generated)
                GeneratorLoss = -fake_criticized.mean()

                # Append to list for logging purpose
                gen_loss.append(GeneratorLoss.item())

                # Backward pass
                critic.zero_grad()
                generator.zero_grad()
                GeneratorLoss.backward()
                G_opt.step()

                # Logs loss scalar to tensorboard
                writer.add_scalars('Generator',
                                   {"g_loss": GeneratorLoss.data.cpu().numpy()},
                                   global_step=step)

                print(
                    f"Epoch {epoch + 1} : {i + 1}/{len(dataloader)}:" +
                    f"{round((time.time() - start_time) / 60, 2)} minutes",
                    end='\r')

        # Switch to evaluation mode and sample new images
        generator.eval()

        # Generate new samples to save
        z_sample = Variable(torch.randn(100, args_dict["latent_dim"]))
        if args_dict["cuda"]:
            z_sample = z_sample.to(DEVICE, non_blocking=True)

        fake_gen_images = (generator(z_sample).data + 1) / 2.0

        torchvision.utils.save_image(
            fake_gen_images,
            args_dict["sample_images"] + '/Epoch ' + str(epoch + 1) + ".jpg",
            nrow=10)

        x = torchvision.utils.make_grid(fake_gen_images, nrow=5)
        writer.add_image("Generated", x, step)

        # Save checkpoints
        save_checkpoint({'epoch': epoch + 1,
                         'D': critic.state_dict(),
                         'Generator': generator.state_dict(),
                         'd_optimizer': C_opt.state_dict(),
                         'g_optimizer': G_opt.state_dict()},
                        f'wgan/epoch_({epoch + 1}).ckpt',
                        max_keep=5)


if __name__ == "__main__":
    args = config()
    train(args)
