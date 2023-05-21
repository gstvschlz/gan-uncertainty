import os
import cv2
import sys
import yaml
import time
import torch
import shutil
import torchvision
import tensorboardX
import torch.nn as nn
import torchvision.transforms as transforms

from tqdm import tqdm
from imageio import imsave
from torch.autograd import grad
from arch.models import CriticModel, GeneratorModel
from torch.autograd import Variable
from skimage.util import view_as_windows

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Filtering warnings
if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")


def gradient_penalty(x, y, f):
    shape = [x.size(0)] + [1] * (x.dim() - 1)
    alpha = torch.rand(shape).to(DEVICE, non_blocking=True)
    z = x + alpha * (y - x)
    z = Variable(z, requires_grad=True)
    z = z.to(DEVICE, non_blocking=True)
    o = f(z)
    g = grad(
        o,
        z,
        grad_outputs=torch.ones(o.size()).to(DEVICE, non_blocking=True),
        create_graph=True,
    )[0].view(z.size(0), -1)
    gp = ((g.norm(p=2, dim=1)) ** 2).mean()
    return gp


def save_checkpoint(state, save_path, is_best=True, max_keep=None):
    # save checkpoint
    torch.save(state, save_path)

    # deal with max_keep
    save_dir = os.path.dirname(save_path)
    list_path = os.path.join(save_dir, "latest_checkpoint")

    save_path = os.path.basename(save_path)
    if os.path.exists(list_path):
        with open(list_path) as f:
            ckpt_list = f.readlines()
            ckpt_list = [save_path + "\n"] + ckpt_list
    else:
        ckpt_list = [save_path + "\n"]

    if max_keep is not None:
        for ckpt in ckpt_list[max_keep:]:
            ckpt = os.path.join(save_dir, ckpt[:-1])
            if os.path.exists(ckpt):
                os.remove(ckpt)
        ckpt_list[max_keep:] = []

    with open(list_path, "w") as f:
        f.writelines(ckpt_list)

    # copy best
    if is_best:
        shutil.copyfile(save_path, os.path.join(save_dir, "best_model.ckpt"))


def load_checkpoint(ckpt_dir_or_file, map_location=None, load_best=False):
    if os.path.isdir(ckpt_dir_or_file):
        if load_best:
            ckpt_path = os.path.join(ckpt_dir_or_file, "best_model.ckpt")
        else:
            with open(os.path.join(ckpt_dir_or_file, "latest_checkpoint")) as f:
                ckpt_path = os.path.join(ckpt_dir_or_file, f.readline()[:-1])
    else:
        ckpt_path = ckpt_dir_or_file
    ckpt = torch.load(ckpt_path, map_location=map_location)
    print("[INFO] Loading checkpoint from %s succeed!" % ckpt_path)
    return ckpt


def config():
    with open("parameters.yaml", "r") as stream:
        try:
            parsed_yaml = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print("Exception occured when opening .yaml config file!")
    return parsed_yaml


""" Creating dataset of sliding patches on Strebelle TI"""

"""Sliding through Strebelle TI and saving as image."""


def generate_windows(
    training_image_path: str, args: dict, img_size: tuple = (128, 128), stride: int = 1
):
    """
    Generate sliding windows using scikit-image function `view_as_windows`.

    Parameters
    ----------
    training_image_path : str
        Path to Strebelle training image.
    args : dict
        Parsed arguments as dictionary.
    img_size : tuple, optional
        Size of windows to be saved. The default is (64, 64).
    stride : int, optional
        Step of which the window walks. The default is 1.

    Returns
    -------
    windows : np.ndarray
        Array with batch size containing saved images.
    """
    ti32 = cv2.imread(args["training_image"], cv2.COLOR_BGR2GRAY)

    _, ti = cv2.threshold(ti32, 127, 255, cv2.THRESH_BINARY)

    windows = view_as_windows(ti, (150, 150))
    return windows


def save_generated_images(windowed_images, args: dict) -> None:
    """
    Save each patch of Strebelle image to output directory.

    Parameters
    ----------
    windowed_images : np.ndarray
        Array of images.
    args : dict
        User defined parameters.
    """
    for i, batch_ti in tqdm(
        enumerate(windowed_images),
        desc="Sliding window, please wait...",
        total=windowed_images.shape[0],
        colour="blue",
    ):
        for j, t in enumerate(batch_ti):
            ti_resized = cv2.resize(t, (128, 128))
            imsave(f"{args['output_dir']}/strebelle_{i}_{j}.png", ti_resized)


def watch_for_checkpoints(args, Critic, Generator, critic_opt, gen_opt):
    """
    Scan directories to see if there are checkpoints saved.

    Parameters
    ----------
    args : argparse.args
        Parameters defined for model training.
    Critic : nn.Module
        The Critic model to be loaded if any checkpoint is found.
    Generator : nn.Module
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
    checkpoint = args["checkpoint"]
    save_dir = args["sample_images"]

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
        start_epoch = ckpt["epoch"]
        Critic.load_state_dict(ckpt["D"])
        Generator.load_state_dict(ckpt["Generator"])
        critic_opt.load_state_dict(ckpt["d_optimizer"])
        gen_opt.load_state_dict(ckpt["g_optimizer"])
    except FileNotFoundError:
        print("[*] No checkpoint!")
        start_epoch = 0

    return start_epoch


def train(args) -> None:
    """
    Train both Generative and Critic model.

    Parameters
    ----------
    args : argparse.args
        Parameters defined for model training.
    """
    Critic = nn.DataParallel(CriticModel(args["num_channels"]))
    Generator = nn.DataParallel(GeneratorModel(args["latent_dim"]))
    Critic.to(DEVICE)
    Generator.to(DEVICE)
    Critic = CriticModel(args["num_channels"]).to(DEVICE, non_blocking=True)
    Generator = GeneratorModel(args["latent_dim"]).to(DEVICE, non_blocking=True)

    # Instantiates optimizers
    G_opt = torch.optim.Adam(
        Generator.parameters(), lr=args["learning_rate"], betas=(0.5, 0.999)
    )
    C_opt = torch.optim.Adam(
        Critic.parameters(), lr=args["learning_rate"], betas=(0.5, 0.999)
    )

    start_epoch = watch_for_checkpoints(args, Critic, Generator, C_opt, G_opt)

    # Loading Dataset
    transf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Grayscale(num_output_channels=1),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    writer = tensorboardX.SummaryWriter(".logs/wgan-gp")

    data = torchvision.datasets.ImageFolder(args["images_path"], transform=transf)
    dataloader = torch.utils.data.DataLoader(
        data,
        batch_size=args["batch_size"],
        shuffle=True,
        num_workers=args["num_workers"],
        pin_memory=True,
    )

    # Starting training loop
    for epoch in tqdm(
        range(start_epoch, args["n_epochs"]),
        desc="Training progress",
        total=args["n_epochs"] - start_epoch,
        position=0,
        ncols=100,
        leave=True,
        colour="green",
    ):
        start_time = time.time()
        critic_loss = []
        gen_loss = []
        Generator.train()

        # Sampling from dataloader
        for i, (images, _) in enumerate(dataloader):
            step = epoch * len(dataloader) + i + 1
            images = Variable(images)
            batch = images.size(0)

            if args["cuda"]:
                images = images.to(DEVICE, non_blocking=True)

            # Creates random noise
            z = Variable(torch.randn(batch, args["latent_dim"]))

            if args["cuda"]:
                z = z.to(DEVICE, non_blocking=True)

            # Sends random noise to Generator and gets Critic output
            generated = Generator(z)
            real_criticized = Critic(images)
            fake_criticized = Critic(generated)

            # Compare distance between critic output
            em_distance = real_criticized.mean() - fake_criticized.mean()
            grad_penalty = gradient_penalty(images.data, generated.data, Critic)

            # Calculates critic loss
            CriticLoss = -em_distance + grad_penalty * 10

            # Append to list for logging purpose
            critic_loss.append(CriticLoss.item())
            Critic.zero_grad()
            CriticLoss.backward()
            C_opt.step()

            # Logs to tensorboard
            writer.add_scalar(
                "Critic/em_dist", em_distance.data.cpu().numpy(), global_step=step
            )
            writer.add_scalar(
                "Critic/gradient_penalty",
                grad_penalty.data.cpu().numpy(),
                global_step=step,
            )
            writer.add_scalar(
                "Critic/critic_loss", CriticLoss.data.cpu().numpy(), global_step=step
            )

            if step % args["n_critic"] == 0:
                # Random latent noise
                z = Variable(torch.randn(batch, args["latent_dim"]))

                if args["cuda"]:
                    z = z.to(DEVICE, non_blocking=True)

                # Generate new images from this latent vector
                generated = Generator(z)

                fake_criticized = Critic(generated)
                GeneratorLoss = -fake_criticized.mean()

                # Append to list for logging purpose
                gen_loss.append(GeneratorLoss.item())

                # Backward pass
                Critic.zero_grad()
                Generator.zero_grad()
                GeneratorLoss.backward()
                G_opt.step()

                # Logs loss scalar to tensorboard
                writer.add_scalars(
                    "Generator",
                    {"g_loss": GeneratorLoss.data.cpu().numpy()},
                    global_step=step,
                )

                print(
                    f"Epoch {epoch+1} : {i+1}/{len(dataloader)}:"
                    + f"{round((time.time()-start_time)/60, 2)} mins",
                    end="\r",
                )

        # Switch to evaluation mode and sample new images
        Generator.eval()

        # Generate new samples to save
        z_sample = Variable(torch.randn(args["latent_dim"], args["latent_dim"]))
        if args["cuda"]:
            z_sample = z_sample.to(DEVICE, non_blocking=True)

        fake_gen_images = (Generator(z_sample).data + 1) / 2.0

        torchvision.utils.save_image(
            fake_gen_images,
            args["sample_images"] + "/Epoch " + str(epoch + 1) + ".jpg",
            nrow=10,
        )

        x = torchvision.utils.make_grid(fake_gen_images, nrow=5)
        writer.add_image("Generated", x, step)

        # Save checkpoints
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "D": Critic.state_dict(),
                "Generator": Generator.state_dict(),
                "d_optimizer": C_opt.state_dict(),
                "g_optimizer": G_opt.state_dict(),
            },
            "%s/Epoch_(%d).ckpt" % (args["checkpoint"], epoch + 1),
            max_keep=5,
        )


if __name__ == "__main__":
    # Get parameters
    param = config()

    # Create necessary folders
    os.makedirs(param["output_dir"], exist_ok=True)

    if not os.path.exists(param["training_image"]):
        raise FileNotFoundError("Could not find Strebelle training image in path!")

    print("Generating sliding windows, please wait...")
    windows = generate_windows(
        training_image_path=param["training_image"], img_size=128, args=param
    )

    print("Saving all sliding windows...")
    save_generated_images(windows, param)
    sys.exit(0)

    # Train the generative model
    train(param)
