import os
import cv2
import sys
import yaml
import time
import torch
import shutil
import pathlib
import torchvision
import tensorboardX
import torch.nn as nn
import torchvision.transforms as transforms

from tqdm import tqdm
from imageio import imsave
from torch.autograd import grad
from torch.autograd import Variable
from skimage.util import view_as_windows
from arch.models import GeneratorModel, CriticModel

# Constants
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CONFIG_FILE = "parameters.yaml"
CHECKPOINT_FILENAME = "latest_checkpoint"

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
    torch.save(state, f"{save_path}/Epoch.ckpt")

    # deal with max_keep
    save_dir = os.path.dirname(save_path)
    list_path = os.path.join(save_dir, CHECKPOINT_FILENAME)

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
      shutil.copyfile("/workspace/checkpoints/Epoch.ckpt", os.path.join(save_dir, "best_model.ckpt"))

def load_checkpoint(ckpt_dir_or_file, map_location=None, load_best=False):
    if os.path.isdir(ckpt_dir_or_file):
        if load_best:
            ckpt_path = os.path.join(ckpt_dir_or_file, "best_model.ckpt")
        else:
            try:
                with open(os.path.join(ckpt_dir_or_file, CHECKPOINT_FILENAME)) as f:
                    ckpt_path = os.path.join(ckpt_dir_or_file, f.readline()[:-1])
            except FileNotFoundError:
                print("Checkpoint file not found. Starting training from scratch.")
                return None
    else:
        ckpt_path = ckpt_dir_or_file
    ckpt = torch.load(ckpt_path, map_location=map_location)
    print("[INFO] Loading checkpoint from %s succeed!" % ckpt_path)
    return ckpt


def config():
    config_file = pathlib.Path(CONFIG_FILE)
    if not config_file.is_file():
        print(f"Config file '{config_file}' not found!")
        sys.exit(-1)

    with config_file.open("r") as stream:
        try:
            parsed_yaml = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print("Error parsing .yaml config file!")
            sys.exit(-1)
    return parsed_yaml


def check_directory(directory: str) -> None:
    """
    Checks if a directory exists, if not, it creates it.

    Parameters
    ----------
    directory : str
        Directory path.
    """
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)


def watch_for_checkpoints(args, Critic, Generator, critic_opt, gen_opt):
    checkpoint = args["checkpoint"]
    save_dir = args["sample_images"]

    # Check and create paths if necessary
    check_directory(checkpoint)
    check_directory(save_dir)

    # Loads checkpoint and changes state dictionary
    ckpt = load_checkpoint(checkpoint)
    start_epoch = ckpt["epoch"]
    Critic.load_state_dict(ckpt["D"])
    Generator.load_state_dict(ckpt["Generator"])
    critic_opt.load_state_dict(ckpt["d_optimizer"])
    gen_opt.load_state_dict(ckpt["g_optimizer"])
    return start_epoch

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
    ti32 = cv2.imread(training_image_path, cv2.COLOR_BGR2GRAY)

    _, ti = cv2.threshold(ti32, 127, 255, cv2.THRESH_BINARY)

    windows = view_as_windows(ti, (150, 150))
    return windows

def save_generated_images(windowed_images, args: dict) -> None:
    output_dir = args['output_dir']
    check_directory(output_dir)

    for i, batch_ti in tqdm(
        enumerate(windowed_images),
        desc="Sliding window, please wait...",
        total=windowed_images.shape[0],
        colour="blue",
    ):
        for j, t in enumerate(batch_ti):
            ti_resized = cv2.resize(t, (128, 128))
            imsave(f"{args['output_dir']}/strebelle_{i}_{j}.png", ti_resized)

# Ignore warnings
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

def train(args) -> None:
    # Get paths from args
    checkpoint_dir = args["checkpoint"]
    sample_images_dir = args["sample_images"]
    images_path = args["images_path"]

    # Check and create paths if necessary
    check_directory(checkpoint_dir)
    check_directory(sample_images_dir)

    Critic = nn.DataParallel(CriticModel(args["num_channels"])).to(DEVICE, non_blocking=True)
    Generator = nn.DataParallel(GeneratorModel(args["latent_dim"])).to(DEVICE, non_blocking=True)

    # Instantiates optimizers
    G_opt = torch.optim.Adam(
        Generator.parameters(), lr=args["learning_rate"], betas=(0.5, 0.999)
    )
    C_opt = torch.optim.Adam(
        Critic.parameters(), lr=args["learning_rate"], betas=(0.5, 0.999)
    )

    # start_epoch = watch_for_checkpoints(args, Critic, Generator, C_opt, G_opt)
    start_epoch = 0
    # Loading Dataset
    transf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Grayscale(num_output_channels=1),
            transforms.Normalize([0.5], [0.5]),
            transforms.Resize((256,256), antialias=True),
        ]
    )

    writer = tensorboardX.SummaryWriter("logs/wgan-gp")

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

        # For mixed precision training
        scaler = torch.cuda.amp.GradScaler()

        # In your training loop:
        for i, (images, _) in enumerate(dataloader):
            step = epoch * len(dataloader) + i + 1
            images = images.to(DEVICE, non_blocking=True)
            batch = images.size(0)
            z = Variable(torch.randn(batch, args["latent_dim"]))
            z = z.to(DEVICE, non_blocking=True)

            # Amp up the process for mixed precision
            with torch.cuda.amp.autocast():
                generated = Generator(z)
                real_criticized = Critic(images)
                fake_criticized = Critic(generated)

                em_distance = real_criticized.mean() - fake_criticized.mean()
                grad_penalty = gradient_penalty(images.data, generated.data, Critic)

                CriticLoss = -em_distance + grad_penalty * 10

            # Scales the loss, and calls backward() to create scaled gradients
            scaler.scale(CriticLoss).backward()

            # Unscales the gradients of optimizer's assigned params in-place, and checks for infs and nans
            scaler.unscale_(C_opt)

            # If the gradients do not contain infs or NaNs, optimizer.step() is then called,
            # otherwise, optimizer.step() is skipped.
            scaler.step(C_opt)

            # Updates the scale for next iteration
            scaler.update()

            # Reset gradients
            Critic.zero_grad()

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
            f'{args["checkpoint"]}')

if __name__ == "__main__":
    # Get parameters
    param = config()

    # Check for necessary files and directories
    training_image = pathlib.Path(param["training_image"])
    if not training_image.is_file():
        print(f"Training image not found at path: '{training_image}'")
        sys.exit(-1)

    print("Generating sliding windows, please wait...")
    windows = generate_windows(
        training_image_path=param["training_image"], img_size=128, args=param
    )

    print("Saving all sliding windows...")
    save_generated_images(windows, param)

    # Train the generative model
    train(param)