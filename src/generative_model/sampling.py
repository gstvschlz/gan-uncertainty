"""Sample the TI catalog: latent vectors -> WGAN-GP generator -> binary TIs."""

import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from arch.models import GeneratorModel

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_generator(model_path: str, latent_size: int) -> torch.nn.Module:
    """Load the Generator weights from a training checkpoint (DataParallel prefix removed)."""
    state = torch.load(model_path, map_location=DEVICE)["Generator"]
    model = GeneratorModel(latent_size).to(DEVICE)
    model.load_state_dict({k.replace("module.", ""): v for k, v in state.items()})
    return model.eval()


def create_ti_files(samples: np.ndarray, output_directory: str) -> None:
    """Write each TI as a 150x150 GSLIB file (`ti_<idx>.out`) for SNESIM."""
    os.makedirs(output_directory, exist_ok=True)
    for idx, im in enumerate(samples):
        im = cv2.resize(im, (150, 150), interpolation=cv2.INTER_NEAREST)
        np.savetxt(
            f"{output_directory}/ti_{idx}.out",
            im.reshape(-1),
            header="150 150 1\n1\nfacies",
            fmt="%1d",
            comments="",
        )


def main(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    generator = load_generator(args.model_path, args.latent_size)

    with torch.no_grad():
        z = torch.randn(args.num_samples, args.latent_size, device=DEVICE)
        images = torch.cat([generator(b) for b in z.split(10)])
    images = F.interpolate(images, size=(250, 250)).squeeze(1).cpu().numpy()

    # Generator output is tanh in [-1, 1]: sand (1) is the positive half, as in the
    # training image where sand is white.
    catalog = (images > 0).astype(np.uint8)

    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    np.save(args.output_file, catalog)
    create_ti_files(catalog, args.output_dir)
    print(
        f"Saved {len(catalog)} TIs {catalog.shape} to {args.output_file} "
        f"(sand proportion {catalog.mean():.4f})"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample the TI catalog")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--latent_size", type=int, default=100)
    parser.add_argument("--model_path", default="checkpoints/Epoch.ckpt")
    parser.add_argument("--output_file", default="outputs/catalog.npy")
    parser.add_argument("--output_dir", default="outputs/catalog")
    parser.add_argument("--seed", type=int, default=69096)
    main(parser.parse_args())
