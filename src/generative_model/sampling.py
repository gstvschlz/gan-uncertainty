"""Sample the TI catalog: latent grids -> WGAN-GP generator -> binary 250x250 TIs."""

import argparse
import math
import os

import numpy as np
import torch

from arch.models import SCALE, GeneratorModel

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TI_SIZE = 250  # the reference TI, so both workflows simulate from same-size TIs


def load_generator(model_path: str) -> GeneratorModel:
    """Load the EMA generator weights from a training checkpoint."""
    ckpt = torch.load(model_path, map_location=DEVICE)
    model = GeneratorModel(ckpt["z_ch"]).to(DEVICE)
    model.load_state_dict(ckpt["Generator"])
    return model.eval()


def latents(n: int, z_ch: int, size: int = TI_SIZE) -> torch.Tensor:
    """n latent grids large enough for a size x size TI."""
    cells = math.ceil(size / SCALE)
    return torch.randn(n, z_ch, cells, cells, device=DEVICE)


@torch.no_grad()
def generate(generator: GeneratorModel, z: torch.Tensor, size: int = TI_SIZE) -> np.ndarray:
    """Generator output, centre-cropped to size x size at native scale, binarized at 0:
    the tanh output is in [-1, 1] and sand (1) is the positive half."""
    out = torch.cat([generator(b) for b in z.split(10)])[:, 0]
    o = (out.shape[-1] - size) // 2
    return (out[:, o : o + size, o : o + size] > 0).cpu().numpy().astype(np.uint8)


def variogram(tis: np.ndarray, lags: int = 50) -> np.ndarray:
    """Mean indicator variogram of binary images along x and y, lags 1..lags: (2, lags).
    Lag 1 measures grain; the sill is p(1-p); the shape carries channel width and length."""
    t = tis.astype(np.float32)
    hs = range(1, lags + 1)
    return np.array(
        [
            [0.5 * np.mean((t[..., h:] - t[..., :-h]) ** 2) for h in hs],
            [0.5 * np.mean((t[..., h:, :] - t[..., :-h, :]) ** 2) for h in hs],
        ]
    )


def main(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    generator = load_generator(args.model_path)
    catalog = generate(generator, latents(args.num_samples, generator.z_ch))

    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    np.save(args.output_file, catalog)
    print(
        f"Saved {len(catalog)} TIs {catalog.shape} to {args.output_file} "
        f"(sand proportion {catalog.mean():.4f})"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample the TI catalog")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--model_path", default="checkpoints/generator.ckpt")
    parser.add_argument("--output_file", default="outputs/catalog.npy")
    parser.add_argument("--seed", type=int, default=69096)
    main(parser.parse_args())
