"""Train the WGAN-GP on random native-scale crops of the reference TI (`mise run train`).

Speed: the TI lives on the GPU and batches are cropped there (no image files, no data
loader); bf16 autocast and TF32; a small fully convolutional pair. Quality: the critic
sees DiffAugment-ed inputs, the sampled generator is an EMA of the trained weights, and
the kept checkpoint is the one whose 250x250 samples best match the reference's sand
proportion and indicator variograms.
"""

import argparse
import copy
import pathlib
import time

import cv2
import numpy as np
import tensorboardX
import torch
import torch.nn.functional as F
import torchvision
import yaml
from torch.autograd import grad

from arch.models import SCALE, CriticModel, GeneratorModel
from sampling import DEVICE, generate, latents, variogram

CONFIG_FILE = pathlib.Path(__file__).with_name("parameters.yaml")


def load_ti(path: str) -> torch.Tensor:
    """Binary TI as a float tensor in {-1, 1} (sand = 1), on the training device."""
    ti = cv2.imread(path, cv2.IMREAD_GRAYSCALE) > 127
    return torch.tensor(ti, dtype=torch.float32, device=DEVICE) * 2 - 1


def crops(ti: torch.Tensor, n: int, size: int) -> torch.Tensor:
    """n random size x size windows of the TI, each randomly mirrored in x and y."""
    k = torch.arange(size, device=DEVICE)

    def axis(length):
        start = torch.randint(0, length - size + 1, (n, 1), device=DEVICE)
        flip = torch.rand(n, 1, device=DEVICE) < 0.5
        return start + torch.where(flip, size - 1 - k, k)

    rows, cols = axis(ti.shape[0]), axis(ti.shape[1])
    return ti[rows[:, :, None], cols[:, None, :]].unsqueeze(1)


def gradient_penalty(critic, real, fake):
    """Two-sided WGAN-GP penalty, E[(||grad D(x_hat)|| - 1)^2] on interpolates."""
    alpha = torch.rand(real.size(0), 1, 1, 1, device=DEVICE)
    x = (real + alpha * (fake - real)).requires_grad_(True)
    (g,) = grad(critic(x).float().sum(), x, create_graph=True)
    return ((g.flatten(1).norm(dim=1) - 1) ** 2).mean()


def diff_augment(x: torch.Tensor) -> torch.Tensor:
    """DiffAugment (Zhao et al. 2020), applied to every critic input, real and fake:
    random translation by up to 1/8 of the side (zero fill) and a random cutout of half
    the side. With one training image the critic otherwise memorises the real crops."""
    n, _, size, _ = x.shape
    k, s, c = torch.arange(size, device=DEVICE), size // 8, size // 2
    xp = F.pad(x, (s, s, s, s))
    rows = torch.randint(0, 2 * s + 1, (n, 1), device=DEVICE) + k
    cols = torch.randint(0, 2 * s + 1, (n, 1), device=DEVICE) + k
    x = xp[torch.arange(n, device=DEVICE)[:, None, None], 0, rows[:, :, None], cols[:, None, :]]
    cy, cx = torch.randint(0, size - c + 1, (2, n, 1), device=DEVICE)
    cut = ((k >= cy) & (k < cy + c))[:, :, None] & ((k >= cx) & (k < cx + c))[:, None, :]
    return (x * ~cut).unsqueeze(1)


@torch.no_grad()
def ema_update(ema, model, decay):
    for e, p in zip(ema.parameters(), model.parameters()):
        e.lerp_(p, 1 - decay)
    for e, b in zip(ema.buffers(), model.buffers()):
        e.copy_(b)


def score(generator, ref: np.ndarray, ref_vario: np.ndarray, n: int = 32) -> float:
    """Distance of n generated 250x250 TIs to the reference: relative error of the sand
    proportion plus mean relative error of the x/y indicator variograms (lags 1..50)."""
    tis = generate(generator, latents(n, generator.z_ch))
    p, p_ref = tis.mean(), ref.mean()
    return float(abs(p - p_ref) / p_ref + np.abs(variogram(tis) - ref_vario).mean() / ref_vario.mean())


def train(args: dict) -> None:
    ckpt_dir, progress_dir = pathlib.Path(args["checkpoint"]), pathlib.Path(args["sample_images"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    progress_dir.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    ti = load_ti(args["training_image"])
    ref = (ti > 0).cpu().numpy()
    ref_vario = variogram(ref)

    G = GeneratorModel(args["latent_channels"]).to(DEVICE)
    D = CriticModel().to(DEVICE)
    G_ema = copy.deepcopy(G).eval().requires_grad_(False)
    opt = dict(lr=args["learning_rate"], betas=tuple(args["betas"]))
    G_opt, D_opt = torch.optim.Adam(G.parameters(), **opt), torch.optim.Adam(D.parameters(), **opt)

    writer = tensorboardX.SummaryWriter("outputs/logs/wgan-gp")
    cells, bs = args["crop"] // SCALE, args["batch_size"]
    z_fixed = torch.randn(100, G.z_ch, cells, cells, device=DEVICE)
    z = lambda: torch.randn(bs, G.z_ch, cells, cells, device=DEVICE)  # noqa: E731
    # bf16 autocast (no loss scaling needed) halves the step time; the penalty's gradient
    # norm is taken on the fp32 input.
    amp = lambda: torch.autocast(DEVICE.type, torch.bfloat16, enabled=DEVICE.type == "cuda")  # noqa: E731
    best, t0 = float("inf"), time.time()

    for step in range(1, args["steps"] + 1):
        D.requires_grad_(True)
        for _ in range(args["n_critic"]):
            real = crops(ti, bs, args["crop"])
            with amp():
                with torch.no_grad():
                    fake = diff_augment(G(z()).float())
                real = diff_augment(real)
                em_distance = D(real).float().mean() - D(fake).float().mean()
                gp = gradient_penalty(D, real, fake)
            D_opt.zero_grad(set_to_none=True)
            (-em_distance + args["gp_weight"] * gp).backward()
            D_opt.step()

        D.requires_grad_(False)
        with amp():
            g_loss = -D(diff_augment(G(z()).float())).float().mean()
        G_opt.zero_grad(set_to_none=True)
        g_loss.backward()
        G_opt.step()
        ema_update(G_ema, G, args["ema_decay"])

        if step % 50 == 0:
            writer.add_scalar("Critic/em_dist", em_distance.item(), step)
            writer.add_scalar("Critic/gradient_penalty", gp.item(), step)
            writer.add_scalar("Generator/g_loss", g_loss.item(), step)

        if step % args["eval_every"] == 0 or step == args["steps"]:
            s = score(G_ema, ref, ref_vario)
            writer.add_scalar("Eval/score", s, step)
            with torch.no_grad():
                torchvision.utils.save_image(
                    (G_ema(z_fixed) + 1) / 2, progress_dir / f"Step {step}.jpg", nrow=10
                )
            kept = s < best
            if kept:
                best = s
                torch.save(
                    {"step": step, "score": s, "z_ch": G.z_ch, "Generator": G_ema.state_dict()},
                    ckpt_dir / "generator.ckpt",
                )
            print(
                f"step {step}/{args['steps']}  W {em_distance.item():.3f}  gp {gp.item():.3f}  "
                f"score {s:.4f}{' (kept)' if kept else ''}  {(time.time() - t0) / 60:.1f} min",
                flush=True,
            )
    print(f"Best score {best:.4f}: {ckpt_dir / 'generator.ckpt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the WGAN-GP")
    parser.add_argument("--seed", type=int, default=69096)
    parser.add_argument("--steps", type=int, help="Overrides steps (generator updates)")
    parser.add_argument("--eval_every", type=int, help="Overrides eval_every")
    cli = parser.parse_args()
    torch.manual_seed(cli.seed)

    param = yaml.safe_load(CONFIG_FILE.read_text())
    param["steps"] = cli.steps or param["steps"]
    param["eval_every"] = cli.eval_every or param["eval_every"]
    train(param)
