"""Generate the four docs/*.gif animations (`mise run gifs`).

One two-tone facies colormap everywhere (sand = 1, mud = 0) on an opaque light
background, so the GIFs read the same on light and dark pages. Sizes stay small:
fixed 27-colour palette, no dithering, ~30-40 frames.

  ti-latent-walk       slerp through the generator latent space, reference TI alongside
  ti-catalog           flipbook of catalog TIs with their sand proportion
  training-progression fixed latent vectors at each saved epoch (outputs/progress)
  realizations         baseline vs GAN-catalog realizations with running proportion histograms

Inputs come from `mise run train | sample | snesim`; the walk re-runs the generator.
"""

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src" / "generative_model"))
from sampling import DEVICE, load_generator  # noqa: E402

SAND, MUD = (240, 196, 84), (
    38,
    52,
    92,
)  # luminance-separated: safe for colour blindness
BG, INK, GRAY = (250, 250, 247), (30, 30, 30), (120, 120, 128)
T, PAD, ROW, HIST_H = 200, 10, 18, 56  # panel size, margin, text row, histogram height
FONT = ImageFont.truetype(
    str(Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf"), 12
)
REFERENCE = ROOT / "src/snesim/data/strebelle.png"


def _palette() -> Image.Image:
    """Fixed palette: facies colours, background, gray, and an ink-to-background ramp for text."""
    ramp = [
        tuple(int(a + (b - a) * k / 12) for a, b in zip(INK, BG)) for k in range(13)
    ]
    colors = [SAND, MUD, BG, INK, GRAY, *ramp[1:-1]]
    pal = Image.new("P", (1, 1))
    pal.putpalette([v for c in colors for v in c])
    return pal


PALETTE = _palette()


def save(frames: list[Image.Image], path: Path, durations: list[int]) -> None:
    """Quantize to the fixed palette (no dithering) and write an endlessly looping GIF."""
    q = [f.quantize(palette=PALETTE, dither=Image.Dither.NONE) for f in frames]
    path.parent.mkdir(parents=True, exist_ok=True)
    q[0].save(
        path,
        save_all=True,
        append_images=q[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )
    print(
        f"{path.name}: {len(q)} frames, {q[0].size[0]}x{q[0].size[1]}, {path.stat().st_size/1e3:.0f} kB"
    )


def timing(n: int, step: int, hold: int) -> list[int]:
    """Frame durations in ms, last frame held longer."""
    return [step] * (n - 1) + [hold]


def tile(a: np.ndarray, size: int) -> np.ndarray:
    """Resize a binary grid to size x size and re-binarize."""
    interp = cv2.INTER_AREA if size < a.shape[0] else cv2.INTER_LINEAR
    return cv2.resize(a.astype(np.float32), (size, size), interpolation=interp) > 0.5


def paint(img: Image.Image, xy: tuple[int, int], a: np.ndarray) -> None:
    img.paste(Image.fromarray(np.where(a[..., None], SAND, MUD).astype(np.uint8)), xy)


def text(
    d: ImageDraw.ImageDraw, xy: tuple[int, int], s: str, anchor: str = "mt"
) -> None:
    d.text(xy, s, font=FONT, fill=INK, anchor=anchor)


def legend(d: ImageDraw.ImageDraw, y: int, w: int, label: str) -> None:
    """Bottom row: colour key on the left, frame position on the right."""
    x = PAD
    for color, name in ((SAND, "sand"), (MUD, "mud")):
        d.rectangle([x, y + 3, x + 9, y + 12], fill=color, outline=GRAY)
        text(d, (x + 14, y + 1), name, "lt")
        x += 60
    text(d, (w - PAD, y + 1), label, "rt")


def pair_frame(
    panels: list[np.ndarray],
    heads: list[str],
    foots: list[str],
    label: str,
    hists: list[Image.Image] | None = None,
    stats: list[str] | None = None,
) -> Image.Image:
    """Two square panels with a header and a footer each, optional histogram strip below."""
    w = 2 * T + 3 * PAD
    h = PAD + ROW + T + ROW + PAD + (HIST_H + 2 * ROW if hists else 0) + ROW
    img = Image.new("RGB", (w, h), BG)
    d = ImageDraw.Draw(img)
    for k, (a, head, foot) in enumerate(zip(panels, heads, foots)):
        x = PAD + k * (T + PAD)
        text(d, (x + T // 2, PAD), head)
        paint(img, (x, PAD + ROW), a)
        text(d, (x + T // 2, PAD + ROW + T + 2), foot)
        if hists:
            img.paste(hists[k], (x, PAD + ROW + T + ROW + PAD))
            text(d, (x + T // 2, PAD + ROW + T + ROW + PAD + HIST_H + ROW), stats[k])
    legend(d, h - ROW - 2, w, label)
    return img


def reference() -> np.ndarray:
    return cv2.imread(str(REFERENCE), cv2.IMREAD_GRAYSCALE) > 127


def generate(z: torch.Tensor, generator: torch.nn.Module) -> np.ndarray:
    """Same path as sampling.py: generator -> 250x250 -> binarize at 0 (sand = 1)."""
    with torch.no_grad():
        out = torch.cat([generator(b) for b in z.split(10)])
    return (F.interpolate(out, size=(250, 250)).squeeze(1) > 0).cpu().numpy()


def slerp(a: torch.Tensor, b: torch.Tensor, t: float) -> torch.Tensor:
    om = torch.acos(torch.clamp((a / a.norm() * b / b.norm()).sum(), -1, 1))
    return (torch.sin((1 - t) * om) * a + torch.sin(t * om) * b) / torch.sin(om)


def latent_walk(a: argparse.Namespace, out: Path) -> None:
    """Closed slerp loop through `--keys` seeded latent vectors, `--steps` frames per leg."""
    torch.manual_seed(a.seed)
    keys = torch.randn(a.keys, a.latent_size, device=DEVICE)
    legs = [(i, j / a.steps) for i in range(a.keys) for j in range(a.steps)]
    z = torch.stack([slerp(keys[i], keys[(i + 1) % a.keys], t) for i, t in legs])
    tis = generate(z, load_generator(a.model_path, a.latent_size))
    ref, p_ref = tile(reference(), T), reference().mean()
    frames = [
        pair_frame(
            [tile(ti, T), ref],
            ["generated TI", "reference TI"],
            [f"sand {ti.mean():.3f}", f"sand {p_ref:.3f}"],
            f"z{i + 1} to z{(i + 1) % a.keys + 1}, t {t:.2f}   step {n + 1}/{len(legs)}",
        )
        for n, ((i, t), ti) in enumerate(zip(legs, tis))
    ]
    save(frames, out / "ti-latent-walk.gif", timing(len(frames), 90, 90))


def catalog(a: argparse.Namespace, out: Path) -> None:
    """Flipbook of `--frames` catalog TIs, evenly spaced through outputs/catalog.npy."""
    cat = np.load(ROOT / "outputs/catalog.npy").astype(bool)
    idx = np.linspace(0, len(cat) - 1, a.frames).round().astype(int)
    ref, p_ref = tile(reference(), T), reference().mean()
    frames = [
        pair_frame(
            [tile(cat[i], T), ref],
            ["catalog TI", "reference TI"],
            [f"sand {cat[i].mean():.3f}", f"sand {p_ref:.3f}"],
            f"TI {i + 1}/{len(cat)}",
        )
        for i in idx
    ]
    save(frames, out / "ti-catalog.gif", timing(len(frames), 150, 1500))


def training(a: argparse.Namespace, out: Path) -> None:
    """Same 4 tiles (fixed latent vectors) from each outputs/progress/Epoch N.jpg."""
    files = sorted(
        (ROOT / "outputs/progress").glob("Epoch *.jpg"),
        key=lambda p: int(p.stem.split()[1]),
    )
    files = [
        files[i]
        for i in np.unique(
            np.linspace(0, len(files) - 1, min(len(files), a.frames))
            .round()
            .astype(int)
        )
    ]
    picks = np.sort(np.random.default_rng(a.seed).choice(100, 4, replace=False))
    size, gap = 128, 4  # tile size and gap of the 2x2 block
    block = 2 * size + gap
    w, h = 2 * block + 3 * PAD, PAD + ROW + block + ROW + PAD + ROW
    ref, p_ref = tile(reference(), block), reference().mean()
    frames = []
    for f in files:
        grid = (
            cv2.imread(str(f), cv2.IMREAD_GRAYSCALE) > 127
        )  # torchvision grid: 10x10, 2 px pad
        pitch = (grid.shape[0] - 2) // 10
        img = Image.new("RGB", (w, h), BG)
        d = ImageDraw.Draw(img)
        sand = []
        for k, p in enumerate(picks):
            r, c = divmod(int(p), 10)
            t = grid[2 + r * pitch : pitch * (r + 1), 2 + c * pitch : pitch * (c + 1)]
            sand.append(t.mean())
            paint(
                img,
                (PAD + (k % 2) * (size + gap), PAD + ROW + (k // 2) * (size + gap)),
                tile(t, size),
            )
        paint(img, (2 * PAD + block, PAD + ROW), ref)
        text(d, (PAD + block // 2, PAD), "generator, fixed latents")
        text(d, (2 * PAD + block + block // 2, PAD), "reference TI")
        text(
            d,
            (PAD + block // 2, PAD + ROW + block + 2),
            f"sand {np.mean(sand):.3f} (mean of 4)",
        )
        text(
            d,
            (2 * PAD + block + block // 2, PAD + ROW + block + 2),
            f"sand {p_ref:.3f}",
        )
        legend(d, h - ROW - 2, w, f"epoch {f.stem.split()[1]}")
        frames.append(img)
    save(
        frames,
        out / "training-progression.gif",
        timing(len(frames), 600 if len(frames) <= 10 else 120, 1500),
    )


def hist_strip(
    p: np.ndarray, lo: float, hi: float, bins: int, ymax: int
) -> Image.Image:
    """Histogram of the sand proportions of the realizations so far."""
    img = Image.new("RGB", (T, HIST_H + 1), BG)
    d = ImageDraw.Draw(img)
    counts, _ = np.histogram(p, bins=bins, range=(lo, hi))
    for j, c in enumerate(counts):
        x0, x1 = round(j * T / bins), round((j + 1) * T / bins) - 2
        d.rectangle([x0, HIST_H - round(c / ymax * HIST_H), x1, HIST_H], fill=GRAY)
    d.line([0, HIST_H, T, HIST_H], fill=INK)
    return img


def realizations(a: argparse.Namespace, out: Path) -> None:
    """Realization i of each workflow side by side; histograms accumulate all realizations up to i."""
    base = np.load(ROOT / "outputs/snesim/snesim.npy") > 0.5
    prop = np.load(ROOT / "outputs/gan/gan.npy") > 0.5
    n = len(base)
    pb, pp = base.mean((1, 2)), prop.mean((1, 2))
    lo = np.floor(min(pb.min(), pp.min()) * 50) / 50
    hi = np.ceil(max(pb.max(), pp.max()) * 50) / 50
    bins = 16
    ymax = max(np.histogram(p, bins=bins, range=(lo, hi))[0].max() for p in (pb, pp))
    frames = []
    for i in np.linspace(0, n - 1, a.frames).round().astype(int):
        hists = [hist_strip(p[: i + 1], lo, hi, bins, ymax) for p in (pb, pp)]
        stats = [
            f"n={i + 1}  std {p[: i + 1].std():.3f}" if i else "n=1" for p in (pb, pp)
        ]
        img = pair_frame(
            [tile(base[i], T), tile(prop[i], T)],
            ["single TI (traditional)", "GAN catalog (proposed)"],
            [f"sand {pb[i]:.3f}", f"sand {pp[i]:.3f}"],
            f"realization {i + 1}/{n}",
            hists,
            stats,
        )
        d = ImageDraw.Draw(img)
        y = PAD + ROW + T + ROW + PAD + HIST_H + 2
        for k in range(2):
            x = PAD + k * (T + PAD)
            text(d, (x, y), f"{lo:.2f}", "lt")
            text(d, (x + T, y), f"{hi:.2f}", "rt")
            text(d, (x + T // 2, y), "sand proportion")
        frames.append(img)
    save(frames, out / "realizations.gif", timing(len(frames), 120, 2000))
    print(
        f"over {n} realizations: sand mean {pb.mean():.3f} / {pp.mean():.3f}, "
        f"std {pb.std():.4f} (single TI) / {pp.std():.4f} (GAN catalog)"
    )


GIFS = {
    "ti-latent-walk": latent_walk,
    "ti-catalog": catalog,
    "training-progression": training,
    "realizations": realizations,
}

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate docs/*.gif")
    p.add_argument(
        "names",
        nargs="*",
        choices=[*GIFS, []],
        metavar="name",
        help="GIFs to build (default: all)",
    )
    p.add_argument("--seed", type=int, default=69096)
    p.add_argument("--out", type=Path, default=ROOT / "docs")
    p.add_argument("--model_path", default=str(ROOT / "checkpoints/Epoch.ckpt"))
    p.add_argument("--latent_size", type=int, default=100)
    p.add_argument("--keys", type=int, default=5, help="latent walk: key vectors")
    p.add_argument("--steps", type=int, default=8, help="latent walk: frames per leg")
    p.add_argument(
        "--frames", type=int, default=30, help="max frames for the other GIFs"
    )
    args = p.parse_args()
    for name in args.names or GIFS:
        GIFS[name](args, args.out)
