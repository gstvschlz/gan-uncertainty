"""SNESIM baseline (all realizations from the Strebelle TI) and proposed workflow
(one realization per catalog TI), both from 250x250 TIs and conditioned to the same
wells. SNESIM is boitata's. Run from the repo root: `mise run snesim`."""

import argparse
import os

import boitata as bt
import cv2
import numpy as np

NX = 150  # simulation grid, 150 x 150 cells of size 1
# The SNESIM settings of the paper's parameter file: 30 conditioning cells, 6 multigrids,
# 10 replicates at least, no servosystem.
OPTS = dict(template_size=30, n_levels=5, min_replicates=10, servo=0.0)


def wells() -> tuple[np.ndarray, np.ndarray]:
    """Well data of `samples50` (GSLIB: x, y, z, facies). Its integer coordinates lie on
    cell boundaries; GSLIB's snesim.exe put a datum at x in cell x - 1, and so do we."""
    data = np.loadtxt("src/snesim/data/samples50", skiprows=6)
    return data[:, :2] - 0.5, data[:, 3].astype(int)


def simulate(ti: np.ndarray, n: int, seed: int) -> np.ndarray:
    """n realizations from one binary TI (rows = y), as (n, 150, 150)."""
    model = bt.BlockModel(
        (0, 0), (1, 1), (ti.shape[1], ti.shape[0]), attributes={"facies": ti.ravel().astype(np.int64)}
    )
    grid = bt.BlockModel((0, 0), (1, 1), (NX, NX))
    snesim = bt.SNESIM(model, "facies", **OPTS).fit(*wells())
    return snesim.simulate(grid, n=n, seed=seed, keep=True).realizations.reshape(n, NX, NX)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--realizations", type=int, default=100, help="Baseline realizations")
    parser.add_argument("--n", type=int, help="Use only the first n catalog TIs")
    parser.add_argument("--seed", type=int, default=69096)
    args = parser.parse_args()

    if not os.path.exists("outputs/catalog.npy"):
        raise SystemExit("No catalog in outputs/catalog.npy: run `mise run sample` first.")
    os.makedirs("outputs/snesim", exist_ok=True)
    os.makedirs("outputs/gan", exist_ok=True)

    # Traditional workflow: N realizations from the single reference TI.
    reference = (cv2.imread("src/snesim/data/strebelle.png", cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
    np.save("outputs/snesim/snesim.npy", simulate(reference, args.realizations, args.seed))

    # Proposed workflow: catalog TI i conditions realization i (seed + i).
    catalog = np.load("outputs/catalog.npy")[: args.n]
    gan = [simulate(ti, 1, args.seed + i)[0] for i, ti in enumerate(catalog)]
    np.save("outputs/gan/gan.npy", np.stack(gan))
    print(f"Baseline: {args.realizations} realizations; catalog: {len(gan)} realizations")
