"""SNESIM baseline (all realizations from the Strebelle TI) and proposed workflow
(one realization per catalog TI). Run from the repo root: `mise run snesim`."""

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src/snesim")
import snesim  # noqa: E402


def simulate(
    name: str, out_dir: str, ti_path: str, ti_dim: int, n: int, seed: int
) -> np.ndarray:
    """Run snesim.exe once; returns the realizations as (n, 150, 150)."""
    os.makedirs(out_dir, exist_ok=True)
    args = argparse.Namespace(
        samples_path="src/snesim/data/samples50",
        ti_path=ti_path,
        par_path=f"{out_dir}/{name}.par",
        exe_path="src/snesim/data/snesim.exe",
        output_path=f"{out_dir}/{name}.out",
        realizations=n,
        max_cond=30,
        min_cond=10,
        ti_dim=ti_dim,
        seed=seed,
    )
    snesim.change_parameters(args)
    snesim.run_simulation(args)
    snesim.save_simulations(args.output_path, n)
    return np.load(f"{out_dir}/{name}.npy")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--realizations", type=int, default=100, help="Baseline realizations"
    )
    parser.add_argument("--n", type=int, help="Use only the first n catalog TIs")
    parser.add_argument("--seed", type=int, default=69096)
    args = parser.parse_args()

    # Traditional workflow: N realizations from the single reference TI.
    simulate(
        "snesim",
        "outputs/snesim",
        "src/snesim/data/strebelle.out",
        250,
        args.realizations,
        args.seed,
    )

    # Proposed workflow: catalog TI i conditions realization i (seed + i).
    tis = sorted(
        Path("outputs/catalog").glob("ti_*.out"),
        key=lambda p: int(re.findall(r"\d+", p.stem)[0]),
    )
    if not tis:
        raise SystemExit("No catalog in outputs/catalog: run `mise run sample` first.")
    tis = tis[: args.n]
    gan = [
        simulate(f"snesim_{i}", "outputs/gan", ti.as_posix(), 150, 1, args.seed + i)[0]
        for i, ti in enumerate(tis)
    ]
    np.save("outputs/gan/gan.npy", np.stack(gan))
    print(
        f"Baseline: {args.realizations} realizations; catalog: {len(gan)} realizations"
    )
