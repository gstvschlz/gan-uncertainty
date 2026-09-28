"""Sanity check (`mise run check`): compile everything, then validate the TI catalog.

The catalog must be (N, 250, 250) binary, and its sand proportion must match the
reference TI. Tolerances come from the reference itself: the generator is trained on
150x150 windows of it, whose sand proportion deviates from the full-TI value by at
most ~0.05 (measured below), so the catalog mean must lie within 0.05 and no single
TI may drift beyond twice that. Skips cleanly when no catalog exists yet.
"""

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from skimage.util import view_as_windows

REFERENCE = "src/snesim/data/strebelle.png"
CATALOG = Path("outputs/catalog.npy")

subprocess.run([sys.executable, "-m", "compileall", "-q", "src", "scripts"], check=True)

ref = (cv2.imread(REFERENCE, cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
ref_p = ref.mean()
windows = view_as_windows(ref, (150, 150)).mean(axis=(2, 3)).ravel()
tol = round(float(np.abs(windows - ref_p).max()), 2)
print(
    f"reference sand proportion {ref_p:.4f}; 150x150 windows {windows.min():.3f}-{windows.max():.3f}"
    f" (max deviation {np.abs(windows - ref_p).max():.3f} -> tolerance {tol})"
)

if not CATALOG.exists():
    print(f"SKIP catalog checks: {CATALOG} not found (run `mise run sample`).")
    sys.exit(0)

catalog = np.load(CATALOG)
assert (
    catalog.ndim == 3 and catalog.shape[1:] == ref.shape
), f"bad shape {catalog.shape}"
assert set(np.unique(catalog)) <= {0, 1}, "catalog is not binary"

props = catalog.mean(axis=(1, 2))
print(
    f"catalog {catalog.shape}: mean sand proportion {props.mean():.4f} "
    f"(min {props.min():.4f}, max {props.max():.4f})"
)
assert (
    abs(props.mean() - ref_p) <= tol
), f"catalog mean off reference by more than {tol}"
assert (
    np.abs(props - ref_p).max() <= 2 * tol
), f"a TI is off reference by more than {2 * tol}"
print("OK")
