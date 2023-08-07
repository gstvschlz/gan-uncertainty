""" SNESIM CLI module. """

import os
import argparse
from typing import Dict
import numpy as np
import pandas as pd


def get_args() -> argparse.Namespace:
    """
    Parses command line arguments using `argparse`.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Perform easy SNESIM simulations with this CLI!"
    )
    parser.add_argument(
        "--samples_path", default="data/samples50", type=str, help="Samples path"
    )

    parser.add_argument(
        "--ti_path",
        default="data/ti_strebelle.out",
        type=str,
        help="Training image path",
    )

    parser.add_argument(
        "--par_path",
        default="data/snesim.par",
        type=str,
        help="SNESIM parameter file path",
    )

    parser.add_argument(
        "--exe_path", default="data/snesim.exe", type=str, help="SNESIM executable path"
    )

    parser.add_argument(
        "--output_path", default="data/snesim.out", type=str, help="Output path"
    )

    parser.add_argument(
        "--realizations",
        default=50,
        type=int,
        help="Number of realizations to simulate",
    )

    parser.add_argument(
        "--max_cond", default=30, type=int, help="Maximum number of points to condition"
    )

    parser.add_argument(
        "--min_cond", default=10, type=int, help="Minimum number of points to condition"
    )

    parser.add_argument("--seed", default=69069, type=int, help="Seed")

    return parser.parse_args()


def change_parameters(arguments: Dict) -> None:
    """
    Updates the parameter file based on the provided arguments.

    Args:
        arguments (Dict): Command line arguments as a dictionary.

    Note:
        This function updates the parameter file in the specified path.
    """
    parameter: str = f"""
Parameters for SNESIM
***********************
START OF PARAMETERS:
{arguments.samples_path}{" "*30}- file with original data
1  2  3  4{" "*30}- columns for x, y, z, variable
2{" "*30}         - number of categories
0   1{" "*30}     - category codes
0.7  0.3{" "*30}  - (target) global pdf
0{" "*30}         - use (target) vertical proportions (0=no, 1=yes)
vertprop.dat                  - file with target vertical proportions
0{" "*30}         - servosystem parameter (0=no correction)
-1{" "*30}        - debugging level: 0,1,2,3
snesim.dbg{" "*30}- debugging file
{arguments.output_path} {" "*30}      - file for simulation output
{arguments.realizations}{" "*30}      - number of realizations to generate
150    0.5    1.0             - nx,xmn,xsiz
150    0.5    1.0             - ny,ymn,ysiz
1     0.5    1.0              - nz,zmn,zsiz
{arguments.seed}              - random number seed
{arguments.max_cond}          - max number of conditioning primary data
{arguments.min_cond}          - min. replicates number
0   0{" "*30}     - condition to LP (0=no, 1=yes), flag for iauto
1.0 1.0{" "*30}   - two weighting factors to combine P(A|B) and P(A|C)
localprop.dat                 - file for local proportions
0{" "*30}         - condition to rotation and affinity (0=no, 1=yes)
rotangle.dat                  - file for rotation and affinity
3{" "*30}         - number of affinity categories
1.0  1.0  1.0                 - affinity factors (X,Y,Z)
1.0  0.6  1.0                 - affinity factors
1.0  2.0  1.0                 - affinity factors
6{" "*30}         - number of multiple grids
{arguments.ti_path}           - file for training image
250  250  1                   - training image dimensions: nxtr, nytr, nztr
1{" "*30}         - column for training variable
10.0   10.0   5.0             - maximum search radii (hmax,hmin,vert)
0.0    0.0   0.0              - angles for search ellipsoid
"""
    with open(arguments.par_path, mode="w", encoding="utf8") as parfile:
        parfile.write(parameter)


def read_conditional_samples(filename: str = "eas.dat", nanval: int = -997799) -> Dict:
    """
    Reads conditional samples from a GSLIB-style file and returns them as a dictionary.

    Args:
        filename (str, optional): Path to the file with conditional samples. Defaults to "eas.dat".
        nanval (int, optional): Value representing NaN samples. Defaults to -997799.

    Returns:
        Dict: Information about the conditional samples.

    Raises:
        RuntimeError: If the data can't be read.
        FileNotFoundError: If the file doesn't exist.
    """
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"File '{filename}' does not exist")

    eas = {}

    with open(filename, mode="r", encoding="utf8") as file:
        eas["title"] = file.readline().strip("\n")
        dimensions = eas["title"].split()

        if len(dimensions) == 3:
            eas["dim"] = {
                "nx": int(dimensions[0]),
                "ny": int(dimensions[1]),
                "nz": int(dimensions[2]),
            }

        eas["n_cols"] = int(file.readline())
        eas["header"] = [file.readline().strip("\n") for _ in range(eas["n_cols"])]

    try:
        eas["D"] = np.genfromtxt(filename, skip_header=2 + eas["n_cols"])
    except Exception as exc:
        raise RuntimeError(f"Could not read data from '{filename}'") from exc

    # Handle NaN values
    eas["D"][eas["D"] == nanval] = np.nan

    # Convert to 2D/3D array if dimensions are given in title
    if "dim" in eas:
        eas["Dmat"] = eas["D"].reshape(
            (eas["dim"]["nz"], eas["dim"]["ny"], eas["dim"]["nx"])
        )
        eas["Dmat"] = np.transpose(eas["Dmat"], (2, 1, 0))

    eas["filename"] = filename

    return eas


def run_simulation(params: argparse.Namespace) -> None:
    """
    Runs the SNESIM simulation using WINE.

    Args:
        params (argparse.Namespace): Parsed command line arguments.
    """
    os.system(f"echo {params.par_path} | wine {params.exe_path}")


def load_ti(filename: str) -> np.ndarray:
    """
    Loads conditional data and returns it as a pandas DataFrame.

    Args:
        filename (str): File name to load the conditional data from.

    Returns:
        pd.DataFrame: DataFrame containing the X, Y coordinates and the class of each sample.
    """
    print("[INFO] Loading TI", end="\r")
    return read_conditional_samples(filename)["D"].reshape(1, 150, 150)[0, :, :]


def load_conditional_data(filename: str) -> pd.DataFrame:
    """
    Load conditional data as pandas dataframe.

    :param filename: Name for the TI conditional data.
    :returns: The pandas dataframe regarding the X, Y and class of the sample.
    """
    print("[INFO] Loading conditional data", end="\r")
    data = read_conditional_samples(filename)["D"]

    samples_im = pd.DataFrame()
    samples_im["x"] = data[:, 0]
    samples_im["y"] = data[:, 1]
    samples_im["class"] = data[:, 3]

    return samples_im


def save_simulations(filename: str, num_realizations: int) -> None:
    """
    Saves the simulations as a numpy pickle file.

    Args:
        filename (str): File name to save the simulations to.
        num_realizations (int): Number of simulations performed to correctly reshape the data.
    """
    print("[INFO] Loading simulations", end="\r")
    data = read_conditional_samples(filename)["D"]
    realizations = data[:, 0].reshape(num_realizations, 150, 150)

    np.save("data/realizations.npy", realizations, allow_pickle=True)


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    # Get arguments for parameter file
    args = get_args()

    # Create parameter file
    change_parameters(args)

    # Calls SNESIM.exe with wine :)
    run_simulation(args)

    # Plots graphs
    ti_file = load_ti("data/reference_ti")
    conditioning_data = load_conditional_data(args.samples_path)

    save_simulations(args.output_path, args.realizations)
    print("--- Ended traditional workflow! ---")