import os
import argparse
import numpy as np
import pandas as pd

from typing import Any

def get_args():
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

    parser.add_argument(
        "--plot",
        default=True,
        type=bool,
        help="Boolean for whether you want to plot graphs or not",
    )

    args = parser.parse_args()
    return args


def change_parameters(arguments: dict) -> Any:
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
    with open(arguments.par_path, "w") as parfile:
        parfile.write(parameter)


def read_conditional_samples(
    filename: object = "eas.dat", nanval: object = -997799
) -> object:
    debug_level = 0
    if not (os.path.isfile(filename)):
        print("Filename:'%s', does not exist" % filename)

    file = open(filename, "r")
    if debug_level > 0:
        print("eas: file ->%20s" % filename)

    eas = {"title": (file.readline()).strip("\n")}

    if debug_level > 0:
        print("eas: title->%20s" % eas["title"])

    dim_arr = eas["title"].split()
    if len(dim_arr) == 3:
        eas["dim"] = {}
        eas["dim"]["nx"] = int(dim_arr[0])
        eas["dim"]["ny"] = int(dim_arr[1])
        eas["dim"]["nz"] = int(dim_arr[2])

    eas["n_cols"] = int(file.readline())

    eas["header"] = []
    for i in range(0, eas["n_cols"]):
        # print (i)
        h_val = (file.readline()).strip("\n")
        eas["header"].append(h_val)

        if debug_level > 1:
            print("eas: header(%2d)-> %s" % (i, eas["header"][i]))

    file.close()

    try:
        eas["D"] = np.genfromtxt(filename, skip_header=2 + eas["n_cols"])
        if debug_level > 1:
            print("eas: Read data from %s" % filename)
    except:
        print("eas: COULD NOT READ DATA FROM %s" % filename)

    # add NaN values
    try:
        eas["D"][eas["D"] == nanval] = np.nan
    except:
        print("eas: FAILED TO HANDLE NaN VALUES (%d(" % nanval)

    # If dimensions are given in title, then convert to 2D/3D array
    if "dim" in eas:
        if eas["dim"]["nz"] == 1:
            eas["Dmat"] = eas["D"].reshape((eas["dim"]["ny"], eas["dim"]["nx"]))
        elif eas["dim"]["nx"] == 1:
            eas["Dmat"] = np.transpose(
                eas["D"].reshape((eas["dim"]["nz"], eas["dim"]["ny"]))
            )
        elif eas["dim"]["ny"] == 1:
            eas["Dmat"] = eas["D"].reshape((eas["dim"]["nz"], eas["dim"]["nx"]))
        else:
            eas["Dmat"] = eas["D"].reshape(
                (eas["dim"]["nz"], eas["dim"]["ny"], eas["dim"]["nx"])
            )

        eas["Dmat"] = eas["D"].reshape(
            (eas["dim"]["nx"], eas["dim"]["ny"], eas["dim"]["nz"])
        )
        eas["Dmat"] = eas["D"].reshape(
            (eas["dim"]["nz"], eas["dim"]["ny"], eas["dim"]["nx"])
        )

        eas["Dmat"] = np.transpose(eas["Dmat"], (2, 1, 0))

        if debug_level > 0:
            print("eas: converted data in matrixes (Dmat)")

    eas["filename"] = filename

    return eas


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    # Get arguments for parameter file
    args = get_args()

    # Create parameter file
    change_parameters(args)

    # Calls SNESIM.exe with wine :)
    os.system(f"echo {args.par_path} | wine {args.exe_path}")

    # Plots graphs
    if args.plot:
        print("[INFO] Loading TI", end="\r")
        file = read_conditional_samples("data/reference_ti")["D"]
        im = file.reshape(1, 150, 150)[0, :, :]

        conditioning_data = read_conditional_samples(args.samples_path)["D"]
        # Hard data
        print("[INFO] Loading conditional data", end="\r")

        # Samples to dataframe
        samples_im = pd.DataFrame()

        samples_im["x"] = conditioning_data[:, 0]
        samples_im["y"] = conditioning_data[:, 1]
        samples_im["class"] = conditioning_data[:, 3]

        print("[INFO] Loading simulations", end="\r")
        file = read_conditional_samples(args.output_path)["D"]
        realizations = file[:, 0].reshape(args.realizations, 150, 150)
        np.save("data/realizations.npy", realizations, allow_pickle=True)
        print("--- Ended traditional workflow! ---")
