import os
import cv2
import numpy
import random
import argparse
import matplotlib
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from typing import Any
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import ImageGrid


def get_args():
    parser = argparse.ArgumentParser(description='Perform easy SNESIM simulations with this CLI!')
  || teste de tabela ||
  || alo ||
    parser.add_argument('--samples_path', default="data/samples50",
                        type=str, help='Samples path')

    parser.add_argument('--ti_path', default="data/ti_strebelle.out",
                        type=str, help='Training image path')

    parser.add_argument('--par_path', default="data/snesim.par",
                        type=str, help='SNESIM parameter file path')

    parser.add_argument('--exe_path', default="data/snesim.exe",
                        type=str, help='SNESIM executable path')

    parser.add_argument('--output_path', default="data/snesim.out",
                        type=str, help='Output path')

    parser.add_argument('--realizations', default=50,
                        type=int, help='Number of realizations to simulate')

    parser.add_argument('--max_cond', default=30,
                        type=int, help='Maximum number of points to condition')

    parser.add_argument('--min_cond', default=10,
                        type=int, help='Minimum number of points to condition')

    parser.add_argument('--seed', default=69069,
                        type=int, help='Seed')

    parser.add_argument('--plot', default=True,
                        type=bool, help='Boolean for whether you want to plot graphs or not')

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

def read_conditional_samples(filename: object = 'eas.dat', nanval: object = -997799) -> object:
    debug_level = 1
    if not (os.path.isfile(filename)):
        print("Filename:'%s', does not exist" % filename)

    file = open(filename, "r")
    if debug_level > 0:
        print("eas: file ->%20s" % filename)

    eas = {'title': (file.readline()).strip('\n')}

    if debug_level > 0:
        print("eas: title->%20s" % eas['title'])

    dim_arr = eas['title'].split()
    if len(dim_arr) == 3:
        eas['dim'] = {}
        eas['dim']['nx'] = int(dim_arr[0])
        eas['dim']['ny'] = int(dim_arr[1])
        eas['dim']['nz'] = int(dim_arr[2])

    eas['n_cols'] = int(file.readline())

    eas['header'] = []
    for i in range(0, eas['n_cols']):
        # print (i)
        h_val = (file.readline()).strip('\n')
        eas['header'].append(h_val)

        if debug_level > 1:
            print("eas: header(%2d)-> %s" % (i, eas['header'][i]))

    file.close()

    try:
        eas['D'] = np.genfromtxt(filename, skip_header=2 + eas['n_cols'])
        if debug_level > 1:
            print("eas: Read data from %s" % filename)
    except:
        print("eas: COULD NOT READ DATA FROM %s" % filename)

    # add NaN values
    try:
        eas['D'][eas['D'] == nanval] = np.nan
    except:
        print("eas: FAILED TO HANDLE NaN VALUES (%d(" % nanval)

    # If dimensions are given in title, then convert to 2D/3D array
    if "dim" in eas:
        if eas['dim']['nz'] == 1:
            eas['Dmat'] = eas['D'].reshape((eas['dim']['ny'], eas['dim']['nx']))
        elif eas['dim']['nx'] == 1:
            eas['Dmat'] = np.transpose(eas['D'].reshape((eas['dim']['nz'], eas['dim']['ny'])))
        elif eas['dim']['ny'] == 1:
            eas['Dmat'] = eas['D'].reshape((eas['dim']['nz'], eas['dim']['nx']))
        else:
            eas['Dmat'] = eas['D'].reshape((eas['dim']['nz'], eas['dim']['ny'], eas['dim']['nx']))

        eas['Dmat'] = eas['D'].reshape((eas['dim']['nx'], eas['dim']['ny'], eas['dim']['nz']))
        eas['Dmat'] = eas['D'].reshape((eas['dim']['nz'], eas['dim']['ny'], eas['dim']['nx']))

        eas['Dmat'] = np.transpose(eas['Dmat'], (2, 1, 0))

        if debug_level > 0:
            print("eas: converted data in matrixes (Dmat)")

    eas['filename'] = filename

    return eas


def get_color_bar():
    cmap = mpl.colors.ListedColormap(['white', "black"])

    col_dict={0:"white", 1:"black"}
    norm_bins = np.sort([*col_dict.keys()]) + 0.5
    norm_bins = np.insert(norm_bins, 0, np.min(norm_bins) - 1.0)

    norm = mpl.colors.BoundaryNorm(norm_bins, 2, clip=True)

    labels = np.array(["Sandstone", "Shale"])
    fmt = matplotlib.ticker.FuncFormatter(lambda x, pos: labels[norm(x)])

    diff = norm_bins[1:] - norm_bins[:-1]
    tickz = norm_bins[:-1] + diff / 2

    colorbar = plt.colorbar(
        mpl.cm.ScalarMappable(cmap=cmap, norm=norm),
        ticks=tickz,
        format=fmt,
        spacing='proportional',
        label='Facies',
    )

    return colorbar

def etype_plot(dict_ti: pd.DataFrame):
    _data = dict_ti.copy()

    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4))

    # Get mean for each row
    _data['etype'] = _data.mean(axis=1)
    ax.imshow(_data['etype'].values.reshape(150, 150),
              cmap='gray', origin='lower')

    # Text settings
    plt.title("Etype of SNESIM simulations")
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")

    # Colorbar settings
    cmap = mpl.cm.gray
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    cb = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                      orientation='vertical', spacing="proportional")
    cb.ax.tick_params()
    cb.set_label("Mean")

    # Saving images to .png format
    plt.grid(False)
    plt.savefig("results/etype.png", dpi=600, bbox_inches='tight')

def calculate_uncertainty(dict_ti: pd.DataFrame):
    _data = dict_ti.copy()

    _data['prob_sand'] = _data.sum(axis=1)
    _data['prob_shale'] = 100 - _data['prob_sand']

    _data["u_max"] = _data[["prob_shale", "prob_sand"]].max(axis=1) / 100
    _data["u_min"] = _data[["prob_shale", "prob_sand"]].min(axis=1) / 100

    _data.drop('prob_sand', axis=1, inplace=True)
    _data.drop('prob_shale', axis=1, inplace=True)

    return _data

def plot_uncertainty(dict_ti: pd.DataFrame):
    _data = dict_ti.copy()

    plt.figure(figsize=(5.5, 4))

    _data = calculate_uncertainty(_data)

    plt.imshow(_data['u_min'].values.reshape(150, 150),
              cmap='gray', origin='lower')

    plt.title("Simulation uncertainty with conditional samples")
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")

    # Colorbar settings
    cmap = mpl.cm.gray
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    cb = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                 orientation='vertical', spacing="proportional")
    cb.ax.tick_params()
    cb.set_label("Uncertainty")

    # Saving images to .png format
    plt.grid(False)
    plt.savefig("results/uncertainty.png", dpi=600, bbox_inches='tight')

def plot_realizations_grid(real):
    rand_idx = [random.randint(0, len(real)-1) for _ in range(len(real))]
    tis_sampled = [real[i] for i in rand_idx]

    fig = plt.figure(figsize=(8, 5))

    sq = round(len(real))

    image_grid = ImageGrid(fig, 111,
                           nrows_ncols=(3, 3),
                           axes_pad=0.1,
                           cbar_location="left",
                           cbar_pad=0.15)

    cb = get_color_bar()

    fig.suptitle("SNESIM realizations")

    for ax, image in zip(image_grid, tis_sampled):
        # Iterating over the grid returns the Axes.
        ax.imshow(image, cmap="gray", origin="lower")

    plt.savefig("results/simulation_grid.png", dpi=600, bbox_inches='tight')

def get_sand_shale_proportion(image):
    image = image.reshape(-1)
    sand_prop = (len(np.where(image==1)[0]) / 150**2)*100
    shale_prop = (len(np.where(image==0)[0]) / 150**2)*100
    return sand_prop, shale_prop

def proportions(real: numpy.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4))

    sand_values = list()
    shale_values = list()

    # Get proportions for all
    for i in range(1, len(real)):
        sand, shale = get_sand_shale_proportion(real[i].reshape(-1))
        sand_values.append(sand)
        shale_values.append(shale)

    df_dict = dict(Sand=sand_values, Shale=shale_values)
    df = pd.DataFrame.from_dict(df_dict, orient='columns')

    # Boxplot
    bp = ax.boxplot(df, labels=["Sandstone", "Shale"], patch_artist=True, showfliers=False)

    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color="black")

    for patch in bp['boxes']:
        patch.set(facecolor='#f1f1f1')

    plt.ylim(20, 80)

    plt.ylabel('Percentage (%)')
    plt.xlabel('Category')
    plt.title('SNESIM Training Images proportions')

    plt.savefig("results/boxplot_snesim.png", dpi=600, bbox_inches='tight')


def plot_ti(ti: numpy.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4))

    ax.imshow(ti, cmap="gray", origin="lower")
    plt.title("Strebelle training image")
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")

    cb = get_color_bar()

    # Saving images to .df format
    plt.grid(False)
    plt.savefig("results/training_image.png", dpi=600, bbox_inches='tight')

def plot_ti_with_samples(ti: numpy.ndarray) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(6, 8),
                           constrained_layout=True)

    ax[0].imshow(im, cmap="gray", origin="lower")
    ax[0].set_title("Reference training image")
    ax[0].set_xlabel("X coordinate")
    ax[0].set_ylabel("Y coordinate")

    for row_sampled in samples_im.values:
        x, y, class_ = row_sampled[0], row_sampled[1], row_sampled[2]
        if class_ == 0:
            ax[1].scatter(x, y, marker='o', color='black', s=10)
        else:
            ax[1].scatter(x, y, marker='o', c='white',
                          edgecolors='black', s=10)

    # Plotting the samples
    ax[1].set_title("Location map")
    ax[1].set_xlabel("X coordinate")
    ax[1].set_ylabel("Y coordinate")
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Shale sample',
                              markerfacecolor='black', markeredgecolor="black"),
                       Line2D([0], [0], marker='o', color='w', label='Sandstone sample',
                              markerfacecolor='white', markeredgecolor="black")]

    ax[1].legend(handles=legend_elements, bbox_to_anchor=(1, 0.5), loc="center left")
    ax[1].grid(True, linestyle="--", color="black", linewidth=0.4)

    #get_color_bar()
    plt.gca().set_aspect('equal')
    # Saving images to .pdf form
    plt.savefig("results/training_image_samples.png", dpi=600, bbox_inches='tight')


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

        samples_im['x'] = conditioning_data[:,0]
        samples_im['y'] = conditioning_data[:,1]
        samples_im['class'] = conditioning_data[:,3]

        ti_dict = dict()

        print("[INFO] Loading simulations", end="\r")
        file = read_conditional_samples(args.output_path)["D"]
        realizations = file[:, 0].reshape(args.realizations, 150, 150)
        np.save("data/realizations.npy", realizations, allow_pickle=True)

        for idx, realization in enumerate(realizations):
            ti_dict[f'ti_{idx+1}'] = np.array(realization.reshape(-1))

        dataframe = pd.DataFrame(ti_dict)

        ti = cv2.imread('data/strebelle.png')
        plot_ti(ti)
        plot_ti_with_samples(ti)
        etype_plot(dataframe)
        plot_uncertainty(dataframe)
        plot_realizations_grid(realizations)