import matplotlib.pyplot as plt
import pandas as pd
import numpy
from mpl_toolkits.axes_grid1 import ImageGrid
import os
import numpy as np
import matplotlib as mpl
import random
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import manhattan_distances, euclidean_distances

def get_color_bar():
    cmap = mpl.colors.ListedColormap(['white', "black"])

    col_dict={0:"white", 1:"black"}
    norm_bins = np.sort([*col_dict.keys()]) + 0.5
    norm_bins = np.insert(norm_bins, 0, np.min(norm_bins) - 1.0)

    norm = mpl.colors.BoundaryNorm(norm_bins, 2, clip=True)

    labels = np.array(["Sandstone", "Shale"])
    fmt = mpl.ticker.FuncFormatter(lambda x, pos: labels[norm(x)])

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
    plt.title("Etype of SNESIM with generated TIs")
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")

    # Colorbar settings
    cmap = mpl.cm.gray
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    cb = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                      orientation='vertical', spacing="proportional")
    cb.ax.tick_params()
    cb.set_label("Mean")

    # Saving images to .pdf format
    plt.grid(False)
    plt.savefig("data/results/etype.png", dpi=600, bbox_inches='tight')


def calculate_uncertainty(dict_ti: pd.DataFrame):
    _data = dict_ti.copy()

    _data['prob_sand'] = _data.mean(axis=1)
    _data['prob_shale'] = 1 - _data['prob_sand']

    _data["uncertainty"] = _data[["prob_shale", "prob_sand"]].min(axis=1)

    _data.drop('prob_sand', axis=1, inplace=True)
    _data.drop('prob_shale', axis=1, inplace=True)

    return _data

def plot_uncertainty(dict_ti: pd.DataFrame):
    _data = dict_ti.copy()

    plt.figure(figsize=(5.5, 4))

    _data = calculate_uncertainty(_data)

    plt.imshow(_data['uncertainty'].values.reshape(150, 150),
              cmap='gray', origin='lower')

    plt.title("Simulation uncertainty from proposed workflow")
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")

    # Colorbar settings
    cmap = mpl.cm.gray
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    cb = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                 orientation='vertical', spacing="proportional")
    cb.ax.tick_params()
    cb.set_label("Uncertainty")

    # Saving images to .pdf format
    plt.grid(False)
    plt.savefig("data/results/uncertainty.png", dpi=600, bbox_inches='tight')


def plot_realizations_grid(real):
    # Samples 25 TI for a grid of 5x5 images
    rand_idx = [random.randint(0, 10) for _ in range(10)]
    tis_sampled = [real[i] for i in rand_idx]

    fig = plt.figure(figsize=(8, 5))

    image_grid = ImageGrid(fig, 111,
                           nrows_ncols=(3, 3),
                           axes_pad=0.1,
                           cbar_location="left",
                           cbar_pad=0.15)

    cb = get_color_bar()

    fig.suptitle("SNESIM realizations from Generative model")

    for ax, image in zip(image_grid, tis_sampled):
        # Iterating over the grid returns the Axes.
        ax.imshow(image, cmap="gray", origin="lower")
    plt.savefig("data/results/simulation_grid.png", dpi=600, bbox_inches='tight')


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
    bp = ax.boxplot(df, labels=["Sandstone", "Shale"], patch_artist=True,
                    showmeans=True, showfliers=False)

    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color="black")

    for patch in bp['boxes']:
        patch.set(facecolor='#f1f1f1')

    plt.ylim(20, 80)

    plt.ylabel('Percentage (%)')
    plt.xlabel('Category')
    plt.title('SNESIM Generated Training Images proportions')

    plt.savefig("data/results/boxplot_snesim.png", dpi=600, bbox_inches='tight')


def concatenate_out_files(directory: str):
    cat_array = []
    for im in os.listdir(directory):
        image = np.loadtxt(directory+im, skiprows=4)[:,0]
        cat_array.append(np.resize(image, (150, 150)))
    return cat_array


def proportions_comparison(real: numpy.ndarray, fake:numpy.ndarray) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(11, 6))

    sand_values = list()
    shale_values = list()

    # Get proportions for real data
    for i in range(1, len(real)):
        sand, shale = get_sand_shale_proportion(real[i].reshape(-1))
        sand_values.append(sand)
        shale_values.append(shale)
    

    df_dict = dict(Sand=sand_values, Shale=shale_values)
    df_snesim = pd.DataFrame.from_dict(df_dict, orient='columns')

    # Boxplot
    bp = ax[0].boxplot(df_snesim, labels=["Sandstone", "Shale"],
                       patch_artist=True, showfliers=False)
    ax[0].set_ylabel('Percentage (%)')
    ax[0].set_xlabel('Category')
    ax[0].set_title('Training Images proportions (Traditional workflow)')

    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color="black")

    for patch in bp['boxes']:
        patch.set(facecolor='#f1f1f1')

    sand_values = list()
    shale_values = list()

    # Get proportions for fake data
    for i in range(1, len(fake)):
        sand, shale = get_sand_shale_proportion(fake[i].reshape(-1))
        sand_values.append(sand)
        shale_values.append(shale)

    df_dict = dict(Sand=sand_values, Shale=shale_values)
    df_gan = pd.DataFrame.from_dict(df_dict, orient='columns')

    # Boxplot
    bp = ax[1].boxplot(df_gan, labels=["Sandstone", "Shale"],
                       patch_artist=True, showfliers=False)

    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color="black")

    for patch in bp['boxes']:
        patch.set(facecolor='#f1f1f1')

    ax[0].set_ylim(20, 80)
    ax[1].set_ylim(20, 80)

    ax[1].set_ylabel('Percentage (%)')
    ax[1].set_xlabel('Category')
    ax[1].set_title('Training Images proportions (Proposed workflow)')
    #plt.savefig("data/results/boxplot_snesim.png", dpi=600, bbox_inches='tight')


def histplots(snesim_realizations, gan_dataframe):
    import seaborn as sns

    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.5))

    ti_dict = dict()
    for idx, realization in enumerate(snesim_realizations):
        ti_dict[f'ti_{idx+1}'] = np.array(realization.reshape(-1))

    snesim_dataframe = pd.DataFrame(ti_dict)

    data_snesim = calculate_uncertainty(snesim_dataframe)
    data_gan = calculate_uncertainty(gan_dataframe)

    sns.kdeplot(data_snesim['uncertainty'], color='black')
    sns.kdeplot(data_gan['uncertainty'], color='blue')
    plt.title("Uncertainty histogram")
    plt.xlabel("Uncertainty values")

    plt.legend([f"Traditional workflow (Mean:{round(data_snesim['uncertainty'].mean()*100, 3)}%)",
     f"Proposed workflow (Mean:{round(data_gan['uncertainty'].mean()*100, 3)}%)"])
    plt.savefig("data/results/histplot.png", dpi=600, bbox_inches='tight')


def plot_mds(original, gan):
    plt.style.use("seaborn")
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4))
    plt.scatter(x=gan[:, 0], y=gan[:, 1], color="blue")
    plt.scatter(x=original[:, 0], y=original[:, 1], color="black")
    plt.title("Multidimensional scaling")
    plt.legend(["Proposed workflow", "Traditional workflow"],
            bbox_to_anchor = (1.05, 0.5))
    plt.savefig("data/results/mds.png", dpi=500, bbox_inches='tight')


def mds_plots(snesim_realizations_path, gan_realizations_path):
    traditional = np.load(snesim_realizations_path).reshape((100, -1))
    proposed = np.load(gan_realizations_path).reshape((100, -1))

    dist_euclid = euclidean_distances(proposed)
    mds = MDS(metric=True, dissimilarity='precomputed', random_state=0)

    # Get the embeddings
    gan = mds.fit_transform(dist_euclid)

    dist_euclid_or = euclidean_distances(traditional)
    # Get the embeddings
    original = mds.fit_transform(dist_euclid_or)

    plot_mds(original, gan)


def read_conditional_samples(filename: object = 'eas.dat', nanval: object = -997799) -> object:
    debug_level = 0
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