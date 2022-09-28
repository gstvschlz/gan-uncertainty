from typing import Any, Tuple
from mpl_toolkits.axes_grid1 import ImageGrid
from helper.plot import *

import cv2
import random
import seaborn
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import matplotlib.pylab as pylab
params = {'legend.fontsize': 'small',
         'axes.labelsize': 'medium',
         'axes.titlesize':'medium',
         'xtick.labelsize':'medium',
         'ytick.labelsize':'medium'}
pylab.rcParams.update(params)

random.seed(69096)

plt.style.use(["science", "ieee", "bright"])

class Handler(object):
    def __init__(self, color):
        self.color = color

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        patch2 = plt.Rectangle(
            [x0 + width / 2.0, y0],
            width / 2.0,
            height,
            facecolor=self.color,
            edgecolor="none",
            transform=handlebox.get_transform(),
        )
        handlebox.add_artist(patch2)
        return patch2


class Plots:
    def __init__(self) -> None:
        self.figsize: Tuple = (3.5, 3.5)
        self.binary_cmap: str = "gray"

        mpl.rcParams["savefig.dpi"] = 600
        mpl.rcParams["figure.dpi"] = 100

        self.reference_ti, self.samples = self.load_samples()
        self.snesim = np.load("/home/gustavo/Github/gan-for-mps/snesim/data/realizations.npy").reshape((100, 150, 150))
        self.gan = np.load("generative_model/data/realizations_old.npy").reshape(
            (100, 150, 150)
        )
        self.snesim_df, self.gan_df = self.get_dict_realizations()

        self.__plot_all()

    def get_dict_realizations(self):
        snesim_df = dict()
        for idx, realization in enumerate(self.snesim):
            snesim_df[f"ti_{idx+1}"] = np.array(realization.reshape(-1))

        gan_df = dict()

        for idx, realization in enumerate(self.gan):
            gan_df[f"ti_{idx+1}"] = np.array(realization.reshape(-1))

        return pd.DataFrame(snesim_df), pd.DataFrame(gan_df)

    def __plot_all(self):
        self.strebelle()
        self.location_map(self.reference_ti, self.samples)
        self.snesim_realizations_grid()
        self.etype()
        self.std_plot()
        self.absolute_difference()
        self.plot_uncertainty()
        self.proportions_comparison(
            self.snesim.reshape(100, -1), self.gan.reshape(100, -1)
        )
        self.gan_realizations_grid()
        self.mds()
        self.histogram()
        return True

    def load_samples(self):
        file = read_conditional_samples("snesim/data/reference_ti")["D"]
        reference_ti = file.reshape(1, 150, 150)[0, :, :]

        conditioning_data = read_conditional_samples("snesim/data/samples50")["D"]

        # Samples to dataframe
        samples = pd.DataFrame()

        samples["x"] = conditioning_data[:, 0]
        samples["y"] = conditioning_data[:, 1]
        samples["class"] = conditioning_data[:, 3]
        return reference_ti, samples

    @staticmethod
    def _get_categorical_cb():
        cmap = mpl.colors.ListedColormap(["white", "black"])

        col_dict = {0: "white", 1: "black"}
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
            spacing="proportional",
            label="Facies",
        )

        return colorbar

    def strebelle(self, ti_path: str = "generative_model/strebelle.png"):
        ti = cv2.imread(ti_path)
        fig, ax = plt.subplots(figsize=self.figsize)

        ax.imshow(ti, cmap=self.binary_cmap, origin="lower")
        plt.xlabel("X coordinate (m)")
        plt.ylabel("Y coordinate (m)")

        cb = self._get_categorical_cb()
        plt.grid(False)
        plt.savefig("results/strebelle_ti.pdf", format="pdf", bbox_inches="tight", dpi=300)

    def location_map(self, reference_ti, samples):
        fig, ax = plt.subplots(1, 2, figsize=self.figsize, sharey=True)

        ax[0].imshow(reference_ti, cmap="gray", origin="lower")
        ax[0].set_title("Ground truth")
        ax[0].set_xlabel("X coordinate (m)")
        ax[0].set_ylabel("Y coordinate (m)")

        for row_sampled in samples.values:
            x, y, class_ = row_sampled[0], row_sampled[1], row_sampled[2]
            if class_ == 0:
                ax[1].scatter(x, y, marker="o", color="black", s=10)
            else:
                ax[1].scatter(x, y, marker="o", c="white", edgecolors="black", s=10)

        # Plotting the samples
        ax[1].set_title("Location map")
        ax[1].set_xlabel("X coordinate (m)")
        ax[1].grid(True, linestyle="--", color="black", linewidth=0.4)

        plt.gca().set_aspect("equal")
        plt.savefig("results/sample_map.pdf", format="pdf", bbox_inches="tight", dpi=300)

    def snesim_realizations_grid(self):
        rand_idx = [
            random.randint(0, len(self.snesim) - 1) for _ in range(len(self.snesim))
        ]
        tis_sampled = [self.snesim[i] for i in rand_idx]

        fig = plt.figure(figsize=(8, 3.5))

        sq = round(len(self.snesim))

        image_grid = ImageGrid(
            fig,
            111,
            nrows_ncols=(2, 2),
            axes_pad=0.05,
            cbar_location="right",
            cbar_pad=0.15,
        )

        cb = self._get_categorical_cb()
        #plt.suptitle("Traditional workflow - simulation results", y=0.99)

        for ax, image in zip(image_grid, tis_sampled):
            # Reshapes image to desired size
            reshaped = image.reshape(150, 150)
            # Iterating over the grid returns the Axes.
            ax.imshow(reshaped, cmap="gray", origin="lower")

            # Adjust axis ticks
            ax.set_xticks(range(0, 150, 50))
            ax.set_yticks(range(0, 150, 50))
        plt.grid(False)
        plt.savefig("results/snesim_grid.pdf", format="pdf", bbox_inches="tight", dpi=300)

    def etype(self):
        fig = plt.figure(figsize=self.figsize)

        grid = ImageGrid(
            fig,
            111,
            nrows_ncols=(1, 2),
            axes_pad=0.15,
            share_all=True,
            cbar_location="right",
            cbar_mode="single",
            cbar_pad=0.15,
        )

        #plt.suptitle("E-type mean plot", y=0.99)

        # Labels and titles
        grid[0].set_title("Traditional workflow")
        grid[1].set_title("Proposed workflow")
        grid[0].set_ylabel("Y coordinate (m)")

        # Add data to image grid
        for ax, im in zip(
            grid,
            (
                self.snesim.reshape(100, 150, 150).mean(axis=0),
                self.gan.reshape(100, 150, 150).mean(axis=0),
            ),
        ):
            im = ax.imshow(im, cmap="gray", origin="lower")
            ax.set_xlabel("X coordinate (m)")

        # Colorbar
        cb = ax.cax.colorbar(im)
        cb.ax.get_yaxis().labelpad = 15
        cb.set_label("Mean value")
        ax.cax.toggle_label(True)
        plt.grid(False)
        fig.tight_layout()
        fig.subplots_adjust(top=1.25)
        plt.savefig("results/etype.pdf", format="pdf", bbox_inches="tight", dpi=300)

    def std_plot(self):
        fig = plt.figure(figsize=self.figsize)

        grid = ImageGrid(
            fig,
            111,
            nrows_ncols=(1, 2),
            axes_pad=0.15,
            share_all=True,
            cbar_location="right",
            cbar_mode="single",
            cbar_pad=0.15,
        )

        #plt.suptitle("Standard deviation comparison", y=0.99)

        # Labels and titles
        grid[0].set_title("Traditional workflow")
        grid[1].set_title("Proposed workflow")
        grid[0].set_ylabel("Y coordinate (m)")

        # Add data to image grid
        for ax, im in zip(
            grid,
            (
                self.snesim.reshape(100, 150, 150).std(axis=0),
                self.gan.reshape(100, 150, 150).std(axis=0),
            ),
        ):
            im = ax.imshow(im, cmap="jet", origin="lower")
            ax.set_xlabel("X coordinate (m)")

        # Colorbar
        cb = ax.cax.colorbar(im)
        cb.ax.get_yaxis().labelpad = 15
        cb.set_label("Standard deviation value")
        ax.cax.toggle_label(True)
        plt.grid(False)
        fig.tight_layout()
        fig.subplots_adjust(top=1.25)
        plt.savefig("results/std_plot.pdf", format="pdf", bbox_inches="tight", dpi=300)

    def absolute_difference(self):
        fig, ax = plt.subplots(1, 1, figsize=self.figsize)

        # Labels and titles
        ax.set_title("Absolute difference between workflows")
        ax.set_xlabel("X coordinate (m)")
        ax.set_ylabel("Y coordinate (m)")

        # Add data to image grid
        im = ax.imshow(
            np.abs(
                self.snesim.reshape(100, 150, 150).mean(axis=0)
                - self.gan.reshape(100, 150, 150).mean(axis=0)
            ),
            cmap="jet",
            origin="lower",
        )

        # Colorbar
        cb = plt.colorbar(im)
        cb.ax.get_yaxis()
        cb.set_label("Delta")
        plt.grid(False)
        plt.savefig("results/absolute_diff.pdf", format="pdf", bbox_inches="tight", dpi=300)

    @staticmethod
    def calculate_uncertainty(dict_ti: pd.DataFrame):
        _data = dict_ti.copy()

        _data["prob_sand"] = _data.mean(axis=1)
        _data["prob_shale"] = 1 - _data["prob_sand"]

        _data["uncertainty"] = _data[["prob_shale", "prob_sand"]].min(axis=1)

        _data.drop("prob_sand", axis=1, inplace=True)
        _data.drop("prob_shale", axis=1, inplace=True)

        return _data

    def histogram(self):
        snesim_unc = calculate_uncertainty(self.snesim_df)
        gan_unc = calculate_uncertainty(self.gan_df)


        trad = [[val[-1], "Traditional"] for val in snesim_unc.values]
        prop = [[val[-1], "Proposed"] for val in gan_unc.values]

        df = pd.DataFrame([*trad, *prop], columns=["Value", "Workflow"])
        fig = plt.figure(figsize=self.figsize)
        seaborn.histplot(
            data=df, x="Value", hue="Workflow", multiple="dodge", element="step"
        )

        #plt.title("Histogram for uncertainty values in both workflows")
        plt.savefig("uncertainty_histogram.pdf", format="pdf", dpi=300, bbox_inch = 'tight', palette=["lightblue", "salmon"])

    def plot_uncertainty(self):
        snesim_unc = calculate_uncertainty(self.snesim_df)
        gan_unc = calculate_uncertainty(self.gan_df)

        fig = plt.figure(figsize=self.figsize)

        grid = ImageGrid(
            fig,
            111,
            nrows_ncols=(1, 2),
            axes_pad=0.15,
            share_all=True,
            cbar_location="right",
            cbar_mode="single",
            cbar_pad=0.15,
        )

        #plt.suptitle("Uncertainty comparison", y=0.99)

        # Labels and titles
        grid[0].set_title("Traditional workflow")
        grid[1].set_title("Proposed workflow")
        grid[0].set_ylabel("Y coordinate (m)")

        # Add data to image grid
        for ax, im in zip(grid, (snesim_unc, gan_unc)):
            im = ax.imshow(
                im["uncertainty"].values.reshape(150, 150), cmap="jet", origin="lower"
            )
            ax.set_xlabel("X coordinate (m)")

        # Colorbar
        cb = ax.cax.colorbar(im)
        cb.ax.get_yaxis().labelpad = 15
        cb.set_label("Uncertainty")
        ax.cax.toggle_label(True)
        plt.grid(False)
        fig.tight_layout()
        fig.subplots_adjust(top=1.25)
        plt.savefig("results/uncertainty.pdf", format="pdf", bbox_inches="tight", dpi=300)

    @staticmethod
    def get_dicts(realization):
        sand_values = list()
        shale_values = list()

        # Get proportions for real data
        for i in range(1, len(realization)):
            sand, shale = get_sand_shale_proportion(realization[i].reshape(-1))
            sand_values.append(sand)
            shale_values.append(shale)

        df_dict = dict(Sand=sand_values, Shale=shale_values)
        df = pd.DataFrame.from_dict(df_dict, orient="columns")
        return df

    def proportions_comparison(self, real: np.ndarray, fake: np.ndarray) -> None:
        import operator as op

        fig, ax = plt.subplots(1, 1, figsize=self.figsize)
        df_snesim = self.get_dicts(real)
        df_gan = self.get_dicts(fake)
        
        handles = [plt.Rectangle((0, 0), 1, 1) for i in range(4)]
        colors = ["cornflowerblue", "lightcoral"]
        hmap = dict(zip(handles, [Handler(color) for color in colors]))

        sh_list = list()
        sh_list.extend(df_snesim["Shale"].values)
        sh_list.extend(df_gan["Shale"].values)

        snd_list = list()
        snd_list.extend(df_snesim["Sand"].values)
        snd_list.extend(df_gan["Sand"].values)

        df_super = {
            "Shale": sh_list,
            "Sand": snd_list,
            "Workflow": ["Traditional"]*99 + ["Proposed"]*99
        }
        df_super = pd.DataFrame(df_super)
        dd=pd.melt(df_super, id_vars=['Workflow'], value_vars=['Shale','Sand'], var_name='Facies')
        seaborn.boxplot(x='Workflow', y='value', data=dd, hue='Facies')
        ax.set_ylabel("Facies proportion in \%")

        plt.legend(
            handles=handles,
            labels=["Shale proportion", "Sand proportion"],
            handler_map=hmap,
            bbox_to_anchor=(0, 1.02, 1, 0.2),
            frameon=True,
            loc="lower left",
            mode="expand", ncol=2
        )
        plt.savefig("results/proportions.pdf", format="pdf", bbox_inches="tight", dpi=300)

    def gan_realizations_grid(self):
        rand_idx = [random.randint(0, len(self.gan) - 1) for _ in range(len(self.gan))]
        tis_sampled = [self.gan[i] for i in rand_idx]

        fig = plt.figure(figsize=(8, 3.5))

        sq = round(len(self.gan))

        image_grid = ImageGrid(
            fig,
            111,
            nrows_ncols=(2, 2),
            axes_pad=0.05,
            cbar_location="right",
            cbar_pad=0.15,
        )

        cb = self._get_categorical_cb()

        #plt.suptitle("Proposed workflow - simulation results", y=0.99)

        for ax, image in zip(image_grid, tis_sampled):
            # Reshapes image to desired size
            reshaped = image.reshape(150, 150)

            # Iterating over the grid returns the Axes.
            ax.imshow(reshaped, cmap="gray", origin="lower")

            # Adjust axis ticks
            ax.set_xticks(range(0, 150, 50))
            ax.set_yticks(range(0, 150, 50))
        plt.savefig("results/gan_grid.pdf", format="pdf", bbox_inches="tight", dpi=300)

    def mds_calculus(self):
        traditional = self.snesim.reshape(100, -1)
        proposed = self.gan.reshape((100, -1))

        mds = MDS(n_components=3, metric=True, random_state=0)

        # Get the embeddings
        original = mds.fit_transform(traditional)

        # Get the embeddings
        gan = mds.fit_transform(proposed)
        return original, gan

    @staticmethod
    def centroidnp(arr):
        length, dim = arr.shape
        return np.array([np.sum(arr[:, i]) / length for i in range(dim)])


    def distance_boxplot(self, traditional, proposed):
        import operator as op

        fig, ax = plt.subplots(1, 1, figsize=self.figsize)


        plot_data = {
            "Traditional": traditional,
            "Proposed": proposed,
        }

        # sort keys and values together
        sorted_keys, sorted_vals = zip(*sorted(plot_data.items(), key=op.itemgetter(1)))

        ax.set(xlabel="Workflow", ylabel="Distance to centroid")
        box = seaborn.boxplot(
            data=sorted_vals, width=0.1, showfliers=False
        )
        # category labels
        plt.text(0.1, 62.5, f'{round(np.mean(traditional), 2)}\%')
        plt.text(1.1, 62.5, f'{round(np.mean(proposed), 2)}\%')
        plt.xticks(plt.xticks()[0], sorted_keys)
        

        plt.savefig("results/mds_distance.pdf", format="pdf", bbox_inches="tight", dpi=300)

        

    def mds3d(self, traditional_coord, proposed_coord):
        # set up a figure twice as wide as it is tall
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(1, 1, 1, projection="3d")

        #ax.set_title("3D Multidimensional Scaling")
        ax.set_xlabel("X dimension")
        ax.set_ylabel("Y dimension")
        ax.set_zlabel("Z dimension")

        x_traditional, y_traditional, z_traditional = traditional_coord.T
        x_proposed, y_proposed, z_proposed = proposed_coord.T

        ax.scatter3D(x_traditional, y_traditional, z_traditional)

        ax.scatter3D(x_proposed, y_proposed, z_proposed)

        ax.legend(["Traditional workflow", "Proposed workflow"])
        plt.xlabel("Uncertainty values")
        plt.savefig("results/mds3d.pdf", format="pdf", bbox_inches="tight", dpi=300)

    def mds(self):
        original, gan = self.mds_calculus()

        trad_centroid = self.centroidnp(original)
        prop_centroid = self.centroidnp(gan)

        import math

        dist_trad = [math.dist(point, trad_centroid) for point in original]
        dist_prop = [math.dist(point, prop_centroid) for point in gan]

        self.distance_boxplot(dist_trad, dist_prop)
        self.mds3d(original, gan)


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    # Plot everything!
    plotter = Plots()
