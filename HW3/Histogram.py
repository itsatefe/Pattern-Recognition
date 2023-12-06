import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math


class Histogram:
    def __init__(self, data, bandwidths):
        self.data = data
        self.bandwidths = bandwidths


    def histogram_density_estimation(self, bandwidth=0.1):
        bin_edges = np.arange(self.data.min(), self.data.max() + bandwidth, bandwidth)
        counts = np.zeros(len(bin_edges) - 1, dtype=int)
        for value in self.data:
            bin_index = int((value - self.data.min()) / bandwidth)
            counts[bin_index] += 1

        normalized_counts = counts / np.sum(counts * bandwidth)
        return normalized_counts, bin_edges
  

    def plot_1d_histogram(self):
        fig, axs = plt.subplots(1, len(self.bandwidths), figsize=(15, 5))

        for i, bandwidth in enumerate(self.bandwidths):
            counts, bin_edges = self.histogram_density_estimation(bandwidth=bandwidth)
            axs[i].bar(bin_edges[:-1], counts, width=bandwidth, label=f'bandwidth={bandwidth}', edgecolor='black', align='edge')
            axs[i].set_title(f'with Bandwidth={bandwidth}')
            axs[i].set_xlabel('Grade')
            axs[i].set_ylabel('Probability Density')
            axs[i].legend()

        plt.tight_layout()
        plt.show()




    def histogram_2d_density_estimation(self, bandwidth=0.1):
        data1, data2 = self.data.iloc[:, 0], self.data.iloc[:, 1]
        x_edges = np.arange(data1.min(), data1.max() + bandwidth, bandwidth)
        y_edges = np.arange(data2.min(), data2.max() + bandwidth, bandwidth)

        counts = np.zeros((len(x_edges) - 1, len(y_edges) -1), dtype=int)

        for x, y in zip(data1, data2):
            x_bin_index = int((x - data1.min()) / bandwidth)
            y_bin_index = int((y - data2.min()) / bandwidth)

            counts[x_bin_index, y_bin_index] += 1

        normalized_counts = counts / np.sum(counts * bandwidth**2)
        return normalized_counts, x_edges, y_edges



    def plot_2d_histogram(self):
        data1, data2 = self.data.iloc[:, 0], self.data.iloc[:, 1]
        fig, axes = plt.subplots(2, len(self.bandwidths), figsize=(15, 10), subplot_kw={'projection': '3d'})

        for i, bandwidth in enumerate(self.bandwidths):
            counts, x_edges, y_edges = self.histogram_2d_density_estimation(bandwidth=bandwidth)
            ylabel = 'Probability Density'

            # Front view
            ax_front = axes[0, i]
            xpos, ypos = np.meshgrid(x_edges[:-1] + bandwidth / 2, y_edges[:-1] + bandwidth / 2, indexing="ij")
            dx = dy = bandwidth
            dz = counts.ravel()
            ax_front.bar3d(xpos.ravel(), ypos.ravel(), np.zeros_like(dz), dx, dy, dz, shade=True)

            ax_front.set_title(f'2D Histogram with Bandwidth={bandwidth}')
            ax_front.set_xlabel('Feature 1')
            ax_front.set_ylabel('Feature 2')
            ax_front.grid(True)
            ax_front.set_zlabel(ylabel)
            ax_front.view_init(elev=0, azim=50)

            # Elevated view
            ax_elevated = axes[1, i]
            surface = ax_elevated.bar3d(xpos.ravel(), ypos.ravel(), np.zeros_like(dz), dx, dy, dz, shade=True, linewidth=1)

            ax_elevated.set_title(f'2D Histogram with Bandwidth={bandwidth}')
            ax_elevated.set_xlabel('Feature 1')
            ax_elevated.set_ylabel('Feature 2')
            ax_elevated.grid(True)
            ax_elevated.set_zlabel(ylabel)
            ax_elevated.view_init(elev=25, azim=50)

        plt.tight_layout()
        plt.show()


  





