import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

class ParzenWindow:
    def __init__(self, data, bandwidths):
        self.data = data
        self.bandwidths = bandwidths

    def parzen_window(self, x, data, bandwidth):
        n, d = data.shape
        kernel_values = np.all(np.abs(x - data) <= bandwidth / 2, axis=1)
        sum_kernels = np.sum(kernel_values)
        return sum_kernels / (n * (bandwidth**d))

    def kde_1d(self, bandwidth):
        x_min, x_max = self.data.min(), self.data.max()
        x_values = np.linspace(x_min, x_max, 1000)
        y_values = [self.parzen_window(x, self.data, bandwidth) for x in x_values]
        return x_values, y_values

    def plot_1d_parzen_hist(self):
        fig, axs = plt.subplots(1, len(self.bandwidths), figsize=(15,7))
        axs = axs.ravel()
        for i, bandwidth in enumerate(self.bandwidths):
            x_1d, y_1d = self.kde_1d(bandwidth)
            bin_edges = np.arange(self.data.min(), self.data.max() + bandwidth, bandwidth)
            n, bins, patches = axs[i].hist(self.data, bins=bin_edges, density=True, alpha=0.5, color='gray', edgecolor='black')
            axs[i].plot(x_1d, y_1d, color='blue')
            axs[i].set_title(f'Bandwidth {bandwidth}')
            axs[i].set_xlabel('Feature1')
            axs[i].set_ylabel('Density')
        plt.tight_layout()
        plt.show()
    
    def plot_1d_parzen(self):
        fig, axs = plt.subplots(1, len(self.bandwidths), figsize=(15, 7))
        axs = axs.ravel()
        for i, bandwidth in enumerate(self.bandwidths):
            x_values, y_values = self.kde_1d(bandwidth)
            axs[i].plot(x_values, y_values, label=f'Bandwidth = {bandwidth}')
            axs[i].set_title(f'Bandwidth = {bandwidth}')
            axs[i].set_xlabel('Data Points')
            axs[i].set_ylabel('Density')
            axs[i].legend()
        plt.tight_layout()
        plt.show()

    def parzen_window_2d(self, x, y, data, bandwidth):
        n = len(data)
        kernel_values = np.sqrt((x - data[:, 0])**2 + (y - data[:, 1])**2) <= bandwidth / 2
        sum_kernels = np.sum(kernel_values)
        return sum_kernels / (n * bandwidth**2)

    def kde_2d(self, bandwidth):
        x_min, x_max = self.data[:, 0].min(), self.data[:, 0].max()
        y_min, y_max = self.data[:, 1].min(), self.data[:, 1].max()
        x_values = np.linspace(x_min, x_max, 50)
        y_values = np.linspace(y_min, y_max, 50)
        X, Y = np.meshgrid(x_values, y_values)
        Z = np.zeros(X.shape)

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = self.parzen_window_2d(X[i, j], Y[i, j], self.data, bandwidth)

        return X, Y, Z

    def plot_2d_parzen(self):
        fig, axes = plt.subplots(2, len(self.bandwidths), figsize=(15, 7), subplot_kw={'projection': '3d'})
        for i, bandwidth in enumerate(self.bandwidths):
            x, y, density = self.kde_2d(bandwidth)

            ax_front = axes[0, i]
            surface_front = ax_front.plot_surface(x, y, density, cmap='viridis', rstride=5, cstride=5, alpha=1, antialiased=True)
            ax_front.set_xlabel('Feature 1')
            ax_front.set_ylabel('Feature 2')
            ax_front.set_zlabel('Density')
            ax_front.view_init(elev=0, azim=50)
            ax_front.set_title(f'Bandwidth: {bandwidth}')

            ax_elevated = axes[1, i]
            surface_elevated = ax_elevated.plot_surface(x, y, density, cmap='viridis', rstride=5, cstride=5, alpha=1, antialiased=True)
            ax_elevated.view_init(elev=25, azim=50)
            ax_elevated.set_xlabel('Feature 1')
            ax_elevated.set_ylabel('Feature 2')
            ax_elevated.set_zlabel('Density')
            ax_elevated.set_title(f'Bandwidth: {bandwidth}')

        plt.tight_layout()
        plt.show()


