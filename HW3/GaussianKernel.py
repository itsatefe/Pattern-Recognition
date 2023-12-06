import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class GaussianKernel:

    def __init__(self, data, bandwidths):
        self.data = data
        self.bandwidths = bandwidths
        
    def gaussian_kernel(self,x):
        d = x.shape[0]
        exponent = -0.5 * np.dot(x.T, x)
        return np.exp(exponent) / ((2 * np.pi) ** (d / 2))


    def gaussian_kernel_density_estimate(self, x, bandwidth):
        n = len(self.data)
        d = self.data.shape[1] if len(self.data.shape) > 1 else 1
        kernel_values = np.zeros(n)
        for point in self.data:
            kernel_values += self.gaussian_kernel((x - point) / bandwidth)
        return np.sum(kernel_values) / (n * (bandwidth**d))


    def kde_2d(self, bandwidth, grid_size=50):
        feature1_range = np.linspace(self.data[:, 0].min(), self.data[:, 0].max(), grid_size)
        feature2_range = np.linspace(self.data[:, 1].min(), self.data[:, 1].max(), grid_size)
        feature1_values, feature2_values = np.meshgrid(feature1_range, feature2_range)
        density_values = np.zeros_like(feature1_values)

        for i in range(grid_size):
            for j in range(grid_size):
                point = np.array([feature1_range[i], feature2_range[j]])
                density_values[i, j] = self.gaussian_kernel_density_estimate(point, bandwidth)

        return feature1_values, feature2_values, density_values


    def plot_2d_kde(self):
        fig, axes = plt.subplots(2, len(self.bandwidths), figsize=(15, 7), subplot_kw={'projection': '3d'})

        for i, bandwidth in enumerate(self.bandwidths):
            x, y, density = self.kde_2d(bandwidth)

            # Front view
            ax_front = axes[0, i]
            surface_front = ax_front.plot_surface(x, y, density, cmap='viridis', rstride=5, cstride=5, alpha=1, antialiased=True)
            ax_front.set_xlabel('Feature 1')
            ax_front.set_ylabel('Feature 2')
            ax_front.set_zlabel('Density')
            ax_front.view_init(elev=0, azim=50)
            ax_front.set_title(f'Bandwidth: {bandwidth}')

            # Elevated view
            ax_elevated = axes[1, i]
            surface_elevated = ax_elevated.plot_surface(x, y, density, cmap='viridis', rstride=5, cstride=5, alpha=1, antialiased=True)
            ax_elevated.view_init(elev=25, azim=50)
            ax_elevated.set_xlabel('Feature 1')
            ax_elevated.set_ylabel('Feature 2')
            ax_elevated.set_zlabel('Density')
            ax_elevated.set_title(f'Bandwidth: {bandwidth}')

        plt.tight_layout()
        plt.show()
        
    def kde_1d(self, bandwidth, grid_size=100):
        feature_range = np.linspace(self.data.min(), self.data.max(), grid_size)
        density_values = np.zeros_like(feature_range)
        for i, point in enumerate(feature_range):
            density_values[i] = self.gaussian_kernel_density_estimate(point, bandwidth)

        return feature_range, density_values

    def plot_1d_kde(self):
        fig, axes = plt.subplots(1, len(self.bandwidths), figsize=(15, 4))
        for i, bandwidth in enumerate(self.bandwidths):
            x, density = self.kde_1d(bandwidth)
            ax = axes[i]
            ax.plot(x, density, label=f'Bandwidth: {bandwidth}', color='blue')
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Density')
            ax.set_title(f'Bandwidth: {bandwidth}')

        plt.tight_layout()
        plt.legend()
        plt.show()





