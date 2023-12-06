import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import math
import seaborn as sns

class KNN:
    def __init__(self, data, K):
        self.data = data
        self.K = K
        
    def KNN_density_estimation(self, x, k):
        N, D = self.data.shape
        distance = np.array([math.dist(row, x) for index, row in self.data.iterrows()])
        filtered_distance = [value for value in distance if value != 0]
        R_k = np.sort(filtered_distance)[k - 1]
        v = (np.pi ** (D/2)) / np.math.gamma((D/2) + 1)
        return k / (N * v * (R_k ** D))


    def plot_2d_knn(self):
        fig, axes = plt.subplots(2, len(self.K), figsize=(15, 10), subplot_kw={'projection': '3d'})

        for i, k_value in enumerate(self.K):
            # Front view
            ax_front = axes[0, i]
            feature1_range = np.linspace(self.data['Feature1'].min(), self.data['Feature1'].max(), 50)
            feature2_range = np.linspace(self.data['Feature2'].min(), self.data['Feature2'].max(), 50)
            feature1_values, feature2_values = np.meshgrid(feature1_range, feature2_range)
            density_values = np.zeros_like(feature1_values)

            for j in range(len(feature1_range)):
                for l in range(len(feature2_range)):
                    density_values[j, l] = self.KNN_density_estimation(np.array([feature1_range[j], feature2_range[l]]), k_value)

            surface = ax_front.plot_surface(feature1_values, feature2_values, density_values, cmap='viridis')
            ax_front.view_init(elev=0, azim=50)
            ax_front.set_xlabel('Feature1')
            ax_front.set_ylabel('Feature2')
            ax_front.set_zlabel('Density')
            ax_front.set_title(f'Front View: K={k_value}')

            # Elevated view
            ax_elevated = axes[1, i]
            surface = ax_elevated.plot_surface(feature1_values, feature2_values, density_values, cmap='viridis')
            ax_elevated.view_init(elev=25, azim=50)
            ax_elevated.set_xlabel('Feature1')
            ax_elevated.set_ylabel('Feature2')
            ax_elevated.set_zlabel('Density')
            ax_elevated.set_title(f'Elevated View: K={k_value}')

        plt.tight_layout()
        plt.show()



    def plot_1d_knn(self):
        fig, axes = plt.subplots(1, len(self.K), figsize=(15, 8))

        for i, k_value in enumerate(self.K):
            ax_front = axes[i]
            feature_range = np.linspace(self.data.min(), self.data.max(), 100)
            density_values = np.zeros_like(feature_range)

            for j, feature_value in enumerate(feature_range):
                density_values[j] = self.KNN_density_estimation(feature_value, k_value)

            ax_front.plot(feature_range, density_values, color='blue')
            ax_front.set_xlabel('Feature')
            ax_front.set_ylabel('Density')
            ax_front.set_title(f'Front View: K={k_value}')

        plt.tight_layout()
        plt.show()

