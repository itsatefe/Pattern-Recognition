#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class DataProcessor:
    def __init__(self, data):
        self.data = data

    def normalization(self):
        mean = self.data.mean()
        self.data = self.data - mean
        std_dev = self.data.std()
        self.data = self.data / std_dev
        return self.data

    def plot_1d_dataset(self):
        plt.scatter(self.data, [0] * len(self.data), marker='o')
        plt.axhline(0, color='black', linestyle='--', linewidth=1)
        plt.xlabel('Values')
        plt.title('1D Dataset Plot on X-axis Only')
        plt.yticks([])
        plt.show()

    def plot_2d_dataset(self):
        plt.scatter(self.data['Feature1'], self.data['Feature2'], marker='o')
        plt.xlabel('Values')
        plt.title('2D Synthetic Plot')
        plt.show()

