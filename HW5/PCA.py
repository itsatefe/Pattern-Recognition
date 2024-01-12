import numpy as np
import pandas as pd

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.eigenvectors = None
        self.eigenvalues = None
        self.mean = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        X = (X - self.mean)
        cov_matrix = np.cov(X.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        eigenvectors = eigenvectors.T
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]
        self.eigenvectors = eigenvectors
        self.eigenvalues = eigenvalues
        self.components = np.real(eigenvectors[:self.n_components])

    def transform(self, X):
        X = (X - self.mean)
        return np.dot(X, self.components.T)
    
    def inverse_transform(self, X):
        return np.dot(X, self.components) + self.mean

