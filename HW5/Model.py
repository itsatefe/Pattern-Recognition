import numpy as np
from scipy.stats import multivariate_normal

class GaussianMixtureModel:
    def __init__(self, n_components, max_iter=100, tol=1e-4):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol

    def initialize_parameters(self, data):
        self.n_samples, self.n_features = data.shape
        self.means = data[np.random.choice(self.n_samples, self.n_components, replace=False)]
        self.covariances = np.tile(np.eye(self.n_features), (self.n_components, 1, 1))
        self.weights = np.ones(self.n_components) / self.n_components

    def expectation_step(self, data):
        self.posteriors = np.zeros((self.n_samples, self.n_components))
        for i in range(self.n_components):
            self.posteriors[:, i] = self.weights[i] * multivariate_normal.pdf(data, mean=self.means[i], cov=self.covariances[i])
        self.posteriors /= self.posteriors.sum(axis=1, keepdims=True)

    def maximization_step(self, data):
        self.weights = self.posteriors.mean(axis=0)
        self.means = np.dot(self.posteriors.T, data) / self.posteriors.sum(axis=0)[:, np.newaxis]
        for i in range(self.n_components):
            diff = data - self.means[i]
            self.covariances[i] = np.dot(self.posteriors[:, i] * diff.T, diff) / self.posteriors[:, i].sum()

    def fit(self, data):
        self.initialize_parameters(data)
        for iteration in range(self.max_iter):
            self.expectation_step(data)
            self.maximization_step(data)
            if iteration > 0:
                if np.abs(self.prev_log_likelihood - self.log_likelihood(data)) < self.tol:
                    break
            self.prev_log_likelihood = self.log_likelihood(data)

    def log_likelihood(self, data):
        likelihoods = np.zeros((self.n_samples, self.n_components))
        for i in range(self.n_components):
            likelihoods[:, i] = multivariate_normal.pdf(data, mean=self.means[i], cov=self.covariances[i])
        return np.log(np.dot(likelihoods, self.weights)).sum()

    def predict(self, data):
        return np.argmax(self.posteriors, axis=1)



