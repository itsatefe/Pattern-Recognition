import numpy as np
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans

class GaussianMixtureModel:
    def __init__(self, n_components, max_iter=100, tol=1e-4, reg_covar=1e-6):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar

    def initialize_parameters(self, data):
        # Using KMeans for better initialization of means
        kmeans = KMeans(n_clusters=self.n_components, n_init=1)
        labels = kmeans.fit_predict(data)
        self.means = kmeans.cluster_centers_

        self.n_samples, self.n_features = data.shape
        self.covariances = np.tile(np.eye(self.n_features), (self.n_components, 1, 1))
        self.weights = np.ones(self.n_components) / self.n_components

    def expectation_step(self, data):
        self.posteriors = np.zeros((self.n_samples, self.n_components))
        for i in range(self.n_components):
            # Adding regularization term for numerical stability
            covar = self.covariances[i] + self.reg_covar * np.eye(self.n_features)
            self.posteriors[:, i] = self.weights[i] * multivariate_normal.pdf(data, mean=self.means[i], cov=covar)
        self.posteriors /= self.posteriors.sum(axis=1, keepdims=True)

    def maximization_step(self, data):
        self.weights = self.posteriors.mean(axis=0)
        self.means = np.dot(self.posteriors.T, data) / self.posteriors.sum(axis=0)[:, np.newaxis]
        for i in range(self.n_components):
            diff = data - self.means[i]
            covar = np.dot(self.posteriors[:, i] * diff.T, diff) / self.posteriors[:, i].sum()
            # Adding regularization term for numerical stability
            self.covariances[i] = covar + self.reg_covar * np.eye(self.n_features)

    def fit(self, data):
        self.initialize_parameters(data)
        self.prev_log_likelihood = -np.inf
        for iteration in range(self.max_iter):
            self.expectation_step(data)
            self.maximization_step(data)
            log_likelihood = self.log_likelihood(data)
            if np.abs(self.prev_log_likelihood - log_likelihood) < self.tol:
                break
            self.prev_log_likelihood = log_likelihood
          

    def log_likelihood(self, data):
        likelihoods = np.zeros((self.n_samples, self.n_components))
        for i in range(self.n_components):
            covar = self.covariances[i] + self.reg_covar * np.eye(self.n_features)
            likelihoods[:, i] = multivariate_normal.pdf(data, mean=self.means[i], cov=covar)
        return np.log(np.dot(likelihoods, self.weights)).sum()

    def predict(self, data):
        self.expectation_step(data)
        return np.argmax(self.posteriors, axis=1)

