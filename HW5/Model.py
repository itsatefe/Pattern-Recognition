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


class KMeans:
    def __init__(self, n_clusters=2, max_iter=100, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.centroids = None
        self.inertia_ = None  # Adding inertia attribute

    def _initialize_centroids(self, X):
        np.random.seed(self.random_state)
        random_idx = np.random.permutation(X.shape[0])
        centroids = X[random_idx[:self.n_clusters]]
        return centroids

    def _compute_distances(self, X, centroids):
        distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
        return distances

    def _assign_clusters(self, distances):
        return np.argmin(distances, axis=0)

    def _update_centroids(self, X, labels):
        centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])
        return centroids

    def fit(self, X):
        self.centroids = self._initialize_centroids(X)
        self.inertia_ = 0  # Initialize inertia

        for _ in range(self.max_iter):
            old_centroids = self.centroids
            distances = self._compute_distances(X, old_centroids)
            self.labels_ = self._assign_clusters(distances)
            self.centroids = self._update_centroids(X, self.labels_)
            self.inertia_ = self._compute_inertia(X, self.labels_)

            if np.all(old_centroids == self.centroids):
                break

        return self

    def _compute_inertia(self, X, labels):
        inertia = 0
        for i in range(self.n_clusters):
            cluster_points = X[labels == i]
            inertia += ((cluster_points - self.centroids[i]) ** 2).sum()
        return inertia

    def predict(self, X):
        distances = self._compute_distances(X, self.centroids)
        return self._assign_clusters(distances)

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_
