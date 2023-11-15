import numpy as np

class GaussianLinearClassifier:
    def __init__(self):
        self.pi_mle = None
        self.mu_mle = None
        self.sigma_mle = None
        self.num_class = None

    def fit(self, X, y):
        self.num_class = y.unique()
        self.pi_mle = self.pi_mle_j(y)
        self.mu_mle = [np.mean(X[y == i], axis=0) for i in range(len(self.num_class))]
        self.sigma_mle = self.mle_covariance_matrix(X)

    def predict_likelihood(self, X):
        likelihood = [self.likelihood(X, self.mu_mle[i], self.sigma_mle) for i in range(len(self.num_class))]
        posterior = np.column_stack([self.pi_mle[i] * np.array(likelihood[i]) for i in range(len(self.num_class))])
        predictions = np.argmax(posterior, axis=1)
        return predictions

    def pi_mle_j(self, y_j):
        m = len(y_j)
        pi_j = [sum(y_j == i) / m for i in range(len(self.num_class))]
        return pi_j

    def mle_covariance_matrix(self, data):
        mean_vector = np.mean(data, axis=0)
        diff_matrix = data - mean_vector
        # normalized 1 / n-1
        mle_cov_matrix = (1 / (len(data)-1)) * np.dot(diff_matrix.T, diff_matrix)
        return mle_cov_matrix

    def likelihood(self, x, mean, covariance):
        D = len(mean)
        cov_det = np.linalg.det(covariance)
        coefficient = 1 / ((2 * np.pi) ** (D / 2) * np.sqrt(cov_det))
        sigma_inv = np.linalg.inv(covariance)
        likelihood_matrix = [coefficient * np.exp(-0.5 * np.dot(np.dot((sample - mean), sigma_inv), (sample - mean).T)) for sample in x]
        return likelihood_matrix

    def mahalanobis_distance(self, x, mean, covariance):
        sigma_inv = np.linalg.inv(covariance)
        distances = [np.dot(np.dot((sample - mean), sigma_inv), (sample - mean).T) for sample in x]
        return distances

    def predict_distance(self, X):
        mahalanobis_distances = [self.mahalanobis_distance(X, self.mu_mle[i], self.sigma_mle) for i in range(len(self.num_class))]
        posteriors = np.column_stack([mahalanobis_distances[i] for i in range(len(self.num_class))])
        predictions = np.argmin(posteriors, axis=1)
        return predictions

class QuadraticDiscriminantAnalysis:
    def __init__(self):
        self.pi_mle = None
        self.mu_mle = None
        self.sigma_mle = None
        self.num_class = None

    def fit(self, X, y):
        self.num_class = np.unique(y)
        self.pi_mle = self.pi_mle_j(y)
        self.mu_mle = [np.mean(X[y == i], axis=0) for i in self.num_class]
        self.sigma_mle = self.mle_covariance_matrix(X, y)

    def predict_likelihood(self, X):
        likelihood = [self.likelihood(X, self.mu_mle[i], self.sigma_mle[i]) for i in range(len(self.num_class))]
        posterior = np.column_stack([self.pi_mle[i] * np.array(likelihood[i]) for i in range(len(self.num_class))])
        predictions = np.argmax(posterior, axis=1)
        return predictions

    def pi_mle_j(self, y_j):
        m = len(y_j)
        pi_j = [np.sum(y_j == i) / m for i in self.num_class]
        return pi_j

    def mle_covariance_matrix(self, data, labels):
        unique_labels = np.unique(labels)
        # normalized 1 / n-1   
        covariance_matrices = [ (1 / (len(data[labels == i])-1)) * np.dot((data[labels == i] - np.mean(data[labels == i], axis=0)).T, data[labels == i] - np.mean(data[labels == i], axis=0)) for i in unique_labels]
        return covariance_matrices

    def likelihood(self, x, mean, covariance):
        D = len(mean)
        cov_det = np.linalg.det(covariance)
        coefficient = 1 / ((2 * np.pi) ** (D / 2) * np.sqrt(cov_det))
        sigma_inv = np.linalg.inv(covariance)
        likelihood_matrix = [coefficient * np.exp(-0.5 * np.dot(np.dot((sample - mean), sigma_inv), (sample - mean).T)) for sample in x]
        return likelihood_matrix
    
    def mahalanobis_distance(self, x, mean, covariance):
        sigma_inv = np.linalg.inv(covariance)
        distances = [np.dot(np.dot((sample - mean), sigma_inv), (sample - mean).T) for sample in x]
        return distances
    
    def predict_distance(self, X):
        mahalanobis_distances = [self.mahalanobis_distance(X, self.mu_mle[i], self.sigma_mle[i]) for i in range(len(self.num_class))]
        posteriors = np.column_stack([mahalanobis_distances[i] for i in range(len(self.num_class))])
        predictions = np.argmin(posteriors, axis=1)
        return predictions

