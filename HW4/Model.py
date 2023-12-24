import numpy as np
from sklearn.preprocessing import LabelEncoder


class QuadraticDiscriminantAnalysis:
    def __init__(self,reg_param = 0):
        self.pi_mle = None
        self.mu_mle = None
        self.sigma_mle = None
        self.num_class = None
        self.reg_param = reg_param

    def fit(self, X, y):
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
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
        reg_param= self.reg_param
        variances = np.diag(covariance)
        regularized_covariance = covariance + np.eye(D) * variances * reg_param
        cov_det = np.linalg.det(regularized_covariance)
        if cov_det <= 0:
            cov_det = 1e-6
        log_coefficient = -0.5 * D * np.log(2 * np.pi) - 0.5 * np.log(cov_det)
        sigma_inv = np.linalg.inv(regularized_covariance)
        likelihood_matrix = [np.exp(log_coefficient - 0.5 * np.dot(np.dot((sample - mean), sigma_inv), (sample - mean).T)) for sample in x]
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

