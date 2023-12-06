import numpy as np

class SoftmaxRegression:
    def __init__(self, num_classes, learning_rate=0.01, num_iterations=1000):
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.theta = None
        self.cost_history = []

    def _one_hot_encode(self, y):
        one_hot_labels = np.zeros((len(y), self.num_classes))
        for i, label in enumerate(y):
            one_hot_labels[i, label] = 1
        return one_hot_labels

    def _softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    # The categorical cross-entropy loss.
    def _compute_cost(self, X, y, theta):
        m = len(y)
        scores = X.dot(theta)
        probabilities = self._softmax(scores)
        # Multinomial distribution
        log_probabilities = -np.log(probabilities[range(m), y])
        cost = np.sum(log_probabilities) / m
        return cost

    def _compute_gradient(self, X, y, theta):
        m = len(y)
        scores = X.dot(theta)
        probabilities = self._softmax(scores)
        gradient = -X.T.dot(self._one_hot_encode(y) - probabilities) / m
        return gradient

    def train(self, X, y, epsilon = 1e-6):
        X_bias = np.c_[np.ones((X.shape[0], 1)), X]
        self.theta = np.zeros((X_bias.shape[1], self.num_classes))

        for i in range(self.num_iterations):
            gradient = self._compute_gradient(X_bias, y, self.theta)
            self.theta -= self.learning_rate * gradient
            cost = self._compute_cost(X_bias, y, self.theta)
            self.cost_history.append(cost)
            if i > 0 and abs(self.cost_history[i - 1] - self.cost_history[i]) < epsilon:
                print(f"Converged after {i} iterations.")
                break

    def predict(self, X):
        X_bias = np.c_[np.ones((X.shape[0], 1)), X]
        scores = X_bias.dot(self.theta)
        probabilities = self._softmax(scores)
        return np.argmax(probabilities, axis=1)


