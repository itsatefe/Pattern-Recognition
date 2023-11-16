import numpy as np

class OvA:
    def __init__(self, binary_classifier):
        self.binary_classifiers = []
        self.binary_classifier = binary_classifier
        self.cost_history = []

    def train(self, X, y):
        unique_classes = np.unique(y)
        for class_label in unique_classes:
            binary_labels = (y == class_label).astype(int)
            classifier = self.binary_classifier()
            classifier.fit(X, binary_labels)
            self.binary_classifiers.append((class_label, classifier))
            self.cost_history.append((classifier.cost_history,f'{class_label} vs All'))

    def predict(self, X):
        predictions = []
        for _, classifier in self.binary_classifiers:
            prediction = classifier.predict(X)
            predictions.append(prediction)

        predicted_classes = np.argmax(predictions, axis=0)
        return predicted_classes
    
class OvO:
    def __init__(self, binary_classifier):
        self.binary_classifiers = []
        self.binary_classifier = binary_classifier
        self.unique_classes = None
        self.cost_history = []

    def train(self, X, y):
        self.unique_classes = np.unique(y)
        num_classes = len(self.unique_classes)

        for i in range(num_classes):
            for j in range(i + 1, num_classes):
                class_indices = np.logical_or(y == self.unique_classes[i], y == self.unique_classes[j])
                X_pair, y_pair = X[class_indices], y[class_indices]
                binary_labels = (y_pair == self.unique_classes[i]).astype(int)
                classifier = self.binary_classifier()
                classifier.fit(X_pair, binary_labels)
                self.binary_classifiers.append((self.unique_classes[i], self.unique_classes[j], classifier))
                self.cost_history.append((classifier.cost_history,f"class {i} vs {j}"))

    def predict(self, X):
        num_instances = X.shape[0]
        num_classifiers = len(self.binary_classifiers)

        votes = np.zeros((num_classifiers, num_instances))
        for k, (_, _, classifier) in enumerate(self.binary_classifiers):
            binary_prediction = classifier.predict(X)
            votes[k, :] = binary_prediction

        class_votes = np.zeros((len(self.unique_classes), num_instances))
        for k, (class1, class2, _) in enumerate(self.binary_classifiers):
            class_votes[class1, :] += (votes[k, :] == 1)
            class_votes[class2, :] += (votes[k, :] == 0)

        predicted_classes = np.argmax(class_votes, axis=0)
        return predicted_classes






