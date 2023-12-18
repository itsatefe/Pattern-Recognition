import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class Metrics:
    def __init__(self, y_true):
        self.y_true = np.array(y_true)
        self.num_class = np.unique(self.y_true)

    def confusion_matrix(self, y_pred):
        unique_labels = np.unique(np.concatenate((self.y_true, y_pred)))
        matrix = np.zeros((len(unique_labels), len(unique_labels)), dtype=int)
        for true, pred in zip(self.y_true, y_pred):
            matrix[true, pred] += 1
        return matrix

    def accuracy(self,y_pred):
        cm = self.confusion_matrix(y_pred)
        return np.trace(cm) / np.sum(cm)

    def precision(self,y_pred):
        cm = self.confusion_matrix(y_pred)
        prec = []
        for label in range(len(self.num_class)):
            true_positive = cm[label, label]
            false_positive = np.sum(cm[:, label]) - true_positive
            prec.append(true_positive / (true_positive + false_positive))
        return prec 

    def recall(self,y_pred):
        cm = self.confusion_matrix(y_pred)
        rec = []
        for label in range(len(self.num_class)):
            true_positive = cm[label, label]
            false_negative = np.sum(cm[label, :]) - true_positive
            rec.append(true_positive / (true_positive + false_negative))
        return rec

    def f1_score(self,y_pred):
        precision_value = self.precision(y_pred)
        recall_value = self.recall(y_pred)
        f1 = []
        for label in range(len(self.num_class)):
            f1.append(2 * (precision_value[label] * recall_value[label]) / (precision_value[label] + recall_value[label]))
        return f1
    
    def misclassified_samples(self,y_pred):
        cm = self.confusion_matrix(y_pred)
        return np.sum(cm) - np.trace(cm)
    
    def display_confusion_matrix(self, y_pred, type_data):
        colors = ["#e74c3c", "#2ecc71"]
        plt.figure(figsize=(2, 2))
        cm = self.confusion_matrix(y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap=colors, cbar=False, xticklabels=self.num_class, yticklabels=self.num_class)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(type_data)
        plt.show()



