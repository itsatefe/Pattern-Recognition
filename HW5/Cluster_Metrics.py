import numpy as np

def pair_confusion_matrix(predicted_labels, true_labels):
    tp = tn = fp = fn = 0
    for i in range(len(predicted_labels)):
        for j in range(i + 1, len(predicted_labels)):
            if predicted_labels[i] == predicted_labels[j] and true_labels[i] == true_labels[j]:
                tp += 1
            elif predicted_labels[i] != predicted_labels[j] and true_labels[i] != true_labels[j]:
                tn += 1
            elif predicted_labels[i] == predicted_labels[j] and true_labels[i] != true_labels[j]:
                fp += 1
            elif predicted_labels[i] != predicted_labels[j] and true_labels[i] == true_labels[j]:
                fn += 1
    return np.array([[2 * tp, 2 * tn], [2 * fp, 2 * fn]], dtype=np.float64)

def adjusted_rand_score(predicted_labels, true_labels):
    table = pair_confusion_matrix(predicted_labels, true_labels)
    tp, tn, fp, fn = table[0][0], table[0][1], table[1][0], table[1][1]
    numerator = 2 * (tp * tn - fn * fp)
    denominator = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    return numerator / denominator if denominator != 0 else 0