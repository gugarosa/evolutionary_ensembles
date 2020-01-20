import numpy as np


def accuracy(preds, labels):
    """Calculates the accuracy between predictions and ground truth labels.

    Args:
        preds (np.array): Array of predictions.
        labels (np.array): Array of labels.

    Returns:
        An accuracy score between 0 and 1.

    """

    # Calculating accuracy over the entire array
    acc = (preds == labels).sum() / preds.size

    return acc
