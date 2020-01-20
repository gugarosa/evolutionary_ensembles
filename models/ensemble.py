import numpy as np


def majority_voting(preds):
    """Gathers the majority votes by finding the most frequent number in an array.

    Args:
        preds (np.array): An array of predictions.

    Returns:
        An array of votes (labels).

    """

    print('Calculating majority voting ...')

    # Calculate the majority votes by finding the most frequent number in array
    votes = [np.argmax(np.bincount(pred)) for pred in preds]

    print('Votes calculated.')

    return np.asarray(votes)
