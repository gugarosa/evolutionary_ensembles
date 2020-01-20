import numpy as np

import utils.load as l


def load_candidates(dataset, step, fold):
    """Loads candidates from a particular dataset, step (validation or test) and fold number.

    Args:
        dataset (str): Dataset's identifier.
        step (str): Whether it should load from validation or test.
        fold (int): Number of fold to be loaded.

    Returns:
        Numpy arrays holding the ground truth and predicted labels.

    """

    print(f'Loading candidates from ({dataset}, {step}, {fold}) ...')

    # Loads the ground truth labels from desired dataset, step and fold
    labels = l.load_labels(dataset, step, fold)

    # Loads the predictions from desired dataset, step and fold
    preds = l.load_predictions(dataset, step, fold)

    # Checks if amount of loaded samples are equal
    if labels.shape[0] != preds.shape[0]:
        # If not, raises a RuntimeError
        raise RuntimeError(
            'Amount of ground truth labels differ from predictions.')

    print(f'Predictions: {preds.shape} | Ground Truth: {labels.shape}.')

    return preds, labels


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
