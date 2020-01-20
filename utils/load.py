import numpy as np

import utils.dictionary as d


def load_labels(dataset, step, fold):
    """Loads ground truth labels from a particular dataset, step (validation or test) and fold number.

    Args:
        dataset (str): Dataset's identifier.
        step (str): Whether it should load from validation or test.
        fold (int): Number of fold to be loaded.

    Returns:
        A numpy array holding the loaded ground truth labels.

    """

    # Creates a dictionary of the desired dataset
    dictionary = d.create_dictionary(dataset)

    # Defines the file input path
    file_path = f'data/{dataset}/{step}/ground_{fold + 1}.txt'

    # Creates a list of labels
    labels = []

    # For every possible line in the file
    for line in open(file_path, 'r'):
        # Appends the label already mapped with the dictionary
        labels.append(dictionary[line.strip()])

    return np.asarray(labels)


def load_predictions(dataset, step, fold):
    """Loads predictions from a particular dataset, step (validation or test) and fold number.

    Args:
        dataset (str): Dataset's identifier.
        step (str): Whether it should load from validation or test.
        fold (int): Number of fold to be loaded.

    Returns:
        A numpy array holding the predicted labels.

    """

    # Creates a dictionary of the desired dataset
    dictionary = d.create_dictionary(dataset)

    # Defines the file input path
    file_path = f'data/{dataset}/{step}/pred_{fold + 1}.txt'

    # Creates a list of labels
    preds = []

    # For every possible line in the file
    for line in open(file_path, 'r'):
        # Appends each line as a new list
        cat_preds = list(line.split())

        # Maps the categorical predictions using the dictionary
        preds.append([dictionary[c] for c in cat_preds])

    return np.asarray(preds)
