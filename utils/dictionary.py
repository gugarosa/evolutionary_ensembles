import numpy as np


def create_dictionary(dataset):
    """It creates a dictionary for a particular dataset.

    Args:
        dataset (string): Dataset's identifier (should be the same as found in data/ folder).

    Returns:
        A dictionary mapping categorical labels to integer labels.

    """

    # Defining basis file path
    file_path = f'data/{dataset}/test/ground_1.txt'

    # Creating an empty list to hold the loaded labels
    loaded_labels = []

    # For each possible line
    for line in open(file_path, 'r'):
        # Appends the content of the line to the list
        loaded_labels.append(line.strip())

    # Creates the categorical labels
    cats = np.unique(loaded_labels)

    # Creates the integer labels
    labels = [i for i in range(cats.shape[0])]

    return dict(zip(cats, labels))
