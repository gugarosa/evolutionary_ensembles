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


def relationship_matrix(i_preds, j_preds, labels):
    """Calculates the relationship matrix between the predictions from two classifiers.

    Args:
        i_preds (np.array): Predictions from classifier #1.
        j_preds (np.array): Predictions from classifier #2.
        labels (np.array): Ground truth labels.

    Returns:
        The relationship matrix between classifier #1 and classifier #2.

    """

    # Initializing relationship matrix
    r_matrix = np.zeros((2, 2))

    # For every prediction of `i`, `j` and ground truth label
    for i, j, l in zip(i_preds, j_preds, labels):
        # If classifier `i` got a hit and classifier `j` got a hit
        if i == l and j == l:
            # Increase `a`
            r_matrix[0][0] += 1

        # If classifier `i` got a miss and classifier `j` got a miss
        if i != l and j != l:
            # Increase `d`
            r_matrix[1][1] += 1

        # If classifier `i` got a hit and classifier `j` got a miss
        if i == l and j != l:
            # Increase `c`
            r_matrix[1][0] += 1

        # If classifier `i` got a miss and classifier `j` got a hit
        if i != l and j == l:
            # Increase `b`
            r_matrix[0][1] += 1

    return r_matrix


def correlation(i_preds, j_preds, labels):
    """Calculates the correlation between the predictions from two classifiers.

    Args:
        i_preds (np.array): Predictions from classifier #1.
        j_preds (np.array): Predictions from classifier #2.
        labels (np.array): Ground truth labels.

    Returns:
        The correlation between classifier #1 and classifier #2.

    """

    # Calculating the relationship matrix between `i` and `j`
    r_matrix = relationship_matrix(i_preds, j_preds, labels)

    # Gathering `a`, `b`, `c` and `d`
    a, b, c, d = r_matrix[0][0], r_matrix[0][1], r_matrix[1][0], r_matrix[1][1]

    # Calculating the correlation between classifiers
    corr = ((a * d) - (b * c)) / \
        (np.sqrt((a + b) * (c + d) * (a + c) * (b + d)))

    return corr


def disagreement_measure(i_preds, j_preds, labels):
    """Calculates the disagreement measure between the predictions from two classifiers.

    Args:
        i_preds (np.array): Predictions from classifier #1.
        j_preds (np.array): Predictions from classifier #2.
        labels (np.array): Ground truth labels.

    Returns:
        The disagreement measure between classifier #1 and classifier #2.

    """

    # Calculating the relationship matrix between `i` and `j`
    r_matrix = relationship_matrix(i_preds, j_preds, labels)

    # Gathering `a`, `b`, `c` and `d`
    a, b, c, d = r_matrix[0][0], r_matrix[0][1], r_matrix[1][0], r_matrix[1][1]

    # Calculating disagreement measure between classifiers
    dm = (b + c) / (a + b + c + d)

    return dm


def double_fault_measure(i_preds, j_preds, labels):
    """Calculates the double-fault measure between the predictions from two classifiers.

    Args:
        i_preds (np.array): Predictions from classifier #1.
        j_preds (np.array): Predictions from classifier #2.
        labels (np.array): Ground truth labels.

    Returns:
        The double-fault measure between classifier #1 and classifier #2.

    """

    # Calculating the relationship matrix between `i` and `j`
    r_matrix = relationship_matrix(i_preds, j_preds, labels)

    # Gathering d`
    d = r_matrix[1][1]

    # Calculating double-fault measure between classifiers
    dfm = d

    return dfm


def interrater_agreement(i_preds, j_preds, labels):
    """Calculates the interrater agreement between the predictions from two classifiers.

    Args:
        i_preds (np.array): Predictions from classifier #1.
        j_preds (np.array): Predictions from classifier #2.
        labels (np.array): Ground truth labels.

    Returns:
        The interrater agreement between classifier #1 and classifier #2.

    """

    # Calculating the relationship matrix between `i` and `j`
    r_matrix = relationship_matrix(i_preds, j_preds, labels)

    # Gathering `a`, `b`, `c` and `d`
    a, b, c, d = r_matrix[0][0], r_matrix[0][1], r_matrix[1][0], r_matrix[1][1]

    # Calculating interrater agreement between classifiers
    ia = (2 * ((a * c) - (b * d))) / ((a + b) * (c + d) + (a + c) * (b + d))

    return ia


def q_statistics(i_preds, j_preds, labels):
    """Calculates the q-statistics between the predictions from two classifiers.

    Args:
        i_preds (np.array): Predictions from classifier #1.
        j_preds (np.array): Predictions from classifier #2.
        labels (np.array): Ground truth labels.

    Returns:
        The q-statistics between classifier #1 and classifier #2.

    """

    # Calculating the relationship matrix between `i` and `j`
    r_matrix = relationship_matrix(i_preds, j_preds, labels)

    # Gathering `a`, `b`, `c` and `d`
    a, b, c, d = r_matrix[0][0], r_matrix[0][1], r_matrix[1][0], r_matrix[1][1]

    # Calculating q-statistics between classifiers
    q_stat = ((a * d) - (b * c)) / ((a * d) + (b * c))

    return q_stat
