import numpy as np
import utils.metrics as m


def weighted_classifier(preds, labels):
    """Creates a function that weights classifier's predictions and minimizes it accuracy.

    Args:
        preds (np.array): Array of predictions of shape (n_samples, n_classifiers).
        labels (np.array): Array of ground truth labels of shape (n_samples, 1).

    Returns:
        A function to be optimized.
    """

    def f(w):
        """Weights predictions and returns an accuracy score.

        Args:
            w (float): Array of weights.

        Returns:
            1 - accuracy.

        """

        # Ensures that the sum of candidates is one and avoids division by zero
        # w = w / max(w.sum(), 1e-10)

        # Gathering the maximum label identifier
        max_label = np.max(labels)

        # Creating an array to hold the weighted predictions
        w_preds = np.zeros((preds.shape[0], max_label+1))

        # For every possible prediction
        for i in range(preds.shape[0]):
            # For every possible classifier
            for j in range(preds.shape[1]):
                # Sums the weighted classifier's prediction to its position in the final array
                w_preds[i][preds[i][j]] += preds[i][j] * w[j]

        # Gathers the most weighted prediction
        hat_preds = np.argmax(w_preds, axis=1)

        # Calculates the accuracy
        acc = m.accuracy(hat_preds, labels)

        return 1 - acc

    return f


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
