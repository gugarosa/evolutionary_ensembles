import argparse

import models.ensemble as e
import utils.load as l
import utils.metrics as m

# Defining a constant to fulfill number of folds
N_FOLDS = 5


def get_arguments():
    """Gets arguments from the command line.

    Returns:
        A parser with the input arguments.

    """

    # Creates the ArgumentParser
    parser = argparse.ArgumentParser(
        usage='Creates an ensemble of classifiers based on majority voting.')

    # Adds a dataset argument with pre-defined choices
    parser.add_argument('dataset', help='Dataset identifier', choices=[
                        'RSDataset', 'RSSCN7', 'UCMerced_LandUse'])

    return parser.parse_args()


if __name__ == '__main__':

    # Gathers the input arguments
    args = get_arguments()

    # Gathering dataset
    dataset = args.dataset

    # Creating empty lists to save outputs
    val_accs, test_accs = [], []

    print(f'\nPerforming majority voting over {dataset} ...\n')

    # For each possible fold
    for k in range(N_FOLDS):
        print(f'Fold {k+1}/{N_FOLDS}:')

        # Loads the validation step predictions and labels
        val_pred, val_y = l.load_candidates(dataset, 'val', k+1)

        # Loads the testing step predictions and labels
        test_pred, test_y = l.load_candidates(dataset, 'test', k+1)

        # Gather the majority votes between validation predictions
        val_votes = e.majority_voting(val_pred)

        # Gather the majority votes between test predictions
        test_votes = e.majority_voting(test_pred)

        # Calculates the validation accuracy
        val_acc = m.accuracy(val_votes, val_y)

        # Calculates the testing accuracy
        test_acc = m.accuracy(test_votes, test_y)

        print(f'Val. Accuracy: {val_acc} | Test Accuracy: {test_acc}.\n')

        # Appending values to lists
        val_accs.append(val_acc)
        test_accs.append(test_acc)

    print('Saving outputs ...')

    # Saving validation accuracies to file
    with open(f'output/MV_{dataset}_val.txt', 'w') as f:
        f.writelines([f'{acc}\n' for acc in val_accs])

    # Saving testing accuracies to file
    with open(f'output/MV_{dataset}_test.txt', 'w') as f:
        f.writelines([f'{acc}\n' for acc in test_accs])

    print('Outputs saved.')
