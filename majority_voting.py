import argparse
import time

import models.ensemble as e
import utils.constants as c
import utils.load as l
import utils.metrics as m


def get_arguments():
    """Gets arguments from the command line.

    Returns:
        A parser with the input arguments.

    """

    # Creates the ArgumentParser
    parser = argparse.ArgumentParser(usage='Creates an ensemble of classifiers based on majority voting.')

    # Adds a dataset argument with pre-defined choices
    parser.add_argument('dataset', help='Dataset identifier', choices=['RSDataset', 'RSSCN7', 'UCMerced_LandUse'])

    return parser.parse_args()


if __name__ == '__main__':
    # Gathers the input arguments
    args = get_arguments()

    # Gathering variables from arguments
    dataset = args.dataset

    # Creating empty lists to save outputs
    val_accs, test_accs = [], []
    val_time, test_time = [], []

    print(f'\nPerforming majority voting over {dataset} ...\n')

    # For each possible fold
    for k in range(c.N_FOLDS):
        print(f'Fold {k+1}/{c.N_FOLDS}:')

        # Loads the validation step predictions and labels
        val_pred, val_y = l.load_candidates(dataset, 'val', k+1)

        # Loads the testing step predictions and labels
        test_pred, test_y = l.load_candidates(dataset, 'test', k+1)
        
        # Defining the starting time of validation majority voting
        start_v = time.time()

        # Gather the majority votes between validation predictions
        val_votes = e.majority_voting(val_pred)

        # Defining the ending time of validation majority voting
        end_v = time.time()

        # Defining the starting time of testing majority voting
        start_t = time.time()

        # Gather the majority votes between test predictions
        test_votes = e.majority_voting(test_pred)

        # Defining the ending time of testing majority voting
        end_t = time.time()

        # Calculates the validation accuracy
        val_acc = m.accuracy(val_votes, val_y)

        # Calculates the testing accuracy
        test_acc = m.accuracy(test_votes, test_y)

        print(f'Val. Accuracy: {val_acc} | Test Accuracy: {test_acc}.\n')

        # Appending values to lists
        val_accs.append(val_acc)
        test_accs.append(test_acc)
        val_time.append(end_v - start_v)
        test_time.append(end_t - start_t)

    print('Saving outputs ...')

    # Saving validation accuracies to file
    with open(f'output/mv_{dataset}_val.txt', 'w') as f:
        f.writelines([f'{acc}\n' for acc in val_accs])

    # Saving validation times to file
    with open(f'output/mv_{dataset}_val.time', 'w') as f:
        f.writelines([f'{time}\n' for time in val_time])

    # Saving testing accuracies to file
    with open(f'output/mv_{dataset}_test.txt', 'w') as f:
        f.writelines([f'{acc}\n' for acc in test_accs])

    # Saving testing times to file
    with open(f'output/mv_{dataset}_test.time', 'w') as f:
        f.writelines([f'{time}\n' for time in test_time])

    print('Outputs saved.')
