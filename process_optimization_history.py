import argparse
import time

import numpy as np
from opytimizer.utils.history import History

import models.ensemble as e
import utils.constants as c
import utils.load as l


def get_arguments():
    """Gets arguments from the command line.

    Returns:
        A parser with the input arguments.

    """

    # Creates the ArgumentParser
    parser = argparse.ArgumentParser(usage='Process the post-optimization information into real results.')

    # Adds a dataset argument with pre-defined choices
    parser.add_argument('dataset', help='Dataset identifier', choices=['RSDataset', 'RSSCN7', 'UCMerced_LandUse'])

    # Adds an identifier argument to the desired fold identifier
    parser.add_argument('fold', help='Fold identifier',type=int, choices=range(1, 6))

    # Adds an identifier argument to the desired type of ensemble
    parser.add_argument('type', help='Ensemble type identifier', choices=['weight', 'boolean'])

    # Adds an identifier argument to the desired meta-heuristic
    parser.add_argument('mh', help='Meta-heuristic identifier',
                        choices=['abc', 'ba', 'bh', 'cs', 'fa', 'fpa', 'gp', 'pso', 'umda'])

    return parser.parse_args()


if __name__ == '__main__':
    # Gathers the input arguments
    args = get_arguments()

    # Gathering variables from arguments
    dataset = args.dataset
    fold = args.fold
    type = args.type
    meta = args.mh

    # Defining an input file
    input_file = f'output/{meta}_{type}_{dataset}_val_{fold}.pkl'

    # Creating a History object
    h = History()

    # Loading the input file
    h.load(input_file)

    # Loading the predictions and labels
    preds, y = l.load_candidates(dataset, 'test', fold)

    # Gathering the best weights
    best_weights = np.asarray(h.best_agent[-1][0])

    # Checks if the type of used ensemble was weight-based
    if type == 'weight':
        # Ensuring that the sum of weights is one and avoids division by zero
        best_weights = best_weights / max(best_weights.sum(), c.EPSILON)

    # Or if it was boolean-based
    elif type == 'boolean':
        # Rounding weights to fulfill the boolean-based ensemble
        best_weights = np.round(best_weights)

    # Defining the starting time of ensemble creation
    start = time.time()

    # Evaluating ensemble
    acc = e.evaluate(best_weights, preds, y)

    # Defining the ending time of ensemble creation
    end = time.time()

    print(f'Ensemble accuracy: {acc}')
    print(f'Time to create: {end-start} seconds')

    print('\nSaving outputs ...')

    # Saving outputs
    with open(f'output/{meta}_{type}_{dataset}_test_{fold}.txt', 'w') as f:
        f.write(f'{acc}\n{best_weights}\n{np.where(best_weights>0,1,0).sum()}') 


    with open(f'output/{meta}_{type}_{dataset}_test_{fold}.time', 'w') as f:
        f.write(f'{end-start}')

    print('Outputs saved.')
