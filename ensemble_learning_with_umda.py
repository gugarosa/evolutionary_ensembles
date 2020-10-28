import argparse

import numpy as np

import models.ensemble as e
import utils.load as l
import utils.metrics as m
import utils.wrapper as w


def get_arguments():
    """Gets arguments from the command line.

    Returns:
        A parser with the input arguments.

    """

    # Creates the ArgumentParser
    parser = argparse.ArgumentParser(
        usage='Optimizes a boolean-based ensemble using Univariate Marginal Distribution Algorithm.')

    # Adds a dataset argument with pre-defined choices
    parser.add_argument('dataset', help='Dataset identifier', choices=['RSDataset', 'RSSCN7', 'UCMerced_LandUse'])

    # Adds a descriptor argument with pre-defined choices
    parser.add_argument('descriptor', help='Descriptor identifier', choices=['global', 'cnn', 'all'])

    # Adds an identifier argument to the desired fold identifier
    parser.add_argument('fold', help='Fold identifier', type=int, choices=range(1, 6))

    # Adds an identifier argument to the desired number of agents
    parser.add_argument('-n_agents', help='Number of meta-heuristic agents', type=int, default=10)

    # Adds an identifier argument to the desired number of iterations
    parser.add_argument('-n_iter', help='Number of meta-heuristic iterations', type=int, default=10)

    return parser.parse_args()


if __name__ == '__main__':
    # Gathers the input arguments
    args = get_arguments()

    # Gathering variables from arguments
    dataset = args.dataset
    descriptor = args.descriptor
    step = 'val'
    fold = args.fold

    # Random seed for experimental consistency
    np.random.seed(fold-1)

    # Loads the predictions and labels
    preds, y = l.load_candidates(dataset, step, fold)

    # If descriptor is global-based
    if descriptor == 'global':
        # Gets the global predictors
        preds = preds[:, :35]

    # If descriptor is cnn-based
    elif descriptor == 'cnn':
        # Gets the CNN predictors
        preds = preds[:, 35:]

    # Defining function to be optimized
    opt_fn = e.boolean_classifiers(preds, y)

    # Defining number of agents, number of variables and number of iterations
    n_agents = args.n_agents
    n_variables = preds.shape[1]
    n_iterations = args.n_iter

    # Defining meta-heuristic hyperparameters
    hyperparams = dict(p_selection=0.75, lower_bound=0.05, upper_bound=0.95)

    # Running the optimization task
    history = w.optimize_umda(opt_fn, n_agents, n_variables, n_iterations, hyperparams)

    # Saves the history object to an output file
    history.save(f'output/umda_boolean_{dataset}_{step}_{fold}.pkl')
