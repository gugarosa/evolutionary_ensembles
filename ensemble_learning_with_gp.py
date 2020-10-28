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
    parser = argparse.ArgumentParser(usage='Optimizes an ensemble using Genetic Programming.')

    # Adds a dataset argument with pre-defined choices
    parser.add_argument('dataset', help='Dataset identifier', choices=['RSDataset', 'RSSCN7', 'UCMerced_LandUse'])

    # Adds a descriptor argument with pre-defined choices
    parser.add_argument('descriptor', help='Descriptor identifier', choices=['global', 'cnn', 'all'])

    # Adds an identifier argument to the desired fold identifier
    parser.add_argument('fold', help='Fold identifier', type=int, choices=range(1, 6))

    # Adds an identifier argument to the desired type of ensemble
    parser.add_argument('type', help='Ensemble type identifier', choices=['weight', 'boolean'])

    # Adds an identifier argument to the desired number of trees
    parser.add_argument('-n_trees', help='Number of Genetic Programming trees', type=int, default=10)

    # Adds an identifier argument to the desired number of terminals
    parser.add_argument('-n_terminals', help='Number of Genetic Programming terminals', type=int, default=2)

    # Adds an identifier argument to the desired number of iterations
    parser.add_argument('-n_iter', help='Number of Genetic Programming iterations', type=int, default=10)

    # Adds an identifier argument to the desired minimum depth
    parser.add_argument('-min_depth', help='Minimum depth of Genetic Programming trees', type=int, default=2)

    # Adds an identifier argument to the desired maximum depth
    parser.add_argument('-max_depth', help='Maximum depth of Genetic Programming trees', type=int, default=5)

    return parser.parse_args()


if __name__ == '__main__':
    # Gathers the input arguments
    args = get_arguments()

    # Gathering variables from arguments
    dataset = args.dataset
    descriptor = args.descriptor
    step = 'val'
    fold = args.fold
    type = args.type

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

    # Checks if the type of used ensemble was weight-based
    if type == 'weight':
        # Defining function to be optimized
        opt_fn = e.weighted_classifiers(preds, y)

    # Or if it was boolean-based
    elif type == 'boolean':
        # Defining function to be optimized
        opt_fn = e.boolean_classifiers(preds, y)

    # Defining number of trees, number of terminals, number of variables and number of iterations
    n_trees = args.n_trees
    n_terminals = args.n_terminals
    n_variables = preds.shape[1]
    n_iterations = args.n_iter

    # Defining minimum and maximum depth of trees
    min_depth = args.min_depth
    max_depth = args.max_depth

    # Defining functions nodes
    functions = ['SUM', 'SUB', 'MUL', 'DIV']

    # Defining lower and upper bounds
    lb = [0] * n_variables
    ub = [1] * n_variables

    # Defining meta-heuristic hyperparameters
    hyperparams = dict(p_reproduction=0.25, p_mutation=0.1,
                       p_crossover=0.2, prunning_ratio=0.0)

    # Running the optimization task
    history = w.optimize_gp(opt_fn, n_trees, n_terminals, n_variables, n_iterations,
                            min_depth, max_depth, functions, lb, ub, hyperparams)

    # Saves the history object to an output file
    history.save(f'output/gp_{type}_{dataset}_{step}_{fold}.pkl')
