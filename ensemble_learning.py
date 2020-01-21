import argparse

import models.ensemble as e
import utils.load as l
import utils.metrics as m
import utils.mh as mh
import utils.wrapper as w


def get_arguments():
    """Gets arguments from the command line.

    Returns:
        A parser with the input arguments.

    """

    # Creates the ArgumentParser
    parser = argparse.ArgumentParser(
        usage='Optimizes an weighted-based ensemble using a meta-heuristic technique.')

    # Adds a dataset argument with pre-defined choices
    parser.add_argument('dataset', help='Dataset identifier', choices=[
                        'RSDataset', 'RSSCN7', 'UCMerced_LandUse'])

    # Adds a step argument with pre-defined choices
    parser.add_argument(
        'step', help='Whether it should load from validation', choices=['val'])

    # Adds an identifier argument to the desired fold identifier
    parser.add_argument('fold', help='Fold identifier',
                        type=int, choices=range(1, 6))

    # Adds an identifier argument to the desired meta-heuristic
    parser.add_argument('mh', help='Meta-heuristic identifier',
                        choices=['abc', 'ba', 'bha', 'cs', 'fa', 'fpa', 'pso'])

    # Adds an identifier argument to the desired number of agents
    parser.add_argument(
        '-n_agents', help='Number of meta-heuristic agents', type=int, default=10)

    # Adds an identifier argument to the desired number of iterations
    parser.add_argument(
        '-n_iter', help='Number of meta-heuristic iterations', type=int, default=10)

    return parser.parse_args()


if __name__ == '__main__':
    # Gathers the input arguments
    args = get_arguments()

    # Gathering variables from arguments
    dataset = args.dataset
    step = args.step
    fold = args.fold
    meta = args.mh

    # Loads the predictions and labels
    preds, y = l.load_candidates(dataset, step, fold)

    # Defining function to be optimized
    opt_fn = e.weighted_classifiers(preds, y)

    # Defining number of agents, number of variables, number of iterations, meta-heuristic and hyperparams
    n_agents = args.n_agents
    n_variables = preds.shape[1]
    n_iterations = args.n_iter
    meta_heuristic = mh.get_mh(meta).obj
    hyperparams = mh.get_mh(meta).hyperparams

    # Defining lower and upper bounds
    lb = [0] * n_variables
    ub = [1] * n_variables

    # Running the optimization task
    history = w.optimize(meta_heuristic, opt_fn, n_agents, n_variables,
                         n_iterations, lb, ub, hyperparams)

    # Saves the history object to an output file
    history.save(f'output/{meta}_{dataset}_{step}_{fold}.pkl')
