import numpy as np
from opytimizer import Opytimizer
from opytimizer.core.function import Function
from opytimizer.core.optimizer import Optimizer
from opytimizer.optimizers.boolean.umda import UMDA
from opytimizer.optimizers.evolutionary.gp import GP
from opytimizer.spaces.boolean import BooleanSpace
from opytimizer.spaces.search import SearchSpace
from opytimizer.spaces.tree import TreeSpace


def optimize(opt, target, n_agents, n_variables, n_iterations, lb, ub, hyperparams):
    """Abstracts all Opytimizer's mechanisms into a single method.

    Args:
        opt (Optimizer): An Optimizer-child class.
        target (callable): The method to be optimized.
        n_agents (int): Number of agents.
        n_variables (int): Number of variables.
        n_iterations (int): Number of iterations.
        lb (list): List of lower bounds.
        ub (list): List of upper bounds.
        hyperparams (dict): Dictionary of hyperparameters.

    Returns:
        A History object containing all optimization's information.

    """

    # Creating the SearchSpace
    space = SearchSpace(n_agents=n_agents, n_variables=n_variables,
                        n_iterations=n_iterations, lower_bound=lb, upper_bound=ub)

    # Creating the Function
    function = Function(pointer=target)

    # Creating Optimizer
    if opt.__name__ is not 'BH':
        optimizer = opt(hyperparams=hyperparams)
    else:
        optimizer = opt()

    # Creating the optimization task
    task = Opytimizer(space=space, optimizer=optimizer, function=function)

    return task.start(store_best_only=True)


def optimize_gp(target, n_trees, n_terminals, n_variables, n_iterations, min_depth, max_depth,
                functions, lb, ub, hyperparams):
    """Abstracts Opytimizer's Genetic Programming into a single method.

    Args:
        target (callable): The method to be optimized.
        n_trees (int): Number of agents.
        n_terminals (int): Number of terminals
        n_variables (int): Number of variables.
        n_iterations (int): Number of iterations.
        min_depth (int): Minimum depth of trees.
        max_depth (int): Maximum depth of trees.
        functions (list): Functions' nodes.
        lb (list): List of lower bounds.
        ub (list): List of upper bounds.
        hyperparams (dict): Dictionary of hyperparameters.

    Returns:
        A History object containing all optimization's information.

    """

    # Creating the TreeSpace
    space = TreeSpace(n_trees=n_trees, n_terminals=n_terminals, n_variables=n_variables,
                      n_iterations=n_iterations, min_depth=min_depth, max_depth=max_depth,
                      functions=functions, lower_bound=lb, upper_bound=ub)

    # Creating the Function
    function = Function(pointer=target)

    # Creating GP's optimizer
    optimizer = GP(hyperparams=hyperparams)

    # Creating the optimization task
    task = Opytimizer(space=space, optimizer=optimizer, function=function)

    return task.start(store_best_only=True)


def optimize_umda(target, n_agents, n_variables, n_iterations, hyperparams):
    """Abstracts Opytimizer's Univariate Marginal Distribution Algorithm into a single method.

    Args:
        target (callable): The method to be optimized.
        n_agents (int): Number of agents.
        n_variables (int): Number of variables.
        n_iterations (int): Number of iterations.
        hyperparams (dict): Dictionary of hyperparameters.

    Returns:
        A History object containing all optimization's information.

    """

    # Creating the BooleanSpace
    space = BooleanSpace(n_agents=n_agents, n_iterations=n_iterations, n_variables=n_variables)

    # Creating the Function
    function = Function(pointer=target)

    # Creating UMDA's optimizer
    optimizer = UMDA(hyperparams=hyperparams)

    # Creating the optimization task
    task = Opytimizer(space=space, optimizer=optimizer, function=function)

    return task.start(store_best_only=True)
