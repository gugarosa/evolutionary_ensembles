from opytimizer.optimizers.pso import PSO

import models.ensemble as e
import utils.load as l
import utils.metrics as m
import utils.wrapper as w

# Defining dataset to be used
DATASET = 'RSDataset'

# Defining number of folds to be used
FOLD = 0

# Loads the validation step predictions and labels
val_pred, val_y = l.load_candidates(DATASET, 'validation', FOLD)

# Loads the testing step predictions and labels
# test_pred, test_y = l.load_candidates(DATASET, 'test', FOLD)

# Defining function to be optimized
opt_fn = e.weighted_classifier(val_pred, val_y)

# Defining number of agents, number of variables and number of iterations
n_agents = 100
n_variables = val_pred.shape[1]
n_iterations = 1000

# Defining lower and upper bounds
lb = [0] * n_variables
ub = [1] * n_variables

# Defining meta-heuristic hyperparameters
hyperparams = dict(w=0.7, c1=1.7, c2=1.7)

# Running the optimization task
history = w.optimize(PSO, opt_fn, n_agents, n_variables,
                     n_iterations, lb, ub, hyperparams)
