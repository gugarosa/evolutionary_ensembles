import models.ensemble as e
import utils.load as l
import utils.metrics as m
import utils.wrapper as w

# Defining dataset to be used
DATASET = 'RSDataset'

# Defining step to be used
STEP = 'validation'

# Defining number of folds to be used
FOLD = 0

# Loads the predictions and labels
preds, y = l.load_candidates(DATASET, STEP, FOLD)

# Defining function to be optimized
opt_fn = e.weighted_classifier(preds, y)

# Defining number of trees, number of terminals, number of variables and number of iterations
n_trees = 10
n_terminals = 2
n_variables = preds.shape[1]
n_iterations = 100

# Defining minimum and maximum depth of trees
min_depth = 2
max_depth = 5

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
history.save(f'output/GP_{DATASET}_{STEP}_{FOLD}.pkl')
