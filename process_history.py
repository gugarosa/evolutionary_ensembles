import numpy as np
from opytimizer.utils.history import History

import models.ensemble as e
import utils.load as l

# Defining algorithm that was used
ALGORITHM = 'PSO'

# Defining dataset that was used
DATASET = 'RSDataset'

# Defining step that was used
STEP = 'validation'

# Defining number of folds that was used
FOLD = 0

# Defining an input file
input_file = f'output/{ALGORITHM}_{DATASET}_{STEP}_{FOLD}.pkl'

# Creating a History object
h = History()

# Loading the input file
h.load(input_file)

# Loading the predictions and labels
preds, y = l.load_candidates(DATASET, STEP, FOLD)

# Gathering the best weights
best_weights = np.asarray(h.best_agent[-1][0])

# Ensuring that the sum of weights is one and avoids division by zero
best_weights = best_weights / max(best_weights.sum(), 1e-10)

# Evaluating ensemble
acc = e.evaluate(best_weights, preds, y)

print(f'Ensemble accuracy: {acc}')
