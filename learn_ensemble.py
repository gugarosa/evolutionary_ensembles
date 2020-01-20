import utils.dictionary as d
import utils.load as l

# Defining dataset to be used
DATASET = 'RSDataset'

# Defining step to be used
STEP = 'validation'

# Defining number of folds to be used
N_FOLDS = 5

# For each possible fold
for k in range(N_FOLDS):
    # Loads the ground truth labels from desired dataset, step and fold
    labels = l.load_labels(DATASET, STEP, k)

    # Loads the predictions from desired dataset, step and fold
    preds = l.load_predictions(DATASET, STEP, k)
