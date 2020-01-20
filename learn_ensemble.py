import models.ensemble as e
import utils.metrics as m

# Defining dataset to be used
DATASET = 'RSDataset'

# Defining number of folds to be used
N_FOLDS = 1

# For each possible fold
for k in range(N_FOLDS):
    # Loads the validation step predictions and labels
    val_pred, val_y = e.load_candidates(DATASET, 'validation', k)

    # Loads the testing step predictions and labels
    test_pred, test_y = e.load_candidates(DATASET, 'test', k)

    # Gather the majority votes between predictions
    val_votes = e.majority_voting(val_pred)