import models.ensemble as e
import utils.load as l
import utils.metrics as m

# Defining dataset to be used
DATASET = 'RSDataset'

# Defining number of folds to be used
N_FOLDS = 5

# For each possible fold
for k in range(N_FOLDS):
    print(f'Running fold {k+1}/{N_FOLDS} ...')

    # Loads the validation step predictions and labels
    val_pred, val_y = l.load_candidates(DATASET, 'validation', k)

    # Loads the testing step predictions and labels
    test_pred, test_y = l.load_candidates(DATASET, 'test', k)

    # Gather the majority votes between validation predictions
    val_votes = e.majority_voting(val_pred)

    # Gather the majority votes between test predictions
    test_votes = e.majority_voting(test_pred)

    # Calculates the validation accuracy
    val_acc = m.accuracy(val_votes, val_y)

    # Calculates the testing accuracy
    test_acc = m.accuracy(test_votes, test_y)

    print(f'Validation Accuracy: {val_acc} | Testing Accuracy: {test_acc}.\n')
