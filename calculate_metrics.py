import utils.load as l
import utils.metrics as m

# Defining dataset to be used
DATASET = 'RSDataset'

# Defining fold identifier to be used
FOLD = 0

CLASSIFIER_I = 1
CLASSIFIER_J = 44

# Loads the validation step predictions and labels
val_pred, val_y = l.load_candidates(DATASET, 'validation', FOLD)

# Loads the testing step predictions and labels
test_pred, test_y = l.load_candidates(DATASET, 'test', FOLD)

# Calculating correlation between classifier #0 and classifier #1 (validation)
corr = m.correlation(val_pred[:, CLASSIFIER_I],
                     val_pred[:, CLASSIFIER_J], val_y)

# Calculating disagreement measure between classifier #0 and classifier #1 (validation)
dm = m.disagreement_measure(
    val_pred[:, CLASSIFIER_I], val_pred[:, CLASSIFIER_J], val_y)

# Calculating double-fault measure between classifier #0 and classifier #1 (validation)
dfm = m.double_fault_measure(
    val_pred[:, CLASSIFIER_I], val_pred[:, CLASSIFIER_J], val_y)

# Calculating interrater agreement between classifier #0 and classifier #1 (validation)
ia = m.interrater_agreement(
    val_pred[:, CLASSIFIER_I], val_pred[:, CLASSIFIER_J], val_y)

# Calculating q-statistics between classifier #0 and classifier #1 (validation)
q_stat = m.q_statistics(
    val_pred[:, CLASSIFIER_I], val_pred[:, CLASSIFIER_J], val_y)

print(
    f'Correlation between classifiers {CLASSIFIER_I} and {CLASSIFIER_J}: {corr}.')
print(
    f'Disagreement Measure between classifiers {CLASSIFIER_I} and {CLASSIFIER_J}: {dm}.')
print(
    f'Double-Fault Measure between classifiers {CLASSIFIER_I} and {CLASSIFIER_J}: {dfm}.')
print(
    f'Interrater Agreement between classifiers {CLASSIFIER_I} and {CLASSIFIER_J}: {ia}.')
print(
    f'Q-Statistics between classifiers {CLASSIFIER_I} and {CLASSIFIER_J}: {q_stat}.')
