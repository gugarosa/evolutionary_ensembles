import argparse

import utils.load as l
import utils.metrics as m


def get_arguments():
    """Gets arguments from the command line.

    Returns:
        A parser with the input arguments.

    """

    # Creates the ArgumentParser
    parser = argparse.ArgumentParser(
        usage='Creates an ensemble of classifiers based on majority voting.')

    # Adds a dataset argument with pre-defined choices
    parser.add_argument('dataset', help='Dataset identifier', choices=[
                        'RSDataset', 'RSSCN7', 'UCMerced_LandUse'])

    # Adds a step argument with pre-defined choices
    parser.add_argument(
        'step', help='Whether it should load from validation or test', choices=['val', 'test'])

    # Adds an identifier argument to the desired fold identifier
    parser.add_argument('fold', help='Fold identifier',
                        type=int, choices=range(1, 6))

    # Adds an identifier argument to classifier `i`
    parser.add_argument('i', help='Classifier `i` identifier',
                        type=int, choices=range(0, 70))

    # Adds an idenfier argument to classifier `j`
    parser.add_argument('j', help='Classifier `j` identifier',
                        type=int, choices=range(0, 70))

    return parser.parse_args()


if __name__ == '__main__':

    # Gathers the input arguments
    args = get_arguments()

    # Gathering variables from arguments
    dataset = args.dataset
    step = args.step
    fold = args.fold
    classifier_i = args.i
    classifier_j = args.j

    print(f'\nCalculating diversity metrics over {dataset} ...')
    print(
        f'Step: {step} | Fold: {fold} | Classifiers: ({classifier_i}, {classifier_j})\n')

    # Loads the predictions and labels
    preds, y = l.load_candidates(dataset, step, fold)

    # Calculating correlation between classifier `i` and classifier `j`
    corr = m.correlation(preds[:, classifier_i],
                         preds[:, classifier_j], y)

    # Calculating disagreement measure between classifier `i` and classifier `j`
    dm = m.disagreement_measure(
        preds[:, classifier_i], preds[:, classifier_j], y)

    # Calculating double-fault measure between classifier `i` and classifier `j`
    dfm = m.double_fault_measure(
        preds[:, classifier_i], preds[:, classifier_j], y)

    # Calculating interrater agreement between classifier `i` and classifier `j`
    ia = m.interrater_agreement(
        preds[:, classifier_i], preds[:, classifier_j], y)

    # Calculating q-statistics between classifier `i` and classifier `j`
    q_stat = m.q_statistics(
        preds[:, classifier_i], preds[:, classifier_j], y)

    print(f'Correlation: {corr}.')
    print(f'Disagreement Measure: {dm}.')
    print(f'Double-Fault Measure: {dfm}.')
    print(f'Interrater Agreement: {ia}.')
    print(f'Q-Statistics: {q_stat}.')

    print(f'\nSaving outputs ...')

    # Saving outputs
    with open(f'output/metrics_{dataset}_{step}_{fold}_{classifier_i}_{classifier_j}.txt', 'w') as f:
        f.write(f'COR {corr}\nDM {dm}\nDFM {dfm}\nIA {ia}\nQSTAT {q_stat}')

    print('Outputs saved.')
