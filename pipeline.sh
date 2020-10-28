#!/bin/bash

# Datasets
DATASETS=("RSDataset" "RSSCN7" "UCMerced_LandUse")

# Descriptor
DESCRIPTOR="all"

# Number of runnings
FOLDS=(1 2 3 4 5)

# Type of ensemble
TYPE="boolean"

# Meta-heuristics
MH=("abc" "ba" "bh" "cs" "fa" "fpa" "pso")

# For every dataset
for DATA in "${DATASETS[@]}"; do
    # For every fold
    for FOLD in "${FOLDS[@]}"; do
        # For every meta-heuristic
        for M in "${MH[@]}"; do
            # Learns an ensemble with meta-heuristics
            python3 ensemble_learning.py $DATA $DESCRIPTOR $FOLD $TYPE $M -n_agents 10 -n_iter 10

            # Processes the optimization history
            python3 process_optimization_history.py $DATA $FOLD $TYPE $M
        done

        # Learns an ensemble with GP
        python3 ensemble_learning_with_gp.py $DATA $DESCRIPTOR $FOLD $TYPE -n_trees 10 -n_terminals 2 -n_iter 10 -min_depth 2 -max_depth 5

        # Processes GP optimization history
        python3 process_optimization_history.py $DATA $FOLD $TYPE gp

        # Learns an ensemble with UMDA
        python3 ensemble_learning_with_umda.py $DATA $DESCRIPTOR $FOLD -n_agents 10 -n_iter 10

        # Processes UMDA optimization history
        python3 process_optimization_history.py $DATA $FOLD $TYPE umda
    done
done