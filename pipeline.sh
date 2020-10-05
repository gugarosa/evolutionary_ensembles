# Datasets
DATASETS=("RSDataset" "RSSCN7" "UCMerced_LandUse")

# Number of runnings
FOLDS=(1 2 3 4 5 6)

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
            python ensemble_learning.py $DATA $FOLD $TYPE $M -n_agents 10 -n_iter 10
        done

        # Learns an ensemble with GP
        python ensemble_learning_with_gp.py $DATA $FOLD $TYPE -n_trees 10 -n_terminals 2 -n_iter 10 -min_depth 2 -max_depth 5
    done
done

# For every running
for RUN in $(seq 1 $N_RUNS); do
    # For every dataset
    for DATA in "${DATASETS[@]}"; do
        # Performs the classification procedure
        python classify_with_opf.py $DATA -tr_split $SPLIT -seed $RUN --normalize

        # Process the classification report
        python process_report.py $DATA $RUN
    done
done