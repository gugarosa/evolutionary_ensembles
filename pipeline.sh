#!/bin/bash

# Datasets
DATASETS=("RSDataset" "RSSCN7" "UCMerced_LandUse")

DESCRIPTORS=("all" "cnn" "global") #'global', 'cnn', 'all'

# Number of runnings
FOLDS=(1 2 3 4 5)

# Type of ensemble
TYPE="boolean"

# Meta-heuristics
MH=("abc" "ba" "bh" "cs" "fa" "fpa" "pso")

# For every DESCRIPTORS
for DESC in "${DESCRIPTORS[@]}"; do
    echo "#### DESCRIPTORS: "$DESC" ####" >> "./output/1_accuracies.txt";
    echo "#### DESCRIPTORS: "$DESC" ####" >> "./output/2_time.txt"; 
    echo "#### DESCRIPTORS: "$DESC" ####" >> "./output/3_ensemble_sizes.txt";
# For every dataset
for DATA in "${DATASETS[@]}"; do
    echo "#### DATASET: "$DATA" ####" >> "./output/1_accuracies.txt";
    echo "#### DATASET: "$DATA" ####" >> "./output/2_time.txt"; 
    echo "#### DATASET: "$DATA" ####" >> "./output/3_ensemble_sizes.txt"; 

    # For every meta-heuristic
    for M in "${MH[@]}"; do
        echo "<- "$M" ->" >> "./output/1_accuracies.txt";
        echo "<- "$M" ->" >> "./output/2_time.txt";
        echo "<- "$M" ->" >> "./output/3_ensemble_sizes.txt"; 
    
        # For every fold
        for FOLD in "${FOLDS[@]}"; do

            # Learns an ensemble with meta-heuristics
            #python3 ensemble_learning.py $DATA $FOLD $TYPE $M -n_agents 10 -n_iter 10
            python3 ensemble_learning.py $DATA $DESC $FOLD $TYPE $M -n_agents 10 -n_iter 10
            # Processes the optimization history
            python3 process_optimization_history.py $DATA $DESC $FOLD $TYPE $M
            head -n 1 "./output/"$M"_"$TYPE"_"$DATA"_test_"$FOLD".txt"  >> "./output/1_accuracies.txt";              
            head -n 1 "./output/"$M"_"$TYPE"_"$DATA"_test_"$FOLD".time"  >> "./output/2_time.txt"; echo "" >> "./output/2_time.txt";
            tail -n 1 "./output/"$M"_"$TYPE"_"$DATA"_test_"$FOLD".txt"  >> "./output/3_ensemble_sizes.txt"; echo "" >> "./output/3_ensemble_sizes.txt";
            echo
        done
    done

    M="gp"
    echo "<- "$M" ->" >> "./output/1_accuracies.txt";
    echo "<- "$M" ->" >> "./output/2_time.txt";  
    echo "<- "$M" ->" >> "./output/3_ensemble_sizes.txt"; 

    # For every fold
    for FOLD in "${FOLDS[@]}"; do

        # Learns an ensemble with GP
        python3 ensemble_learning_with_gp.py $DATA $DESC $FOLD $TYPE -n_trees 10 -n_terminals 2 -n_iter 10 -min_depth 2 -max_depth 5

        # Processes GP optimization history
        python3 process_optimization_history.py $DATA $DESC $FOLD $TYPE gp
        head -n 1 "./output/"$M"_"$TYPE"_"$DATA"_test_"$FOLD".txt"  >> "./output/1_accuracies.txt";              
        head -n 1 "./output/"$M"_"$TYPE"_"$DATA"_test_"$FOLD".time"  >> "./output/2_time.txt"; echo "" >> "./output/2_time.txt";
        tail -n 1 "./output/"$M"_"$TYPE"_"$DATA"_test_"$FOLD".txt"  >> "./output/3_ensemble_sizes.txt"; echo "" >> "./output/3_ensemble_sizes.txt";
    done

    M="umda"
    echo "<- "$M" ->" >> "./output/1_accuracies.txt";
    echo "<- "$M" ->" >> "./output/2_time.txt";  
    echo "<- "$M" ->" >> "./output/3_ensemble_sizes.txt"; 

    # For every fold
    for FOLD in "${FOLDS[@]}"; do


        # Learns an ensemble with UMDA
        python3 ensemble_learning_with_umda.py $DATA $DESC $FOLD -n_agents 10 -n_iter 10

        # Processes UMDA optimization history
        python3 process_optimization_history.py $DATA $DESC $FOLD $TYPE umda
        head -n 1 "./output/"$M"_"$TYPE"_"$DATA"_test_"$FOLD".txt"  >> "./output/1_accuracies.txt";              
        head -n 1 "./output/"$M"_"$TYPE"_"$DATA"_test_"$FOLD".time"  >> "./output/2_time.txt"; echo "" >> "./output/2_time.txt";
        tail -n 1 "./output/"$M"_"$TYPE"_"$DATA"_test_"$FOLD".txt"  >> "./output/3_ensemble_sizes.txt";  echo "" >> "./output/3_ensemble_sizes.txt";
    done
done
done # DESCRIPTORS
