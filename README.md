# Creating Classifier Ensembles through Meta-heuristic Algorithms for Aerial Scene Classification

*This repository holds all the necessary code to run the very-same experiments described in the paper "Creating Classifier Ensembles through Meta-heuristic Algorithms for Aerial Scene Classification".*

---

## References

If you use our work to fulfill any of your needs, please cite us:

```
```

---

## Structure

  * `data/`
    * `RSDataset`: Folder containing the RSDataset data;
    * `RSSCN7`: Folder containing the RSSCN7 data;
    * `UCMerced_LandUse`: Folder containing the UCMerced_LandUse data;
  * `models/`
    * `ensemble.py`: Ensemble-based methods, such as weight-based and majority voting;
  * `utils/`
    * `constants.py`: Constants definitions;
    * `dictionary.py`: Creates a dictionary of classes and labels;
    * `load.py`: Loads the dataset according to desired format;
    * `metrics.py`: Provides several metrics calculations;
    * `mh.py`: Wraps the meta-heuristic classes;
    * `wrapper.py`: Wraps the optimization tasks;

---

## How-to-Use

There are 4+1 simple steps in order to accomplish the same experiments described in the paper:

 * Install the requirements;
 * Perform the majority voting;
 * Optimize weight-based or boolean-based ensembles;
 * Process post-optimization information for further comparison;
 * (Optional) Calculate diversity metrics between classifiers.
 
### Installation

Please install all the pre-needed requirements using:

```pip install -r requirements.txt```

### Majority Voting

Our first ensemble-based baseline is to use the Majority Voting in order to create a count-based ensemble. With that in mind, just run the following script with the input arguments:

```python majority_voting.py -h```

*Note that it will output a helper file in order to assist in choosing the correct arguments for the script.*

### Optimizing Weight-Based or Boolean-Based Ensembles

After defining the Majority Voting baselines, now we can proceed and try to find the most suitable weights for the ensemble (one can use a weight-based or a boolean-based approach) through a meta-heuristic optimization process. Just choose the following scripts and invoke their helper:

```python ensemble_learning.py -h```

```python ensemble_learning_with_gp.py -h```

*Note that Genetic Programming-based optimization is included in a different script due to its particular structure defined in the Opytimizer library.*

### Post-Optimization Processing

Finally, after concluding the optimization step over the validation sets, it is now possible to load back the best weights found during the optimization procedure and apply them into a weight-based ensemble over the testing set. Run the following script in order to fulfill that purpose:

```python process_optimization_history.py -h```

*Note that the optimization process will always output a `.pkl` file, while the other scripts will output a `*.txt` file.*

### (Optional) Diversity Metrics

As an optional procedure, one can also calculate some diversity metrics between classifiers. Please use the following script in order to accomplish such an approach:

```python diversity_metrics.py -h```

*Note that this script will also calculate both classifier accuracies over the desired dataset and fold.*

---
