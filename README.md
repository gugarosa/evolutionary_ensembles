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

## Package Guidelines

### Installation

You may just install the pre-needed requirements under your most preferred Python 3+ environment (raw, conda, virtualenv, whatever):

```Python
pip3 install -r requirements.txt
```

---

## Usage

### Majority Voting

Our first ensemble-based baseline is to use the Majority Voting in order to create a count-based ensemble. With that in mind, just run the following script with the input arguments:

```Python
python3 majority_voting.py -h
```

*Note that it will output a helper file in order to assist in choosing the correct arguments for the script.*

### Optimizing Weight-Based or Boolean-Based Ensembles

After defining the Majority Voting baselines, now we can proceed and try to find the most suitable weights for the ensemble (one can use a weight-based or a boolean-based approach) through a meta-heuristic optimization process. Just choose the following scripts and invoke their helper:

```Python
python3 ensemble_learning.py -h
```

and

```Python
python3 ensemble_learning_with_gp.py -h
```

and

```Python
python3 ensemble_learning_with_umda.py -h
```

*Note that Genetic Programming- and Univariate Marginal Distribution Algorithm-based optimization are included in different scripts due to their particular structure defined in the Opytimizer library.*

### Post-Optimization Processing

Finally, after concluding the optimization step over the validation sets, it is now possible to load back the best weights found during the optimization procedure and apply them into a weight-based ensemble over the testing set. Run the following script in order to fulfill that purpose:

```Python
python3 process_optimization_history.py -h
```

*Note that the optimization process will always output a `.pkl` file, while the other scripts will output a `*.txt` file.*

### Bash Script

Instead of invoking every script to conduct the experiments, it is also possible to use the provided shell script, as follows:

```Bash
./pipeline.sh
```

Such a script will conduct every step needed to accomplish the experimentation used throughout this paper. Furthermore, one can change any input argument that is defined in the script.

### (Optional) Diversity Metrics

As an optional procedure, one can also calculate some diversity metrics between classifiers. Please use the following script in order to accomplish such an approach:

```Python
python3 diversity_metrics.py -h
```

*Note that this script will also calculate both classifier accuracies over the desired dataset and fold.*

---

## Support

We know that we do our best, but it is inevitable to acknowledge that we make mistakes. If you ever need to report a bug, report a problem, talk to us, please do so! We will be available at our bests at this repository or gustavo.rosa@unesp.br, aalvin10@gmail.com and ffaria@unifesp.br.

---
