# Electron-Diffraction-CNN



A small ML Training and Evaluation Framework for reliable experimentation.


## Setup

This project targets a GPU-enabled Linux Workstation. Additional work may be required for testing on other operating systems or on CPUs. 

We target Python 3.6+. 

1. Create/Activate a virtual environment (via [anaconda](https://docs.conda.io/en/latest/miniconda.html), [virtualenv](https://virtualenv.pypa.io/en/latest/), or [pyenv](https://github.com/pyenv/pyenv)) Recommended: Anaconda 
2. `pip install -e .`


### Dev Install

Development installations are recommended if you'd like to contribute to this library.

`pip install -e .[dev]`

### Testing

`python setup.py test`

Thereafter, coverage reporting can be found at `docs/cov/`

## How to Run Experiments

### Collecting data into a manifest file

This project includes a `make_manifest.py` script that assumes the following:

- All of your EBSD data exists under a single directory (likely called `data`) on your filesystem.
- The folders that contain image data (`.tiff` files) also contain a `Grain List.txt` file with metadata about the images in the folder.
- The folder structure for the images, while arbitrary, will not change at the time experiments are run.

Before training starts, first create a manifest file to represent your dataset.

### Splitting data manifest into groups

We also include a `split_manifest.py` script whose purpose is to divide the EBSD image dataset into train, test, and validation groups. 

After creating a manifest file and before starting experimenting, please use this script to mutate the manifest file for repeatable experimentation.  

This script should add a column (likely called `_split`) that will be used in the training process.

### Train a Model

Define a model through subclassing either a Regression or Classification type in the `src/vecchio/models.py` file. 
Thereafter, train your model in a repeatble way using the `train.py` or `regression_train.py` scripts below. 
Consider saving the exact command that is run, because the specific arguments will be integral in the evaluation phase.
See documentation below for arguments and example uages.  

### Evaluate the model performance. 

For classification or regression, please make use of the `eval.py` and `regression_eval.py` scripts  (docs below).


## Docs

## `make_manifest.py --help`

```
usage: make_manifest.py [-h] [-o OUTPUT] source_directory

Produce an EBSD data manifest file from a root directory.

positional arguments:
  source_directory      Root directory to begin parsing data

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        /path/to/output.csv Default: `manifest.csv`
```

Example usage: 
```
python src/make_manifest.py data/
```

```
python src/make_manifest.py data/ -o my_manifest.csv
```

## `split_manifest.py --help`
```
usage: split_manifest.py [-h] [-ts TEST_SIZE] [-vs VAL_SIZE] [-s SEED]
                         [-sm {shuffle,stratified-shuffle}] [-o OUTPUT]
                         manifest label_column

Produce an EBSD data manifest file from a root directory.

positional arguments:
  manifest              path/to/manifest.csv
  label_column          Name of column considered the label or `y`.

optional arguments:
  -h, --help            show this help message and exit
  -ts TEST_SIZE, --test-size TEST_SIZE
                        Ratio of dataset to include in test set
  -vs VAL_SIZE, --validation-size VAL_SIZE
                        Ratio of dataset to include in validation set
  -s SEED, --seed SEED  Random seed
  -sm {shuffle,stratified-shuffle}, --split-method {shuffle,stratified-shuffle}
  -o OUTPUT, --output OUTPUT
                        (optional) /path/to/mutated/copy/of/manifest.csv

```

Example usage: 
```
python src/split_manifest.py manifest.csv 'Phase'
```

```
python src/split_manifest.py manifest.csv 'Phase' -s 42 -sm 'stratified-shuffle'
```

```
python src/split_manifest.py manifest.csv 'Phase' -o split_manifest.csv -vs 0.1 -sm 'stratified-shuffle'
```
## `train.py --help`
```
usage: train.py [-h] [--weight-classes] [-bs BATCH_SIZE] [-e EPOCHS]
                [-md MIN_DELTA] [-p PATIENCE] [-o OUTPUT]
                {XceptionClassifier,ResNet50Model} manifest label_column

Train a Classification ML Model

positional arguments:
  {XceptionClassifier,ResNet50Model}
                        The model to use for classification.
  manifest              path/to/manifest.csv
  label_column          Column name for label (y) from manifest file.

optional arguments:
  -h, --help            show this help message and exit
  --weight-classes      Flag that turns on class balancing (should not be used
                        with regression).
  -bs BATCH_SIZE, --batch-size BATCH_SIZE
                        Training batch size (default: 32)
  -e EPOCHS, --epochs EPOCHS
                        Number of epochs (default: 10000)
  -md MIN_DELTA, --min-delta MIN_DELTA
                        Minimum change int the monitored quantity to qualify
                        as an improvement. Default: 0.001
  -p PATIENCE, --patience PATIENCE
                        Number of epochs with no improvement after which
                        training will be stopped Default: 25
  -o OUTPUT, --output OUTPUT
                        (optional) path/to/output/directory/
```

Example Usage: 

```
python src/train.py XceptionClassifier manifest.csv Phase -e 1
```


## `regression_train.py --help`
```
usage: regression_train.py [-h] [-bs BATCH_SIZE] [-e EPOCHS] [-md MIN_DELTA]
                           [-p PATIENCE] [-o OUTPUT]
                           {MultiLabelLinearRegressor,XceptionRegressor}
                           manifest label_columns [label_columns ...]

Train a Regression ML Model

positional arguments:
  {MultiLabelLinearRegressor,XceptionRegressor}
                        The model to use for regression.
  manifest              path/to/manifest.csv
  label_columns         Column(s) name for label (y) from manifest file.

optional arguments:
  -h, --help            show this help message and exit
  -bs BATCH_SIZE, --batch-size BATCH_SIZE
                        Training batch size (default: 32)
  -e EPOCHS, --epochs EPOCHS
                        Number of epochs (default: 10000)
  -md MIN_DELTA, --min-delta MIN_DELTA
                        Minimum change int the monitored quantity to qualify
                        as an improvement. Default: 0.001
  -p PATIENCE, --patience PATIENCE
                        Number of epochs with no improvement after which
                        training will be stopped Default: 25
  -o OUTPUT, --output OUTPUT
                        (optional) path/to/output/directory/
```

Example usage: 

```
python src/regression_train.py MultiLabelLinearRegressor tests/test_regression.csv Lattice_a -e 1
```

```
python src/regression_train.py XceptionRegressor tests/test_regression.csv Lattice_a Lattice_b Lattice_c -bs 128
```

## `eval.py --help`

```
usage: eval.py [-h] [-o OUTPUT] [-bs BATCH_SIZE]
               trained_model manifest label_column

Evaluate a Classification ML Model

positional arguments:
  trained_model         /path/to/model_checkpoint.h5
  manifest              path/to/manifest.csv
  label_column          Column name for label (y) from manifest file.

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        (optional) path/to/output/directory/
  -bs BATCH_SIZE, --batch-size BATCH_SIZE
                        Training batch size (default: 32)

```

Example usage:

```
python src/eval.py models/XceptionClassifier-0.0.0/2019-04-07-06-37/model_checkpoint.h5 manifest.csv Phase -o class_eval.csv
```

## `regression_eval.py --help`
```
usage: regression_eval.py [-h] [-o OUTPUT] [-bs BATCH_SIZE]
                          trained_model manifest label_columns
                          [label_columns ...]

Evaluate a Regression ML Model

positional arguments:
  trained_model         /path/to/model_checkpoint.h5
  manifest              path/to/manifest.csv
  label_columns         Column(s) name for label (y) from manifest file.

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        (optional) path/to/output/directory/
  -bs BATCH_SIZE, --batch-size BATCH_SIZE
                        Training batch size (default: 32)
```

Example usage: 

```
python src/regression_eval.py models/XceptionRegressor-0.0.0/2019-04-07-06-37_small/model_checkpoint.h5 manifest.csv Lattice_a Lattice_b Lattice_c -o regress_eval.csv
```

# FAQ

Q: Error: `Failed to load the native TensorFlow runtime.` (I don't have a GPU / Haven't set up GPU Drivers)
A: Try installing a non-gpu version of tensorflow: `pip install tensorflow`
