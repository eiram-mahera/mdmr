# MDMR-Paper

This folder contains the full experimental framework used in our paper.  
It includes dataset loaders, feature extraction, selection methods, training routines, evaluation metrics, and scripts for reproducing results.

The standalone MRMD algorithm (`mdmr/`) is available at the project root.  
This package (`mdmr_paper`) builds on top of it to provide the full subset selection and fine-tuning pipeline.

---

## Repository Layout

```
mrmd/                                              # Repository root
├── mdmr/
│   └── mrmd.py                                    # Core MDMR algorithm
│
├── paper/                                         # Experimental framework
│   ├── cache/                                     # Cached features (e.g. style vectors, npy files)
│   ├── configs/                                   # Configuration files
│   │   └── datasets/ctc/                          # Dataset-specific YAML configs
│   ├── logs/                                      # Runtime logs
│   ├── models/                                    # Trained model checkpoints
│   ├── results/                                   # Experiment results (CSV files)
│   ├── src/                                       # All source code
│   │   └── mdmr_paper/                            # Main folder containing all the source code for experiments
│   │       ├── datasets/                          # Dataset loaders
│   │       ├── evaluation/                        # Evaluation metrics
│   │       ├── features/                          # Feature extraction
│   │       ├── scripts/                           # CLI entrypoints
│   │       │   ├── fine_tune.py                   # Subset selection and fine-tuning of the model
│   │       │   ├── eval_pretrained.py             # Evaluate the performance of the pre-trained model
│   │       │   └── train_fully_supervised.py      # Training of the model in a fully supervised manner
│   │       ├── selection/                         # Subset selection methods
│   │       ├── training/                          # Training utilities
│   │       ├── utils/                             # General utilities
│   ├── tools/SEGMeasure                           # Official CTC SEG metric tool
│   └── README.md                                  # Paper experiments documentation
├── pyproject.toml                                 # Project metadata and dependencies
├── README.md                                      # MDMR algorithm documentation
└── LICENSE                                        # License file
```

---

## Installation

### 0. Prerequisites
- Python ≥ 3.8
- Tested on Ubuntu 22.04
- (Optional) NVIDIA driver if you want GPU acceleration
- Clone the repository
```
git clone https://github.com/eiram-mahera/mdmr.git
cd mdmr/mdmr-paper
```
---

### 1. [Optional] Create & activate environment
##### Option A) Conda environment
```
conda create -n mdmr
conda activate mdmr
```

#### Option B) pip + venv
```
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install PyTorch
Follow instructions from the official [PyTorch](https://pytorch.org/get-started/locally/) website.


### 2. Install dependencies
```
# Regular install
pip install .[paper]

# Or editable/dev install
pip install -e .[paper]
```

### 3. Optional Dependencies

TypiClust requires `faiss`. Use conda for installation:

```
conda install -c pytorch faiss-gpu     # if you have a GPU
# or
conda install -c pytorch faiss-cpu     # for CPU-only
```

---

## Dataset Setup

1. Download datasets from the [Cell Tracking Challenge](https://celltrackingchallenge.net/2d-datasets/).
2. Place them in a directory, for example:

   ```
   /path/to/CTC/
   └── DIC-C2DH-HeLa/
       ├── 01/          # training images
       ├── 01_GT/SEG/   # training masks
       ├── 02/          # test images
       └── 02_GT/SEG/   # test masks
   ```
3. Dataset-specific configurations are defined in the YAML configs under `mdmr-paper/configs/datasets/ctc/`.

---

## Usage
All scripts are located in `src/mdmr_paper/scripts/`.

### 1. Subset Selection & Fine-tuning
Run the main pipeline (`fine_tune.py`):

```
python -m mdmr_paper.scripts.fine_tune \
  --dataset-config configs/datasets/ctc/DIC-C2DH-HeLa.yaml \
  --model-type cyto2 \
  --selector ALL \
  --budget 2 \
  --train-mode fixed --epochs 100 \
  --features-dir ./cache \
  --results-csv results/results_fine_tune.csv
```

Training Modes:

- Fixed epochs (`--train-mode fixed --epochs-fixed 100`)
- K-fold CV (`--train-mode cv --cv-start 10 --cv-max 200 --cv-step 10 --cv-kfolds 5`)

Key options:
- `--selector`: comma-separated list of subset selection methods. `ALL` will run all the methods.
- `--budget`: query budget (number of training samples to select).
- `--train-mode`: fixed (train for fixed epochs) or cv (cross-validation to pick epochs).
- `--features-dir`: cache directory for style vectors.
- `--runs`: numer of times to run the experiment.
- `--seed`: base seed value for the experiments.
- `--cuda`: GPU to use.
- `--overwrite_features`: recompute and overwrite cached features.
- `--ctc_app`: path to the official CTC SEGMeasure binary.
- `--results-csv`: file to append results.

### 2. Evaluate Pre-trained Cellpose Model

```
python -m mdmr_paper.scripts.eval_pretrained \
  --dataset-config configs/datasets/ctc/DIC-C2DH-HeLa.yaml \
  --model-type cyto2 \
  --results-csv results/results_pretrained.csv
```

### 3. Fully Supervised Training

```
python -m mdmr_paper.scripts.train_fully_supervised \
  --dataset-config configs/datasets/ctc/DIC-C2DH-HeLa.yaml \
  --model-type cyto2 \
  --train-mode fixed --epochs 200 \
  --results-csv results/results_fully.csv
```

Training Modes:

- Fixed epochs (`--train-mode fixed --epochs N`)
- Early stopping (`--train-mode early --max-epochs 500 --start-epochs 10 --patience 20`)
- K-fold CV (`--train-mode cv --cv-start 10 --cv-max 200 --cv-step 10 --cv-kfolds 5`)


---

## Metrics

* **SEG**: the official CTC segmentation metric (using SEGMeasure).
* If the SEGMeasure binary fails execution, a Python fallback implementation is used.

---

## Results
Results are appended to csv file with header:

`Dataset, Model, Train, Test, Run, Seed, Epochs, Budget, Indices, Selector, SEG`

---

## Extending the Framework

* **Datasets**: add YAML configs under `configs/datasets/` and extend `src/mdmr_paper/datasets/` if custom preprocessing is required.
* **Selection methods**: add a script under `src/mdmr_paper/selectors/`.
* **Metrics**: define new metrics under `src/mdmr_paper/evaluation/`.
* **Training**: extend `src/mdmr_paper/training/` for new training regimes.

---
