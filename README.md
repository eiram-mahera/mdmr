# MDMR: Maximize Diversity Minimize Redundancy

This repository is the official implementation of the MDMR algorithm proposed in the paper.  

- The **`mdmr/`** folder provides a minimal standalone implementation of MDMR.  
- The **`paper/`** folder contains all code, scripts, configs, and instructions to reproduce the experiments and results from our paper.  
  Please see the [paper/README.md](mdmr-paper/README.md) for detailed instructions.

---

## Installation
```
git clone https://github.com/eiram-mahera/mdmr.git
cd mdmr
pip install .
```

This only installs the minimal dependencies required for the MDMR algorithm itself.

Dependencies for reproducing paper experiments are described in [paper/README.md](mdmr-paper/README.md).

---

## Usage
```
import numpy as np
from mdmr.mdmr import MDMR

# Example feature matrix with 100 samples, each of 256 dimensions
X = np.random.randn(100, 256)

# Create selector object
selector = MDMR(X)

# Select 5 samples
indices = selector.select(budget=5)

print("Selected indices:", indices)
```
---

## Reproducing Paper Results

For instructions on dataset setup, feature extraction, training, and evaluation,
please see [paper/README.md](mdmr-paper/README.md).

---

## Citation

If you use the MDMR algorithm in your work, please cite:

```bibtex
@article{she2025mdmr,
  title   = {MDMR: BALANCING DIVERSITY AND REDUNDANCY FOR ANNOTATION-EFFICIENT FINE-TUNING OF PRETRAINED CELL SEGMENTATION MODELS},
  author  = {Eiram Mahera Sheikh, Alaa Tharwat, Constanze Schwan, Wolfram Schenck},
  journal = {name},
  year    = {year}
}
```

---

## License
This project is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.

---
