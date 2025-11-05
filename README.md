# MDMR: Maximum Diversity Minimum Redundancy

This repository contains the official implementation of the MDMR algorithm, as proposed in the paper [MDMR: Balancing Diversity and Redundancy for Annotation-Efficient Fine-Tuning of Pretrained Cell Segmentation Models](https://www.biorxiv.org/content/10.1101/2025.11.04.686267v1).  

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
@article {Sheikh2025.11.04.686267,
	author = {Sheikh, Eiram Mahera and Tharwat, Alaa and Schwan, Constanze and Schenck, Wolfram},
	title = {MDMR: Balancing Diversity and Redundancy for Annotation-Efficient Fine-Tuning of Pretrained Cell Segmentation Models},
	elocation-id = {2025.11.04.686267},
	year = {2025},
	doi = {10.1101/2025.11.04.686267},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2025/11/05/2025.11.04.686267},
	eprint = {https://www.biorxiv.org/content/early/2025/11/05/2025.11.04.686267.full.pdf},
	journal = {bioRxiv}
}

```

---

## License
This project is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.

---
