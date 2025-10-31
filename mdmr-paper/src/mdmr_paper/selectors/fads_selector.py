"""
FADSSelector — Wrapper for the Diversity Subsampling method.

Implements the algorithm described in:
"Diversity Subsampling: Custom Subsamples from Large Data Sets"
 by Boyang Shang, Daniel W. Apley, Sanjay Mehrotra (2023),
 INFORMS Journal on Data Science 2(2):161-182.


This method selects a representative subset of samples by maximizing
diversity in feature space, making it suitable for scenarios where
annotation budgets are very limited.

Usage
-----
- Construct with a feature matrix (NumPy array or compatible sequence).
- Call `.select(budget)` to obtain a list of indices of the selected subset.

Example
-------
>>> selector = FADSSelector(features, seed=42)
>>> picks = selector.select(budget=5)

Notes
-----
- This class is a thin wrapper around the `FADS` package implementation.
- `budget` must be ≤ the number of available samples.
- Warnings from the underlying package are suppressed by default.
"""


import numpy as np
from typing import List, Optional, Sequence, Union
import FADS
import warnings

class FADSSelector:
    def __init__(
            self,
            features: Union[np.ndarray, Sequence[float], Sequence[Sequence[float]]],
            *,
            seed: Optional[Union[int, np.random.Generator]] = 42,
    ):
        self.features = features
        self.seed = seed

    def select(self, budget: int = 2) -> List[int]:
        # perform diversity subsampling
        fastds = FADS.FADS(self.features)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            selected_indices = fastds.DS(budget)

        return selected_indices


