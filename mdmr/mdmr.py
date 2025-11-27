"""
MDMR: Maximum Diversity Minimum Redundancy
=============================================

This module implements the MDMR subset selection algorithm introduced in:

    "MDMR: Balancing Diversity and Redundancy for Annotation-Efficient Fine-Tuning of Pretrained Cell Segmentation Models."
    by Eiram Mahera Sheikh, Alaa Tharwat, Constanze Schwan, Wolfram Schenck, 2025.

MDMR selects a small, maximally diverse, and minimally redundant subset of samples from a pool of unlabeled data,
making it particularly suitable for active learning.

Core Idea
---------
- Constructs a similarity matrix via an RBF kernel.
- Selects the most "central" sample first, then iteratively selects
  samples that maximize diversity while minimizing redundancy.
- Includes tie-breaking with randomization for reproducibility.

Inputs
------
- features : np.ndarray of shape (N, D)
    Feature matrix for N samples, each with D-dimensional embeddings.
- budget : int
    Number of samples to select.
- seed : int, optional
    Seed for random tie-breaking.

Outputs
-------
- List[int]
    Indices of the selected subset.

"""


from __future__ import annotations
from typing import List, Optional, Sequence, Union
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel


class MDMR:
    """
    MDMR: Maximize Diversity Minimize Redundancy
    """

    def __init__(
        self,
        features: Union[np.ndarray, Sequence[float], Sequence[Sequence[float]]],
        gamma: Optional[float] = None,
        seed: Optional[Union[int, np.random.Generator]] = 42,
        dtype: str = "float64",
    ) -> None:
        """
        features : array-like
            Feature matrix of shape (N, d) or (N,). Accepts NumPy arrays or lists/tuples.
            1D inputs are reshaped to (N, 1).
        gamma : float or None, default=None
            RBF kernel width. If None, uses 1/d (d = number of feature dimensions).
        seed : int, np.random.Generator, or None, default=42
            Seed/Generator for reproducible tie-breaking.
        dtype : {"float32", "float64"}, default="float64"
            Numeric precision for the kernel matrix K (float32 saves memory).
        """

        # Normalize input to a 2D float64 array of shape (N, d)
        X = np.asarray(features, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2:
            raise ValueError(f"Features must be (N,) or (N,d); got shape {X.shape}")

        N, d = X.shape
        if N == 0:
            raise ValueError("Features must contain at least one sample.")
        if dtype not in ("float32", "float64"):
            raise ValueError("dtype must be 'float32' or 'float64'.")

        # Store basic attributes
        self.N, self.d = N, d
        self.gamma = (1.0 / d) if gamma is None else float(gamma)
        self.dtype = np.float32 if dtype == "float32" else np.float64

        # RNG for tiebreaking (uniform without replacement)
        if isinstance(seed, np.random.Generator):
            self.rng = seed
        else:
            self.rng = np.random.default_rng(seed)

        # Step 1) Build the RBF kernel matrix
        self.K = rbf_kernel(X, gamma=self.gamma)

        # Step 2) Find the most central sample c
        self.c = int(np.argmax(np.sum(self.K, axis=1)))

        # Step 3) Initialize the selected set S
        self.S: List[int] = [self.c]

        # bookkeeping
        self.in_S_mask = np.zeros(self.N, dtype=bool)
        self.in_S_mask[self.c] = True

    def select(self, budget: int = 2) -> List[int]:
        """
        :param budget: number of samples to query
        :return: indices of the samples selected by the MDMR algorithm
        """
        # Validate input
        if not isinstance(budget, int) or budget < 0:
            raise ValueError("budget must be a non-negative integer.")
        if budget == 0:
            return []
        if budget > self.N - 1:
            # Cannot return more than N-1 because c is excluded
            raise ValueError("budget exceeds available samples (excluding central).")

        # Snapshot how many elements were already in S before this call
        previously_selected = len(self.S)

        # Target size for S at the end of this query:
        # S currently has 'previously_selected' elements; we need to add 'budget' more.
        target_len = previously_selected + budget

        # Greedy loop: keep adding until |S| == target_len
        while len(self.S) < target_len:
            # Diversity for all i: D(i) = 1 - mean_{j in S} K[i, j].
            # NOTE: We recompute from scratch (no incremental sum) for clarity.
            redundancy = self.K[:, self.S].mean(axis=1)   # shape (N,)
            diversity = 1.0 - redundancy

            # Exclude indices already in S from being (re)selected.
            # They still contribute to the mean via K[:, S] above.
            diversity[self.in_S_mask] = -np.inf

            # Max diversity and its tie set T
            D_max = float(np.max(diversity))
            T = np.flatnonzero(diversity == D_max)

            if T.size == 0:
                # Degenerate case (e.g., everything equal or already selected).
                break

            # Remaining slots we still need to reach target_len
            r = target_len - len(self.S)

            # Simple, readable tie rule:
            # - If there are multiple ties AND they don't all fit, sample exactly r.
            # - Otherwise (single best OR all ties fit), take T as-is.
            if T.size > 1 and T.size > r:
                chosen = self.rng.choice(T, size=r, replace=False)
            else:
                chosen = T

            # Update S and the membership mask
            self.S.extend(chosen.tolist())
            self.in_S_mask[chosen] = True

        # Return only the indices added by this call (exclude central and prior picks)
        return self.S[previously_selected:]


# Minimal example
if __name__ == "__main__":
    X = np.random.default_rng(0).standard_normal((200, 64))

    selector = MDMR(X)         # builds K, finds central sample, S = [c]

    s1 = selector.select(5)     # select 5 indices
    print("s1:", s1)

    s2 = selector.select(3)     # select 3 more indices
    print("s2:", s2)

    s3 = selector.select(7)     # select 7 more indices
    print("s3:", s3)

