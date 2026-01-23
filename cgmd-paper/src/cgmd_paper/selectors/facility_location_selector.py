import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from apricot import FacilityLocationSelection
from typing import List, Optional, Sequence, Union


class FacilityLocationSelector:
    """
    Facility Location subset selection using submodlib on a precomputed RBF similarity.

    f(S) = sum_i max_{j in S} K[i, j]

    Args:
        style_vectors: (N, D) array of Cellpose style vectors.
        budget: number of samples to select.
        gamma: RBF kernel width (default 1/D).
        optimizer: 'NaiveGreedy' or 'LazyGreedy' (faster).
        random_state: optional seed for any internal randomness.

    Returns:
        List of selected indices (length == budget).
    """

    def __init__(
            self,
            features: Union[np.ndarray, Sequence[float], Sequence[Sequence[float]]],
            *,
            seed: Optional[Union[int, np.random.Generator]] = 42,
            gamma: Optional[float] = None,
            optimizer: Optional[str] = "LazyGreedy",
    ):
        self.features = features
        self.seed = seed
        self.gamma = gamma
        self.optimizer = optimizer

    def select(self, budget: int = 2) -> List[int]:

        # Input validation
        if not isinstance(self.features, np.ndarray) or self.features.ndim != 2:
            raise ValueError("style_vectors must be a 2D numpy array of shape (N, D).")

        N, D = self.features.shape
        if N == 0:
            raise ValueError("style_vectors must contain at least one sample.")
        if not isinstance(budget, int) or budget < 0:
            raise ValueError("query_budget must be a non-negative integer.")
        if budget > N - 1:
            raise ValueError("query_budget exceeds number of available samples (excluding central sample).")

        # Set default gamma for RBF kernel
        if self.gamma is None:
            self.gamma = 1.0 / D

        # Precompute RBF similarity matrix (dense)
        K = rbf_kernel(self.features, gamma=self.gamma)

        # Apricot facility location with precomputed similarity
        selector = FacilityLocationSelection(
            n_samples=budget,
            metric='precomputed',
            optimizer='lazy',  # greedy with lazy updates (faster)
            random_state=self.seed
        )

        selector.fit(K)  # pass similarity matrix directly
        return list(selector.ranking)  # indices in selection order

