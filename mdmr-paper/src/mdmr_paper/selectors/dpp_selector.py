import numpy as np
from typing import List, Sequence, Union, Optional
from dppy.finite_dpps import FiniteDPP
from sklearn.metrics.pairwise import rbf_kernel


class DPPSelector:
    def __init__(
            self,
            features: Union[np.ndarray, Sequence[float], Sequence[Sequence[float]]],
            *,
            gamma: Optional[float] = None,
    ):
        self.features = features
        self.gamma = gamma

    def select(self, budget: int = 2) -> List[int]:

        # Step 1: Compute the RBF kernel matrix (similarity matrix)
        # gamma is 1/(2*sigma^2);
        if self.gamma is None:
            # self.gamma = 1.0 / (2 * np.mean(np.var(self.features, axis=0)))
            self.gamma = 1.0 / (self.features.shape[1])

        print(f"gamma: {self.gamma}")
        L = rbf_kernel(self.features, gamma=None)

        # 2. Add a small epsilon to the diagonal for numerical stability
        epsilon = 1e-5
        L_stabilized = L + epsilon * np.eye(L.shape[0])

        # Step 2: Create a DPP instance
        dpp = FiniteDPP('likelihood', L=L_stabilized)

        # Step 3: Sample a fixed-size subset of query_budget
        dpp.sample_exact_k_dpp(size=budget)

        selected_indices = dpp.list_of_samples[-1]

        return selected_indices


