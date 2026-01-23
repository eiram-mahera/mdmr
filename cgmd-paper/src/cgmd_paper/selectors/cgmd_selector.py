from __future__ import annotations
import numpy as np
from typing import List, Optional, Sequence, Union

from cgmd.cgmd import CGMD

class CGMDSelector:
    def __init__(
            self,
            features: Union[np.ndarray, Sequence[float], Sequence[Sequence[float]]],
            *,
            seed: Optional[Union[int, np.random.Generator]] = 42,
            gamma: Optional[float] = None,
            dtype: str = "float64",
    ):
        self._algo = CGMD(features=features, gamma=gamma, dtype=dtype, seed=seed)

    def select(self, budget: int = 2) -> List[int]:
        return self._algo.select(budget=budget)

