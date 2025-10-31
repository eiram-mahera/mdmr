from __future__ import annotations
import numpy as np
from typing import List, Optional, Sequence, Union

class RandomSelector:
    def __init__(
            self,
            features: Union[np.ndarray, Sequence[float], Sequence[Sequence[float]]],
            *,
            seed: Optional[Union[int, np.random.Generator]] = 42
        ):
        self.features = features
        self._rng = np.random.default_rng(seed)

    def select(self, budget: int = 2) -> List[int]:
        return self._rng.choice(len(self.features), size=budget, replace=False).tolist()
