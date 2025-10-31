from __future__ import annotations
import numpy as np
from typing import List, Protocol, Sequence, Union

class Selector(Protocol):
    def __init__(
            self,
            features: Union[np.ndarray, Sequence[float], Sequence[Sequence[float]]]
    ): ...
    def select(self, budget: int = 2) -> List[int]: ...
