from dataclasses import dataclass, field
from typing import Any
import numpy as np

@dataclass
class VectorItem:
    id: int | str
    vector: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.vector = np.asarray(self.vector, dtype=np.float32)
        if self.vector.ndim != 1:
            raise ValueError("Vector must be one-dimensional.")
