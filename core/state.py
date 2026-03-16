# core/state.py
from dataclasses import dataclass
import numpy as np

@dataclass
class State:
    time: float = 0.0
    y: np.ndarray | None = None
