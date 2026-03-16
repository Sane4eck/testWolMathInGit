from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_one(history: dict[str, np.ndarray], key: str, out_path: str | Path | None = None) -> None:
    t = history["1_time"]
    y = history[key]
    plt.figure(figsize=(10, 4))
    plt.plot(t, y)
    plt.xlabel("t, s")
    plt.ylabel(key)
    plt.title(key)
    plt.grid(True)
    if out_path is None:
        plt.show()
    else:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
