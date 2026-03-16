# core/system.py
import os
import importlib

# DEFAULT_SYSTEM = "sys_C=0"
DEFAULT_SYSTEM = "sys_wolmath"


def _load_module(name: str):
    return importlib.import_module(f"core.systems.{name}")

SYSTEM_NAME = os.getenv("DYNAMICS_SYSTEM", DEFAULT_SYSTEM)
_mod = _load_module(SYSTEM_NAME)

# якщо в системі є __all__ — експортуємо тільки його
if hasattr(_mod, "__all__"):
    __all__ = list(_mod.__all__)
    for k in __all__:
        globals()[k] = getattr(_mod, k)
else:
    # fallback: експортуємо все, що не починається з "_"
    __all__ = [k for k in dir(_mod) if not k.startswith("_")]
    for k in __all__:
        globals()[k] = getattr(_mod, k)
