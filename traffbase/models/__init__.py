from .DLinear import DLinear
from .SMamba import SMamba
from .DSTMambaV1 import DSTMambaV1
from .PatchTST import PatchTST


_MODEL_REGISTRY = {
    'DLinear': DLinear,
    'SMamba': SMamba,
    'DSTMambaV1': DSTMambaV1,
    'PatchTST': PatchTST,
}


def select_model(name: str) -> type:
    if name not in _MODEL_REGISTRY:
        available = ', '.join(_MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model '{name}'. Available models: {available}")

    return _MODEL_REGISTRY[name]
