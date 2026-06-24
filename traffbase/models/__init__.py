from .DLinear import DLinear
from .SMamba import SMamba
from .DSTMambaV1 import DSTMambaV1
from .PatchTST import PatchTST
from .iTransformer import iTransformer
from .CycleNet import CycleNet
from .MTGNN import MTGNN
from .Mamba import Mamba
from .TimesNet import TimesNet
from .FilterNet import FilterNet
from .Amplifier import Amplifier
from .FoMoV1 import FoMoV1
from .CAM import CAM


_MODEL_REGISTRY = {
    'DLinear': DLinear,
    'SMamba': SMamba,
    'DSTMambaV1': DSTMambaV1,
    'PatchTST': PatchTST,
    'iTransformer': iTransformer,
    'CycleNet': CycleNet,
    'MTGNN': MTGNN,
    'Mamba': Mamba,
    'TimesNet': TimesNet,
    'FilterNet': FilterNet,
    'Amplifier': Amplifier,
    'FoMoV1': FoMoV1,
    'CAM': CAM,
}


def select_model(name: str) -> type:
    if name not in _MODEL_REGISTRY:
        available = ', '.join(_MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model '{name}'. Available models: {available}")

    return _MODEL_REGISTRY[name]
