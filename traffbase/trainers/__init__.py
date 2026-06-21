from .base_trainer import BaseTrainer
from .ltsf_trainer import LTSFTrainer

__all__ = ['BaseTrainer', 'LTSFTrainer']


def select_trainer(trainer: str) -> type:
    trainer_map = {'LTSFTrainer': LTSFTrainer}

    return trainer_map[trainer]
