from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseTrainer(ABC):
    @abstractmethod
    def train_one_epoch(
        self,
        model: Any,
        train_loader: Any,
        optimizer: Any,
        scheduler: Any,
        criterion: Any,
    ) -> float:
        ...

    @abstractmethod
    def train_model(
        self,
        model: Any,
        train_loader: Any,
        val_loader: Any,
        optimizer: Any,
        scheduler: Any,
        criterion: Any,
        max_epochs: int = 10,
        early_stop_patience: int = 3,
        verbose: int = 1,
        save: str | None = None,
    ) -> Any:
        ...

    @abstractmethod
    def eval_model(
        self, model: Any, val_loader: Any, criterion: Any
    ) -> float:
        ...

    @abstractmethod
    def test_model(self, model: Any, test_loader: Any) -> None:
        ...

    @abstractmethod
    def predict(
        self,
        model: Any,
        loader: Any,
        input_mask_steps: int | None = None,
        mask_seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        ...

    @abstractmethod
    def model_summary(self, model: Any, dataloader: Any) -> str:
        ...
