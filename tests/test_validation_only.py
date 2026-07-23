import io
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from traffbase.main import run


class _TinyModel(nn.Module):
    def __init__(self, **_: object) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))


class _FakeTrainer:
    def __init__(self) -> None:
        self.epoch_time = 1.25
        self.saved_to: str | None = None
        self.test_calls = 0

    def model_summary(self, model: nn.Module, loader: object) -> str:
        return 'summary'

    def train_model(
        self,
        model: nn.Module,
        *_: object,
        save: str | None = None,
        **__: object,
    ) -> tuple[nn.Module, float, float]:
        self.saved_to = save
        return model, 0.1, 0.2

    def test_model(
        self, model: nn.Module, loader: object
    ) -> dict[str, float]:
        self.test_calls += 1
        return {'clean_mse': 0.3, 'clean_mae': 0.4, 'infer_time': 0.5}


class ValidationOnlyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = {
            'GENERAL': {'runner': 'LTSFTrainer', 'max_epochs': 1},
            'DATA': {'in_steps': 12, 'out_steps': 12},
            'OPTIM': {
                'lr_scheduler_type': 'ExponentialLR',
                'initial_lr': 0.001,
            },
            'MODEL_PARAM': {},
        }
        self.device = torch.device('cpu')

    def _run(self, validation_only: bool) -> tuple[dict[str, float], _FakeTrainer]:
        trainer = _FakeTrainer()
        loaders = (object(), object(), object())
        with (
            patch('traffbase.main.select_model', return_value=_TinyModel),
            patch('traffbase.main.build_LTSF_dataloader', return_value=loaders),
            patch('traffbase.main.LTSFTrainer', return_value=trainer),
            patch('traffbase.main.create_log_file', return_value=io.StringIO()),
            patch(
                'traffbase.main.create_checkpoint_path',
                return_value=Path('checkpoint.pt'),
            ) as create_checkpoint,
            patch('traffbase.main.print_log'),
        ):
            metrics = run(
                'Tiny',
                'TEST',
                self.cfg,
                2024,
                self.device,
                validation_only=validation_only,
            )

        if validation_only:
            create_checkpoint.assert_not_called()
        else:
            create_checkpoint.assert_called_once()
        return metrics, trainer

    def test_validation_only_skips_checkpoint_and_test(self) -> None:
        metrics, trainer = self._run(validation_only=True)

        self.assertIsNone(trainer.saved_to)
        self.assertEqual(trainer.test_calls, 0)
        self.assertEqual(metrics['val_mse'], 0.1)
        self.assertNotIn('test_mse', metrics)

    def test_normal_run_keeps_checkpoint_and_test(self) -> None:
        metrics, trainer = self._run(validation_only=False)

        self.assertEqual(trainer.saved_to, 'checkpoint.pt')
        self.assertEqual(trainer.test_calls, 1)
        self.assertEqual(metrics['test_mse'], 0.3)


if __name__ == '__main__':
    unittest.main()
