import io
import unittest

import numpy as np
import torch

from traffbase.input_mask import (
    apply_random_time_mask,
    resolve_input_mask_settings,
)
from traffbase.trainers.ltsf_trainer import LTSFTrainer


class InputMaskSettingsTests(unittest.TestCase):
    def test_disabled_mask_returns_none(self) -> None:
        settings = resolve_input_mask_settings({'enabled': False}, 96)

        self.assertIsNone(settings)

    def test_ratio_uses_round_half_up(self) -> None:
        settings = resolve_input_mask_settings(
            {'enabled': True, 'ratio': 0.05, 'steps': None, 'repeats': 5},
            96,
        )

        self.assertIsNotNone(settings)
        self.assertEqual(settings.steps, 5)
        self.assertEqual(settings.repeats, 5)

    def test_steps_are_used_exactly(self) -> None:
        settings = resolve_input_mask_settings(
            {'enabled': True, 'ratio': None, 'steps': 12, 'repeats': 3},
            96,
        )

        self.assertIsNotNone(settings)
        self.assertEqual(settings.steps, 12)
        self.assertEqual(settings.repeats, 3)

    def test_ratio_and_steps_are_mutually_exclusive(self) -> None:
        with self.assertRaisesRegex(ValueError, 'Exactly one'):
            resolve_input_mask_settings(
                {'enabled': True, 'ratio': 0.1, 'steps': 12},
                96,
            )

    def test_steps_cannot_exceed_sequence_length(self) -> None:
        with self.assertRaisesRegex(ValueError, 'between 1'):
            resolve_input_mask_settings(
                {'enabled': True, 'ratio': None, 'steps': 97},
                96,
            )


class RandomTimeMaskTests(unittest.TestCase):
    def test_masks_exact_steps_shared_by_all_nodes(self) -> None:
        inputs = torch.ones(4, 12, 5, 1)

        masked = apply_random_time_mask(inputs, steps=3, seed=2024)
        masked_times = masked[..., 0].eq(0).all(dim=2)

        self.assertTrue(torch.equal(inputs, torch.ones_like(inputs)))
        self.assertTrue(torch.equal(masked_times.sum(dim=1), torch.full((4,), 3)))
        self.assertGreater(torch.unique(masked_times, dim=0).shape[0], 1)

        expanded_mask = masked_times[:, :, None, None].expand_as(masked)
        self.assertTrue(torch.all(masked[expanded_mask] == 0))
        self.assertTrue(torch.all(masked[~expanded_mask] == 1))

    def test_same_seed_is_reproducible(self) -> None:
        inputs = torch.ones(8, 24, 3, 1)

        first = apply_random_time_mask(inputs, steps=6, seed=2024)
        second = apply_random_time_mask(inputs, steps=6, seed=2024)
        different = apply_random_time_mask(inputs, steps=6, seed=2025)

        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.equal(first, different))


class LTSFTrainerMaskFlowTests(unittest.TestCase):
    class _Model:
        def eval(self) -> None:
            return None

    class _RecordingTrainer(LTSFTrainer):
        def __init__(self, cfg: dict, seed: int, log: io.StringIO) -> None:
            super().__init__(
                cfg=cfg,
                device=torch.device('cpu'),
                scaler=None,
                log=log,
                seed=seed,
            )
            self.predict_calls: list[tuple[int | None, int | None]] = []

        def predict(
            self,
            model,
            loader,
            input_mask_steps: int | None = None,
            mask_seed: int | None = None,
        ) -> tuple[np.ndarray, np.ndarray]:
            self.predict_calls.append((input_mask_steps, mask_seed))
            y_true = np.ones((2, 2, 1, 1), dtype=np.float32)
            if input_mask_steps is None:
                offset = 0.0
            else:
                assert mask_seed is not None
                offset = mask_seed / 10000
            y_pred = np.full_like(y_true, 1.0 + offset)
            return y_true, y_pred

    def test_clean_evaluation_precedes_mask_repeats(self) -> None:
        cfg = {
            'OPTIM': {},
            'TEST': {
                'input_mask': {
                    'enabled': True,
                    'ratio': None,
                    'steps': 3,
                    'repeats': 3,
                }
            },
        }
        log = io.StringIO()
        trainer = self._RecordingTrainer(cfg=cfg, seed=2024, log=log)
        loader = [(torch.ones(2, 12, 4, 1), torch.ones(2, 2, 4, 1))]

        trainer.test_model(self._Model(), loader)

        self.assertEqual(
            trainer.predict_calls,
            [(None, None), (3, 2024), (3, 2025), (3, 2026)],
        )
        self.assertIn('Clean', log.getvalue())
        self.assertIn('Masked summary', log.getvalue())

    def test_disabled_mask_only_runs_clean_evaluation(self) -> None:
        cfg = {'OPTIM': {}, 'TEST': {'input_mask': {'enabled': False}}}
        trainer = self._RecordingTrainer(
            cfg=cfg,
            seed=2024,
            log=io.StringIO(),
        )
        loader = [(torch.ones(2, 12, 4, 1), torch.ones(2, 2, 4, 1))]

        trainer.test_model(self._Model(), loader)

        self.assertEqual(trainer.predict_calls, [(None, None)])


class LTSFTrainerFitMetricsTests(unittest.TestCase):
    class _RecordingTrainer(LTSFTrainer):
        def __init__(self, log_fit_metrics: bool) -> None:
            super().__init__(
                cfg={
                    'GENERAL': {'log_fit_metrics': log_fit_metrics},
                    'OPTIM': {},
                },
                device=torch.device('cpu'),
                scaler=None,
                log=io.StringIO(),
            )
            self.predict_loaders: list[object] = []

        def train_one_epoch(
            self, model, train_loader, optimizer, scheduler, criterion
        ) -> float:
            return 0.5

        def eval_model(self, model, val_loader, criterion) -> float:
            return 0.25

        def predict(
            self,
            model,
            loader,
            input_mask_steps: int | None = None,
            mask_seed: int | None = None,
        ) -> tuple[np.ndarray, np.ndarray]:
            self.predict_loaders.append(loader)
            y_true = np.ones((2, 2, 1, 1), dtype=np.float32)
            return y_true, y_true.copy()

    def _train(self, log_fit_metrics: bool) -> tuple[float, float]:
        trainer = self._RecordingTrainer(log_fit_metrics)
        train_loader = object()
        val_loader = object()
        model = torch.nn.Linear(1, 1)

        _, val_mse, val_mae = trainer.train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=None,
            scheduler=None,
            criterion=None,
            max_epochs=1,
            save=None,
        )

        expected_loaders = (
            [train_loader, val_loader] if log_fit_metrics else [val_loader]
        )
        self.assertEqual(trainer.predict_loaders, expected_loaders)
        return val_mse, val_mae

    def test_validation_metrics_are_computed_once_without_fit_logging(self) -> None:
        self.assertEqual(self._train(log_fit_metrics=False), (0.0, 0.0))

    def test_fit_logging_reuses_validation_metrics(self) -> None:
        self.assertEqual(self._train(log_fit_metrics=True), (0.0, 0.0))


if __name__ == '__main__':
    unittest.main()
