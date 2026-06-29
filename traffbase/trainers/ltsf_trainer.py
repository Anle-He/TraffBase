import datetime
import time
from typing import Any, TextIO

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchinfo import summary

from .base_trainer import BaseTrainer
from traffbase.input_mask import (
    apply_random_time_mask,
    resolve_input_mask_settings,
)
from traffbase.utils import compute_mse_mae, print_log, banner, StandardScaler


class LTSFTrainer(BaseTrainer):
    def __init__(
        self,
        cfg: dict[str, Any],
        device: torch.device,
        scaler: StandardScaler,
        log: TextIO | None = None,
        seed: int = 2024,
    ) -> None:
        super().__init__()

        self.cfg = cfg
        self.device = device
        # Retained for API symmetry with the runner/dataloader and possible future
        # inverse scaling. Metrics are computed on the normalized scale, so the
        # trainer itself never inverts (see AGENTS.md); this field is currently unused.
        self.scaler = scaler
        self.log = log
        self.seed = seed

        # Mean wall-clock seconds per training epoch, populated by ``fit``.
        self.epoch_time = float('nan')

        self.clip_grad = self.cfg['OPTIM'].get('clip_grad')

        # FreDF: optional frequency-domain auxiliary loss added to the prediction
        # loss during training (Wang et al., "FreDF"). Disabled unless OPTIM.use_fredf
        # is set, so models that do not request it train exactly as before.
        optim = self.cfg['OPTIM']
        self.use_fredf = optim.get('use_fredf', False)
        self.fredf_loss = optim.get('fredf_loss', 'MAE')
        self.fredf_mode = optim.get('fredf_mode', 'fft')
        self.fredf_weight = optim.get('fredf_weight', 0.0)
        self.module_first = optim.get('module_first', True)

        # Whether to run an extra full pass over train/val after training to log
        # their MSE/MAE. Purely diagnostic, so it can be disabled to save compute.
        self.log_fit_metrics = self.cfg['GENERAL'].get('log_fit_metrics', True)

    def train_one_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        criterion: nn.Module,
    ) -> float:
        model.train()

        # Accumulate on-device and sync once per epoch; calling .item() per batch
        # would force a GPU->CPU sync every step and stall the pipeline.
        total_loss = torch.zeros((), device=self.device)
        total_samples = 0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.float().to(self.device, non_blocking=True)
            y_batch = y_batch.float().to(self.device, non_blocking=True)

            out_batch = model(x_batch)

            loss = criterion(out_batch, y_batch)
            if self.use_fredf:
                loss = loss + self.fredf_weight * self._fredf_loss(out_batch, y_batch)
            batch_size = x_batch.size(0)
            total_loss += loss.detach() * batch_size
            total_samples += batch_size

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.clip_grad:
                nn.utils.clip_grad_norm_(model.parameters(), self.clip_grad)
            optimizer.step()

        epoch_loss = (total_loss / total_samples).item()
        scheduler.step()

        return epoch_loss

    def _fredf_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Frequency-domain discrepancy between prediction and target.

        Both tensors are ``[B, T_out, N, 1]``; the transform is taken over the time
        axis (``dim=1``). ``module_first`` controls whether the reduction happens on
        the complex magnitude (True) or on the complex difference before taking the
        magnitude (False), matching the original FreDF formulation.
        """
        if self.fredf_mode == 'fft':
            diff = torch.fft.fft(pred, dim=1) - torch.fft.fft(target, dim=1)
        elif self.fredf_mode == 'rfft-2D':
            diff = torch.fft.rfft2(pred) - torch.fft.rfft2(target)
        else:
            raise ValueError(f"Unknown fredf_mode '{self.fredf_mode}'")

        if self.fredf_loss == 'MAE':
            return diff.abs().mean() if self.module_first else diff.mean().abs()
        elif self.fredf_loss == 'MSE':
            return (
                (diff.abs() ** 2).mean()
                if self.module_first
                else (diff**2).mean().abs()
            )
        else:
            raise ValueError(f"Unknown fredf_loss '{self.fredf_loss}'")

    @torch.inference_mode()
    def eval_model(
        self, model: nn.Module, val_loader: DataLoader, criterion: nn.Module
    ) -> float:
        model.eval()

        total_loss = torch.zeros((), device=self.device)
        total_samples = 0
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.float().to(self.device, non_blocking=True)
            y_batch = y_batch.float().to(self.device, non_blocking=True)

            out_batch = model(x_batch)

            loss = criterion(out_batch, y_batch)
            batch_size = x_batch.size(0)
            total_loss += loss * batch_size
            total_samples += batch_size

        return (total_loss / total_samples).item()

    @torch.inference_mode()
    def predict(
        self,
        model: nn.Module,
        loader: DataLoader,
        input_mask_steps: int | None = None,
        mask_seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        model.eval()

        if (input_mask_steps is None) != (mask_seed is None):
            raise ValueError(
                'input_mask_steps and mask_seed must be provided together'
            )

        y_list = []
        out_list = []

        for x_batch, y_batch in loader:
            x_batch = x_batch.float()
            if input_mask_steps is not None and mask_seed is not None:
                x_batch = apply_random_time_mask(
                    x_batch,
                    steps=input_mask_steps,
                    seed=mask_seed,
                )
                mask_seed += x_batch.size(0)

            x_batch = x_batch.to(self.device, non_blocking=True)
            y_batch = y_batch.float().to(self.device, non_blocking=True)

            out_batch = model(x_batch)

            out_list.append(out_batch.cpu().numpy())
            y_list.append(y_batch.cpu().numpy())

        # (samples, out_steps, num_nodes, output_dim)
        y = np.vstack(y_list)
        out = np.vstack(out_list)

        return y, out

    def train_model(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        criterion: nn.Module,
        max_epochs: int = 10,
        early_stop_patience: int = 3,
        verbose: int = 1,
        save: str | None = None,
    ) -> nn.Module:
        if max_epochs <= 0:
            raise ValueError('max_epochs must be greater than 0')

        wait = 0
        min_val_loss = np.inf
        best_epoch = 0
        completed_epochs = 0
        # Snapshot weights to CPU so the best checkpoint does not keep a second copy
        # resident in GPU memory. copy=True forces an independent copy in both cases:
        # off-GPU it is the device->host transfer, on a CPU model it avoids aliasing
        # the live parameter (where .cpu() would be a no-op).
        best_state_dict = {
            k: v.detach().to('cpu', copy=True) for k, v in model.state_dict().items()
        }

        train_loss_list = []
        val_loss_list = []

        print_log(banner('Training'), log=self.log)

        start = time.time()
        for epoch in range(max_epochs):
            completed_epochs = epoch + 1
            train_loss = self.train_one_epoch(
                model, train_loader, optimizer, scheduler, criterion
            )
            train_loss_list.append(train_loss)

            val_loss = self.eval_model(model, val_loader, criterion)
            val_loss_list.append(val_loss)

            if verbose and (epoch + 1) % verbose == 0:
                now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print_log(
                    f'[{now}] Epoch {epoch + 1:>3}/{max_epochs}  '
                    f'Train Loss = {train_loss:.5f}  Val Loss = {val_loss:.5f}',
                    log=self.log,
                )

            if val_loss < min_val_loss:
                wait = 0
                min_val_loss = val_loss
                best_epoch = epoch
                best_state_dict = {
                    k: v.detach().to('cpu', copy=True)
                    for k, v in model.state_dict().items()
                }
            else:
                wait += 1
                if wait >= early_stop_patience:
                    print_log(
                        f'Early stopping triggered at epoch {epoch + 1} '
                        f'(patience={early_stop_patience})',
                        log=self.log,
                    )
                    break
        end = time.time()

        model.load_state_dict(best_state_dict)

        if save:
            torch.save(best_state_dict, save)

        train_loss_best = train_loss_list[best_epoch]
        val_loss_best = val_loss_list[best_epoch]

        print_log(banner('Best Model'), log=self.log)
        print_log(f'{"Epoch":<11}: {best_epoch + 1}/{completed_epochs}', log=self.log)
        if self.log_fit_metrics:
            train_mse, train_mae = compute_mse_mae(*self.predict(model, train_loader))
            val_mse, val_mae = compute_mse_mae(*self.predict(model, val_loader))
            print_log(
                f'{"Train":<11}: Loss = {train_loss_best:.5f}   '
                f'MSE = {train_mse:.5f}   MAE = {train_mae:.5f}',
                log=self.log,
            )
            print_log(
                f'{"Val":<11}: Loss = {val_loss_best:.5f}   '
                f'MSE = {val_mse:.5f}   MAE = {val_mae:.5f}',
                log=self.log,
            )
        else:
            print_log(f'{"Train":<11}: Loss = {train_loss_best:.5f}', log=self.log)
            print_log(f'{"Val":<11}: Loss = {val_loss_best:.5f}', log=self.log)
        self.epoch_time = (end - start) / completed_epochs
        print_log(
            f'{"Epoch time":<11}: {self.epoch_time:.3f} s',
            log=self.log,
        )

        return model

    @torch.inference_mode()
    def test_model(self, model: nn.Module, test_loader: DataLoader) -> dict[str, float]:
        model.eval()

        print_log(banner('Test'), log=self.log)

        start = time.time()
        y_true, y_pred = self.predict(model, test_loader)
        end = time.time()

        out_steps = y_pred.shape[1]

        clean_mse, clean_mae = compute_mse_mae(y_true, y_pred)
        print_log(
            f'{"Clean":<11}: MSE = {clean_mse:.5f}   MAE = {clean_mae:.5f}   '
            f'(steps 1-{out_steps})',
            log=self.log,
        )
        infer_time = end - start
        print_log(f'{"Infer time":<11}: {infer_time:.3f} s', log=self.log)

        metrics = {
            'clean_mse': clean_mse,
            'clean_mae': clean_mae,
            'infer_time': infer_time,
        }

        mask_config = self.cfg.get('TEST', {}).get('input_mask', {})
        if not mask_config.get('enabled', False):
            return metrics

        sample_batch = next(iter(test_loader))[0]
        settings = resolve_input_mask_settings(
            mask_config,
            sequence_length=sample_batch.shape[1],
        )
        if settings is None:
            return metrics

        print_log(banner('Input Mask Test'), log=self.log)
        print_log(
            f'Mask condition: {settings.description}, repeats={settings.repeats}',
            log=self.log,
        )

        masked_mse = []
        masked_mae = []
        for repeat in range(settings.repeats):
            repeat_seed = self.seed + repeat
            start = time.time()
            y_true, y_pred = self.predict(
                model,
                test_loader,
                input_mask_steps=settings.steps,
                mask_seed=repeat_seed,
            )
            end = time.time()

            repeat_mse, repeat_mae = compute_mse_mae(y_true, y_pred)
            masked_mse.append(repeat_mse)
            masked_mae.append(repeat_mae)
            print_log(
                f'Repeat {repeat + 1}/{settings.repeats} '
                f'(seed={repeat_seed}): MSE = {repeat_mse:.5f}, '
                f'MAE = {repeat_mae:.5f}, inference time = {end - start:.3f} s',
                log=self.log,
            )

        mse_mean = float(np.mean(masked_mse))
        mse_std = float(np.std(masked_mse))
        mae_mean = float(np.mean(masked_mae))
        mae_std = float(np.std(masked_mae))
        mse_change = mse_mean - clean_mse
        mae_change = mae_mean - clean_mae
        mse_percent = self._percentage_change(mse_mean, clean_mse)
        mae_percent = self._percentage_change(mae_mean, clean_mae)

        print_log(
            f'Masked summary: MSE = {mse_mean:.5f} +/- {mse_std:.5f}, '
            f'MAE = {mae_mean:.5f} +/- {mae_std:.5f}',
            log=self.log,
        )
        print_log(
            f'Degradation vs clean: MSE = {mse_change:+.5f} '
            f'({mse_percent:+.2f}%), MAE = {mae_change:+.5f} '
            f'({mae_percent:+.2f}%)',
            log=self.log,
        )

        metrics.update(
            {
                'masked_mse': mse_mean,
                'masked_mae': mae_mean,
                'masked_mse_std': mse_std,
                'masked_mae_std': mae_std,
            }
        )

        return metrics

    @staticmethod
    def _percentage_change(value: float, baseline: float) -> float:
        if baseline == 0:
            return 0.0 if value == 0 else float('inf')

        return (value - baseline) / baseline * 100

    def model_summary(self, model: nn.Module, dataloader: DataLoader) -> str:
        x_shape = next(iter(dataloader))[0].shape

        return str(summary(model, x_shape, verbose=0, device=self.device))
