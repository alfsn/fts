# src/plugins/nets/evaluator.py

import logging
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from trading_bot.core.schemas import BarData

from ..enums import PredictionSignal
from ..output_selectors import BaseOutputSelector

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """
    Computes trading performance metrics (Loss, IC, Directional Accuracy, IC Decay)
    from raw prediction and target arrays.
    """

    @staticmethod
    def calculate_metrics(
        preds: np.ndarray,
        targets: np.ndarray,
        output_selector: Optional[BaseOutputSelector] = None,
        val_bars: Optional[Sequence[Sequence[BarData]]] = None,
    ) -> Dict[str, Any]:
        """
        Computes MSE, MAE, Huber, and Cross Entropy losses, Information Coefficient (IC) via Pearson correlation,
        Directional Accuracy (sign agreement), and IC Decay over time (lags 0 to 5).
        If output_selector and val_bars are provided, computes selector signal metrics.
        """
        if len(preds) == 0 or len(targets) == 0:
            return {
                "loss": 0.0,
                "mse": 0.0,
                "mae": 0.0,
                "huber": 0.0,
                "cross_entropy": 0.0,
                "ic": 0.0,
                "directional_accuracy": 0.5,
                "ic_decay": {i: 0.0 for i in range(6)},
            }

        preds_flat = preds.flatten()
        targets_flat = targets.flatten()

        # Compute losses
        mse = float(np.mean((preds_flat - targets_flat) ** 2))
        mae = float(np.mean(np.abs(preds_flat - targets_flat)))

        # Huber loss
        delta = 1.0
        diff = np.abs(preds_flat - targets_flat)
        huber_loss = np.where(
            diff < delta, 0.5 * (diff**2), delta * (diff - 0.5 * delta)
        )
        huber = float(np.mean(huber_loss))

        # Cross Entropy (for 3-class probability predictions)
        cross_entropy = 0.0
        if len(preds.shape) > 1 and preds.shape[1] == 3:
            # Check if predictions are valid probabilities
            is_prob = (
                np.all(preds >= 0)
                and np.all(preds <= 1)
                and np.allclose(np.sum(preds, axis=1), 1.0, atol=1e-2)
            )
            if is_prob:
                threshold = 0.001
                targets_cls = np.zeros(len(targets_flat), dtype=np.int32)
                targets_cls[targets_flat > threshold] = 2
                targets_cls[
                    (targets_flat >= -threshold) & (targets_flat <= threshold)
                ] = 1
                targets_cls[targets_flat < -threshold] = 0

                eps = 1e-15
                clipped_preds = np.clip(preds, eps, 1 - eps)
                log_loss = -np.mean(
                    np.log(clipped_preds[np.arange(len(targets_cls)), targets_cls])
                )
                cross_entropy = float(log_loss)

        loss = mse

        std_preds = np.std(preds_flat)
        std_targets = np.std(targets_flat)
        ic = (
            float(np.corrcoef(preds_flat, targets_flat)[0, 1])
            if std_preds > 1e-8 and std_targets > 1e-8
            else 0.0
        )

        directional_accuracy = float(
            np.mean(np.sign(preds_flat) == np.sign(targets_flat))
        )

        # IC Decay over time (lag 0 to 5)
        ic_decay = {}
        for lag in range(6):
            if lag == 0:
                ic_decay[0] = ic
            elif len(preds_flat) > lag:
                p_sub = preds_flat[:-lag]
                t_sub = targets_flat[lag:]
                s_p = np.std(p_sub)
                s_t = np.std(t_sub)
                ic_decay[lag] = (
                    float(np.corrcoef(p_sub, t_sub)[0, 1])
                    if s_p > 1e-8 and s_t > 1e-8
                    else 0.0
                )
            else:
                ic_decay[lag] = 0.0

        metrics = {
            "loss": loss,
            "mse": mse,
            "mae": mae,
            "huber": huber,
            "cross_entropy": cross_entropy,
            "ic": ic,
            "directional_accuracy": directional_accuracy,
            "ic_decay": ic_decay,
        }

        # Calculate metrics specific to output selector
        if output_selector is not None and val_bars is not None:
            selected_signals = []
            for pred_val, bars_val in zip(preds, val_bars):
                sig, _ = output_selector.select_output(pred_val, bars_val)
                selected_signals.append(sig)

            correct_signals = 0
            non_flat_signals = 0
            for sig, target_ret in zip(selected_signals, targets_flat):
                if sig == PredictionSignal.UP:
                    non_flat_signals += 1
                    if target_ret > 0:
                        correct_signals += 1
                elif sig == PredictionSignal.DOWN:
                    non_flat_signals += 1
                    if target_ret < 0:
                        correct_signals += 1

            metrics["selector_accuracy"] = (
                correct_signals / non_flat_signals if non_flat_signals > 0 else 0.5
            )
            metrics["selector_signal_counts"] = {
                "up": sum(1 for s in selected_signals if s == PredictionSignal.UP),
                "flat": sum(1 for s in selected_signals if s == PredictionSignal.FLAT),
                "down": sum(1 for s in selected_signals if s == PredictionSignal.DOWN),
            }

        return metrics


class ValidationEvaluator:
    """
    Evaluator for out-of-sample validation metrics.
    Adheres to the Single Responsibility Principle.
    """

    @staticmethod
    def evaluate(
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        output_selector: Optional[BaseOutputSelector] = None,
        val_bars: Optional[Sequence[Sequence[BarData]]] = None,
    ) -> Dict[str, Any]:
        """
        Runs the evaluation loop on the validation dataloader.
        Collects predictions and targets and delegates metrics calculation.
        """
        model.eval()
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch_X, batch_y in dataloader:
                outputs = model(batch_X)
                val_preds.append(outputs.cpu().numpy())
                val_targets.append(batch_y.cpu().numpy())

        preds_arr = np.concatenate(val_preds, axis=0)
        targets_arr = np.concatenate(val_targets, axis=0)

        # Delegate metrics calculation
        metrics = MetricsCalculator.calculate_metrics(
            preds=preds_arr,
            targets=targets_arr,
            output_selector=output_selector,
            val_bars=val_bars,
        )

        # If criterion is MAE or Huber, recalculate loss with that criterion
        # (Since MetricsCalculator defaults to MSE loss)
        if not isinstance(criterion, nn.MSELoss):
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in dataloader:
                    outputs = model(batch_X)
                    val_loss += criterion(outputs, batch_y).item()
            metrics["loss"] = val_loss / len(dataloader)

        return metrics

    @staticmethod
    def log_to_tensorboard(
        writer: Any,
        metrics: Dict[str, Any],
        epoch: int,
    ) -> None:
        """
        Logs the computed metrics to TensorBoard SummaryWriter.
        """
        if writer is None:
            return

        writer.add_scalar("Loss/val_epoch", metrics["loss"], epoch)
        writer.add_scalar("Metrics/val_ic", metrics["ic"], epoch)
        writer.add_scalar(
            "Metrics/val_directional_accuracy",
            metrics["directional_accuracy"],
            epoch,
        )
        if "selector_accuracy" in metrics:
            writer.add_scalar(
                "Metrics/val_selector_accuracy", metrics["selector_accuracy"], epoch
            )
        if "selector_signal_counts" in metrics:
            for sig, count in metrics["selector_signal_counts"].items():
                writer.add_scalar(f"Metrics/val_selector_count_{sig}", count, epoch)

        for lag, lag_ic in metrics["ic_decay"].items():
            writer.add_scalar(f"Metrics/val_ic_decay_lag_{lag}", lag_ic, epoch)
