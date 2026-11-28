# src/plugins/nets/evaluator.py

import logging
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from quant_core.metrics import compute_forecast_metrics
from quant_core.models import ForecastMetrics
from torch.utils.data import DataLoader
from trading_bot.core.schemas import BarData

from ..enums import PredictionSignal
from ..output_selectors import BaseOutputSelector

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """
    Validation metrics from raw prediction and target arrays: the losses here,
    the forecast scores (rank IC with its SE, hit rate, IC decay) from
    ``quant_core.metrics.compute_forecast_metrics`` (R1).
    """

    @staticmethod
    def calculate_metrics(
        preds: np.ndarray,
        targets: np.ndarray,
        horizon: int,
        output_selector: Optional[BaseOutputSelector] = None,
        val_bars: Optional[Sequence[Sequence[BarData]]] = None,
    ) -> Dict[str, Any]:
        """
        Computes MSE, MAE, Huber and Cross Entropy losses, and puts the
        forecast scores under ``forecast`` (a ``ForecastMetrics``); ``ic``,
        ``directional_accuracy`` and ``ic_decay`` read from it. If
        output_selector and val_bars are provided, the selector's signals are
        scored the same way. Undefined values are NaN, never 0.0.
        """
        preds_flat = np.asarray(preds, dtype=float).flatten()
        targets_flat = np.asarray(targets, dtype=float).flatten()
        forecast = compute_forecast_metrics(preds_flat, targets_flat, horizon=horizon)

        nan = float("nan")
        mse = mae = huber = cross_entropy = nan
        if len(preds_flat) > 0 and len(targets_flat) > 0:
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

        metrics = {
            "loss": mse,
            "mse": mse,
            "mae": mae,
            "huber": huber,
            "cross_entropy": cross_entropy,
            "forecast": forecast,
            "ic": forecast.rank_ic,
            "ic_se": forecast.rank_ic_se,
            "directional_accuracy": forecast.hit_rate,
            "ic_decay": forecast.ic_decay,
        }

        # Score the output selector's signals (+1 / 0 / -1) the same way:
        # flat signals have no direction and are left out of the hit rate.
        if output_selector is not None and val_bars is not None:
            selected_signals = [
                output_selector.select_output(pred_val, bars_val)[0]
                for pred_val, bars_val in zip(preds, val_bars)
            ]
            direction = {PredictionSignal.UP: 1.0, PredictionSignal.DOWN: -1.0}
            signal_values = [direction.get(s, 0.0) for s in selected_signals]
            selector = compute_forecast_metrics(
                signal_values, targets_flat[: len(signal_values)], horizon=horizon
            )
            metrics["selector_accuracy"] = selector.hit_rate
            metrics["selector_signal_counts"] = {
                "up": sum(1 for s in selected_signals if s == PredictionSignal.UP),
                "flat": sum(1 for s in selected_signals if s == PredictionSignal.FLAT),
                "down": sum(1 for s in selected_signals if s == PredictionSignal.DOWN),
            }

        return metrics


def registry_metrics(metrics: Optional[Dict[str, Any]]) -> Dict[str, float]:
    """The validation scores the model registry stores (R1-D7), from a
    trainer's ``best_val_metrics``. NaN when there was no validation set."""
    metrics = metrics or {}
    forecast = metrics.get("forecast") or ForecastMetrics()
    return {
        "val_loss": metrics.get("loss", float("nan")),
        "val_ic": forecast.rank_ic,
        "val_ic_se": forecast.rank_ic_se,
        "val_ic_n": forecast.n_obs,
        "val_hit_rate": forecast.hit_rate,
    }


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
        horizon: int,
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
            horizon=horizon,
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
        writer.add_scalar("Metrics/val_ic_se", metrics["ic_se"], epoch)
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
