# src/plugins/nets/evaluator.py

import logging
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """
    Computes trading performance metrics (Loss, IC, Directional Accuracy, IC Decay)
    from raw prediction and target arrays.
    """

    @staticmethod
    def calculate_metrics(preds: np.ndarray, targets: np.ndarray) -> Dict[str, Any]:
        """
        Computes MSE loss, Information Coefficient (IC) via Pearson correlation,
        Directional Accuracy (sign agreement), and IC Decay over time (lags 0 to 5).
        """
        if len(preds) == 0 or len(targets) == 0:
            return {
                "loss": 0.0,
                "ic": 0.0,
                "directional_accuracy": 0.5,
                "ic_decay": {i: 0.0 for i in range(6)},
            }

        preds = preds.flatten()
        targets = targets.flatten()

        loss = float(np.mean((preds - targets) ** 2))

        std_preds = np.std(preds)
        std_targets = np.std(targets)
        ic = (
            float(np.corrcoef(preds, targets)[0, 1])
            if std_preds > 1e-8 and std_targets > 1e-8
            else 0.0
        )

        directional_accuracy = float(np.mean(np.sign(preds) == np.sign(targets)))

        # IC Decay over time (lag 0 to 5)
        ic_decay = {}
        for lag in range(6):
            if lag == 0:
                ic_decay[0] = ic
            elif len(preds) > lag:
                p_sub = preds[:-lag]
                t_sub = targets[lag:]
                s_p = np.std(p_sub)
                s_t = np.std(t_sub)
                ic_decay[lag] = (
                    float(np.corrcoef(p_sub, t_sub)[0, 1])
                    if s_p > 1e-8 and s_t > 1e-8
                    else 0.0
                )
            else:
                ic_decay[lag] = 0.0

        return {
            "loss": loss,
            "ic": ic,
            "directional_accuracy": directional_accuracy,
            "ic_decay": ic_decay,
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
        metrics = MetricsCalculator.calculate_metrics(preds_arr, targets_arr)

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
        for lag, lag_ic in metrics["ic_decay"].items():
            writer.add_scalar(f"Metrics/val_ic_decay_lag_{lag}", lag_ic, epoch)
