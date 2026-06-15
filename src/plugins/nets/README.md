# Nets Plugin (Forecasting & ML)

The `nets` plugin provides a standardized, S.O.L.I.D. framework for implementing machine learning based forecasting strategies within the FTS (Financial Trading System) core. It leverages **ONNX** for cross-framework inference and a **UFD (Up/Flat/Down)** classification system for robust signal generation.

---

## 1. Architecture & Design

The plugin follows a decoupled pipeline to ensure that trading logic is isolated from ML model specifics:

1.  **Structural Conversion**: `DatasetBuilder` in core converts raw bars into feature matrices and sliding windows.
2.  **Inference (ONNX)**: An `.onnx` model predicts the next return. This standardized approach allows models trained in PyTorch, XGBoost, or Scikit-Learn to be used interchangeably, with preprocessing like scaling baked directly into the model artifact.
3.  **Classification (UFD)**: The predicted return is classified as `UP`, `FLAT`, or `DOWN` based on dynamic or fixed thresholds.
4.  **Signal Generation**: The classification is mapped to a `TradeSignal` (BUY, SELL, or FLAT).

---

## 2. Core Components

### `ONNXPredictor`
The heart of the inference engine. It loads an ONNX model file and performs standardized inference. 
*   **Contract**: Accepts arbitrary `numpy.ndarray` or `Dict[str, np.ndarray]` inputs and returns an `ndarray` of predictions. Input preparation is delegated to the `Transform` layer or the Strategy.

### `BaseClassifier` (UFD)
Classifies model output into actionable signals.
*   **`SimpleThresholdClassifier`**: Uses a fixed threshold (e.g., 0.1%) to filter noise.
*   **`DynamicThresholdClassifier`**: Uses ATR (Average True Range) to adjust the "Flat" zone based on current market volatility.

### `BaseModelTrainer`
Standardized interfaces for offline training and ONNX export:
*   **`LinearRegressionTrainer`**: Baseline statistical model.
*   **`XGBoostTrainer`**: Gradient boosting for non-linear patterns.
*   **`CNNTrainer`**: Convolutional Neural Network for sequence/pattern recognition.
*   **`RNNTrainer`**: Recurrent Neural Network (Elman RNN) for sequence modeling.
*   **`LSTMTrainer`**: Long Short-Term Memory Network for tracking long-term time-series dependencies.

---

## 3. Configuration & Parameterization

Neural network models are dynamically configured and validated using Pydantic models defined in `schemas.py`, while their network structures reside in `models.py`.

### Configuration Schemas (`schemas.py`)
*   **`BaseTrainerConfig`**: Holds basic options common to all trainers (e.g. `lookback_period` and `feature_cols`).
*   **`NNTrainingConfig`**: Deep learning training configurations (epochs, batch size, learning rate, optimizer, loss function, and TensorBoard logging directory).
*   **`CNNConfig`**: Specific hyperparameters for CNNs (channels, kernels, pooling sizes, dropout).
*   **`RNNConfig` & `LSTMConfig`**: Specific recurrent architecture parameters (hidden dimension, layers, dropout, bidirectional options).

### Neural Network Architectures (`models.py`)
*   **`SimpleCNN`**, **`SimpleRNN`**, and **`SimpleLSTM`**: Standard PyTorch model classes.
*   *Note: These models save training dataset parameters (`mean` and `std`) as internal PyTorch parameters, baking feature scaling directly into the exported `.onnx` file. No separate scaling pipeline is needed during live inference.*

---

## 4. Model Visualization & Monitoring

### Graph Inspection (Netron)
[Netron](https://netron.app/) is the open-source standard for visualizing exported model structures. You can drag and drop any exported `.onnx` file into Netron to interactively inspect the layers, tensor dimensions, and weights.

### Metrics & Graph Tracking (TensorBoard)
If `tensorboard_log_dir` is configured in `NNTrainingConfig`, the training loop logs metrics and registers the computational graph using PyTorch's native `SummaryWriter`.
*   **Launch TensorBoard**:
    ```bash
    uv run tensorboard --logdir=runs/
    ```

---

## 5. Usage & Configuration

To use the `nets` plugin, define a **Task** in a YAML configuration file.

### Example Configuration (`nets_task.yaml`)

```yaml
name: "nets_forecasting_example"
market_ids: ["BTC_USD"]

strategies:
  - class_path: "nets.strategies.nets_strategy.NetsStrategy"
    params:
      lookback_period: 20
      name_suffix: "v1"
      predictor:
        class_path: "nets.inference.ONNXPredictor"
        params:
          model_path: "models/my_lstm_model.onnx"
      transform:
        class_path: "trading_bot.core.transforms.LogReturnTransform"
      classifier:
        class_path: "nets.classifiers.DynamicThresholdClassifier"
        params:
          k: 0.5
          period: 10

sizing_strategy:
  class_path: "nets.sizing.confidence_sizer.ConfidenceSizer"
  params:
    base_amount_quote: 1000.0
```

---

## 6. Development Workflow

### Adding a New Trainer
1.  Implement `BaseModelTrainer` (or inherit from `BasePyTorchTrainer`) in `training.py`.
2.  Define any custom architecture in `models.py` and its configuration schema in `schemas.py`.
3.  Ensure the `train()` method returns the serialized ONNX bytes.
4.  Add any new dependencies to `pyproject.toml`.

### Adding a New Classifier
1.  Implement `BaseClassifier` in `classifiers.py`.
2.  Return a `PredictionSignal` (UP, FLAT, DOWN).

---

## 7. Advanced Training Features

All model trainers support advanced time-series training and validation capabilities designed for quantitative finance:

### 1. Leakage Prevention (Purging & Embargoing)
Standard cross-validation and splits leak information when label windows overlap or if features are serially correlated. 
- **Purging**: Removes training observations whose target horizons overlap with the validation set.
- **Embargoing**: Removes training observations that immediately succeed the validation set to prevent spillover correlation.
- **Scaling Isolation**: Training features' mean and standard deviation are computed **strictly on the training split** to prevent look-ahead bias from entering preprocessing.

### 2. Checkpointing & Early Stopping
- **PyTorch Models**: The training loop tracks validation loss at the end of each epoch, keeps a deep copy of the model weights that achieved the best validation score, and halts training early if validation loss does not improve for `early_stopping_patience` epochs. The best model state is restored before ONNX export.
- **XGBoost**: Employs XGBoost's native `early_stopping_rounds` and automatically returns the best iteration's state.

### 3. Gradient Norm Clipping
- Prevents extreme time-series spikes (e.g. flash crashes) from destabilizing network weights by capping the gradients using `torch.nn.utils.clip_grad_norm_` during training.

### 4. Metrics & Evaluation
- **`TimeSeriesDataset`**: An idiomatic PyTorch dataset that converts inputs and targets to float32 tensors and transposes inputs to `[batch, features, sequence_length]`.
- **`MetricsCalculator`**: A decoupled, model-agnostic class that calculates validation loss, Information Coefficient (IC) via Pearson correlation, Directional Accuracy (sign agreement), and IC Decay (correlation at lags 0 to 5) on raw predictions and targets.
- **`ValidationEvaluator`**: Coordinates the PyTorch validation loop and forwards prediction/target tensors to `MetricsCalculator`.

---

## 8. Verification

Run the test suite to ensure architectural integrity:
```bash
uv run pytest
```
