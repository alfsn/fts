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

## 2. Package Layout

The plugin codebase is organized into domain-focused subfolders under `src/plugins/nets/`:

*   **`output_selectors/`**: Definitions and implementations for return/probability selectors and thresholding:
    *   `abc.py`: Abstract Base Classes (`BaseOutputSelector`, `BaseRegressionOutputSelector`).
    *   `output_selectors.py`: Concrete selectors (`SimpleThresholdClassifier`, `DynamicThresholdClassifier`, `ClassificationOutputSelector`, `QuantileOutputSelector`).
*   **`models/`**: Neural network structures and validation schemas:
    *   `models.py`: Concrete PyTorch model structures (`SimpleCNN`, `SimpleRNN`, `SimpleLSTM`).
    *   `schemas.py`: Pydantic config schemas (`CNNConfig`, `LSTMConfig`, `NNTrainingConfig`, `BaseTrainerConfig`).
*   **`training/`**: ML model trainers and validation/evaluation utilities:
    *   `training.py`: Concrete trainers (`CNNTrainer`, `LSTMTrainer`, `XGBoostTrainer`, etc.).
    *   `evaluator.py`: Cross-validation helpers (`MetricsCalculator`, `ValidationEvaluator`).
*   **`inference/`**: ONNX engine:
    *   `inference.py`: Standardized `ONNXPredictor`.
*   **`sizing/`**: Sizer plugin implementation (`ConfidenceSizer`).

---

## 3. Core Components

### `ONNXPredictor`
The heart of the inference engine. It loads an ONNX model file and performs standardized inference. 
*   **Contract**: Accepts arbitrary `numpy.ndarray` or `Dict[str, np.ndarray]` inputs and returns an `ndarray` of predictions. Input preparation is delegated to the `Transform` layer or the Strategy.

### `BaseOutputSelector`
Converts model predictions into unified signals and confidence.
*   **`BaseRegressionOutputSelector`**: Abstract base for regression return predictions. Subclasses:
    *   **`SimpleThresholdClassifier`**: Uses a fixed threshold to filter noise and computes return-based confidence.
    *   **`DynamicThresholdClassifier`**: Uses ATR to dynamically adjust the "Flat" zone.
*   **`ClassificationOutputSelector`**: Converts probabilities to signals using configured class labels.
*   **`QuantileOutputSelector`**: Converts `[q10, q50, q90]` quantile predictions using spread-based confidence.

### `BaseModelTrainer`
Standardized interfaces for offline training and ONNX export:
*   **`LinearRegressionTrainer`**: Baseline statistical model.
*   **`XGBoostTrainer`**: Gradient boosting for non-linear patterns.
*   **`CNNTrainer`**: Convolutional Neural Network for sequence/pattern recognition.
*   **`RNNTrainer`**: Recurrent Neural Network (Elman RNN) for sequence modeling.
*   **`LSTMTrainer`**: Long Short-Term Memory Network for tracking long-term time-series dependencies.

---

## 4. Configuration & Parameterization

Neural network models are dynamically configured and validated using Pydantic models defined in `models/schemas.py`, while their network structures reside in `models/models.py`.

### Configuration Schemas (`models/schemas.py`)
*   **`BaseTrainerConfig`**: Holds basic options common to all trainers (e.g. `lookback_period` and `feature_cols`).
*   **`NNTrainingConfig`**: Deep learning training configurations (epochs, batch size, learning rate, optimizer, loss function, and TensorBoard logging directory).
*   **`CNNConfig`**: Specific hyperparameters for CNNs (channels, kernels, pooling sizes, dropout). Inherits from `NNTrainingConfig`.
*   **`RNNConfig` & `LSTMConfig`**: Specific recurrent architecture parameters (hidden dimension, layers, dropout, bidirectional options). Both inherit from `NNTrainingConfig`.

### Neural Network Architectures (`models/models.py`)
*   **`SimpleCNN`**, **`SimpleRNN`**, and **`SimpleLSTM`**: Standard PyTorch model classes.
*   *Note: These models save training dataset parameters (`mean` and `std`) as internal PyTorch parameters, baking feature scaling directly into the exported `.onnx` file. No separate scaling pipeline is needed during live inference.*

---

## 5. Model Visualization & Monitoring

### Graph Inspection (Netron)
[Netron](https://netron.app/) is the open-source standard for visualizing exported model structures. You can drag and drop any exported `.onnx` file into Netron to interactively inspect the layers, tensor dimensions, and weights.

### Metrics & Graph Tracking (TensorBoard)
If `tensorboard_log_dir` is configured in `NNTrainingConfig`, the training loop logs metrics and registers the computational graph using PyTorch's native `SummaryWriter`.
*   **Launch TensorBoard**:
    ```bash
    uv run tensorboard --logdir=runs/
    ```

---

## 6. Usage & Configuration

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
      output_selector:
        class_path: "nets.output_selectors.DynamicThresholdClassifier"
        params:
          k: 0.5
          period: 10

### Alternative Output Selector Configurations

#### 1. For Softmax Probability Models (Classification)
If your ONNX model outputs discrete class probabilities (such as softmax [DOWN, FLAT, UP]):
```yaml
      output_selector:
        class_path: "nets.output_selectors.ClassificationOutputSelector"
        params:
          class_labels:
            - "down"
            - "flat"
            - "up"
```

#### 2. For Quantile Forecasts (q10, q50, q90)
If your ONNX model outputs three conditional quantiles representing uncertainty:
```yaml
      output_selector:
        class_path: "nets.output_selectors.QuantileOutputSelector"
        params:
          threshold: 0.001
          spread_scale: 1.5
```

sizing_strategy:
  class_path: "nets.sizing.confidence_sizer.ConfidenceSizer"
  params:
    base_amount_quote: 1000.0
```

---

## 7. Development Workflow

### Adding a New Trainer
1.  Implement `BaseModelTrainer` (or inherit from `BasePyTorchTrainer`) in `training/training.py`.
2.  Define any custom architecture in `models/models.py` and its configuration schema in `models/schemas.py`.
3.  Ensure the `train()` method returns the serialized ONNX bytes.
4.  Add any new dependencies to `pyproject.toml`.

### Adding a New Output Selector
1.  Implement `BaseOutputSelector` (or inherit from `BaseRegressionOutputSelector`) in `output_selectors/output_selectors.py`.
2.  Return a `(PredictionSignal, confidence)` tuple from `select_output()`.

---

## 8. Advanced Training Features

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

## 9. Verification

Run the test suite to ensure architectural integrity:
```bash
uv run pytest
```
