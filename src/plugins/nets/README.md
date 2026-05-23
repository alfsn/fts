# Nets Plugin (Forecasting & ML)

The `nets` plugin provides a standardized, S.O.L.I.D. framework for implementing machine learning based forecasting strategies within the FTS (Financial Trading System) core. It leverages **ONNX** for cross-framework inference and a **UFD (Up/Flat/Down)** classification system for robust signal generation.

## 1. Architecture & Design

The plugin follows a decoupled pipeline to ensure that trading logic is isolated from ML model specifics:

1.  **Structural Conversion**: `DatasetBuilder` in core converts raw bars into feature matrices and sliding windows.
2.  **Inference (ONNX)**: An `.onnx` model (often an exported `sklearn.pipeline.Pipeline`) predicts the next return. This standardized approach allows models trained in PyTorch, XGBoost, or Scikit-Learn to be used interchangeably, often with preprocessing like scaling baked directly into the artifact.
3.  **Classification (UFD)**: The predicted return is classified as `UP`, `FLAT`, or `DOWN` based on dynamic or fixed thresholds.
4.  **Signal Generation**: The classification is mapped to a `TradeSignal` (BUY, SELL, or FLAT).

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

## 3. Usage & Configuration

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
          model_path: "models/my_xgboost_model.onnx"
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

## 4. Development Workflow

### Adding a New Trainer
1.  Implement `BaseModelTrainer` in `training.py`.
2.  Ensure the `train()` method returns the serialized ONNX bytes.
3.  Add any new dependencies to `pyproject.toml`.

### Adding a New Classifier
1.  Implement `BaseClassifier` in `classifiers.py`.
2.  Return a `PredictionSignal` (UP, FLAT, DOWN).

## 5. Verification

Run the test suite to ensure architectural integrity:
```bash
uv run pytest tests/unit/test_nets_plugin.py tests/unit/test_nets_integration.py tests/unit/test_argentina_plugin.py
```

