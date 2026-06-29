import logging
from typing import Dict, Union

import numpy as np
import onnxruntime as ort

from ..models import ONNXModelMetadata

logger = logging.getLogger(__name__)


class ONNXPredictor:
    """
    Generic inference engine using ONNX.
    Accepts arbitrary input shapes and types, delegating preparation to the caller.
    Automatically reshapes 2D raw feature matrices based on the model's metadata.
    """

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        try:
            self.session = ort.InferenceSession(model_path)
            self.input_metadata = {i.name: i for i in self.session.get_inputs()}
            meta = self.session.get_modelmeta()
            custom_map = (
                meta.custom_metadata_map if hasattr(meta, "custom_metadata_map") else {}
            )
            try:
                self.model_metadata = ONNXModelMetadata.from_custom_metadata(custom_map)
                self.pipeline = None
                if self.model_metadata and self.model_metadata.feature_pipeline:
                    try:
                        import json

                        from trading_bot.core.transforms import BaseTransform

                        pipeline_data = json.loads(self.model_metadata.feature_pipeline)
                        self.pipeline = BaseTransform.from_dict(pipeline_data)
                        logger.info(
                            "Successfully deserialized FeaturePipeline from ONNX metadata."
                        )
                    except Exception as pe:
                        logger.warning(f"Failed to deserialize FeaturePipeline: {pe}")
            except Exception as e:
                logger.warning(
                    f"Could not parse ONNX model metadata for {model_path}: {e}"
                )
                self.model_metadata = None
                self.pipeline = None
            logger.info(f"Loaded ONNX model from {model_path}")
        except Exception as e:
            logger.critical(f"Failed to load ONNX model from {model_path}: {e}")
            raise RuntimeError(
                f"Failed to load ONNX model from {model_path}: {e}"
            ) from e

    def predict(self, inputs: Union[np.ndarray, Dict[str, np.ndarray]]) -> np.ndarray:
        """
        Performs inference on the provided inputs.
        If inputs is an ndarray, it's assumed to be the first input.
        """
        if self.session is None:
            return np.array(0.0, dtype=np.float32)

        try:
            # 1. Normalize input to Dict[str, ndarray]
            if isinstance(inputs, np.ndarray):
                first_input_name = list(self.input_metadata.keys())[0]
                expected_shape = self.input_metadata[first_input_name].shape

                # Automatically shape 2D inputs of (lookback, n_features)
                if len(inputs.shape) == 2:
                    lookback, n_features = inputs.shape
                    if len(expected_shape) == 3:
                        # PyTorch models expect (batch, n_features, lookback)
                        inputs = inputs.T.reshape(1, n_features, lookback).astype(
                            np.float32
                        )
                    elif len(expected_shape) == 2:
                        # sklearn/XGBoost models expect (batch, lookback * n_features)
                        inputs = inputs.flatten().reshape(1, -1).astype(np.float32)

                onnx_inputs = {first_input_name: inputs}
            else:
                onnx_inputs = inputs

            # 2. Run session
            outputs = self.session.run(None, onnx_inputs)

            # 3. Return the first output as ndarray
            return np.array(outputs[0], dtype=np.float32)
        except Exception as e:
            logger.error(f"Inference error with model {self.model_path}: {e}")
            return np.array(0.0, dtype=np.float32)
