import logging
from typing import Dict, Union

import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)


class ONNXPredictor:
    """
    Generic inference engine using ONNX.
    Accepts arbitrary input shapes and types, delegating preparation to the caller.
    """

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        try:
            self.session = ort.InferenceSession(model_path)
            self.input_metadata = {i.name: i for i in self.session.get_inputs()}
            logger.info(f"Loaded ONNX model from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load ONNX model from {model_path}: {e}")
            self.session = None

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
