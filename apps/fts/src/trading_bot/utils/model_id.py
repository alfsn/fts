import hashlib

from .run_id import calculate_params_hash, generate_backtest_run_id


def generate_model_id(onnx_bytes: bytes) -> str:
    """Generates a deterministic 12-character SHA-256 hex digest for ONNX model bytes."""
    return hashlib.sha256(onnx_bytes).hexdigest()[:12]


__all__ = ["generate_model_id", "generate_backtest_run_id", "calculate_params_hash"]
