import hashlib


def generate_model_id(onnx_bytes: bytes) -> str:
    """Generates a deterministic 12-character SHA-256 hex digest for ONNX model bytes."""
    return hashlib.sha256(onnx_bytes).hexdigest()[:12]
