import hashlib
import json
from typing import Any, Dict, Optional


def calculate_params_hash(params: Optional[Dict[str, Any]]) -> str:
    """
    Computes a deterministic 6-character SHA-256 hex digest for a dictionary of parameters.
    Returns empty string if params is None or empty.
    """
    if not params:
        return ""
    serialized = json.dumps(params, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()[:6]


def generate_backtest_run_id(
    model_id: str,
    test_dataset_sha: str,
    params: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Generates a deterministic backtest run identifier.

    Format: `bt_{model_id[:8]}_{test_dataset_sha[:6]}_{param_hash[:6]}`
    (or `bt_{model_id[:8]}_{test_dataset_sha[:6]}` if params is None/empty).

    :param model_id: Non-empty model identifier.
    :param test_dataset_sha: Non-empty test dataset SHA-256 hash.
    :param params: Optional parameter dictionary for execution/strategy settings.
    :return: Formatted backtest run ID string.
    :raises ValueError: If model_id or test_dataset_sha is missing or empty.
    """
    if not model_id or not str(model_id).strip():
        raise ValueError("model_id must be a non-empty string")
    if not test_dataset_sha or not str(test_dataset_sha).strip():
        raise ValueError("test_dataset_sha must be a non-empty string")

    m_id = str(model_id).strip()[:8]
    d_sha = str(test_dataset_sha).strip()[:6]
    p_hash = calculate_params_hash(params)

    if p_hash:
        return f"bt_{m_id}_{d_sha}_{p_hash[:6]}"
    return f"bt_{m_id}_{d_sha}"
