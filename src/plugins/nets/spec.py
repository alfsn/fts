# src/plugins/nets/spec.py

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import yaml
from pydantic import BaseModel, Field


class HParamStudySpec(BaseModel):
    """
    Specification model for hyperparameter search & optimization studies.
    Parses study bounds, market selection, dataset dates, feature transformation specs,
    and Optuna search space definitions.
    """

    # Optuna Study Parameters
    study_name: str = Field(..., description="Unique name identifier for Optuna study")
    direction: Literal["minimize", "maximize"] = Field(
        default="minimize",
        description="Optimization direction (minimize validation loss)",
    )
    n_trials: int = Field(
        default=10, ge=1, description="Number of Optuna search trials"
    )

    # Target Model Architecture & Market Selection
    model_type: str = Field(
        default="lstm", description="Target model architecture name"
    )
    market_id: str = Field(default="BTC/USDT", description="Target market pair")
    interval: str = Field(default="30m", description="Bar aggregation timeframe")

    # Training Data Range Boundaries
    start_date: Optional[datetime] = Field(
        default=None, description="Start date timestamp for historical training data"
    )
    end_date: Optional[datetime] = Field(
        default=None, description="End date timestamp for historical training data"
    )

    # Feature & Transformation Parameters
    lookback_period: int = Field(
        default=20, ge=1, description="Lookback window size in bars"
    )
    feature_cols: List[str] = Field(
        default_factory=lambda: ["close"],
        description="Raw OHLCV attributes used for feature engineering",
    )
    feature_pipeline: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Serialized BaseTransform/FeaturePipeline dict specification",
    )

    # Optuna Search Space Definitions
    search_space: Dict[str, Any] = Field(
        default_factory=dict,
        description="Hyperparameter ranges and distributions for Optuna trial sampling",
    )

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "HParamStudySpec":
        """Loads and validates an HParamStudySpec from a YAML specification file."""
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"HParam study spec file not found at: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return cls(**(data or {}))
