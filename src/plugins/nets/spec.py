# src/plugins/nets/spec.py

from typing import Any, Dict, List

from pydantic import Field

from trading_bot.core.spec_base import BaseComposableSpec
from trading_bot.core.spec_models import (
    DateRangeSpec,
    ExecutionRiskSpec,
    FeatureSetSpec,
    MarketSpec,
    OptunaStudySpec,
)


class HParamStudySpec(BaseComposableSpec):
    """
    Specification model for hyperparameter search & optimization studies.
    Composes Optuna study bounds, market selection, dataset dates, feature transformation specs,
    and Optuna search space definitions.
    """

    study: OptunaStudySpec = Field(default_factory=OptunaStudySpec)
    market: MarketSpec = Field(default_factory=MarketSpec)
    dates: DateRangeSpec = Field(default_factory=DateRangeSpec)
    features: FeatureSetSpec = Field(default_factory=FeatureSetSpec)
    search_space: Dict[str, Any] = Field(
        default_factory=dict,
        description="Hyperparameter ranges and distributions for Optuna trial sampling",
    )


class SweepSpec(BaseComposableSpec):
    """
    Specification model for controlled parameter sweep experiments.
    Composes market selection, feature set parameters, date ranges, sweep parameters,
    base model/training parameters, and execution risk parameters.
    """

    sweep_name: str = Field(
        ..., description="Unique name identifier for parameter sweep"
    )
    model_type: str = Field(
        default="lstm", description="Target model architecture name"
    )
    sweep_param: str = Field(
        ..., description="Hyperparameter name to sweep across grid"
    )
    sweep_values: List[Any] = Field(
        ..., description="List of parameter values to sweep"
    )

    market: MarketSpec = Field(default_factory=MarketSpec)
    features: FeatureSetSpec = Field(default_factory=FeatureSetSpec)
    train_dates: DateRangeSpec = Field(default_factory=DateRangeSpec)
    test_dates: DateRangeSpec = Field(default_factory=DateRangeSpec)
    execution: ExecutionRiskSpec = Field(default_factory=ExecutionRiskSpec)

    base_model_params: Dict[str, Any] = Field(
        default_factory=dict, description="Fixed non-swept model hyperparameters"
    )
    base_train_params: Dict[str, Any] = Field(
        default_factory=dict, description="Fixed non-swept training parameters"
    )
