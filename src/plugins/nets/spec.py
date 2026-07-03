# src/plugins/nets/spec.py

from typing import Any, Dict

from pydantic import Field

from trading_bot.core.spec_base import BaseComposableSpec
from trading_bot.core.spec_models import (
    DateRangeSpec,
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
