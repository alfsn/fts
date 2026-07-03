from typing import Dict, Tuple, Type

from nets.models import (
    BaseTrainerConfig,
    CNNConfig,
    LSTMConfig,
    NNTrainingConfig,
    RNNConfig,
)
from nets.training import (
    CNNTrainer,
    LinearRegressionTrainer,
    LSTMTrainer,
    RNNTrainer,
    XGBoostTrainer,
)

# OCP Trainer & Config Registry Map
TRAINER_REGISTRY: Dict[str, Tuple[Type, Type]] = {
    "lstm": (LSTMTrainer, LSTMConfig),
    "rnn": (RNNTrainer, RNNConfig),
    "cnn": (CNNTrainer, CNNConfig),
    "linear_regression": (LinearRegressionTrainer, BaseTrainerConfig),
    "xgboost": (XGBoostTrainer, BaseTrainerConfig),
}


def get_trainer_and_config(model_type: str) -> Tuple[Type, Type]:
    """Retrieves the Trainer class and Config model class for a given model type."""
    if model_type not in TRAINER_REGISTRY:
        raise ValueError(
            f"Model type '{model_type}' is not registered in TRAINER_REGISTRY. "
            f"Available types: {list(TRAINER_REGISTRY.keys())}"
        )
    return TRAINER_REGISTRY[model_type]
