import argparse
import logging
import os
import uuid
from datetime import datetime, timezone

import optuna
import yaml
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

from trading_bot.config import settings
from trading_bot.core.database import init_db
from trading_bot.core.repository import MarketDataRepository, ModelRepository

logger = logging.getLogger(__name__)

# OCP Trainer Registry Map
TRAINER_REGISTRY = {
    "lstm": (LSTMTrainer, LSTMConfig),
    "rnn": (RNNTrainer, RNNConfig),
    "cnn": (CNNTrainer, CNNConfig),
    "linear_regression": (LinearRegressionTrainer, BaseTrainerConfig),
    "xgboost": (XGBoostTrainer, BaseTrainerConfig),
}


def generate_model_id(model_type: str, market_id: str, interval: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    rand_id = uuid.uuid4().hex[:6]
    clean_market = market_id.replace("_", "").lower()
    return f"model_{model_type}_{clean_market}_{interval.lower()}_{timestamp}_{rand_id}"


def parse_search_space(trial: optuna.Trial, space_config: dict) -> dict:
    params = {}
    for param_name, cfg in space_config.items():
        cfg_type = cfg["type"]
        if cfg_type == "float":
            params[param_name] = trial.suggest_float(
                param_name, cfg["low"], cfg["high"], log=cfg.get("log", False)
            )
        elif cfg_type == "int":
            params[param_name] = trial.suggest_int(param_name, cfg["low"], cfg["high"])
        elif cfg_type == "categorical":
            params[param_name] = trial.suggest_categorical(param_name, cfg["choices"])
    return params


def run_hparam_search(config_path: str):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Dynamic SQLite engine and SessionLocal bound at run-time
    engine = create_engine(settings.DATABASE_URL, pool_pre_ping=True)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    # Initialize DB (creates model_registry table if not exists)
    init_db(extra_models=["trading_bot.core.models"], bind_engine=engine)

    # Load data ONCE with short-lived session to prevent session leak
    market_id = config["market_id"]
    interval = config.get("interval", "1h")
    with SessionLocal() as db:
        market_repo = MarketDataRepository(db)
        raw_bars = market_repo.get_bars(market_id, interval=interval)
        if not raw_bars:
            raise ValueError(f"No bar data found in database for market {market_id}")

        # Map BarData logs back to schema structures
        from trading_bot.core.schemas import BarData

        bar_schemas = [
            BarData(
                timestamp=b.timestamp,
                open=b.open,
                high=b.high,
                low=b.low,
                close=b.close,
                volume=b.volume,
                bar_type=b.bar_type,
                ticks_count=b.ticks_count,
                dollar_volume=b.dollar_volume,
                interval=b.interval,
            )
            for b in raw_bars
        ]

    model_type = config["model_type"]
    if model_type not in TRAINER_REGISTRY:
        raise ValueError(f"Model type {model_type} not found in TRAINER_REGISTRY.")

    trainer_cls, config_cls = TRAINER_REGISTRY[model_type]

    def objective(trial: optuna.Trial) -> float:
        trial_params = parse_search_space(trial, config.get("search_space") or {})
        run_id = f"trial_{trial.number}_{uuid.uuid4().hex[:8]}"

        # Separate NNTrainingConfig vs architecture config_cls
        train_params = {
            k: v for k, v in trial_params.items() if k in NNTrainingConfig.model_fields
        }
        train_params["tensorboard_log_dir"] = (
            f"runs/optuna/{config['study_name']}/{run_id}"
        )
        train_params["feature_cols"] = config["feature_cols"]
        train_params["validation_split"] = 0.2
        training_config = NNTrainingConfig(**train_params)

        model_params = {
            k: v for k, v in trial_params.items() if k in config_cls.model_fields
        }
        model_config = config_cls(**model_params)

        import inspect

        sig = inspect.signature(trainer_cls.__init__)
        if "model_config" in sig.parameters:
            trainer = trainer_cls(
                lookback_period=config["lookback_period"],
                model_config=model_config,
                training_config=training_config,
            )
        else:
            trainer = trainer_cls(
                lookback_period=config["lookback_period"],
                config=model_config,
            )

        # Train model
        onnx_bytes = trainer.train(bar_schemas)
        if not onnx_bytes:
            raise optuna.TrialPruned("Training failed or returned empty bytes.")

        # Save ONNX artifact
        os.makedirs("models/registry", exist_ok=True)
        model_id = generate_model_id(model_type, market_id, interval)
        onnx_filename = f"models/registry/{model_id}.onnx"
        with open(onnx_filename, "wb") as f:
            f.write(onnx_bytes)

        # Retrieve validation metrics
        val_loss = getattr(trainer, "best_val_loss", 999.0)
        metrics = {
            "val_loss": val_loss,
            "ic": (
                trainer.best_val_metrics.get("ic", 0.0)
                if getattr(trainer, "best_val_metrics", None)
                else 0.0
            ),
            "directional_accuracy": (
                trainer.best_val_metrics.get("directional_accuracy", 0.5)
                if getattr(trainer, "best_val_metrics", None)
                else 0.5
            ),
        }

        # Log details to SQLite Model Registry inside a context-managed session per trial
        with SessionLocal() as trial_db:
            model_repo = ModelRepository(trial_db)
            try:
                model_repo.register_model(
                    model_id=model_id,
                    model_type=model_type,
                    market_id=market_id,
                    interval=interval,
                    horizon=training_config.horizon,
                    onnx_path=onnx_filename,
                    hyperparameters=trial_params,
                    metrics=metrics,
                    run_id=run_id,
                    status="candidate",
                )
                trial_db.commit()  # Decoupled commit managed at caller level
            except Exception as e:
                trial_db.rollback()  # Decoupled rollback managed at caller level
                logger.error(f"Failed to register model in database: {e}")
                raise e

        return val_loss

    # Dynamic Optuna storage path mapping
    optuna_storage = settings.DATABASE_URL.replace("sqlite+pysqlite://", "sqlite://")

    study = optuna.create_study(
        study_name=config["study_name"],
        direction=config["direction"],
        storage=optuna_storage,  # Coherent settings database
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=config["n_trials"])
    print(f"Best trial: {study.best_trial.number} with loss {study.best_value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/hparam_search.yaml")
    args = parser.parse_args()
    run_hparam_search(args.config)
