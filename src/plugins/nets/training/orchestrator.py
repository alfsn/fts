import inspect
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Union

from nets.models import NNTrainingConfig
from nets.training.registry import get_trainer_and_config
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from trading_bot.config import settings
from trading_bot.core.database import SessionLocal, create_db_engine, init_db
from trading_bot.core.dataset import calculate_dataset_hash
from trading_bot.core.repository import MarketDataRepository, ModelRepository
from trading_bot.core.schemas import BarData
from trading_bot.core.transforms import BaseTransform
from trading_bot.utils.model_id import generate_model_id

logger = logging.getLogger(__name__)


def parse_datetime_param(val: Optional[Union[str, datetime]]) -> Optional[datetime]:
    if val is None:
        return None
    if isinstance(val, datetime):
        if val.tzinfo is None:
            return val.replace(tzinfo=timezone.utc)
        return val
    if isinstance(val, str):
        dt = datetime.fromisoformat(val)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt
    return None


@dataclass
class TrainingResult:
    model_id: str
    onnx_path: str
    val_ic: Optional[float]
    val_loss: Optional[float]
    model_type: str
    model_config: Dict[str, Any]
    training_config: Dict[str, Any]


def train_and_register_candidate(
    model_type: str,
    market_id: str,
    interval: str,
    lookback_period: int,
    feature_cols: list,
    start_date: Union[str, datetime],
    end_date: Union[str, datetime],
    model_params: Dict[str, Any],
    train_params: Optional[Dict[str, Any]] = None,
    feature_pipeline: Optional[Dict[str, Any]] = None,
    run_id: Optional[str] = None,
    status: str = "candidate",
    horizon: int = 1,
) -> TrainingResult:
    """
    SOLID & DRY Orchestrator:
    Loads market data, trains a model candidate using dynamic registry lookups,
    saves the ONNX artifact, and registers metadata in ModelRegistryLog.
    """
    train_params = train_params or {}

    init_db(extra_models=["trading_bot.core.models"])

    start_dt = parse_datetime_param(start_date)
    end_dt = parse_datetime_param(end_date)

    # 1. Fetch bar data & register dataset
    with SessionLocal() as db:
        market_repo = MarketDataRepository(db)
        raw_bars = market_repo.get_bars(
            market_id,
            interval=interval,
            start_date=start_dt,
            end_date=end_dt,
        )
        if not raw_bars:
            raise ValueError(f"No bar data found in database for market {market_id}")

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

        sorted_raw_bars = sorted(raw_bars, key=lambda b: b.timestamp)
        dataset_full_hash = calculate_dataset_hash(sorted_raw_bars)
        model_repo = ModelRepository(db)
        dataset_obj = model_repo.get_or_create_dataset(
            market_id=market_id,
            interval=interval,
            start_time=sorted_raw_bars[0].timestamp,
            end_time=sorted_raw_bars[-1].timestamp,
            hash_val=dataset_full_hash,
        )
        db.commit()
        dataset_id = dataset_obj.dataset_id

    # 2. Lookup Trainer & Config Class from Registry (OCP)
    trainer_cls, config_cls = get_trainer_and_config(model_type)

    # 3. Build Config Pydantic objects & Transform
    full_train_params = train_params.copy()
    full_train_params["feature_cols"] = feature_cols
    if "validation_split" not in full_train_params:
        full_train_params["validation_split"] = 0.2

    training_config_obj = NNTrainingConfig(**full_train_params)
    model_config_obj = config_cls(**model_params)

    transform = None
    if feature_pipeline:
        transform = BaseTransform.from_dict(feature_pipeline)

    # Instantiate Trainer dynamically
    sig = inspect.signature(trainer_cls.__init__)
    trainer_kwargs = {"lookback_period": lookback_period}
    if "model_config" in sig.parameters:
        trainer_kwargs["model_config"] = model_config_obj
        trainer_kwargs["training_config"] = training_config_obj
    else:
        trainer_kwargs["config"] = model_config_obj

    if "transform" in sig.parameters and transform is not None:
        trainer_kwargs["transform"] = transform

    trainer = trainer_cls(**trainer_kwargs)

    # 4. Train Model
    onnx_bytes = trainer.train(bar_schemas)
    if not onnx_bytes:
        raise RuntimeError("Model training failed or returned empty ONNX bytes.")

    # 5. Save ONNX Artifact & Register candidate
    model_id = generate_model_id(onnx_bytes)
    registry_dir = os.path.join(settings.MODELS_DIR, "registry", "trials")
    os.makedirs(registry_dir, exist_ok=True)
    onnx_filename = os.path.join(registry_dir, f"{model_id}.onnx")
    with open(onnx_filename, "wb") as f:
        f.write(onnx_bytes)

    val_ic = getattr(trainer, "last_val_ic", None)
    val_loss = getattr(trainer, "last_val_loss", None)

    with SessionLocal() as db:
        model_repo = ModelRepository(db)
        model_repo.register_model(
            model_id=model_id,
            run_id=run_id,
            model_type=model_type,
            market_id=market_id,
            interval=interval,
            horizon=horizon,
            onnx_path=onnx_filename,
            hyperparameters={
                **model_config_obj.model_dump(),
                **training_config_obj.model_dump(),
            },
            metrics={"val_ic": val_ic, "val_loss": val_loss},
            status=status,
            dataset_id=dataset_id,
        )
        db.commit()

    logger.info(f"Model {model_id} ({model_type}) successfully trained & registered.")

    return TrainingResult(
        model_id=model_id,
        onnx_path=onnx_filename,
        val_ic=val_ic,
        val_loss=val_loss,
        model_type=model_type,
        model_config=model_config_obj.model_dump(),
        training_config=training_config_obj.model_dump(),
    )
