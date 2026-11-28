# src/trading_bot/core/repository.py

import logging
from datetime import datetime
from typing import Optional, Sequence

from .models import ModelRegistryLog, TimeSeriesDataset

logger = logging.getLogger(__name__)


from quant_data.db.repositories import BaseRepository


class ModelRepository(BaseRepository):
    """Encapsulates database operations for the Model Registry. Decoupled from transaction control."""

    def get_or_create_dataset(
        self,
        instrument_id: str,
        interval: str,
        start_time: datetime,
        end_time: datetime,
        hash_val: str,
    ) -> TimeSeriesDataset:
        """Retrieves an existing dataset by its content hash or creates a new entry."""
        existing = self.db.query(TimeSeriesDataset).filter_by(hash=hash_val).first()
        if existing:
            return existing

        dataset_id = f"ds_{hash_val[:12]}"
        dataset = TimeSeriesDataset(
            dataset_id=dataset_id,
            instrument_id=instrument_id,
            interval=interval,
            start_time=start_time,
            end_time=end_time,
            hash=hash_val,
        )
        self.db.add(dataset)
        return dataset

    def register_model(
        self,
        model_id: str,
        model_type: str,
        instrument_id: str,
        interval: str,
        horizon: int,
        onnx_path: str,
        hyperparameters: dict,
        metrics: dict,
        run_id: Optional[str] = None,
        status: str = "candidate",
        dataset_id: Optional[str] = None,
    ) -> ModelRegistryLog:
        """Adds a trained model metadata entry to the registry session idempotently."""
        existing = self.db.query(ModelRegistryLog).filter_by(model_id=model_id).first()
        if existing:
            logger.info(
                f"Model '{model_id}' already registered in model_registry. "
                "Returning existing entry."
            )
            return existing

        log_entry = ModelRegistryLog(
            model_id=model_id,
            run_id=run_id,
            model_type=model_type,
            instrument_id=instrument_id,
            interval=interval,
            horizon=horizon,
            onnx_path=onnx_path,
            hyperparameters=hyperparameters,
            metrics=metrics,
            status=status,
            dataset_id=dataset_id,
        )
        self.db.add(log_entry)
        return log_entry

    def get_production_model(
        self,
        model_type: str,
        instrument_id: str,
        interval: str,
        horizon: int,
        feature_cols: Optional[Sequence[str]] = None,
    ) -> Optional[ModelRegistryLog]:
        """Fetches the active 'production' model matching the logical signature and optional feature_cols."""
        query = (
            self.db.query(ModelRegistryLog)
            .filter_by(
                model_type=model_type,
                instrument_id=instrument_id,
                interval=interval,
                horizon=horizon,
                status="production",
            )
            .order_by(ModelRegistryLog.created_at.desc())
        )
        candidates = query.all()
        if not candidates:
            return None

        if feature_cols is None:
            return candidates[0]

        target_features = list(feature_cols)
        for model in candidates:
            if model.hyperparameters and isinstance(model.hyperparameters, dict):
                model_features = model.hyperparameters.get("feature_cols")
                if (
                    model_features is not None
                    and list(model_features) == target_features
                ):
                    return model
        return None

    def get_candidate_model(
        self,
        model_type: str,
        instrument_id: str,
        interval: Optional[str] = None,
        horizon: Optional[int] = None,
        feature_cols: Optional[Sequence[str]] = None,
    ) -> Optional[ModelRegistryLog]:
        """Fetches the latest candidate model matching the logical signature and optional feature_cols."""
        query = self.db.query(ModelRegistryLog).filter_by(
            model_type=model_type,
            instrument_id=instrument_id,
            status="candidate",
        )
        if interval:
            query = query.filter_by(interval=interval)
        if horizon:
            query = query.filter_by(horizon=horizon)

        query = query.order_by(ModelRegistryLog.created_at.desc())
        candidates = query.all()
        if not candidates:
            return None

        if feature_cols is None:
            return candidates[0]

        target_features = list(feature_cols)
        for model in candidates:
            if model.hyperparameters and isinstance(model.hyperparameters, dict):
                model_features = model.hyperparameters.get("feature_cols")
                if (
                    model_features is not None
                    and list(model_features) == target_features
                ):
                    return model
        return None

    def get_model(self, model_id: str) -> Optional[ModelRegistryLog]:
        """Fetches a model by its unique model ID."""
        return self.db.query(ModelRegistryLog).filter_by(model_id=model_id).first()

    def promote_to_production(self, model_id: str) -> None:
        """Promotes a candidate model, moves its ONNX file to permanent registry,
        archives previous production models, and cleans up other candidate files/records.
        """
        model = self.get_model(model_id)
        if not model:
            raise ValueError(f"Model with model_id {model_id} not found.")

        import os
        import shutil

        # Move ONNX file from trials/ to permanent registry if applicable
        source_path = model.onnx_path
        if source_path and "registry/trials" in source_path:
            # permanent registry is the parent directory of registry/trials/
            registry_dir = os.path.dirname(os.path.dirname(source_path))
            permanent_path = os.path.join(registry_dir, f"{model_id}.onnx")

            if os.path.exists(source_path):
                shutil.move(source_path, permanent_path)
                model.onnx_path = permanent_path
            else:
                logger.warning(
                    f"ONNX file not found at expected trial path: {source_path}"
                )

        # Demote current production model(s) with matching logical signature
        self.db.query(ModelRegistryLog).filter_by(
            model_type=model.model_type,
            instrument_id=model.instrument_id,
            interval=model.interval,
            horizon=model.horizon,
            status="production",
        ).update({"status": "archived"})

        # Promote this model
        model.status = "production"

        # Clean up files of other candidates with the same logical signature
        other_candidates = (
            self.db.query(ModelRegistryLog)
            .filter(
                ModelRegistryLog.model_type == model.model_type,
                ModelRegistryLog.instrument_id == model.instrument_id,
                ModelRegistryLog.interval == model.interval,
                ModelRegistryLog.horizon == model.horizon,
                ModelRegistryLog.model_id != model_id,
                ModelRegistryLog.status == "candidate",
            )
            .all()
        )

        for candidate in other_candidates:
            if candidate.onnx_path and os.path.exists(candidate.onnx_path):
                try:
                    os.remove(candidate.onnx_path)
                except Exception as e:
                    logger.warning(
                        f"Failed to remove unpromoted candidate file {candidate.onnx_path}: {e}"
                    )
            self.db.delete(candidate)
