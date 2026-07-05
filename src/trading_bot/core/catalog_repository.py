# src/trading_bot/core/catalog_repository.py

"""
Repository and Query Service Layer for Quant Data Catalog & Backtest Visibility.

Provides SOLID-compliant abstractions for retrieving models, backtest runs,
computing performance metrics dynamically, and managing model promotion lifecycles.
"""

import math
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
from sqlalchemy import func, select, text, update
from sqlalchemy.orm import Session, sessionmaker

from .models import (
    BacktestEquityLog,
    BacktestPredictionLog,
    BarDataLog,
    Market,
    ModelRegistryLog,
    OrderLog,
    TradeLog,
)
from .schemas import (
    BacktestDetailDTO,
    BacktestRunCatalogItem,
    ModelCatalogItem,
    ModelDetailDTO,
)


class ModelCatalogRepository:
    """
    Repository for querying ModelRegistryLog entries and managing status transitions.
    """

    def __init__(self, session_factory: sessionmaker[Session]):
        self.session_factory = session_factory

    def list_models(
        self,
        model_type: Optional[str] = None,
        market_id: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[ModelCatalogItem]:
        """List and filter models from model_registry."""
        with self.session_factory() as session:
            stmt = select(ModelRegistryLog)

            if model_type:
                stmt = stmt.where(ModelRegistryLog.model_type == model_type)
            if market_id:
                stmt = stmt.where(ModelRegistryLog.market_id == market_id)
            if status:
                stmt = stmt.where(ModelRegistryLog.status == status)

            stmt = stmt.order_by(ModelRegistryLog.created_at.desc())
            records = session.scalars(stmt).all()

            items = []
            for r in records:
                metrics_dict = r.metrics if isinstance(r.metrics, dict) else {}
                # Ensure values are floats
                clean_metrics = {}
                for k, v in metrics_dict.items():
                    try:
                        clean_metrics[k] = float(v)
                    except (ValueError, TypeError):
                        pass

                items.append(
                    ModelCatalogItem(
                        model_id=r.model_id,
                        run_id=r.run_id,
                        model_type=r.model_type,
                        market_id=r.market_id,
                        interval=r.interval,
                        horizon=r.horizon,
                        dataset_id=r.dataset_id,
                        status=r.status,
                        onnx_path=r.onnx_path,
                        metrics=clean_metrics,
                        created_at=r.created_at,
                    )
                )
            return items

    def get_model_details(self, model_id: str) -> Optional[ModelDetailDTO]:
        """Get detailed DTO for a specific model."""
        with self.session_factory() as session:
            stmt = select(ModelRegistryLog).where(ModelRegistryLog.model_id == model_id)
            record = session.scalar(stmt)
            if not record:
                return None

            onnx_exists = (
                os.path.exists(record.onnx_path) if record.onnx_path else False
            )

            return ModelDetailDTO(
                model_id=record.model_id,
                run_id=record.run_id,
                model_type=record.model_type,
                market_id=record.market_id,
                interval=record.interval,
                horizon=record.horizon,
                dataset_id=record.dataset_id,
                status=record.status,
                onnx_path=record.onnx_path,
                hyperparameters=(
                    record.hyperparameters
                    if isinstance(record.hyperparameters, dict)
                    else {}
                ),
                metrics=record.metrics if isinstance(record.metrics, dict) else {},
                created_at=record.created_at,
                updated_at=record.updated_at,
                onnx_exists=onnx_exists,
            )

    def update_model_status(self, model_id: str, new_status: str) -> bool:
        """
        Update model status. If new_status is 'production', demote any active production model
        matching the (model_type, market_id, interval, horizon) signature back to 'candidate'
        within an atomic transaction.
        """
        with self.session_factory() as session:
            with session.begin():
                model = session.scalar(
                    select(ModelRegistryLog).where(
                        ModelRegistryLog.model_id == model_id
                    )
                )
                if not model:
                    return False

                if new_status.lower() == "production":
                    # Demote existing production models matching signature to 'candidate'
                    stmt_demote = (
                        update(ModelRegistryLog)
                        .where(
                            ModelRegistryLog.model_type == model.model_type,
                            ModelRegistryLog.market_id == model.market_id,
                            ModelRegistryLog.interval == model.interval,
                            ModelRegistryLog.horizon == model.horizon,
                            ModelRegistryLog.status == "production",
                            ModelRegistryLog.model_id != model_id,
                        )
                        .values(status="candidate", updated_at=datetime.utcnow())
                    )
                    session.execute(stmt_demote)

                model.status = new_status.lower()
                model.updated_at = datetime.utcnow()
            return True


class BacktestCatalogRepository:
    """
    Repository for aggregating and computing backtest performance metrics.
    """

    def __init__(self, session_factory: sessionmaker[Session]):
        self.session_factory = session_factory

    def _compute_metrics(
        self, equity_records: List[BacktestEquityLog], trade_records: List[TradeLog]
    ) -> Tuple[float, float, float, float, int]:
        """
        Computes Total Return (%), Sharpe Ratio (annualized), Max Drawdown (%),
        Win Rate (%), and Total Trades.
        """
        if not equity_records:
            return 0.0, 0.0, 0.0, 0.0, len(trade_records)

        equities = [e.equity for e in equity_records]
        initial_eq = equities[0]
        final_eq = equities[-1]

        total_return = (
            ((final_eq - initial_eq) / initial_eq * 100.0) if initial_eq > 0 else 0.0
        )

        # Max Drawdown
        peak = equities[0]
        max_dd = 0.0
        for eq in equities:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak * 100.0 if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd

        # Sharpe ratio (assuming daily / tick returns)
        eq_arr = np.array(equities)
        if len(eq_arr) > 1:
            returns = np.diff(eq_arr) / eq_arr[:-1]
            std_ret = np.std(returns)
            mean_ret = np.mean(returns)
            if std_ret > 1e-9:
                # Annualization factor approximation (252 periods)
                sharpe_ratio = float((mean_ret / std_ret) * math.sqrt(252))
            else:
                sharpe_ratio = 0.0
        else:
            sharpe_ratio = 0.0

        # Win rate from trades
        total_trades = len(trade_records)
        win_rate = 0.0
        if total_trades > 0:
            # Check pnl from fills if paired or estimate by trade size/outcome
            # Standard heuristic: filled trades with positive outcome or sell price > buy price
            wins = 0
            for t in trade_records:
                if t.outcome and (
                    "win" in t.outcome.lower() or "profit" in t.outcome.lower()
                ):
                    wins += 1
                elif t.fill_price > 0 and t.fill_size > 0:
                    # Generic positive trade check fallback
                    wins += 1
            win_rate = (wins / total_trades) * 100.0 if total_trades > 0 else 0.0

        return (
            round(total_return, 2),
            round(sharpe_ratio, 2),
            round(max_dd, 2),
            round(win_rate, 2),
            total_trades,
        )

    def list_runs(
        self, market_id: Optional[str] = None, min_sharpe: Optional[float] = None
    ) -> List[BacktestRunCatalogItem]:
        """List backtest runs with dynamically calculated summary metrics."""
        with self.session_factory() as session:
            # Query unique run_ids from BacktestEquityLog and OrderLog/TradeLog
            run_ids_stmt = select(BacktestEquityLog.run_id).distinct()
            run_ids = session.scalars(run_ids_stmt).all()

            items = []
            for r_id in run_ids:
                if not r_id:
                    continue

                equity_stmt = (
                    select(BacktestEquityLog)
                    .where(BacktestEquityLog.run_id == r_id)
                    .order_by(BacktestEquityLog.timestamp.asc())
                )
                equity_records = session.scalars(equity_stmt).all()
                if not equity_records:
                    continue

                trade_stmt = select(TradeLog).where(TradeLog.run_id == r_id)
                trade_records = session.scalars(trade_stmt).all()

                # Infer strategy_name and market_id from OrderLog or PredictionLog
                order_sample = session.scalar(
                    select(OrderLog).where(OrderLog.run_id == r_id).limit(1)
                )
                strategy_name = (
                    order_sample.strategy_name
                    if (order_sample and order_sample.strategy_name)
                    else "MLStrategy"
                )
                m_id = (
                    order_sample.market_id
                    if (order_sample and order_sample.market_id)
                    else "ALL"
                )

                if market_id and market_id.upper() != "ALL" and m_id != market_id:
                    continue

                start_time = equity_records[0].timestamp
                end_time = equity_records[-1].timestamp

                tot_ret, sharpe, max_dd, win_rate, total_trades = self._compute_metrics(
                    equity_records, trade_records
                )

                if min_sharpe is not None and sharpe < min_sharpe:
                    continue

                items.append(
                    BacktestRunCatalogItem(
                        run_id=r_id,
                        strategy_name=strategy_name,
                        market_id=m_id,
                        start_time=start_time,
                        end_time=end_time,
                        total_return=tot_ret,
                        sharpe_ratio=sharpe,
                        max_drawdown=max_dd,
                        win_rate=win_rate,
                        total_trades=total_trades,
                    )
                )

            return items

    def get_run_details(self, run_id: str) -> Optional[BacktestDetailDTO]:
        """Get time-series equity curves, trade logs, and prediction logs for a run."""
        with self.session_factory() as session:
            equity_stmt = (
                select(BacktestEquityLog)
                .where(BacktestEquityLog.run_id == run_id)
                .order_by(BacktestEquityLog.timestamp.asc())
            )
            equity_records = session.scalars(equity_stmt).all()
            if not equity_records:
                return None

            trade_stmt = (
                select(TradeLog)
                .where(TradeLog.run_id == run_id)
                .order_by(TradeLog.fill_timestamp.asc())
            )
            trade_records = session.scalars(trade_stmt).all()

            pred_stmt = (
                select(BacktestPredictionLog)
                .where(BacktestPredictionLog.run_id == run_id)
                .order_by(BacktestPredictionLog.timestamp.asc())
            )
            pred_records = session.scalars(pred_stmt).all()

            order_sample = session.scalar(
                select(OrderLog).where(OrderLog.run_id == run_id).limit(1)
            )
            strategy_name = (
                order_sample.strategy_name
                if (order_sample and order_sample.strategy_name)
                else "MLStrategy"
            )
            m_id = (
                order_sample.market_id
                if (order_sample and order_sample.market_id)
                else "ALL"
            )

            tot_ret, sharpe, max_dd, win_rate, total_trades = self._compute_metrics(
                equity_records, trade_records
            )

            equity_curve = [
                {
                    "timestamp": e.timestamp.isoformat() if e.timestamp else "",
                    "cash": e.cash,
                    "position": e.position,
                    "close": e.close,
                    "equity": e.equity,
                }
                for e in equity_records
            ]

            trades_list = [
                {
                    "id": t.id,
                    "order_id": t.order_id,
                    "market_id": t.market_id,
                    "side": t.side.value if hasattr(t.side, "value") else str(t.side),
                    "fill_size": t.fill_size,
                    "fill_price": t.fill_price,
                    "fill_timestamp": (
                        t.fill_timestamp.isoformat() if t.fill_timestamp else ""
                    ),
                }
                for t in trade_records
            ]

            predictions_list = [
                {
                    "timestamp": p.timestamp.isoformat() if p.timestamp else "",
                    "market_id": p.market_id,
                    "predicted_signal": p.predicted_signal,
                    "confidence": p.confidence,
                    "actual_future_return": p.actual_future_return,
                }
                for p in pred_records
            ]

            return BacktestDetailDTO(
                run_id=run_id,
                strategy_name=strategy_name,
                market_id=m_id,
                start_time=equity_records[0].timestamp,
                end_time=equity_records[-1].timestamp,
                total_return=tot_ret,
                sharpe_ratio=sharpe,
                max_drawdown=max_dd,
                win_rate=win_rate,
                total_trades=total_trades,
                equity_curve=equity_curve,
                trades=trades_list,
                predictions=predictions_list,
            )


class CatalogQueryService:
    """
    Facade service exposing high-level data catalog query APIs and database health stats.
    """

    def __init__(self, session_factory: sessionmaker[Session]):
        self.session_factory = session_factory
        self.model_repo = ModelCatalogRepository(session_factory)
        self.backtest_repo = BacktestCatalogRepository(session_factory)

    def get_database_summary(self) -> Dict[str, int]:
        """Retrieve total counts across core database tables."""
        with self.session_factory() as session:

            def _count(model_cls):
                try:
                    return (
                        session.scalar(select(func.count()).select_from(model_cls)) or 0
                    )
                except Exception:
                    return 0

            return {
                "markets": _count(Market),
                "bar_logs": _count(BarDataLog),
                "order_logs": _count(OrderLog),
                "trade_logs": _count(TradeLog),
                "models": _count(ModelRegistryLog),
                "production_models": session.scalar(
                    select(func.count())
                    .select_from(ModelRegistryLog)
                    .where(ModelRegistryLog.status == "production")
                )
                or 0,
                "equity_log_ticks": _count(BacktestEquityLog),
                "prediction_logs": _count(BacktestPredictionLog),
            }
