import logging
from typing import Any, Dict, Optional, Union

import pandas as pd
import yaml
from nets.inference import ONNXPredictor
from nets.output_selectors import DynamicThresholdClassifier
from nets.strategies.nets_strategy import NetsStrategy
from nets.training.orchestrator import (
    TrainingResult,
    parse_datetime_param,
    train_and_register_candidate,
)

from trading_bot.backtesting.engine import BacktestEngine
from trading_bot.backtesting.readers import SQLBacktestDataReader
from trading_bot.config import settings
from trading_bot.core.database import SessionLocal, init_db
from trading_bot.core.loop import HistoricalReplayLoop
from trading_bot.core.models import (
    BacktestPredictionLog,
    ModelRegistryLog,
    OrderLog,
    Position,
    TimeSeriesDataset,
    TradeLog,
)
from trading_bot.core.pipeline import TradingPipeline
from trading_bot.core.repository import (
    ModelRepository,
    OrderRepository,
    PositionRepository,
)
from trading_bot.core.schemas import BarData
from trading_bot.execution.delay import KBarExecuteDelay
from trading_bot.execution.engine import ExecutionEngine
from trading_bot.execution.handlers.simulated_handler import SimulatedExecutionHandler
from trading_bot.execution.slippage import FlatPriceSlip
from trading_bot.monitoring.prediction_logger import DatabasePredictionLogger
from trading_bot.risk_management.manager import RiskManager
from trading_bot.risk_management.portfolio import Portfolio
from trading_bot.risk_management.sizing.fixed_percentage import FixedPercentageSizer
from trading_bot.strategy.engine import StrategyEngine
from trading_bot.utils.run_id import generate_backtest_run_id

logger = logging.getLogger(__name__)


def run_model_backtest(
    model_entry: ModelRegistryLog,
    test_start_date: Union[str, Any],
    test_end_date: Union[str, Any],
    backtest_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Runs an out-of-sample backtest simulation for a specific candidate ModelRegistryLog entry.
    Returns strategy performance summary metrics.
    """
    backtest_params = backtest_params or {}
    start_dt = parse_datetime_param(test_start_date)
    end_dt = parse_datetime_param(test_end_date)

    with SessionLocal() as db:
        test_dataset_sha = None
        if getattr(model_entry, "dataset", None) and getattr(
            model_entry.dataset, "hash", None
        ):
            test_dataset_sha = model_entry.dataset.hash
        elif getattr(model_entry, "dataset_id", None):
            ds = (
                db.query(TimeSeriesDataset)
                .filter_by(dataset_id=model_entry.dataset_id)
                .first()
            )
            if ds:
                test_dataset_sha = ds.hash

        run_id = generate_backtest_run_id(
            model_id=model_entry.model_id,
            test_dataset_sha=test_dataset_sha,
            params=backtest_params,
        )

        # Clear prior backtest logs for clean calculation
        db.query(BacktestPredictionLog).filter_by(run_id=run_id).delete()
        db.query(OrderLog).filter_by(run_id=run_id).delete()
        db.query(TradeLog).filter_by(run_id=run_id).delete()
        db.query(Position).filter_by(run_id=run_id).delete()
        db.commit()

        # Build Strategy Components
        onnx_path = model_entry.onnx_path
        predictor = ONNXPredictor(onnx_path)
        output_selector = DynamicThresholdClassifier(
            k=backtest_params.get("classifier_k", 0.03),
            period=backtest_params.get("period", 10),
            confidence_multiplier=backtest_params.get("confidence_multiplier", 20.0),
        )

        transform = None
        if getattr(predictor, "pipeline", None) is not None:
            transform = predictor.pipeline
        elif backtest_params.get("feature_pipeline"):
            from trading_bot.core.transforms import BaseTransform

            transform = BaseTransform.from_dict(backtest_params["feature_pipeline"])

        feature_cols = backtest_params.get("feature_cols", None)
        if (
            feature_cols is None
            and hasattr(model_entry, "feature_cols")
            and model_entry.feature_cols
        ):
            feature_cols = model_entry.feature_cols
        if feature_cols is None:
            feature_cols = ["open", "high", "low", "close", "volume"]

        lookback_period = backtest_params.get(
            "lookback_period",
            (
                predictor.model_metadata.lookback_period
                if predictor.model_metadata
                else (getattr(model_entry, "lookback_period", None) or 20)
            ),
        )

        strategy = NetsStrategy(
            predictor=predictor,
            output_selector=output_selector,
            transform=transform,
            lookback_period=lookback_period,
            feature_cols=feature_cols,
            name_suffix=model_entry.model_type,
            allow_in_sample=True,
        )
        strategy_engine = StrategyEngine(strategies=[strategy])

        # Build Portfolio & Risk Management
        pos_repo = PositionRepository(db)
        order_repo = OrderRepository(db)
        portfolio = Portfolio(
            initial_balance=backtest_params.get("initial_balance", 10000.0),
            quote_currency="USD",
            pos_repo=pos_repo,
            order_repo=order_repo,
        )
        portfolio._positions = {}
        sizer = FixedPercentageSizer(
            default_percentage=backtest_params.get("position_size_pct", 0.10)
        )
        risk_manager = RiskManager(portfolio=portfolio, sizer=sizer)

        # Build Execution Engine with delay & slippage
        exec_delay = KBarExecuteDelay(k=backtest_params.get("execution_delay_bars", 1))
        slippage = FlatPriceSlip(
            slippage_pct=backtest_params.get("slippage_bps", 5.0) / 10000.0
        )
        handler = SimulatedExecutionHandler(
            delay_model=exec_delay,
            slippage_model=slippage,
        )
        exec_engine = ExecutionEngine(
            execution_handler=handler,
            portfolio=portfolio,
            run_id=run_id,
        )

        prediction_logger = DatabasePredictionLogger(db=db, run_id=run_id)

        pipeline = TradingPipeline(
            ingestion=None,
            strategy=strategy_engine,
            risk=risk_manager,
            execution=exec_engine,
            portfolio=portfolio,
            prediction_logger=prediction_logger,
        )

        # Execute Historical Replay Loop via BacktestEngine
        data_reader = SQLBacktestDataReader(
            session=db,
            market_id=model_entry.market_id,
            start_date=start_dt,
            end_date=end_dt,
        )

        bt_engine = BacktestEngine(
            pipeline=pipeline,
            data_reader=data_reader,
            db=db,
            market_id=model_entry.market_id,
        )
        bt_result = bt_engine.run(run_id=run_id, clear_previous_run=True)

        total_pnl = bt_result.final_equity - bt_result.initial_equity
        trades = db.query(TradeLog).filter_by(run_id=run_id).all()
        winning_trades = [t for t in trades if t.realized_pnl > 0] if trades else []
        win_rate = (len(winning_trades) / len(trades)) if trades else 0.0

        return {
            "run_id": run_id,
            "final_equity": bt_result.final_equity,
            "total_pnl": total_pnl,
            "total_trades": len(trades) if trades else bt_result.total_trades,
            "win_rate": win_rate,
            "sharpe_ratio": bt_result.sharpe_ratio,
            "max_drawdown": abs(bt_result.max_drawdown_pct),
        }


import os
from pathlib import Path

from trading_bot.backtesting import SweepResult, SweepTrialResult


def run_parameter_sweep(
    spec_yaml_path: Union[str, Path, Any],
    output_dir: Optional[str] = "runs/reports",
) -> SweepResult:
    """
    Executes a controlled parameter sweep from a YAML spec file.
    All non-swept parameters are held constant (ceteris paribus).

    :param spec_yaml_path: Path to YAML spec file or loaded SweepSpec object.
    :param output_dir: Optional directory to save summary JSON (defaults to 'runs/reports').
                       If set to None, runs purely in-memory.
    :return: Evaluated SweepResult object.
    """
    from nets.spec import SweepSpec

    try:
        from plugins.nets.spec import SweepSpec as PluginsSweepSpec
    except ImportError:
        PluginsSweepSpec = SweepSpec

    if (
        isinstance(spec_yaml_path, (SweepSpec, PluginsSweepSpec))
        or type(spec_yaml_path).__name__ == "SweepSpec"
        or hasattr(spec_yaml_path, "sweep_name")
    ):
        spec = spec_yaml_path
    else:
        spec = SweepSpec.from_yaml(spec_yaml_path)

    sweep_result = SweepResult(
        sweep_name=spec.sweep_name,
        sweep_param=spec.sweep_param,
        sweep_values=spec.sweep_values,
        market_id=spec.market.market_id,
    )

    for i, val in enumerate(spec.sweep_values):
        logger.info(
            f"--- Running Sweep Trial [{i+1}/{len(spec.sweep_values)}]: {spec.sweep_param} = {val} ---"
        )

        # 1. Override target parameter while keeping all other hyperparams fixed
        model_params = spec.base_model_params.copy()
        model_params[spec.sweep_param] = val

        # 2. Train and register candidate model
        train_res = train_and_register_candidate(
            model_type=spec.model_type,
            market_id=spec.market.market_id,
            interval=spec.market.interval,
            lookback_period=spec.features.lookback_period,
            feature_cols=spec.features.feature_cols,
            start_date=spec.train_dates.start_date,
            end_date=spec.train_dates.end_date,
            model_params=model_params,
            train_params=spec.base_train_params,
            feature_pipeline=spec.features.feature_pipeline,
            run_id=spec.sweep_name,
            status="candidate",
        )

        # Fetch model DB log
        with SessionLocal() as db:
            model_repo = ModelRepository(db)
            model_entry = model_repo.get_model(train_res.model_id)

        # 3. Execute Out-of-Sample Backtest
        backtest_params = spec.execution.model_dump()
        backtest_params["feature_cols"] = spec.features.feature_cols
        backtest_params["feature_pipeline"] = spec.features.feature_pipeline
        backtest_params["lookback_period"] = spec.features.lookback_period

        bt_res = run_model_backtest(
            model_entry=model_entry,
            test_start_date=spec.test_dates.start_date,
            test_end_date=spec.test_dates.end_date,
            backtest_params=backtest_params,
        )

        trial_result = SweepTrialResult(
            trial_index=i,
            param_value=val,
            model_id=str(train_res.model_id),
            run_id=bt_res["run_id"],
            val_ic=train_res.val_ic if train_res.val_ic is not None else 0.0,
            val_loss=train_res.val_loss if train_res.val_loss is not None else 0.0,
            oos_pnl=bt_res["total_pnl"],
            oos_sharpe=bt_res["sharpe_ratio"],
            oos_max_dd=bt_res["max_drawdown"],
            win_rate=bt_res["win_rate"],
            total_trades=bt_res["total_trades"],
            final_equity=bt_res["final_equity"],
        )
        sweep_result.add_trial(trial_result)

    # Optional summary JSON persistence
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        json_path = os.path.join(output_dir, f"sweep_{spec.sweep_name}.json")
        sweep_result.save_summary(json_path)

    return sweep_result
