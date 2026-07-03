import pytest

from trading_bot.utils.model_id import (
    calculate_params_hash,
    generate_backtest_run_id,
)
from trading_bot.utils.run_id import calculate_params_hash as calc_hash_direct
from trading_bot.utils.run_id import generate_backtest_run_id as gen_id_direct


def test_calculate_params_hash_deterministic():
    params1 = {"slippage_pct": 0.001, "initial_balance": 10000.0, "k": 0.03}
    params2 = {"k": 0.03, "initial_balance": 10000.0, "slippage_pct": 0.001}
    params3 = {"slippage_pct": 0.005, "initial_balance": 10000.0, "k": 0.03}

    hash1 = calculate_params_hash(params1)
    hash2 = calculate_params_hash(params2)
    hash3 = calculate_params_hash(params3)

    assert len(hash1) == 6
    assert hash1 == hash2
    assert hash1 != hash3
    assert calculate_params_hash(None) == ""
    assert calculate_params_hash({}) == ""


def test_generate_backtest_run_id_formatting():
    model_id = "a1b2c3d4e5f67890"
    dataset_sha = "9876543210fedcba"

    # Without params
    run_id_no_params = generate_backtest_run_id(model_id, dataset_sha)
    assert run_id_no_params == "bt_a1b2c3d4_987654"

    # With params
    params = {"slippage_pct": 0.001, "k": 0.03}
    p_hash = calculate_params_hash(params)
    run_id_with_params = generate_backtest_run_id(model_id, dataset_sha, params)
    assert run_id_with_params == f"bt_a1b2c3d4_987654_{p_hash}"


def test_generate_backtest_run_id_validation():
    with pytest.raises(ValueError, match="model_id must be a non-empty string"):
        generate_backtest_run_id("", "9876543210")

    with pytest.raises(ValueError, match="model_id must be a non-empty string"):
        generate_backtest_run_id(None, "9876543210")

    with pytest.raises(ValueError, match="test_dataset_sha must be a non-empty string"):
        generate_backtest_run_id("a1b2c3d4e5f6", "")

    with pytest.raises(ValueError, match="test_dataset_sha must be a non-empty string"):
        generate_backtest_run_id("a1b2c3d4e5f6", None)


def test_module_reexport():
    assert gen_id_direct is generate_backtest_run_id
    assert calc_hash_direct is calculate_params_hash
