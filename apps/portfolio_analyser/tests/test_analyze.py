"""`analyze` values the snapshot at the latest market close (PR 2b-2), not at
the YAML `current_price`, and reconciles the ledger against it."""

from datetime import timedelta
from pathlib import Path

import portfolio_analyser.review as review
import pytest
from click.testing import CliRunner
from portfolio_analyser.main import cli
from quant_core.enums import BarType
from quant_core.models import BarData
from quant_data.providers import YFinanceMarketDataProvider

APP_DIR = Path(__file__).resolve().parents[1]

PORTFOLIO = """
base_currency: USD
positions:
  - instrument_id: YPF
    provider_symbol: YPFD.BA
    currency: ARS
    quantity: 100.0
    cost_basis: 12000.0
cash_balances:
  - currency: ARS
    total: 250000.0
    available: 250000.0
"""

LEDGER = """
transactions:
  - {date: 2024-01-02, type: deposit, currency: ARS, amount: 1450000}
  - {date: 2024-01-03, type: buy, currency: ARS, instrument_id: YPF, provider_symbol: YPFD.BA, quantity: %s, price: 12000}
"""

# Both pairs imply 1666.67 ARS per USD (ratio 10).
CLOSES = {"GGAL.BA": 5000.0, "GGAL": 30.0, "YPFD.BA": 7500.0, "YPF": 45.0}


def _stub_get_bars(closes):
    def get_bars(self, request):
        if request.instrument_id not in closes:
            return []
        start, end = request.resolve_bounds()
        days = range(0, (end - start).days, 7)  # weekly is enough for the test
        return [
            BarData(
                timestamp=start + timedelta(days=d),
                open=closes[request.instrument_id],
                high=closes[request.instrument_id],
                low=closes[request.instrument_id],
                close=closes[request.instrument_id],
                volume=0.0,
                bar_type=BarType.TIME,
                interval="1d",
                ticks_count=1,
                dollar_volume=0.0,
            )
            for d in days
        ]

    return get_bars


def _analyze(monkeypatch, tmp_path, closes, yaml=PORTFOLIO, actions=None):
    monkeypatch.setattr(YFinanceMarketDataProvider, "get_bars", _stub_get_bars(closes))
    monkeypatch.setattr(
        YFinanceMarketDataProvider, "rate_limit", None
    )  # no real sleeps
    monkeypatch.setattr(
        YFinanceMarketDataProvider,
        "get_corporate_actions",
        lambda self, symbol: (actions or {}).get(symbol, []),
    )
    path = tmp_path / "portfolio.yaml"
    path.write_text(yaml)
    args = ["analyze", "--broker", "yaml", "--file", str(path), "--db-url", "sqlite://"]
    return CliRunner().invoke(cli, args, terminal_width=200)


def test_analyze_values_at_the_market_close_not_the_yaml_price(monkeypatch, tmp_path):
    result = _analyze(
        monkeypatch,
        tmp_path,
        CLOSES,
        yaml=PORTFOLIO.replace(
            "cost_basis: 12000.0", "cost_basis: 12000.0\n    current_price: 15500.0"
        ),
    )

    assert result.exit_code == 0, result.output
    assert "1,666.67 per USD" in result.output
    assert "Valued at market close" in result.output
    assert "7,500.00" in result.output  # the close, not 15,500
    assert "150.00" in result.output  # 250,000 ARS cash / 1666.67
    assert "450.00" in result.output  # 100 x 7,500 ARS / 1666.67
    assert "600.00" in result.output  # total
    assert "930.00" not in result.output


def test_analyze_reports_missing_fx_instead_of_guessing(monkeypatch, tmp_path):
    result = _analyze(monkeypatch, tmp_path, {"YPFD.BA": 7500.0})

    assert result.exit_code == 0, result.output
    assert "No market FX rate for ARS" in result.output
    assert "Total Portfolio Value" not in result.output


def test_analyze_reports_a_listing_without_prices(monkeypatch, tmp_path):
    closes = {k: v for k, v in CLOSES.items() if k != "YPFD.BA"}
    result = _analyze(monkeypatch, tmp_path, closes)

    assert "YPFD.BA" in result.output and "no market price" in result.output.lower()
    assert "Total Portfolio Value" not in result.output


def test_analyze_reconciles_a_matching_ledger(monkeypatch, tmp_path):
    result = _analyze(monkeypatch, tmp_path, CLOSES, yaml=PORTFOLIO + LEDGER % 100)

    assert result.exit_code == 0, result.output
    assert "Reconciliation: OK" in result.output
    assert "Total Portfolio Value" in result.output


def test_analyze_stops_on_a_quantity_break(monkeypatch, tmp_path):
    result = _analyze(monkeypatch, tmp_path, CLOSES, yaml=PORTFOLIO + LEDGER % 90)

    assert result.exit_code != 0
    assert "quantity break YPF (ARS): ledger 90, snapshot 100" in result.output
    assert "Total Portfolio Value" not in result.output


def test_analyze_warns_on_a_cash_break_and_continues(monkeypatch, tmp_path):
    yaml = PORTFOLIO.replace("total: 250000.0", "total: 240000.0") + LEDGER % 100
    result = _analyze(monkeypatch, tmp_path, CLOSES, yaml=yaml)

    assert result.exit_code == 0, result.output
    assert (
        "cash break ARS" in result.output and "unexplained -10,000.00" in result.output
    )
    assert "Total Portfolio Value" in result.output


def test_the_demo_ledger_reconciles_across_the_ypfd_split(monkeypatch, tmp_path):
    """examples/demo_ledger.yaml holds YPFD.BA across its 10:1 split: the
    snapshot's 600 shares only match the ledger's 60 if the split is applied."""
    from datetime import datetime, timezone

    from quant_core.enums import CorporateActionType
    from quant_core.models import CorporateAction

    split = CorporateAction(
        instrument_id="YPFD.BA",
        action_type=CorporateActionType.SPLIT,
        ex_date=datetime(2026, 8, 3, tzinfo=timezone.utc),
        ratio=10.0,
    )
    demo = (APP_DIR / "examples" / "demo_ledger.yaml").read_text()
    closes = {**CLOSES, "AAPL": 200.0}

    result = _analyze(
        monkeypatch, tmp_path, closes, yaml=demo, actions={"YPFD.BA": [split]}
    )
    assert result.exit_code == 0, result.output
    assert "Reconciliation: OK" in result.output
    assert "cash break" not in result.output

    without_split = _analyze(monkeypatch, tmp_path, closes, yaml=demo)
    assert "quantity break YPF (ARS)" in without_split.output
