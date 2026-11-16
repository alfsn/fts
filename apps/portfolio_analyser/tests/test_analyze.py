from datetime import timedelta

from click.testing import CliRunner
from main import cli
from quant_core.enums import BarType
from quant_core.models import BarData
from quant_data.providers import YFinanceMarketDataProvider

PORTFOLIO = """
base_currency: USD
positions:
  - instrument_id: YPF
    provider_symbol: YPFD.BA
    currency: ARS
    quantity: 100.0
    cost_basis: 12000.0
    current_price: 15500.0
cash_balances:
  - currency: ARS
    total: 250000.0
    available: 250000.0
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


def _analyze(monkeypatch, tmp_path, closes):
    monkeypatch.setattr(YFinanceMarketDataProvider, "get_bars", _stub_get_bars(closes))
    path = tmp_path / "portfolio.yaml"
    path.write_text(PORTFOLIO)
    args = ["analyze", "--broker", "yaml", "--file", str(path), "--db-url", "sqlite://"]
    return CliRunner().invoke(cli, args, terminal_width=200)


def test_analyze_values_ars_at_market_ccl(monkeypatch, tmp_path):
    result = _analyze(monkeypatch, tmp_path, CLOSES)

    assert result.exit_code == 0, result.output
    assert "1,666.67 per USD" in result.output
    assert "150.00" in result.output  # 250,000 ARS cash / 1666.67
    assert "930.00" in result.output  # 100 x 15,500 ARS / 1666.67
    assert "1,080.00" in result.output  # total


def test_analyze_reports_missing_fx_instead_of_guessing(monkeypatch, tmp_path):
    result = _analyze(monkeypatch, tmp_path, {"YPFD.BA": 7500.0})

    assert result.exit_code == 0, result.output
    assert "FX error" in result.output
    assert "No market FX rate for ARS" in result.output
    assert "Total Portfolio Value" not in result.output
