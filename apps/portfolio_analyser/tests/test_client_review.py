import numpy as np
import pandas as pd
from click.testing import CliRunner
from main import cli
from visualization.plot_factory import backward_analytics_figures


def test_backward_analytics_figures():
    dates = pd.bdate_range("2022-01-03", periods=400)
    values = pd.Series(
        100 * np.cumprod(1 + np.random.default_rng(2).normal(0, 0.01, 400)),
        index=dates,
    )
    figures = backward_analytics_figures(values, baseline=values * 1.1)
    assert len(figures) == 4


def test_client_review_writes_report():
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(cli, ["client-review", "--broker", "yaml"])
        assert result.exit_code == 0, result.output
        with open("client_review_yaml.html", encoding="utf-8") as f:
            assert "Risk KPIs" in f.read()
