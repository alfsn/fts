import click
from quant_core.calculator import PortfolioCalculator
from quant_core.enums import Currency
from quant_core.models import FXContext
from quant_data.providers.brokers import IBKRConnector, InviuConnector, IOLConnector


@click.group()
def cli():
    """Portfolio Analyser CLI"""
    pass


@cli.command()
@click.option("--broker", type=click.Choice(["ibkr", "inviu", "iol"]), required=True)
def analyze(broker):
    """Analyze a portfolio from a broker"""
    click.echo(f"Analyzing portfolio from {broker}...")

    if broker == "ibkr":
        connector = IBKRConnector()
    elif broker == "inviu":
        connector = InviuConnector()
    elif broker == "iol":
        connector = IOLConnector()

    portfolio = connector.get_portfolio()

    # Example FX Context
    fx_context = FXContext(
        base_currency=portfolio.base_currency,
        rates={
            Currency.ARS: 0.001,
            Currency.EUR: 1.1,
            Currency.USD: 1.0,
        },
    )

    total_value = portfolio.total_value(fx_context)

    click.echo(
        f"Total Portfolio Value ({portfolio.base_currency.value}): {total_value:.2f}"
    )


@cli.command()
@click.option("--broker", type=click.Choice(["ibkr", "inviu", "iol"]), required=True)
def client_review(broker):
    """Generate a client review dashboard"""
    click.echo(f"Generating client review for {broker}...")
    from datetime import datetime

    import numpy as np
    import pandas as pd
    from quant_ui.plots.performance import plot_backtest_performance

    # Mock a PnL dataframe for demonstration
    dates = pd.date_range(end=datetime.today(), periods=100, freq="D")
    df_pnl = pd.DataFrame(
        {
            "timestamp": dates,
            "open": np.random.normal(100, 2, 100).cumprod(),
        }
    )
    df_pnl["high"] = df_pnl["open"] + np.random.uniform(0, 2, 100)
    df_pnl["low"] = df_pnl["open"] - np.random.uniform(0, 2, 100)
    df_pnl["close"] = df_pnl["open"] + np.random.normal(0, 1, 100)
    df_pnl["cum_strat_return"] = (
        df_pnl["close"].pct_change().fillna(0) + np.random.normal(0.001, 0.005, 100)
    ).cumsum()
    df_pnl["cum_baseline_return"] = df_pnl["close"].pct_change().fillna(0).cumsum()

    fig = plot_backtest_performance(df_pnl, f"{broker.upper()} PORTFOLIO")
    output_file = f"client_review_{broker}.html"
    fig.write_html(output_file)
    click.echo(f"Client review dashboard generated at {output_file}")


if __name__ == "__main__":
    cli()
