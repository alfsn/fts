import click
from data_layer.market_history import (
    build_fx_context,
    history_symbols,
    history_window,
    latest_usd_per_unit,
)
from quant_core.models import MissingFXRateError
from quant_data.providers.brokers import (
    IBKRConnector,
    InviuConnector,
    IOLConnector,
    LocalYamlConnector,
)


@click.group()
def cli():
    """Portfolio Analyser CLI"""
    pass


@cli.command()
@click.option(
    "--broker", type=click.Choice(["ibkr", "inviu", "iol", "yaml"]), required=True
)
@click.option(
    "--file",
    "file_path",
    default="portfolio.yaml",
    help="Path to YAML file when broker=yaml",
)
@click.option(
    "--db-url",
    envvar="PA_DATABASE_URL",
    default="sqlite:///portfolio_analyser.db",
    show_default=True,
    help="Market-data cache database",
)
def analyze(broker, file_path, db_url):
    """Analyze a portfolio from a broker"""
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console()
    console.print(
        f"[bold blue]Analyzing portfolio from {broker.upper()}...[/bold blue]"
    )

    if broker == "ibkr":
        connector = IBKRConnector()
    elif broker == "inviu":
        connector = InviuConnector()
    elif broker == "iol":
        connector = IOLConnector()
    elif broker == "yaml":
        connector = LocalYamlConnector(file_path=file_path)

    try:
        portfolio = connector.get_portfolio()
    except Exception as e:
        console.print(f"[bold red]Error fetching portfolio:[/bold red] {e}")
        return

    closes = prefetch_history(portfolio, db_url)
    console.print(history_table(portfolio, closes))

    try:
        fx_context = build_fx_context(portfolio, closes)
    except MissingFXRateError as e:
        console.print(f"[bold red]FX error:[/bold red] {e}")
        return
    for currency, (rate, as_of) in latest_usd_per_unit(closes).items():
        console.print(
            f"FX {currency.value}: {1 / rate:,.2f} per USD (CCL, close {as_of.date()})"
        )

    # Print Cash Balances
    cash_table = Table(
        title="Cash Balances", show_header=True, header_style="bold green"
    )
    cash_table.add_column("Currency")
    cash_table.add_column("Total", justify="right")
    cash_table.add_column("Available", justify="right")
    cash_table.add_column(f"Value in {portfolio.base_currency.value}", justify="right")

    for cb in portfolio.cash_balances:
        rate = fx_context.get_rate(cb.currency)
        val_in_base = cb.total * rate
        cash_table.add_row(
            cb.currency.value,
            f"{cb.total:,.2f}",
            f"{cb.available:,.2f}",
            f"{val_in_base:,.2f}",
        )
    console.print(cash_table)

    # Print Positions
    pos_table = Table(
        title="Open Positions", show_header=True, header_style="bold yellow"
    )
    pos_table.add_column("Instrument")
    pos_table.add_column("Currency")
    pos_table.add_column("Quantity", justify="right")
    pos_table.add_column("Cost Basis", justify="right")
    pos_table.add_column("Current Price", justify="right")
    pos_table.add_column(f"Value in {portfolio.base_currency.value}", justify="right")

    for p in portfolio.positions:
        rate = fx_context.get_rate(p.currency)
        price = p.current_price or 0.0
        val_in_base = price * p.quantity * rate
        pos_table.add_row(
            p.instrument_id,
            p.currency.value,
            f"{p.quantity:,.4f}",
            f"{p.cost_basis:,.2f}",
            f"{price:,.2f}",
            f"{val_in_base:,.2f}",
        )
    console.print(pos_table)

    total_value = portfolio.total_value(fx_context)
    console.print(
        Panel(
            f"[bold white]Total Portfolio Value ({portfolio.base_currency.value}):[/bold white] [bold green]${total_value:,.2f}[/bold green]",
            expand=False,
        )
    )


def prefetch_history(portfolio, db_url):
    """Cache-backed daily closes for everything the snapshot needs valuing."""
    from quant_data.db import Base, create_db_session
    from quant_data.history import fetch_daily_closes
    from quant_data.providers import TieredDataProvider, YFinanceMarketDataProvider

    with create_db_session(db_url) as session:
        Base.metadata.create_all(session.get_bind())
        # A real chain: create_default() puts mock tiers first.
        provider = TieredDataProvider(session, [YFinanceMarketDataProvider()])
        return fetch_daily_closes(
            provider, history_symbols(portfolio), *history_window()
        )


def history_table(portfolio, closes):
    from rich.table import Table

    table = Table(title="Daily History", show_header=True, header_style="bold cyan")
    for col in ("Symbol", "First", "Last", "Bars"):
        table.add_column(col)
    for symbol in sorted(history_symbols(portfolio)):
        series = closes[symbol].dropna() if symbol in closes else None
        if series is None or series.empty:
            table.add_row(f"[red]{symbol}[/red]", "-", "-", "[red]missing[/red]")
        else:
            table.add_row(
                symbol,
                str(series.index[0].date()),
                str(series.index[-1].date()),
                str(len(series)),
            )
    return table


@cli.command()
@click.option(
    "--broker", type=click.Choice(["ibkr", "inviu", "iol", "yaml"]), required=True
)
@click.option(
    "--file",
    "file_path",
    default="portfolio.yaml",
    help="Path to YAML file when broker=yaml",
)
def client_review(broker, file_path):
    """Generate a client review dashboard"""
    click.echo(f"Generating client review for {broker}...")
    import numpy as np
    import pandas as pd
    from visualization.plot_factory import backward_analytics_figures, write_report

    # TODO(PR 2b): replace with real holdings history from the transaction ledger.
    rng = np.random.default_rng()
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=756)
    portfolio = pd.Series(
        100 * np.cumprod(1 + rng.normal(0.0004, 0.01, len(dates))), index=dates
    )
    baseline = pd.Series(
        100 * np.cumprod(1 + rng.normal(0.0003, 0.012, len(dates))), index=dates
    )

    output_file = f"client_review_{broker}.html"
    write_report(
        backward_analytics_figures(portfolio, baseline=baseline, baseline_name="SPY"),
        output_file,
    )
    click.echo(f"Client review dashboard generated at {output_file}")


if __name__ == "__main__":
    cli()
