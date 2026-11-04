import click
from quant_core.enums import Currency
from quant_core.models import FXContext
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
def analyze(broker, file_path):
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

    # Example FX Context
    fx_context = FXContext(
        base_currency=portfolio.base_currency,
        rates={
            Currency.ARS: 0.001,
            Currency.EUR: 1.1,
            Currency.USD: 1.0,
        },
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

    # TODO(PR 2): replace with broker-snapshot valuations from SmartDataFetcher.
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
