from pathlib import Path

import click
from portfolio_analyser import review
from quant_core.enums import Currency
from quant_core.models import MissingFXRateError
from quant_data.providers.brokers import (
    IBKRConnector,
    InviuConnector,
    IOLConnector,
    LocalYamlConnector,
)

# One cache for every run: anchored to the app folder (apps/portfolio_analyser),
# not to the working directory, so running from elsewhere can't start a second DB.
APP_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DB_URL = f"sqlite:///{APP_DIR / 'portfolio_analyser.db'}"


@click.group()
def cli():
    """Portfolio Analyser CLI"""
    pass


def _broker_options(command):
    """`--broker`, `--file` and `--db-url`, shared by analyze and client-review."""
    for option in reversed(
        [
            click.option(
                "--broker",
                type=click.Choice(["ibkr", "inviu", "iol", "yaml"]),
                required=True,
                help="Where the portfolio comes from. ibkr, inviu and iol are "
                "mocks (fixed fake data); use yaml.",
            ),
            click.option(
                "--file",
                "file_path",
                default="portfolio.yaml",
                help="Path to YAML file when broker=yaml",
            ),
            click.option(
                "--db-url",
                envvar="PA_DATABASE_URL",
                default=DEFAULT_DB_URL,
                show_default=True,
                help="Market-data cache database",
            ),
        ]
    ):
        command = option(command)
    return command


def _connector(broker, file_path):
    return {
        "ibkr": IBKRConnector,
        "inviu": InviuConnector,
        "iol": IOLConnector,
        "yaml": lambda: LocalYamlConnector(file_path=file_path),
    }[broker]()


@cli.command()
@_broker_options
def analyze(broker, file_path, db_url):
    """Analyze a portfolio from a broker, valued at the latest market close."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console()
    console.print(
        f"[bold blue]Analyzing portfolio from {broker.upper()}...[/bold blue]"
    )

    try:
        portfolio = _connector(broker, file_path).get_portfolio()
    except Exception as e:
        console.print(f"[bold red]Error fetching portfolio:[/bold red] {e}")
        return

    closes, actions = review.prepare_history(portfolio, db_url)
    console.print(history_table(review.history_rows(portfolio, closes)))
    try:
        snapshot = review.build_snapshot(portfolio, closes, actions)
    except review.QuantityBreakError as e:
        _print_breaks(console, e.breaks)
        raise click.ClickException(str(e))
    except MissingFXRateError as e:
        console.print(f"[bold red]FX error:[/bold red] {e}")
        return
    except ValueError as e:
        console.print(f"[bold red]Price error:[/bold red] {e}")
        return
    as_of, fx_context = snapshot.as_of, snapshot.fx_context
    console.print(f"Valued at market close {as_of.date()}")
    if Currency.ARS in fx_context.rates:
        console.print(
            f"FX ARS: {1 / fx_context.rates[Currency.ARS]:,.2f} per USD (CCL, close {as_of.date()})"
        )
    if snapshot.breaks is not None:
        _print_breaks(console, snapshot.breaks)

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
    pos_table.add_column(f"Close {as_of.date()}", justify="right")
    pos_table.add_column(f"Value in {portfolio.base_currency.value}", justify="right")

    for p in portfolio.positions:
        pos_table.add_row(
            p.instrument_id,
            p.currency.value,
            f"{p.quantity:,.4f}",
            f"{p.cost_basis:,.2f}",
            f"{p.current_price:,.2f}",
            f"{p.market_value(fx_context):,.2f}",
        )
    console.print(pos_table)

    console.print(
        Panel(
            f"[bold white]Total Portfolio Value ({portfolio.base_currency.value}):[/bold white] [bold green]${snapshot.total_value:,.2f}[/bold green]",
            expand=False,
        )
    )


def history_table(rows):
    from rich.table import Table

    table = Table(title="Daily History", show_header=True, header_style="bold cyan")
    for col in ("Symbol", "First", "Last", "Bars"):
        table.add_column(col)
    for row in rows:
        if row.bars == 0:
            table.add_row(f"[red]{row.symbol}[/red]", "-", "-", "[red]missing[/red]")
        else:
            table.add_row(
                row.symbol, str(row.first.date()), str(row.last.date()), str(row.bars)
            )
    return table


@cli.command()
@_broker_options
@click.option(
    "--benchmark",
    "benchmark_symbol",
    default=None,
    help="Benchmark symbol in USD (default: the YAML's `benchmark:`, else SPY).",
)
@click.option(
    "--risk-free-rate",
    type=float,
    default=None,
    help="Annual risk-free rate as a fraction (0.04). Default: daily 13-week "
    "T-bill (^IRX); required when the base currency is not USD.",
)
@click.option(
    "--dividend-withholding",
    type=click.FloatRange(0, 1, max_open=True),
    default=0.0,
    show_default=True,
    help="Tax withheld on the benchmark's dividends (0.30 for a non-US holder).",
)
@click.option(
    "--start",
    type=click.DateTime(["%Y-%m-%d"]),
    default=None,
    help="First day of the review window; returns start from the last close "
    "before it. Default: the first ledger day.",
)
@click.option(
    "--end",
    type=click.DateTime(["%Y-%m-%d"]),
    default=None,
    help="Last day of the review window. Default: the valuation date.",
)
def client_review(
    broker,
    file_path,
    db_url,
    benchmark_symbol,
    risk_free_rate,
    dividend_withholding,
    start,
    end,
):
    """Client review dashboard from the ledger, valued at market, against a
    total-return benchmark (PR 2b, PR 7), over an optional date window (W2)."""
    from portfolio_analyser.visualization.plot_factory import (
        backward_analytics_figures,
        write_report,
    )
    from rich.console import Console

    console = Console()
    console.print(f"[bold blue]Generating client review for {broker}...[/bold blue]")
    try:
        portfolio = _connector(broker, file_path).get_portfolio()
    except Exception as e:
        raise click.ClickException(f"Error fetching portfolio: {e}")

    try:
        result = review.prepare_review(
            portfolio,
            db_url,
            benchmark_symbol=benchmark_symbol,
            risk_free_rate=risk_free_rate,
            withholding=dividend_withholding,
            start=start,
            end=end,
        )
    except review.ReviewError as e:
        if e.breaks is not None:
            _print_breaks(console, e.breaks)
        raise click.ClickException(str(e))
    except (MissingFXRateError, ValueError) as e:
        raise click.ClickException(str(e))

    _print_breaks(console, result.snapshot.breaks)
    for warning in result.warnings:
        console.print(f"[yellow]{warning}[/yellow]")
    output_file = f"client_review_{broker}.html"
    write_report(
        backward_analytics_figures(
            result.values,
            result.flows,
            benchmark=result.benchmark,
            benchmark_name=result.benchmark_name,
            risk_free_rate=result.risk_free,
        ),
        output_file,
    )
    first, last = result.period
    console.print(
        f"Valued from {first.date()} to {last.date()}: "
        f"{len(result.values)} valued days, {len(result.dropped)} dropped."
    )
    console.print(f"Client review dashboard generated at {output_file}")


def _print_breaks(console, breaks):
    """Quantity breaks stop the run (the use case raises); cash breaks are
    warnings (D7)."""
    for line in breaks.describe().splitlines():
        style = "bold red" if line.startswith("quantity") else "yellow"
        console.print(f"[{style}]{line}[/{style}]")
    if breaks.ok:
        suffix = " (cash breaks are warnings)" if breaks.cash else ""
        console.print(f"[green]Reconciliation: OK[/green]{suffix}")


if __name__ == "__main__":
    cli()
