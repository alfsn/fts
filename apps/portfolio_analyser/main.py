import click
from data_layer.ledger_binding import reconcile_snapshot, replay_ledger
from data_layer.market_history import (
    ccl_series,
    fx_context_at,
    history_symbols,
    history_window,
    price_positions,
    usd_per_unit,
    valuation_date,
)
from quant_core.enums import Currency
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


def _broker_options(command):
    """`--broker`, `--file` and `--db-url`, shared by analyze and client-review."""
    for option in reversed(
        [
            click.option(
                "--broker",
                type=click.Choice(["ibkr", "inviu", "iol", "yaml"]),
                required=True,
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
                default="sqlite:///portfolio_analyser.db",
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

    closes, actions = prefetch_history(portfolio, db_url)
    console.print(history_table(portfolio, closes))

    fx = usd_per_unit(ccl_series(closes, actions), closes.index)
    try:
        as_of = valuation_date(portfolio, closes, fx)
        fx_context = fx_context_at(portfolio, fx, as_of)
    except MissingFXRateError as e:
        console.print(f"[bold red]FX error:[/bold red] {e}")
        return
    except ValueError as e:
        console.print(f"[bold red]Price error:[/bold red] {e}")
        return
    price_positions(portfolio, closes, as_of)
    console.print(f"Valued at market close {as_of.date()}")
    if Currency.ARS in fx_context.rates:
        console.print(
            f"FX ARS: {1 / fx_context.rates[Currency.ARS]:,.2f} per USD (CCL, close {as_of.date()})"
        )

    if portfolio.transactions:
        _print_reconciliation(
            console, reconcile_snapshot(portfolio, closes, actions, as_of)
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

    total_value = portfolio.total_value(fx_context)
    console.print(
        Panel(
            f"[bold white]Total Portfolio Value ({portfolio.base_currency.value}):[/bold white] [bold green]${total_value:,.2f}[/bold green]",
            expand=False,
        )
    )


def prefetch_history(portfolio, db_url, extra_symbols=()):
    """Cache-backed raw daily closes and corporate actions for everything the
    snapshot and ledger need valuing, plus `extra_symbols` (benchmark,
    risk-free). PR 2a: prices stay raw, `adjust` on read."""
    from quant_data.db import Base, create_db_session
    from quant_data.history import fetch_corporate_actions, fetch_daily_closes
    from quant_data.providers import TieredDataProvider, YFinanceMarketDataProvider

    symbols = history_symbols(portfolio) | set(extra_symbols)
    with create_db_session(db_url) as session:
        Base.metadata.create_all(session.get_bind())
        provider = TieredDataProvider(session, [YFinanceMarketDataProvider()])
        closes = fetch_daily_closes(provider, symbols, *history_window(portfolio))
        return closes, fetch_corporate_actions(provider, symbols)


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
def client_review(
    broker, file_path, db_url, benchmark_symbol, risk_free_rate, dividend_withholding
):
    """Client review dashboard from the ledger, valued at market, against a
    total-return benchmark (PR 2b, PR 7)."""
    from data_layer.benchmark import (
        benchmark_label,
        benchmark_level,
        treasury_bill_rate,
    )
    from quant_core.models import Benchmark
    from quant_data.reference import RISK_FREE_SYMBOL
    from rich.console import Console
    from visualization.plot_factory import backward_analytics_figures, write_report

    console = Console()
    console.print(f"[bold blue]Generating client review for {broker}...[/bold blue]")
    try:
        portfolio = _connector(broker, file_path).get_portfolio()
    except Exception as e:
        raise click.ClickException(f"Error fetching portfolio: {e}")
    if not portfolio.transactions:
        raise click.ClickException(
            "client-review needs a transactions ledger: performance is rebuilt "
            "day by day from it, never back-cast from today's holdings."
        )

    if risk_free_rate is None and portfolio.base_currency is not Currency.USD:
        raise click.ClickException(
            f"Base currency {portfolio.base_currency.value}: the US T-bill is not "
            "its risk-free rate. Pass --risk-free-rate (annual fraction)."
        )
    benchmark = (
        portfolio.benchmark
        if benchmark_symbol is None
        else Benchmark(symbol=benchmark_symbol)
    )
    extra = [benchmark.symbol] + ([RISK_FREE_SYMBOL] if risk_free_rate is None else [])

    closes, actions = prefetch_history(portfolio, db_url, extra)
    fx = usd_per_unit(ccl_series(closes, actions), closes.index)
    try:
        as_of = valuation_date(portfolio, closes, fx)
    except (MissingFXRateError, ValueError) as e:
        raise click.ClickException(str(e))
    _print_reconciliation(
        console, reconcile_snapshot(portfolio, closes, actions, as_of)
    )

    result = replay_ledger(portfolio, closes, actions, fx, as_of)
    level = benchmark_level(
        closes, actions, benchmark, fx, portfolio.base_currency, dividend_withholding
    )
    if level is None:
        console.print(
            f"[yellow]No prices for benchmark {benchmark.symbol}: "
            "benchmark statistics are n/a.[/yellow]"
        )
    if risk_free_rate is None:
        risk_free_rate = treasury_bill_rate(closes)
        if risk_free_rate is None:
            console.print(
                f"[yellow]No {RISK_FREE_SYMBOL} quotes: Sharpe, Sortino, alpha "
                "and beta are n/a (or pass --risk-free-rate).[/yellow]"
            )
            risk_free_rate = float("nan")
    output_file = f"client_review_{broker}.html"
    write_report(
        backward_analytics_figures(
            result.values,
            result.flows,
            benchmark=level,
            benchmark_name=benchmark_label(benchmark, dividend_withholding),
            risk_free_rate=risk_free_rate,
        ),
        output_file,
    )
    console.print(
        f"Valued from {result.values.index[0].date()} to {as_of.date()}: "
        f"{len(result.values)} valued days, {len(result.dropped)} dropped."
    )
    console.print(f"Client review dashboard generated at {output_file}")


def _print_reconciliation(console, breaks):
    """Quantity breaks stop the run; cash breaks are warnings (D7)."""
    for line in breaks.describe().splitlines():
        style = "bold red" if line.startswith("quantity") else "yellow"
        console.print(f"[{style}]{line}[/{style}]")
    if not breaks.ok:
        raise click.ClickException(
            "Ledger and snapshot disagree on share counts: a trade or a split is missing."
        )
    suffix = " (cash breaks are warnings)" if breaks.cash else ""
    console.print(f"[green]Reconciliation: OK[/green]{suffix}")


if __name__ == "__main__":
    cli()
