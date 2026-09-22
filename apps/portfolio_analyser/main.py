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
    click.echo("UI components not yet implemented (Stage 2).")


if __name__ == "__main__":
    cli()
