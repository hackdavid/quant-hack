#!/usr/bin/env python3
"""Quick smoke test for the MT5 wrapper.

Usage:
    python scripts/test_mt5.py --account 123456 --password "..." --server "XMGlobal-MT5"
"""
from __future__ import annotations

import asyncio
import typer
from rich import print as rprint

from intraday.trader.mt5_wrapper import MT5TradingWrapper

app = typer.Typer()


@app.command()
def main(
    account: int = typer.Option(..., help="MT5 account number"),
    password: str = typer.Option(..., help="MT5 account password"),
    server: str = typer.Option(..., help="MT5 broker server name"),
    path: str = typer.Option(None, help="Path to terminal64.exe (optional)"),
    symbol: str = typer.Option("BTCUSDT", help="Symbol to test"),
    test_order: bool = typer.Option(False, help="Place a tiny market order (0.01 lot)"),
) -> None:
    rprint("[yellow]Connecting to MT5...[/yellow]")
    mt5 = MT5TradingWrapper(account_id=account, password=password, server=server, path=path)
    if not mt5.connect():
        rprint("[red]Failed to connect[/red]")
        raise typer.Exit(1)

    # Account state
    state = mt5.account_state()
    rprint("[green]Connected![/green]")
    rprint(f"  Balance: {state.balance}")
    rprint(f"  Equity:  {state.equity}")
    rprint(f"  Profit:  {state.profit}")

    # Positions
    positions = mt5.get_positions(symbol)
    rprint(f"\n[yellow]Open positions ({symbol}): {len(positions)}[/yellow]")
    for p in positions:
        rprint(f"  {p.side} {p.volume} @ {p.open_price} (profit: {p.profit})")

    # Test order
    if test_order:
        rprint("\n[yellow]Placing test market order (0.01 lot)...[/yellow]")
        result = mt5.market_order(symbol, "buy", volume=0.01, comment="smoke test")
        rprint(f"  Result: {result.to_dict()}")
        if result.success:
            rprint("[green]Order filled. Closing it now...[/green]")
            close = mt5.close_position(result.ticket)
            rprint(f"  Close: {close.to_dict()}")

    # Symbol mapping
    mt5_sym = mt5.to_mt5_symbol(symbol)
    rprint(f"\n[yellow]Symbol mapping: {symbol} → {mt5_sym}[/yellow]")

    mt5.shutdown()
    rprint("\n[green]Done.[/green]")


if __name__ == "__main__":
    app()
