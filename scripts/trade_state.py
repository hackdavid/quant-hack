#!/usr/bin/env python3
"""Shared trade state via SQLite — IPC between trader and monitor.

The main trader writes its state here every cycle.
The monitor reads/writes commands here.

Commands:
    close_all      → Trader closes all positions immediately
    update_tp      → Update take profit target
    update_sl      → Update stop loss
    pause          → Pause trading (don't enter new trades)
    resume         → Resume trading
"""
from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).parent.parent / "data" / "trade_state.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

_lock = threading.Lock()


@dataclass
class TradeState:
    is_running: bool = True
    has_position: bool = False
    position_side: str = ""  # "long" or "short"
    position_lots: float = 0.0
    position_profit: float = 0.0
    position_open_price: float = 0.0
    current_tp: float = 0.0
    current_sl: float = 0.0
    current_hold: float = 0.0
    elapsed_seconds: float = 0.0
    signal_score: float = 0.0
    signal_label: str = ""
    total_pnl: float = 0.0
    win_rate: float = 0.0
    sharpe: float = 0.0
    final_score: float = 9.95
    command: str = ""          # "close_all", "update_tp", "update_sl", "pause", "resume"
    command_value: float = 0.0  # numeric value for update_tp / update_sl
    last_updated: str = ""


def _init_db():
    """Create the state table if not exists."""
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trade_state (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                data TEXT NOT NULL
            )
        """)
        conn.execute("""
            INSERT OR IGNORE INTO trade_state (id, data) VALUES (1, '{}')
        """)


@contextmanager
def _connect():
    conn = sqlite3.connect(str(DB_PATH), timeout=5.0, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
    finally:
        conn.close()


def read_state() -> TradeState:
    """Read current state from SQLite."""
    _init_db()
    with _connect() as conn:
        cursor = conn.execute("SELECT data FROM trade_state WHERE id = 1")
        row = cursor.fetchone()
        if row and row[0]:
            data = json.loads(row[0])
            return TradeState(**data)
        return TradeState()


def write_state(state: TradeState) -> None:
    """Write state to SQLite (thread-safe)."""
    _init_db()
    state.last_updated = datetime.now().isoformat()
    data = json.dumps(asdict(state))
    with _lock:
        with _connect() as conn:
            conn.execute(
                "INSERT INTO trade_state (id, data) VALUES (1, ?) ON CONFLICT(id) DO UPDATE SET data = ?",
                (data, data)
            )


def clear_command() -> None:
    """Clear the pending command after it has been processed."""
    state = read_state()
    state.command = ""
    state.command_value = 0.0
    write_state(state)


def send_command(command: str, value: float = 0.0) -> None:
    """Send a command from the monitor."""
    state = read_state()
    state.command = command
    state.command_value = value
    write_state(state)
    print(f"[OK] Command sent: {command}" + (f"={value}" if value else ""))


def print_status() -> None:
    """Print current trading status."""
    state = read_state()
    print("\n" + "=" * 60)
    print("  TRADE MONITOR STATUS")
    print("=" * 60)
    print(f"\n  Running:        {state.is_running}")
    print(f"  Has Position:   {state.has_position}")
    if state.has_position:
        print(f"  Side:           {state.position_side}")
        print(f"  Lots:           {state.position_lots}")
        print(f"  Open Price:     ${state.position_open_price:,.2f}")
        print(f"  Current P&L:    ${state.position_profit:,.2f}")
        print(f"  TP:             ${state.current_tp:,.2f}")
        print(f"  SL:             ${state.current_sl:,.2f}")
        print(f"  Hold:           {state.elapsed_seconds:.0f}s / {state.current_hold:.0f}s")
        print(f"  Signal:         {state.signal_label} ({state.signal_score:.0f}/100)")
    print(f"\n  Competition:")
    print(f"    Total P&L:    ${state.total_pnl:,.2f}")
    print(f"    Win Rate:     {state.win_rate:.1f}%")
    print(f"    Sharpe:       {state.sharpe:.4f}")
    print(f"    Final Score:  {state.final_score:.2f}")
    if state.command:
        print(f"\n  Pending Command: {state.command}={state.command_value}")
    print(f"\n  Last Updated: {state.last_updated}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Trade State Monitor")
    parser.add_argument("--status", action="store_true", help="Show current status")
    parser.add_argument("--close-all", action="store_true", help="Close all positions")
    parser.add_argument("--update-tp", type=float, default=0, help="Update take profit")
    parser.add_argument("--update-sl", type=float, default=0, help="Update stop loss")
    parser.add_argument("--pause", action="store_true", help="Pause trading")
    parser.add_argument("--resume", action="store_true", help="Resume trading")
    args = parser.parse_args()

    if args.close_all:
        send_command("close_all")
    elif args.update_tp > 0:
        send_command("update_tp", args.update_tp)
    elif args.update_sl > 0:
        send_command("update_sl", args.update_sl)
    elif args.pause:
        send_command("pause")
    elif args.resume:
        send_command("resume")
    else:
        print_status()
