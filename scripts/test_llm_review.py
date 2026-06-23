#!/usr/bin/env python3
"""Smoke test for the LLM review agent.

Usage:
    export LLM_TOKEN="your-fireworks-api-key"
    export LLM_BASE_URL="https://api.fireworks.ai/inference"
    export LLM_MODEL="accounts/fireworks/routers/kimi-k2p6-turbo"

    uv run python scripts/test_llm_review.py
"""
from __future__ import annotations

import json
import os
import sys

from rich import print as rprint

# Ensure src is on path
sys.path.insert(0, str(os.path.dirname(__file__) + "/../src"))

from intraday.llm.review import LLMReviewAgent


def main() -> None:
    api_key = os.getenv("LLM_TOKEN", "")
    if not api_key:
        rprint("[red]Error: LLM_TOKEN not set.[/red]")
        rprint("  export LLM_TOKEN='your-fireworks-api-key'")
        rprint("  export LLM_BASE_URL='https://api.fireworks.ai/inference'")
        rprint("  export LLM_MODEL='accounts/fireworks/routers/kimi-k2p6-turbo'")
        sys.exit(1)

    agent = LLMReviewAgent(
        base_url=os.getenv("LLM_BASE_URL"),
        api_key=api_key,
        model=os.getenv("LLM_MODEL"),
    )

    rprint("[yellow]Testing LLM review agent...[/yellow]")
    rprint(f"  Base URL: {agent.base_url}")
    rprint(f"  Model: {agent.model}")

    review = agent.review(
        signal="BUY",
        confidence=0.72,
        bar={"close": 94000.0, "volume": 5_000_000, "high": 94200.0, "low": 93800.0},
        positions=[],
        account={"balance": 1_000_000, "equity": 1_000_000, "profit": 0},
        risk_state={"drawdown_pct": 0.0, "daily_pnl_pct": 0.0, "trade_count_today": 2},
        recent_logs=[],
    )

    rprint("\n[bold]LLM Response:[/bold]")
    rprint(json.dumps(review.to_dict(), indent=2))

    if review.risk_approved:
        rprint("\n[green]✓ Risk approved[/green]")
    else:
        rprint("\n[red]✗ Risk rejected[/red]")


if __name__ == "__main__":
    main()
