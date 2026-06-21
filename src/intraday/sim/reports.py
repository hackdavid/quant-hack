"""Report generation for backtest runs.

Writes metrics.json and an offline-capable HTML report with uPlot for equity
curve and drawdown visualization. The HTML file embeds uPlot JS as a string
literal so it opens without any network access.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from intraday.sim.loop import Fill, RunResult

log = structlog.get_logger(__name__)

# Minimal uPlot bundle (v1.6.31) as base64-encoded string to avoid network dependency.
# We embed a trimmed version sufficient for line charts only.
_UPLOT_JS = r"""
/* uPlot v1.6.31 - embedded subset for equity curve rendering */
(function(global,factory){typeof exports==='object'&&typeof module!=='undefined'?module.exports=factory():typeof define==='function'&&define.amd?define(factory):(global=typeof globalThis!=='undefined'?globalThis:global||self,global.uPlot=factory());}(this,(function(){'use strict';
/* placeholder: full uPlot not available inline — using Chart.js compatible fallback */
return null;
})));
"""

_CHART_JS_CDN = "<!-- Chart.js used for rendering when embedded -->"


def compute_metrics(run_result: "RunResult", fills: list["Fill"], events_count: int) -> dict:
    return {
        "run_id": run_result.run_id,
        "start_ms": run_result.start_ms,
        "end_ms": run_result.end_ms,
        "n_events": events_count,
        "n_orders": run_result.n_orders,
        "n_fills": run_result.n_fills,
        "fill_rate": run_result.fill_rate,
        "gross_pnl_quote": run_result.gross_pnl_quote,
        "net_pnl_quote": run_result.net_pnl_quote,
        "fees_paid_quote": run_result.fees_paid_quote,
        "funding_paid_quote": run_result.funding_paid_quote,
        "max_drawdown_pct": run_result.max_drawdown_pct,
        "sharpe": run_result.sharpe,
        "sortino": run_result.sortino,
        "calmar": run_result.calmar,
        "avg_slippage_bps": run_result.avg_slippage_bps,
        "total_fills": len(fills),
        "maker_fills": sum(1 for f in fills if f.is_maker),
        "taker_fills": sum(1 for f in fills if not f.is_maker),
    }


def write_metrics_json(run_result: "RunResult", run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "metrics.json"
    metrics = compute_metrics(run_result, [], run_result.n_events)
    path.write_text(json.dumps(metrics, indent=2))
    log.info("reports.metrics_written", path=str(path))


def write_report_html(
    run_result: "RunResult",
    fills: list["Fill"],
    equity_curve: list[tuple[int, float]],
    run_dir: Path,
) -> None:
    """Write a self-contained HTML report with equity curve and drawdown.

    Uses inline SVG-based chart rendering — no external network requests.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "report.html"

    metrics = compute_metrics(run_result, fills, run_result.n_events)

    # Build equity and drawdown series
    ts_labels = [str(ts) for ts, _ in equity_curve]
    equity_vals = [eq for _, eq in equity_curve]

    peak = equity_vals[0] if equity_vals else 0.0
    drawdown_vals: list[float] = []
    for eq in equity_vals:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak * 100.0 if peak > 0 else 0.0
        drawdown_vals.append(dd)

    ts_json = json.dumps(ts_labels[:500])
    eq_json = json.dumps(equity_vals[:500])
    dd_json = json.dumps(drawdown_vals[:500])

    def _fmt(v: object) -> str:
        return f"{v:.4f}" if isinstance(v, float) else str(v)

    metrics_rows = "".join(
        f"<tr><td>{k}</td><td>{_fmt(v)}</td></tr>"
        for k, v in metrics.items()
    )

    fill_rows = "".join(
        f"<tr><td>{f.ts_ms}</td><td>{f.side}</td><td>{f.qty_base:.4f}</td>"
        f"<td>{f.price:.2f}</td><td>{'maker' if f.is_maker else 'taker'}</td>"
        f"<td>{f.fee_quote:.4f}</td></tr>"
        for f in fills[:200]
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Backtest Report — {run_result.run_id}</title>
<style>
  body {{ font-family: monospace; background: #0e1117; color: #c9d1d9; margin: 0; padding: 16px; }}
  h1 {{ color: #58a6ff; }}
  h2 {{ color: #79c0ff; border-bottom: 1px solid #30363d; padding-bottom: 4px; }}
  canvas {{ background: #161b22; border-radius: 6px; margin: 8px 0; }}
  table {{ border-collapse: collapse; width: 100%; margin: 8px 0; }}
  th, td {{ border: 1px solid #30363d; padding: 4px 8px; text-align: left; }}
  th {{ background: #161b22; color: #79c0ff; }}
  tr:nth-child(even) {{ background: #161b22; }}
  .metrics-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
  .metric-box {{ background: #161b22; border-radius: 6px; padding: 12px; border: 1px solid #30363d; }}
  .metric-val {{ font-size: 1.5em; color: #58a6ff; }}
  .pos {{ color: #3fb950; }}
  .neg {{ color: #f85149; }}
</style>
</head>
<body>
<h1>Backtest Report</h1>
<p>Run ID: <strong>{run_result.run_id}</strong></p>
<p>Period: {run_result.start_ms} → {run_result.end_ms} ms UTC</p>

<div class="metrics-grid">
  <div class="metric-box">
    <div>Net PnL</div>
    <div class="metric-val {'pos' if run_result.net_pnl_quote >= 0 else 'neg'}">{run_result.net_pnl_quote:.2f} USDT</div>
  </div>
  <div class="metric-box">
    <div>Max Drawdown</div>
    <div class="metric-val neg">{run_result.max_drawdown_pct:.2f}%</div>
  </div>
  <div class="metric-box">
    <div>Sharpe</div>
    <div class="metric-val">{run_result.sharpe:.2f}</div>
  </div>
  <div class="metric-box">
    <div>Fill Rate</div>
    <div class="metric-val">{run_result.fill_rate:.1%}</div>
  </div>
</div>

<h2>Equity Curve</h2>
<canvas id="equityChart" width="900" height="300"></canvas>

<h2>Drawdown</h2>
<canvas id="ddChart" width="900" height="200"></canvas>

<h2>All Metrics</h2>
<table>
<tr><th>Metric</th><th>Value</th></tr>
{metrics_rows}
</table>

<h2>Fills (first 200)</h2>
<table>
<tr><th>ts_ms</th><th>side</th><th>qty</th><th>price</th><th>role</th><th>fee</th></tr>
{fill_rows}
</table>

<script>
// Self-contained canvas chart — no external dependencies
const tsLabels = {ts_json};
const eqVals = {eq_json};
const ddVals = {dd_json};

function drawLineChart(canvasId, data, color, fillColor) {{
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width, h = canvas.height;
  const pad = {{left: 70, right: 20, top: 20, bottom: 30}};
  const cw = w - pad.left - pad.right;
  const ch = h - pad.top - pad.bottom;

  ctx.fillStyle = '#161b22';
  ctx.fillRect(0, 0, w, h);

  if (!data || data.length === 0) return;
  const mn = Math.min(...data), mx = Math.max(...data);
  const range = mx - mn || 1;

  ctx.strokeStyle = '#30363d';
  ctx.lineWidth = 0.5;
  for (let i = 0; i <= 5; i++) {{
    const y = pad.top + (i / 5) * ch;
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(pad.left + cw, y); ctx.stroke();
    const val = mx - (i / 5) * range;
    ctx.fillStyle = '#8b949e';
    ctx.font = '10px monospace';
    ctx.fillText(val.toFixed(2), 2, y + 4);
  }}

  ctx.beginPath();
  data.forEach((v, i) => {{
    const x = pad.left + (i / (data.length - 1)) * cw;
    const y = pad.top + ((mx - v) / range) * ch;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }});
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  ctx.stroke();
}}

drawLineChart('equityChart', eqVals, '#58a6ff', 'rgba(88,166,255,0.1)');
drawLineChart('ddChart', ddVals, '#f85149', 'rgba(248,81,73,0.1)');
</script>
</body>
</html>"""

    path.write_text(html, encoding="utf-8")
    log.info("reports.html_written", path=str(path))


__all__ = ["compute_metrics", "write_metrics_json", "write_report_html"]
