"""LLM Review Agent for trading decisions.

Connects to Fireworks AI (or any OpenAI-compatible endpoint) to validate
trading signals. Returns a fixed JSON schema that the executor consumes.

Usage:
    from intraday.llm.review import LLMReviewAgent

    agent = LLMReviewAgent(
        base_url="https://api.fireworks.ai/inference",
        api_key="...",
        model="accounts/fireworks/routers/kimi-k2p6-turbo",
    )
    review = agent.review(signal="BUY", confidence=0.72, market_data={...})
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any

import httpx
import structlog

log = structlog.get_logger(__name__)

DEFAULT_BASE_URL = "https://api.fireworks.ai/inference/v1"
DEFAULT_MODEL = "accounts/fireworks/routers/kimi-k2p6-turbo"


@dataclass
class LLMReview:
    """Fixed JSON schema returned by the LLM."""

    action: str          # BUY | SELL | HOLD
    confidence: float    # 0.0-1.0
    reason: str          # human-readable explanation
    position_size: float # 0.0-1.0
    sl_price: float      # stop-loss price
    tp_price: float      # take-profit price
    risk_approved: bool    # True if LLM approves

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "confidence": self.confidence,
            "reason": self.reason,
            "position_size": self.position_size,
            "sl_price": self.sl_price,
            "tp_price": self.tp_price,
            "risk_approved": self.risk_approved,
        }


class LLMReviewAgent:
    """Fireworks/OpenAI-compatible LLM review agent.

    Args:
        base_url: API endpoint (e.g., https://api.fireworks.ai/inference/v1)
        api_key: Bearer token
        model: Model name (e.g., accounts/fireworks/routers/kimi-k2p6-turbo)
        timeout: Request timeout in seconds
        debug: If True, logs the full prompt to the structlog logger
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = 120.0,
        debug: bool = False,
    ) -> None:
        self.base_url = (base_url or os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL)).rstrip("/")
        if not self.base_url.endswith("/v1"):
            self.base_url = self.base_url + "/v1"
        # Only fall back to env if api_key is None (not provided), not if explicitly empty string
        if api_key is None:
            self.api_key = os.getenv("LLM_TOKEN", "")
        else:
            self.api_key = api_key
        self.model = model or os.getenv("LLM_MODEL", DEFAULT_MODEL)
        self.timeout = timeout
        self.debug = debug
        self._client = httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=timeout,
        )

    def _call_llm(self, messages: list[dict[str, str]], temperature: float = 0.2) -> str:
        """Send chat completion request and return assistant text."""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 8192,
            "response_format": {"type": "json_object"},
        }
        try:
            resp = self._client.post("/chat/completions", json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as exc:
            log.error("llm_http_error", status=exc.response.status_code, body=exc.response.text)
            raise
        except Exception as exc:
            log.error("llm_request_error", error=str(exc))
            raise

    def _build_prompt(
        self,
        signal: str,
        confidence: float,
        bar: dict,
        positions: list[dict],
        account: dict,
        risk_state: dict,
        recent_logs: list[dict],
        competition_rules: dict,
        pipeline: dict | None,
    ) -> tuple[str, str]:
        """Build the system + user prompt for the LLM.

        Returns (system_prompt, user_prompt).
        """
        # ── Recent trades ──────────────────────────────────────────────────────
        recent_lines: list[str] = []
        for e in recent_logs[-20:]:
            line = (
                f"- {e.get('ts', '?'):19} | {e.get('action', '?'):>4} | "
                f"conf={e.get('confidence', 0):.2f} | "
                f"pos={e.get('position_size', 0):.2f} | "
                f"bal={e.get('account_balance', 0):,.2f} | "
                f"pnl={e.get('daily_pnl_pct', 0):+.2f}% | "
                f"reason={e.get('reason', '?')[:80]}"
            )
            recent_lines.append(line)
        recent = "\n".join(recent_lines) if recent_lines else "(none)"

        # ── Pipeline context ───────────────────────────────────────────────────
        pipe_text = ""
        if pipeline:
            fc = pipeline.get("forecast", {})
            of = pipeline.get("orderflow", {})
            rg = pipeline.get("regime", {})
            rk = pipeline.get("risk", {})
            so = pipeline.get("stay_out", {})
            de = pipeline.get("decision", {})
            pipe_text = f"""
PIPELINE ANALYSIS:
- Forecast: p_up={fc.get('p_up', 0.5):.4f} | confidence={fc.get('confidence', 0):.4f} | expected_move={fc.get('expected_move', 0):.4f}
- Orderflow: flow_bias={of.get('flow_bias', 0):.2f} | vpin={of.get('vpin', 0):.4f}
- Regime: {rg.get('regime', 'unknown')} | vol_regime={rg.get('vol_regime', 'normal')}
- Risk: allow_trade={rk.get('allow_trade', True)} | multiplier={rk.get('risk_multiplier', 1.0):.2f}
- StayOut: mode={so.get('mode', 'normal')}
- Decision: side={de.get('side', 'flat')} | confidence={de.get('confidence', 0):.4f} | reason={de.get('reason', 'N/A')}
"""

        # ── Positions ──────────────────────────────────────────────────────────
        pos_lines: list[str] = []
        for p in positions:
            pos_lines.append(
                f"  {p.get('side', '?'):>5} {p.get('volume', 0):.4f} lot @ "
                f"{p.get('open_price', 0):.2f} | PnL={p.get('profit', 0):+.2f} | "
                f"SL={p.get('sl', 0):.2f} TP={p.get('tp', 0):.2f}"
            )
        pos_text = "\n".join(pos_lines) if pos_lines else "  (no open positions)"

        # ── Candle ─────────────────────────────────────────────────────────────
        candle_text = (
            f"Open: {bar.get('open', 0):.2f} | High: {bar.get('high', 0):.2f} | "
            f"Low: {bar.get('low', 0):.2f} | Close: {bar.get('close', 0):.2f} | "
            f"Volume: {bar.get('volume', 0):.2f} | Trades: {bar.get('trade_count', 0)} | "
            f"TakerBuy: {bar.get('taker_buy_ratio', 0):.2%}"
        )

        system = """You are a quantitative risk committee for a live trading competition.

CRITICAL INSTRUCTIONS:
1. You MUST output ONLY a raw JSON object. No markdown, no explanation, no code blocks, no thinking tags, no preamble.
2. The JSON must match this EXACT schema with REAL values (not placeholders):
   {"action": "BUY", "confidence": 0.72, "reason": "Uptrend confirmed with acceptable risk metrics", "position_size": 0.50, "sl_price": 92000.0, "tp_price": 98000.0, "risk_approved": true}
3. Do NOT add any text before or after the JSON. Do NOT wrap it in ```json fences.
4. The response must be parseable by json.loads() directly.

Position sizing rules:
- confidence < 0.60 -> action=HOLD, position_size=0.0
- 0.60-0.75 -> position_size=0.25
- 0.75-0.85 -> position_size=0.50
- > 0.85 -> position_size=1.00

If risk rules are violated, set risk_approved=false and action=HOLD.
"""

        user = f"""
COMPETITION RULES:
{json.dumps(competition_rules, indent=2)}

CURRENT STATE:
- Signal: {signal} | Confidence: {confidence}
- Candle: {candle_text}
- Account balance: {account.get('balance', 0):,.2f} | Equity: {account.get('equity', 0):,.2f} | Profit: {account.get('profit', 0):+.2f}
- Open positions: {len(positions)}
{pos_text}
- Daily trades so far: {risk_state.get('trade_count_today', 0)}
- Current drawdown: {risk_state.get('drawdown_pct', 0):.2f}%
- Daily PnL: {risk_state.get('daily_pnl_pct', 0):+.2f}%
- Weekly PnL: {risk_state.get('weekly_pnl_pct', 0):+.2f}%
- Total exposure: {risk_state.get('total_exposure_pct', 0):.2f}%
{pipe_text}

RECENT TRADES (last 20):
{recent}

YOUR TASK:
Review the signal and return ONLY the JSON object.
"""
        if self.debug:
            log.info(
                "llm_prompt_debug",
                system_chars=len(system),
                user_chars=len(user),
                total_chars=len(system) + len(user),
                prompt=user,
            )
        return system, user

    def review(
        self,
        signal: str,
        confidence: float,
        bar: dict,
        positions: list[dict],
        account: dict,
        risk_state: dict,
        recent_logs: list[dict],
        competition_rules: dict | None = None,
        pipeline: dict | None = None,
    ) -> LLMReview:
        """Get LLM review of the trading signal.

        Returns LLMReview with action, confidence, position_size, SL, TP.
        Falls back to rule-based if the LLM call fails.
        """
        if not self.api_key:
            log.warning("no_api_key")
            return self._rule_based(signal, confidence, bar, risk_state)

        rules = competition_rules or _default_competition_rules()
        system, user = self._build_prompt(
            signal, confidence, bar, positions, account, risk_state, recent_logs, rules, pipeline
        )

        try:
            text = self._call_llm(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ]
            )
            return self._parse_response(text)
        except Exception as exc:
            log.error("llm_review_failed", error=str(exc))
            return self._rule_based(signal, confidence, bar, risk_state)

    def _parse_response(self, text: str) -> LLMReview:
        """Parse JSON response from LLM. Handles text wrappers, reasoning, markdown."""
        import re

        # Strategy 1: Try to find JSON block inside markdown fences
        if "```json" in text:
            inner = text.split("```json")[1].split("```")[0]
            try:
                data = json.loads(inner.strip())
                return self._build_review(data)
            except Exception:
                pass

        # Strategy 2: Try to find JSON block inside plain ``` fences
        if "```" in text:
            blocks = text.split("```")
            for block in blocks:
                try:
                    data = json.loads(block.strip())
                    if "action" in data:
                        return self._build_review(data)
                except Exception:
                    pass

        # Strategy 3: Extract the last JSON object from the text using brace matching.
        # Models often emit reasoning text first, then the JSON at the end.
        last_json = self._extract_last_json(text)
        if last_json:
            try:
                data = json.loads(last_json)
                if "action" in data:
                    return self._build_review(data)
            except Exception:
                pass

        # Strategy 4: Try to parse the entire text as JSON
        try:
            data = json.loads(text.strip())
            return self._build_review(data)
        except Exception as exc:
            log.warning("llm_json_parse_failed", text=text[:200], error=str(exc))
            return self._rule_based("HOLD", 0.0, {}, {})

    def _extract_last_json(self, text: str) -> str | None:
        """Find the last top-level JSON object in text by brace matching."""
        # Collect all brace positions by scanning the string directly
        opens: list[int] = []
        closes: list[int] = []
        for i, ch in enumerate(text):
            if ch == '{':
                opens.append(i)
            elif ch == '}':
                closes.append(i)

        if not opens or not closes:
            return None

        # Work backwards from the last close brace to find a balanced open
        for close_idx in reversed(closes):
            depth = 0
            for i in range(close_idx, -1, -1):
                if text[i] == '}':
                    depth += 1
                elif text[i] == '{':
                    depth -= 1
                    if depth == 0:
                        candidate = text[i:close_idx + 1]
                        # Sanity check: must contain "action" key
                        if '"action"' in candidate:
                            return candidate
                        break
        return None

    def _build_review(self, data: dict) -> LLMReview:
        """Build LLMReview from parsed dict."""
        action = str(data.get("action", "HOLD")).upper()
        if action not in ("BUY", "SELL", "HOLD"):
            action = "HOLD"
        return LLMReview(
            action=action,
            confidence=float(data.get("confidence", 0.0)),
            reason=str(data.get("reason", "LLM parse fallback")),
            position_size=float(data.get("position_size", 0.0)),
            sl_price=float(data.get("sl_price", 0.0)),
            tp_price=float(data.get("tp_price", 0.0)),
            risk_approved=bool(data.get("risk_approved", False)),
        )

    def _rule_based(
        self,
        signal: str,
        confidence: float,
        bar: dict,
        risk_state: dict,
    ) -> LLMReview:
        """Fallback when LLM is unavailable."""
        dd = risk_state.get("drawdown_pct", 0)
        daily_pnl = risk_state.get("daily_pnl_pct", 0)

        if dd >= 12.0:
            return LLMReview("HOLD", 0.0, f"Hard drawdown limit ({dd:.1f}%)", 0.0, 0.0, 0.0, False)
        if daily_pnl <= -2.0:
            return LLMReview("HOLD", 0.0, f"Daily loss limit ({daily_pnl:.1f}%)", 0.0, 0.0, 0.0, False)
        if confidence < 0.65:
            return LLMReview("HOLD", confidence, f"Confidence {confidence:.2f} < 0.65", 0.0, 0.0, 0.0, False)

        if confidence < 0.60:
            size = 0.0
        elif confidence < 0.75:
            size = 0.25
        elif confidence < 0.85:
            size = 0.50
        else:
            size = 1.00

        close = bar.get("close", 0.0)
        atr = bar.get("high", close) - bar.get("low", close) if bar.get("high") else close * 0.001
        sl = close - atr * 2 if signal == "BUY" else close + atr * 2
        tp = close + atr * 4 if signal == "BUY" else close - atr * 4

        risk = abs(close - sl)
        reward = abs(tp - close)
        rr = reward / risk if risk > 0 else 0
        if rr < 2.0:
            return LLMReview("HOLD", confidence, f"R/R {rr:.1f} < 2.0", 0.0, 0.0, 0.0, False)

        return LLMReview(
            action=signal.upper(),
            confidence=confidence,
            reason=f"Rule-based: {signal} conf={confidence:.2f} size={size} R/R={rr:.1f}",
            position_size=size,
            sl_price=round(sl, 2),
            tp_price=round(tp, 2),
            risk_approved=True,
        )

    def analyze_chart(
        self,
        candles: list[dict],
        indicators: dict,
        positions: list[dict],
        account: dict,
        risk_state: dict,
        recent_logs: list[dict],
        competition_rules: dict | None = None,
        last_trade_reason: str | None = None,
    ) -> LLMReview:
        """Use LLM as primary decision maker.

        Instead of validating a signal, the LLM analyzes the chart
        and returns a trade decision directly.
        """
        if not self.api_key:
            log.warning("no_api_key")
            return self._rule_based("HOLD", 0.0, {}, risk_state)

        rules = competition_rules or _default_competition_rules()
        system, user = self._build_chart_prompt(
            candles, indicators, positions, account, risk_state, recent_logs, rules, last_trade_reason
        )

        try:
            text = self._call_llm(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ]
            )
            return self._parse_response(text)
        except Exception as exc:
            log.error("llm_analyze_failed", error=str(exc))
            return self._rule_based("HOLD", 0.0, {}, risk_state)

    def _build_chart_prompt(
        self,
        candles: list[dict],
        indicators: dict,
        positions: list[dict],
        account: dict,
        risk_state: dict,
        recent_logs: list[dict],
        competition_rules: dict,
        last_trade_reason: str | None = None,
    ) -> tuple[str, str]:
        """Build a chart analysis prompt for the LLM.

        Describes the last 40 candles as a human trader would see them.
        """
        # Build candle table
        candle_lines = []
        for i, c in enumerate(candles[-40:]):
            candle_lines.append(
                f"  {i+1:2d}. | O:{c['open']:8.2f} | H:{c['high']:8.2f} | L:{c['low']:8.2f} | C:{c['close']:8.2f} | V:{c['volume']:8.2f} | T:{c['trades']:5d} | Taker:{c['taker_buy_pct']:5.1f}%"
            )
        candle_table = "\n".join(candle_lines)

        # Current price and recent stats
        current = candles[-1] if candles else {"close": 0.0, "high": 0.0, "low": 0.0}
        close = current["close"]
        recent_high = max(c["high"] for c in candles[-40:]) if candles else close
        recent_low = min(c["low"] for c in candles[-40:]) if candles else close
        price_range = recent_high - recent_low

        # Indicator summary
        ind_text = ""
        if indicators:
            ind_lines = []
            for k, v in indicators.items():
                if v is not None:
                    ind_lines.append(f"- {k}: {v}")
            ind_text = "\n".join(ind_lines) if ind_lines else "  (no indicators available)"
        else:
            ind_text = "  (no indicators available)"

        # Positions
        pos_lines: list[str] = []
        for p in positions:
            pos_lines.append(
                f"  {p.get('side', '?'):>5} {p.get('volume', 0):.4f} lot @ "
                f"{p.get('open_price', 0):.2f} | PnL={p.get('profit', 0):+.2f} | "
                f"SL={p.get('sl', 0):.2f} TP={p.get('tp', 0):.2f}"
            )
        pos_text = "\n".join(pos_lines) if pos_lines else "  (no open positions)"

        # Recent trades
        recent_lines: list[str] = []
        for e in recent_logs[-10:]:
            line = (
                f"- {e.get('ts', '?'):19} | {e.get('action', '?'):>4} | "
                f"conf={e.get('confidence', 0):.2f} | "
                f"pos={e.get('position_size', 0):.2f} | "
                f"bal={e.get('account_balance', 0):,.2f} | "
                f"pnl={e.get('daily_pnl_pct', 0):+.2f}%"
            )
            recent_lines.append(line)
        recent = "\n".join(recent_lines) if recent_lines else "  (no recent trades)"

        system = """You are a professional BTC/USD intraday trader. You analyze charts and make trade decisions.

CRITICAL INSTRUCTIONS:
1. You MUST output ONLY a raw JSON object. No markdown, no explanation, no code blocks.
2. The JSON must match this EXACT schema:
   {"action": "BUY", "confidence": 0.72, "reason": "Bullish breakout with volume confirmation", "position_size": 0.25, "sl_price": 62000.0, "tp_price": 64000.0, "risk_approved": true}
3. Do NOT add any text before or after the JSON.
4. Available actions: "BUY", "SELL", "HOLD"

Trading Rules:
- Look for ANY edge: momentum, mean reversion, volume spikes, support/resistance tests
- Set stop loss at recent swing low/high or using ATR
- Set take profit at 2:1 risk/reward minimum
- Position size: 0.25 for moderate setups, 0.50 for strong setups, 1.00 for exceptional setups
- IMPORTANT: If you see even a slight directional bias or momentum, take a small position (0.25) rather than HOLD
- The competition requires 5-10 trades per day — don't be too conservative
- If already in a position, consider closing or adding based on the setup
"""

        last_trade_text = f"\nLAST TRADE CLOSE: {last_trade_reason}" if last_trade_reason else ""

        user = f"""
COMPETITION RULES:
{json.dumps(competition_rules, indent=2)}

CHART ANALYSIS — BTCUSDT Recent 40 Candles:

{candle_table}

KEY LEVELS:
- Recent High: {recent_high:.2f}
- Recent Low: {recent_low:.2f}
- Price Range: {price_range:.2f}
- Current Price: {close:.2f}

TECHNICAL INDICATORS:
{ind_text}

ACCOUNT STATUS:
- Balance: {account.get('balance', 0):,.2f}
- Equity: {account.get('equity', 0):,.2f}
- Profit: {account.get('profit', 0):+.2f}
- Open Positions: {len(positions)}
{pos_text}
- Daily trades: {risk_state.get('trade_count_today', 0)}
- Drawdown: {risk_state.get('drawdown_pct', 0):.2f}%
- Daily PnL: {risk_state.get('daily_pnl_pct', 0):+.2f}%

RECENT TRADES:
{recent}
{last_trade_text}

YOUR TASK:
Analyze the chart above. Identify the trend, support/resistance, volume patterns, and any clear trade setups.

Return ONLY a JSON object with your decision:
- action: "BUY" | "SELL" | "HOLD"
- confidence: 0.0-1.0 (how confident you are in this setup)
- reason: brief explanation of the setup
- position_size: 0.0-1.0 (fraction of capital to risk)
- sl_price: stop loss price
- tp_price: take profit price
- risk_approved: true if this trade follows the risk rules

If no clear setup exists, return HOLD with confidence 0.0.
"""
        if self.debug:
            log.info(
                "llm_chart_prompt_debug",
                system_chars=len(system),
                user_chars=len(user),
                total_chars=len(system) + len(user),
                prompt=user,
            )
        return system, user


def _default_competition_rules() -> dict:
    return {
        "competition_context": {
            "name": "AI Quant Trading Competition",
            "initial_capital": 1000000,
            "max_leverage": 30,
            "stop_out_level_percent": 30,
            "primary_asset": "BTCUSD",
            "objective": "Maximize risk-adjusted returns while maintaining low drawdown and sufficient trading activity.",
        },
        "leaderboard_priorities": [
            "Preserve capital",
            "Maintain high Sharpe Ratio",
            "Control Maximum Drawdown",
            "Generate positive returns",
            "Meet trading volume requirements",
        ],
        "risk_constraints": {
            "max_risk_per_trade_percent": 0.5,
            "max_daily_loss_percent": 2.0,
            "max_weekly_loss_percent": 5.0,
            "soft_drawdown_limit_percent": 8.0,
            "hard_drawdown_limit_percent": 12.0,
            "max_total_exposure_percent": 25.0,
            "preferred_leverage": 1,
            "absolute_max_leverage": 5,
        },
        "trade_frequency": {
            "goal": "Generate enough volume for Sharpe and drawdown calculations",
            "minimum_trades_per_day": 5,
            "target_trades_per_day": 10,
            "avoid_overtrading": True,
        },
        "position_sizing": {
            "confidence_below_0_60": 0.0,
            "confidence_0_60_to_0_75": 0.25,
            "confidence_0_75_to_0_85": 0.50,
            "confidence_above_0_85": 1.00,
            "reduce_size_during_high_volatility": True,
        },
        "trade_filtering": {
            "reject_if_confidence_below": 0.65,
            "reject_if_risk_reward_below": 1.5,
            "reject_if_spread_abnormally_high": True,
            "reject_if_drawdown_limit_hit": True,
        },
        "take_profit_and_stop_loss": {
            "mandatory": True,
            "stop_loss_method": "ATR",
            "take_profit_method": "ATR",
            "minimum_risk_reward": 2.0,
        },
        "capital_preservation_rules": [
            "Never martingale",
            "Never average losing positions",
            "Never use max leverage",
            "Never remove stop loss",
            "Never risk account survival for single trade",
        ],
    }
