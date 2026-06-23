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
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = (base_url or os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL)).rstrip("/")
        if not self.base_url.endswith("/v1"):
            self.base_url = self.base_url + "/v1"
        self.api_key = api_key or os.getenv("LLM_TOKEN", "")
        self.model = model or os.getenv("LLM_MODEL", DEFAULT_MODEL)
        self.timeout = timeout
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
            "max_tokens": 1024,
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
    ) -> str:
        """Build the system + user prompt for the LLM."""
        recent = "\n".join(
            f"- {e.get('ts', '?')}: {e.get('action', '?')} conf={e.get('confidence', 0)} reason={e.get('reason', '?')[:60]}"
            for e in recent_logs[-10:]
        )

        system = """You are a quantitative risk committee for a live trading competition.

CRITICAL: You must output ONLY a raw JSON object. No markdown, no explanation, no code blocks, no thinking tags.
The JSON must match this exact schema and contain REAL values (not placeholders):
{"action": "BUY", "confidence": 0.72, "reason": "Uptrend confirmed with acceptable risk metrics", "position_size": 0.50, "sl_price": 92000.0, "tp_price": 98000.0, "risk_approved": true}

Position sizing rules:
- confidence < 0.60 → action=HOLD, position_size=0.0
- 0.60-0.75 → position_size=0.25
- 0.75-0.85 → position_size=0.50
- > 0.85 → position_size=1.00

If risk rules are violated, set risk_approved=false and action=HOLD.
"""

        user = f"""
COMPETITION RULES:
{json.dumps(competition_rules, indent=2)}

CURRENT STATE:
- Signal: {signal} | Confidence: {confidence}
- Bar close: {bar.get('close', '?')} | Volume: {bar.get('volume', '?')}
- Account balance: {account.get('balance', 0)} | Equity: {account.get('equity', 0)}
- Open positions: {len(positions)}
- Daily trades so far: {risk_state.get('trade_count_today', 0)}
- Current drawdown: {risk_state.get('drawdown_pct', 0):.2f}%
- Daily PnL: {risk_state.get('daily_pnl_pct', 0):.2f}%
- Total exposure: {risk_state.get('total_exposure_pct', 0):.2f}%

RECENT TRADES:
{recent}

YOUR TASK:
Review the signal and return ONLY the JSON object.
"""
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
            signal, confidence, bar, positions, account, risk_state, recent_logs, rules
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

        # Strategy 3: Try to find the first JSON object via regex
        json_match = re.search(r'\{[^{}]*"action"[^{}]*\}', text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
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
