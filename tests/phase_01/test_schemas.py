"""Test Pydantic schemas for market data."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from intraday.data.schemas import (
    Depth,
    DepthLevel,
    FundingRate,
    Kline,
    Liquidation,
    MarkPrice,
    Trade,
)


def test_kline_valid() -> None:
    """Test valid kline creation."""
    kline = Kline(
        symbol="BTCUSDT",
        interval="5m",
        open_time_ms=1700000000000,
        close_time_ms=1700000300000,
        open=50000.0,
        high=50100.0,
        low=49900.0,
        close=50050.0,
        volume=100.5,
        quote_volume=5000000.0,
        num_trades=1000,
        taker_buy_base_volume=60.0,
        taker_buy_quote_volume=3000000.0,
    )

    assert kline.symbol == "BTCUSDT"
    assert kline.interval == "5m"
    assert isinstance(kline.open_time, datetime)
    assert isinstance(kline.close_time, datetime)


def test_kline_invalid_timestamp() -> None:
    """Test kline with invalid timestamp."""
    with pytest.raises(ValidationError):
        Kline(
            symbol="BTCUSDT",
            interval="5m",
            open_time_ms=-1,  # Invalid
            close_time_ms=1700000300000,
            open=50000.0,
            high=50100.0,
            low=49900.0,
            close=50050.0,
            volume=100.5,
            quote_volume=5000000.0,
            num_trades=1000,
            taker_buy_base_volume=60.0,
            taker_buy_quote_volume=3000000.0,
        )


def test_trade_side() -> None:
    """Test trade side derivation."""
    # Buyer is maker → sell order filled → aggressive side is SELL
    trade_sell = Trade(
        symbol="BTCUSDT",
        trade_id=123,
        price=50000.0,
        quantity=0.5,
        time_ms=1700000000000,
        is_buyer_maker=True,
    )
    assert trade_sell.side == "sell"

    # Buyer is taker → buy order filled → aggressive side is BUY
    trade_buy = Trade(
        symbol="BTCUSDT",
        trade_id=124,
        price=50001.0,
        quantity=0.3,
        time_ms=1700000001000,
        is_buyer_maker=False,
    )
    assert trade_buy.side == "buy"


def test_depth_properties() -> None:
    """Test depth computed properties."""
    depth = Depth(
        symbol="BTCUSDT",
        time_ms=1700000000000,
        last_update_id=12345,
        bids=[
            DepthLevel(price=50000.0, quantity=1.0),
            DepthLevel(price=49999.0, quantity=2.0),
        ],
        asks=[
            DepthLevel(price=50001.0, quantity=0.5),
            DepthLevel(price=50002.0, quantity=1.5),
        ],
    )

    assert depth.best_bid == 50000.0
    assert depth.best_ask == 50001.0
    assert depth.mid_price == 50000.5
    assert depth.spread_bps is not None
    assert depth.spread_bps > 0  # Should be ~0.2 bps


def test_depth_empty() -> None:
    """Test depth with empty book."""
    depth = Depth(
        symbol="BTCUSDT",
        time_ms=1700000000000,
        last_update_id=12345,
        bids=[],
        asks=[],
    )

    assert depth.best_bid is None
    assert depth.best_ask is None
    assert depth.mid_price is None
    assert depth.spread_bps is None


def test_funding_rate_annualized() -> None:
    """Test funding rate annualization."""
    funding = FundingRate(
        symbol="BTCUSDT",
        funding_time_ms=1700000000000,
        funding_rate=0.0001,  # 0.01% per 8h
        mark_price=50000.0,
    )

    # 3 fundings per day × 365 days = 1095
    assert funding.annualized_rate == pytest.approx(0.0001 * 1095, rel=1e-6)


def test_mark_price_premium() -> None:
    """Test mark price premium calculation."""
    mark = MarkPrice(
        symbol="BTCUSDT",
        time_ms=1700000000000,
        mark_price=50010.0,
        index_price=50000.0,
        last_funding_rate=0.0001,
        next_funding_time_ms=1700028800000,
    )

    # Premium = (50010 - 50000) / 50000 * 10000 = 2 bps
    assert mark.premium_bps == pytest.approx(2.0, rel=1e-6)


def test_liquidation_notional() -> None:
    """Test liquidation notional calculation."""
    liq = Liquidation(
        symbol="BTCUSDT",
        time_ms=1700000000000,
        side="BUY",  # Long liquidation
        order_type="LIMIT",
        price=49500.0,
        quantity=2.0,
        average_price=49450.0,
    )

    assert liq.notional_usd == pytest.approx(2.0 * 49450.0, rel=1e-6)


def test_openinterest_valid() -> None:
    """Test open interest creation."""
    oi = FundingRate(
        symbol="BTCUSDT",
        funding_time_ms=1700000000000,
        funding_rate=0.0001,
    )

    assert oi.symbol == "BTCUSDT"
    assert isinstance(oi.funding_time, datetime)
