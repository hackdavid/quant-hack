#!/usr/bin/env python3
"""Analyze BTC 1m trend for user's position."""
import httpx
import pandas as pd

url = 'https://data-api.binance.vision/api/v3/klines'
params = {'symbol': 'BTCUSDT', 'interval': '1m', 'limit': 100}
r = httpx.get(url, params=params, timeout=30)
r.raise_for_status()

data = []
for row in r.json():
    data.append({
        'timestamp': pd.to_datetime(row[0], unit='ms'),
        'open': float(row[1]),
        'high': float(row[2]),
        'low': float(row[3]),
        'close': float(row[4]),
        'volume': float(row[5]),
    })

df = pd.DataFrame(data)

# EMAs
df['ema25'] = df['close'].ewm(span=25).mean()
df['ema50'] = df['close'].ewm(span=50).mean()
df['ema100'] = df['close'].ewm(span=100).mean()

# RSI (14 period)
delta = df['close'].diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = delta.where(delta < 0, 0).rolling(14).mean()
rs = gain / loss
rsi = 100 - (100 / (1 + rs))

# ATR (14 period)
h1 = df['high'] - df['low']
h2 = abs(df['high'] - df['close'].shift())
h3 = abs(df['low'] - df['close'].shift())
df['tr'] = pd.concat([h1, h2, h3], axis=1).max(axis=1)
atr = df['tr'].rolling(14).mean()

# ADX (14 period)
plus_dm = df['high'].diff().where((df['high'].diff() > df['low'].diff()) & (df['high'].diff() > 0), 0)
minus_dm = -df['low'].diff().where((df['low'].diff() > df['high'].diff()) & (df['low'].diff() > 0), 0)
plus_di = 100 * plus_dm.rolling(14).mean() / atr
minus_di = 100 * minus_dm.rolling(14).mean() / atr
dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
adx = dx.rolling(14).mean()

# Recent candles
recent = df.tail(30)
print('=== LAST 30 MINUTES (1m candles) ===')
header = "Time       | Close      | EMA25      | EMA50      | EMA100     | RSI  | ADX  | Volume"
print(header)
print('-' * 90)
for _, row in recent.iterrows():
    print(f"{row['timestamp'].strftime('%H:%M'):10s} | {row['close']:10.2f} | {row['ema25']:10.2f} | {row['ema50']:10.2f} | {row['ema100']:10.2f} | {rsi.loc[_]:5.0f} | {adx.loc[_]:5.0f} | {row['volume']:12.2f}")

print('')
print('=== CURRENT ANALYSIS ===')
last = df.iloc[-1]
print(f'Current Price: {last["close"]:.2f}')
print(f'EMA25: {last["ema25"]:.2f}')
print(f'EMA50: {last["ema50"]:.2f}')
print(f'EMA100: {last["ema100"]:.2f}')
print(f'RSI: {rsi.iloc[-1]:.0f}')
print(f'ADX: {adx.iloc[-1]:.0f}')

# Trend
if last['ema25'] > last['ema50'] > last['ema100']:
    trend = 'UP (EMA bullish)'
elif last['ema25'] < last['ema50'] < last['ema100']:
    trend = 'DOWN (EMA bearish)'
else:
    trend = 'MIXED / RANGING'

# Price action
last_5 = df.tail(5)
last_10 = df.tail(10)
last_20 = df.tail(20)

up_5 = sum(last_5['close'] > last_5['close'].shift())
up_10 = sum(last_10['close'] > last_10['close'].shift())
up_20 = sum(last_20['close'] > last_20['close'].shift())

print('')
print(f'Trend: {trend}')
print(f'Last 5 candles: {up_5}/5 green (up)')
print(f'Last 10 candles: {up_10}/10 green (up)')
print(f'Last 20 candles: {up_20}/20 green (up)')

# Support / Resistance
recent_20 = df.tail(20)
resistance = recent_20['high'].max()
support = recent_20['low'].min()
print('')
print(f'20-min Resistance: {resistance:.2f}')
print(f'20-min Support: {support:.2f}')
print(f'Current vs Support: {last["close"] - support:.2f} above')
print(f'Current vs Resistance: {resistance - last["close"]:.2f} below')

# Your position
avg_entry = 59228
print('')
print('=== YOUR POSITION ===')
print(f'Your average entry (short): {avg_entry:.2f}')
print(f'Current price: {last["close"]:.2f}')
print(f'Price needs to drop to: {avg_entry:.2f} for breakeven')
print(f'Current loss per lot: {last["close"] - avg_entry:.2f}')
print(f'Current total loss: {(last["close"] - avg_entry) * 200:.2f}')

# Probability assessment
print('')
print('=== PROBABILITY ASSESSMENT ===')
print(f'Trend: {trend}')
print(f'ADX: {adx.iloc[-1]:.0f} (trend strength)')
if adx.iloc[-1] > 25:
    print('  -> Strong trend in place')
else:
    print('  -> Weak trend')

print(f'RSI: {rsi.iloc[-1]:.0f}')
if rsi.iloc[-1] > 70:
    print('  -> Overbought (could reverse down)')
elif rsi.iloc[-1] < 30:
    print('  -> Oversold (could reverse up)')
else:
    print('  -> Neutral')

print('')
print('=== HONEST ASSESSMENT ===')
if last['ema25'] > last['ema50'] and last['close'] > last['ema25']:
    print('Price is ABOVE EMA25 and EMA25 > EMA50. This is an UPTREND.')
    print('Probability of further up move: HIGH')
    print('Probability of dropping to your breakeven: LOW')
if up_5 >= 4:
    print('Last 5 candles mostly UP. Momentum is bullish.')
    print('Shorting into this momentum is dangerous.')

print('')
print('=== WHAT PRICE NEEDS TO DO ===')
print(f'For breakeven: BTC must drop from {last["close"]:.2f} to {avg_entry:.2f}')
print(f'That is a drop of {last["close"] - avg_entry:.2f} ({((last["close"] - avg_entry) / last["close"]) * 100:.2f}%)')
print('')
print('With 200 lots, every $10 move = $2,000 P&L.')
print('Every $100 move = $20,000 P&L.')
