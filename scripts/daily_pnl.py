#!/usr/bin/env python3
"""Show daily P&L breakdown."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from mt5_competition_score import connect_mt5, fetch_deals
from datetime import datetime, timedelta

mt5 = connect_mt5(10408, 'Daudibrahim@123', '3.11.134.149:443')
info = mt5.account_info()

from_date = datetime.now() - timedelta(days=30)
to_date = datetime.now()
all_deals = fetch_deals(mt5, from_date, to_date)
mt5.shutdown()

closed = [d for d in all_deals if d['entry'] == 1]
closed.sort(key=lambda x: x['time'])

initial = 1_000_000.0
if info:
    initial = info.balance - sum(d['profit'] for d in closed)

# Build daily breakdown
daily = {}
for d in closed:
    day = d['time'].strftime('%Y-%m-%d')
    if day not in daily:
        daily[day] = {'profit': 0, 'wins': 0, 'losses': 0}
    daily[day]['profit'] += d['profit']
    if d['profit'] > 0:
        daily[day]['wins'] += 1
    else:
        daily[day]['losses'] += 1

print('=== DAILY BREAKDOWN ===')
header = "Date         | Trades | Wins | Loss | P&L          | Running"
print(header)
print('-' * 65)

running = initial
for day in sorted(daily.keys()):
    d = daily[day]
    running += d['profit']
    sign = '+' if d['profit'] > 0 else ''
    print(f"{day:12s} | {d['wins']+d['losses']:6d} | {d['wins']:4d} | {d['losses']:4d} | {sign}{d['profit']:>11,.2f} | {running:>12,.2f}")

total_profit = sum(d['profit'] for d in closed)
print('-' * 65)
print(f"{'TOTAL':12s} | {len(closed):6d} | {sum(1 for d in closed if d['profit']>0):4d} | {sum(1 for d in closed if d['profit']<0):4d} | {total_profit:>+11,.2f} | {running:>12,.2f}")

print('')
print(f'You started with: ${initial:,.2f}')
print(f'You now have:     ${running:,.2f}')
print(f'Net change:       ${running - initial:,.2f}')
if running > initial:
    print('You ARE in PROFIT!')
elif running < initial:
    print('You ARE in LOSS.')
else:
    print('You are at BREAK EVEN.')
