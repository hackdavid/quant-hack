#!/usr/bin/env python3
"""Check all-time P&L from MT5."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from mt5_competition_score import connect_mt5, fetch_deals
from datetime import datetime, timedelta

mt5 = connect_mt5(10408, 'Daudibrahim@123', '3.11.134.149:443')
info = mt5.account_info()

# Fetch ALL deals (last 30 days)
from_date = datetime.now() - timedelta(days=30)
to_date = datetime.now()
all_deals = fetch_deals(mt5, from_date, to_date)

# Get today's deals
today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
today_deals = [d for d in all_deals if d['time'] >= today]

mt5.shutdown()

print('=' * 60)
print('ALL TIME TRADE HISTORY (Last 30 Days)')
print('=' * 60)

if info:
    print(f'Current Balance:     ${info.balance:,.2f}')
    print(f'Current Equity:      ${info.equity:,.2f}')
    print(f'Account Name:        {info.name}')
    print(f'Account Number:      {info.login}')
    print('')

# Calculate ALL P&L
all_wins = [d for d in all_deals if d['profit'] > 0 and d['entry'] == 1]
all_losses = [d for d in all_deals if d['profit'] < 0 and d['entry'] == 1]
all_profit = sum(d['profit'] for d in all_deals if d['entry'] == 1)

print(f'Total Trades (30d):    {len(all_wins) + len(all_losses)}')
print(f'Winning Trades:        {len(all_wins)}')
print(f'Losing Trades:         {len(all_losses)}')
if (len(all_wins)+len(all_losses)) > 0:
    print(f'Win Rate:              {len(all_wins)/(len(all_wins)+len(all_losses))*100:.1f}%')
print(f'Total P&L (30d):       ${all_profit:,.2f}')
print('')

# Calculate TODAY only
today_wins = [d for d in today_deals if d['profit'] > 0 and d['entry'] == 1]
today_losses = [d for d in today_deals if d['profit'] < 0 and d['entry'] == 1]
today_profit = sum(d['profit'] for d in today_deals if d['entry'] == 1)

print('--- TODAY ONLY ---')
print(f'Today Trades:          {len(today_wins) + len(today_losses)}')
print(f'Today Wins:            {len(today_wins)}')
print(f'Today Losses:          {len(today_losses)}')
print(f'Today P&L:             ${today_profit:,.2f}')
print('')

# Show last 10 trades
print('--- LAST 10 CLOSED TRADES ---')
closed_deals = [d for d in all_deals if d['entry'] == 1]
closed_deals.sort(key=lambda x: x['time'], reverse=True)
for d in closed_deals[:10]:
    color = '+' if d['profit'] > 0 else ''
    print(f"  {d['time'].strftime('%m-%d %H:%M')} | {d['type']:4s} | Vol: {d['volume']:.2f} | P&L: {color}${d['profit']:,.2f}")

print('')
print('=' * 60)
if all_profit > 0:
    status = 'PROFIT'
elif all_profit < 0:
    status = 'LOSS'
else:
    status = 'BREAK EVEN'
print(f'OVERALL STATUS: {status}')
print(f'OVERALL P&L:    ${all_profit:,.2f}')
print('=' * 60)
