import re
import json

log_file = 'logs/autonomous_trader/1m_primary_run_v7.log'

with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
    content = f.read()

trades = []
lines = content.split('\n')

for i, line in enumerate(lines):
    if 'position_closed' in line or 'profit_lock_closed' in line or 'trailing_stop_closed' in line or 'stop_loss_closed' in line:
        for j in range(i, min(i+5, len(lines))):
            profit_match = re.search(r'profit[=:]\s*([-\d.]+)', lines[j])
            if profit_match:
                profit = float(profit_match.group(1))
                ticket_match = re.search(r'ticket[=:]\s*(\d+)', lines[j])
                ticket = ticket_match.group(1) if ticket_match else '?'
                reason = 'profit_lock' if 'profit_lock' in lines[j] else 'trailing_stop' if 'trailing_stop' in lines[j] else 'stop_loss' if 'stop_loss' in lines[j] else 'other'
                trades.append({'ticket': ticket, 'profit': profit, 'reason': reason})
                break

if trades:
    total = sum(t['profit'] for t in trades)
    wins = [t for t in trades if t['profit'] > 0]
    losses = [t for t in trades if t['profit'] < 0]
    print(f'Total closed trades: {len(trades)}')
    print(f'Total P&L: ${total:.2f}')
    print(f'Wins: {len(wins)} | Losses: {len(losses)}')
    print(f'Win rate: {len(wins)/len(trades)*100:.1f}%')
    if wins:
        print(f'Average win: ${sum(t["profit"] for t in wins)/len(wins):.2f}')
    if losses:
        print(f'Average loss: ${sum(t["profit"] for t in losses)/len(losses):.2f}')
    print(f'Profit factor: {abs(sum(t["profit"] for t in wins)/sum(t["profit"] for t in losses)):.2f}' if losses else 'inf')
    print('---')
    for t in trades:
        print(f"Ticket {t['ticket']}: ${t['profit']:.2f} ({t['reason']})")
else:
    print('No trades found')
