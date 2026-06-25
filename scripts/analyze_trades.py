import MetaTrader5 as mt5
print("Initializing MT5...")
if not mt5.initialize():
    print("MT5 init failed")
    exit(1)
print("MT5 initialized")

info = mt5.account_info()
if info:
    print(f"Account: {info.login}, Balance: {info.balance}, Equity: {info.equity}")

history = mt5.history_deals_get(0, 0)
print(f"History deals: {len(history) if history else 0}")

if history:
    total_profit = sum(d.profit for d in history)
    wins = [d for d in history if d.profit > 0]
    losses = [d for d in history if d.profit < 0]
    print(f'Total trades: {len(history)}')
    print(f'Total profit: {total_profit:.2f}')
    print(f'Win rate: {len(wins)}/{len(history)} = {len(wins)/len(history)*100:.1f}%')
    if wins:
        print(f'Average win: {sum(d.profit for d in wins)/len(wins):.2f}')
    if losses:
        print(f'Average loss: {sum(d.profit for d in losses)/len(losses):.2f}')
    print('---')
    for d in history[-15:]:
        print(f'{d.time} | {d.symbol} | type={d.type} | vol={d.volume} | ${d.profit:.2f}')
else:
    print("No history deals found")

mt5.shutdown()
