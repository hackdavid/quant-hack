import os
from intraday.trader.mt5_wrapper import MT5TradingWrapper

wrapper = MT5TradingWrapper(
    account_id=int(os.getenv("MT5_ACCOUNT", "0")),
    password=os.getenv("MT5_PASSWORD", ""),
    server=os.getenv("MT5_SERVER", ""),
    magic=999999,
)

if wrapper.connect():
    print('MT5 connected')
    positions = wrapper.get_positions('BTCUSDT')
    print(f'Positions: {len(positions)}')
    for p in positions:
        print(f'  Ticket {p.ticket}: {p.side} {p.volume} lot @ {p.open_price} | PnL={p.profit}')
        print(f'  Closing position...')
        result = wrapper.close_position(p.ticket)
        print(f'  Result: {result}')
    wrapper.shutdown()
else:
    print('MT5 connection failed')
