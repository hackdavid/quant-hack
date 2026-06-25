import argparse
import os
from dotenv import load_dotenv
from intraday.trader.mt5_wrapper import MT5TradingWrapper

load_dotenv()

parser = argparse.ArgumentParser(description="Close a specific MT5 position by ticket")
parser.add_argument("--ticket", type=int, required=True, help="Position ticket to close")
parser.add_argument("--mt5-account", type=int, default=int(os.getenv("MT5_ACCOUNT", "0")))
parser.add_argument("--mt5-password", type=str, default=os.getenv("MT5_PASSWORD", ""))
parser.add_argument("--mt5-server", type=str, default=os.getenv("MT5_SERVER", ""))
args = parser.parse_args()

wrapper = MT5TradingWrapper(
    account_id=args.mt5_account,
    password=args.mt5_password,
    server=args.mt5_server,
    magic=999999,
)

if wrapper.connect():
    print('MT5 connected')
    positions = wrapper.get_positions('BTCUSDT')
    target = None
    for p in positions:
        if p.ticket == args.ticket:
            target = p
            break
    if target:
        print(f'Closing ticket #{args.ticket}: {target.side} {target.volume} lot @ {target.open_price} | PnL={target.profit}')
        result = wrapper.close_position(args.ticket)
        print(f'Result: {result}')
    else:
        print(f'Ticket #{args.ticket} not found in open positions')
        print('Open positions:')
        for p in positions:
            print(f'  Ticket {p.ticket}: {p.side} {p.volume} lot | PnL={p.profit}')
    wrapper.shutdown()
else:
    print('MT5 connection failed')
