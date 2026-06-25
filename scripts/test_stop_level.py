import MetaTrader5 as mt5

mt5.initialize()
info = mt5.symbol_info("BTCUSD")
if info:
    print(f"symbol: {info.name}")
    print(f"trade_stops_level: {info.trade_stops_level}")
    print(f"point: {info.point}")
    print(f"digits: {info.digits}")
    print(f"spread: {info.spread}")
    tick = mt5.symbol_info_tick("BTCUSD")
    if tick:
        print(f"ask: {tick.ask}")
        print(f"bid: {tick.bid}")
else:
    print("BTCUSD not found")
    print("Available symbols:", [s.name for s in mt5.symbols_get() if "BTC" in s.name])
mt5.shutdown()
