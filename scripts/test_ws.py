import asyncio
import json
import websockets

async def test():
    # Try futures single stream
    url = "wss://fstream.binance.com/ws/btcusdt@kline_1m"
    print(f"Connecting to {url}")
    try:
        async with websockets.connect(url, ping_interval=20, close_timeout=10) as ws:
            print("Futures single connected!")
            count = 0
            closed_count = 0
            while closed_count < 2:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=120)
                    count += 1
                    msg = json.loads(raw)
                    if msg.get("e") != "kline":
                        print(f"[{count}] Non-kline: {msg.get('e')}")
                        continue
                    k = msg["k"]
                    interval = k["i"]
                    closed = k.get("x", False)
                    if closed:
                        closed_count += 1
                        print(f"[{count}] CLOSED {interval} price={k['c']} closed_count={closed_count}")
                    else:
                        print(f"[{count}] open {interval} price={k['c']}", end="\r")
                except asyncio.TimeoutError:
                    print(f"\n[{count}] Timeout waiting for message")
                    break
    except Exception as exc:
        print(f"Futures single error: {exc}")

    # Try futures combined stream with explicit /stream endpoint
    url2 = "wss://fstream.binance.com/stream?streams=btcusdt@kline_1m"
    print(f"\nConnecting to {url2}")
    try:
        async with websockets.connect(url2, ping_interval=20, close_timeout=10) as ws:
            print("Futures combined (single) connected!")
            count = 0
            closed_count = 0
            while closed_count < 2:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=120)
                    count += 1
                    msg = json.loads(raw)
                    data = msg.get("data", msg)
                    if data.get("e") != "kline":
                        print(f"[{count}] Non-kline: {data.get('e')}")
                        continue
                    k = data["k"]
                    interval = k["i"]
                    closed = k.get("x", False)
                    if closed:
                        closed_count += 1
                        print(f"[{count}] CLOSED {interval} price={k['c']} closed_count={closed_count}")
                    else:
                        print(f"[{count}] open {interval} price={k['c']}", end="\r")
                except asyncio.TimeoutError:
                    print(f"\n[{count}] Timeout waiting for message")
                    break
    except Exception as exc:
        print(f"Futures combined error: {exc}")

    print("\nDone")

if __name__ == "__main__":
    asyncio.run(test())
