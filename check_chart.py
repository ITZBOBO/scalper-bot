"""
check_chart.py - Fetches live MT5 account status, current quotes, candles, indicators, and positions/deals.
"""
import sys
from datetime import datetime, timezone, timedelta
import pandas as pd
import MetaTrader5 as mt5
from config import config
from mt5_connector import MT5Connector
from signal_engine import SignalEngine

def main():
    print("=" * 65)
    print("  LIVE MT5 MARKET & CHART STATUS (GOLD SCALPER)")
    print("=" * 65)

    connector = MT5Connector(config)
    if not connector.initialize():
        print("[-] Failed to connect to MetaTrader 5 terminal.")
        return

    acc = connector.get_account_summary()
    mode_str = "DEMO" if acc.get("trade_mode") == 0 else ("CONTEST" if acc.get("trade_mode") == 1 else "LIVE/REAL")
    print(f"\n[ACCOUNT OVERVIEW]")
    print(f" • Login       : {acc.get('login')} ({mode_str})")
    print(f" • Server      : {acc.get('server', config.mt5_server)}")
    print(f" • Balance     : ${acc.get('balance', 0):,.2f} {acc.get('currency', 'USD')}")
    print(f" • Equity      : ${acc.get('equity', 0):,.2f}")
    print(f" • Free Margin : ${acc.get('margin_free', 0):,.2f}")
    print(f" • Floating PnL: ${acc.get('profit', 0):+,.2f}")

    symbol = connector.resolved_symbol
    print(f"\n[SYMBOL & SPREAD]")
    print(f" • Symbol      : {symbol} (Timeframe: {config.timeframe_str})")
    
    tick = connector.get_current_tick()
    if tick:
        spread_ok = tick['spread_price'] <= config.max_spread_price
        print(f" • Live Bid    : {tick['bid']:.2f}")
        print(f" • Live Ask    : {tick['ask']:.2f}")
        print(f" • Live Spread : ${tick['spread_price']:.3f} ({tick['spread_points']:.1f} pts) -> {'[OK - Within Limit]' if spread_ok else '[HIGH - Rejects Trading]'}")
        print(f" • Max Allowed : ${config.max_spread_price:.2f}")
    else:
        print("[-] Could not fetch live tick.")

    df = connector.get_candles(count=30)
    if df is not None and not df.empty:
        engine = SignalEngine(config)
        df_ind = engine.compute_indicators(df)
        print(f"\n[LATEST 6 CANDLES WITH INDICATORS]")
        cols = ['time', 'open', 'high', 'low', 'close', 'ema_fast', 'ema_slow', 'rsi', 'atr']
        sub_df = df_ind[cols].tail(6).copy()
        sub_df['time'] = sub_df['time'].dt.strftime('%Y-%m-%d %H:%M:%S UTC')
        print(sub_df.to_string(index=False))

        # Check signal on last closed candle
        signal = engine.evaluate_signals(df)
        print(f"\n[SIGNAL ENGINE EVALUATION (Latest Closed Bar)]")
        print(f" • Signal Type : {signal.signal_type.value}")
        print(f" • Reason      : {signal.reason}")
        print(f" • Fast EMA(9) : {signal.fast_ema:.3f}")
        print(f" • Slow EMA(21): {signal.slow_ema:.3f}")
        print(f" • RSI (14)    : {signal.rsi:.2f} (Overbought > {config.rsi_overbought}, Oversold < {config.rsi_oversold})")
        print(f" • ATR (14)    : {signal.atr_value:.3f}")
    else:
        print("[-] Could not fetch candle data.")

    # Open Positions
    raw_positions = mt5.positions_get(symbol=symbol)
    print(f"\n[OPEN POSITIONS]")
    if raw_positions:
        for p in raw_positions:
            p_type = "BUY" if p.type == mt5.ORDER_TYPE_BUY else "SELL"
            print(f" • Ticket #{p.ticket} | {p_type} {p.volume} lots @ {p.price_open:.2f} | Current: {p.price_current:.2f} | SL: {p.sl:.2f} | TP: {p.tp:.2f} | Profit: ${p.profit:+,.2f} | Magic: {p.magic}")
    else:
        print(" • No active open positions.")

    # Recent Closed Deals (past 7 days)
    from_date = datetime.now(timezone.utc) - timedelta(days=7)
    deals = mt5.history_deals_get(from_date, datetime.now(timezone.utc))
    print(f"\n[RECENT CLOSED DEALS (Past 7 Days)]")
    if deals:
        # Filter deals with profit != 0 or entry == 1 (out)
        exit_deals = [d for d in deals if d.entry in (1, 2) and d.symbol == symbol]
        if exit_deals:
            for d in exit_deals[-5:]:
                deal_time = datetime.fromtimestamp(d.time, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
                d_type = "BUY" if d.type == mt5.ORDER_TYPE_BUY else "SELL"
                print(f" • {deal_time} | Deal #{d.ticket} | {d_type} {d.volume} lots @ {d.price:.2f} | PnL: ${d.profit:+,.2f} | Comment: {d.comment}")
        else:
            print(" • No recent exit deals found for this symbol.")
    else:
        print(" • No history deals found.")

    connector.shutdown()
    print("\n" + "=" * 65)

if __name__ == "__main__":
    main()
