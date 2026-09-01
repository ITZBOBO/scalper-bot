"""
execute_test_group.py - Triggers an immediate live test trade group on MT5.
Uses the exact risk manager, executor, and logger components.
"""
import sys
from datetime import datetime, timezone
import MetaTrader5 as mt5
from config import config
from mt5_connector import MT5Connector
from signal_engine import Signal, SignalType
from risk_manager import RiskManager, RiskAssessmentResult
from executor import TradeExecutor
from logger import TradeLogger
from kill_switch import KillSwitch

def main():
    print("=" * 65)
    print("  EXECUTING LIVE TEST TRADE GROUP (MT5 GOLD SCALPER)")
    print("=" * 65)

    connector = MT5Connector(config)
    if not connector.initialize():
        print("[-] Failed to initialize MT5 connector.")
        sys.exit(1)

    symbol = connector.resolved_symbol
    sym_info = connector.symbol_info
    tick = connector.get_current_tick()
    account = connector.get_account_summary()

    if not tick or not sym_info or not account:
        print("[-] Failed to retrieve tick, symbol_info, or account summary.")
        connector.shutdown()
        sys.exit(1)

    print(f"\n[1] Account & Symbol Info:")
    print(f" • Login       : {account.get('login')} ({account.get('margin_mode_str')})")
    print(f" • Balance     : ${account.get('balance', 0):,.2f}")
    print(f" • Symbol      : {symbol}")
    print(f" • Current Ask : {tick['ask']:.3f}")
    print(f" • Spread      : ${tick['spread_price']:.3f} ({tick['spread_points']:.1f} pts)")

    # Reset kill switch state for demo testing
    ks = KillSwitch(config)
    server_dt = connector.get_server_time()
    ks.server_date = server_dt.strftime("%Y-%m-%d")
    ks.starting_daily_balance = account["balance"]
    ks.daily_realized_pnl = 0.0
    ks.daily_trade_count = 0
    ks.consecutive_losses = 0
    ks.is_tripped = False
    ks.trip_reason = None
    ks._save_state()

    # Get recent ATR from candles
    df = connector.get_candles(count=20)
    atr = 2.0
    if df is not None and len(df) >= 14:
        from signal_engine import SignalEngine
        se = SignalEngine(config)
        df_ind = se.compute_indicators(df)
        atr = float(df_ind.iloc[-2]["atr"])
        print(f" • Live ATR(14): ${atr:.3f}")

    risk_manager = RiskManager(config)
    executor = TradeExecutor(config)
    trade_logger = TradeLogger(config)

    # Construct test BUY signal
    signal = Signal(
        signal_type=SignalType.BUY,
        entry_price=tick["ask"],
        atr_value=atr,
        candle_time=datetime.now(timezone.utc),
        fast_ema=tick["ask"] + 0.5,
        slow_ema=tick["ask"] - 0.5,
        rsi=50.0,
        reason="Manual Test Trade Group Execution",
    )

    # Evaluate risk & sizing
    print(f"\n[2] Evaluating Risk & Sizing:")
    risk_res = risk_manager.evaluate_signal(
        signal=signal,
        tick=tick,
        account_summary=account,
        symbol_info=sym_info,
        open_groups_count=0,
    )

    if not risk_res.approved:
        print(f"[-] Risk Manager rejected test signal: {risk_res.rejection_reason}")
        connector.shutdown()
        sys.exit(1)

    print(f" • Approved          : {risk_res.approved}")
    print(f" • Total Lot Size    : {risk_res.total_lot_size} lots")
    print(f" • Split Lots        : {risk_res.position_lots} ({risk_res.positions_count} position(s))")
    print(f" • Entry Price       : {risk_res.entry_price:.3f}")
    print(f" • Stop Loss (SL)    : {risk_res.sl_price:.3f}")
    print(f" • Take Profit (TP)  : {risk_res.tp_price:.3f}")
    print(f" • Risk Target       : ${risk_res.risk_amount_currency:.2f}")
    print(f" • Theoretical Risk  : ~${risk_res.theoretical_group_risk:.2f}")

    # Execute trade group
    print(f"\n[3] Sending Group Orders to MT5 Terminal...")
    group_res = executor.execute_trade_group(
        symbol=symbol,
        signal_type=signal.signal_type,
        risk_result=risk_res,
        symbol_info=sym_info,
    )

    if group_res.success and group_res.group:
        print(f"\n✅ TRADE GROUP EXECUTED SUCCESSFULLY!")
        print(f" • Group ID          : {group_res.group.group_id}")
        print(f" • Order Tickets     : {group_res.group.tickets}")
        print(f" • Positions Lots    : {group_res.group.position_lots}")
        print(f" • Total Volume      : {group_res.group.total_volume} lots")
        
        # Dispatch Telegram alert
        trade_logger.notify_group_opened(group_res.group, risk_res)
        print(f" • Telegram Alert    : Sent!")
    else:
        print(f"\n❌ Execution Failed: {group_res.rejection_reason}")

    connector.shutdown()
    print("\n" + "=" * 65)

if __name__ == "__main__":
    main()
