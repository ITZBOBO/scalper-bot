"""
diagnose_risk_decision.py - Detailed Risk Assessment & Trade Invariant Diagnostic
Outputs the exact breakdown of sizing, risk limits, and trade decisions.
"""
import sys
from datetime import datetime, timezone
from config import config
from mt5_connector import MT5Connector
from risk_manager import RiskManager
from signal_engine import Signal, SignalType, SignalEngine


def main():
    print("=" * 70)
    print("      MT5 GOLD SCALPER — RISK INVARIANT & SIZING DIAGNOSTIC")
    print("=" * 70)

    connector = MT5Connector(config)
    if not connector.initialize():
        print("[-] Failed to initialize MT5 connector.")
        sys.exit(1)

    account = connector.get_account_summary()
    symbol_info = connector.symbol_info
    tick = connector.get_current_tick()

    if not tick or not symbol_info or not account:
        print("[-] Failed to retrieve tick or symbol info.")
        connector.shutdown()
        sys.exit(1)

    df = connector.get_candles(count=20)
    atr = 2.0
    if df is not None and len(df) >= 14:
        se = SignalEngine(config)
        df_ind = se.compute_indicators(df)
        atr = float(df_ind.iloc[-2]["atr"])

    rm = RiskManager(config)
    sig = Signal(
        signal_type=SignalType.BUY,
        entry_price=tick["ask"],
        atr_value=atr,
        candle_time=datetime.now(timezone.utc),
        fast_ema=tick["ask"] + 0.5,
        slow_ema=tick["ask"] - 0.5,
        rsi=50.0,
        reason="Diagnostic Test Signal",
    )

    res = rm.evaluate_signal(sig, tick, account, symbol_info, open_groups_count=0)

    # Calculate metrics for 1 position and 3 positions under current market ATR
    sl_distance = atr * config.sl_atr_multiplier
    tick_val = float(getattr(symbol_info, "trade_tick_value", 0.0))
    tick_sz = float(getattr(symbol_info, "trade_tick_size", 0.0))
    point_val = (tick_val / tick_sz) * float(symbol_info.point) if (tick_val > 0 and tick_sz > 0) else float(symbol_info.trade_contract_size) * float(symbol_info.point)
    loss_per_1_lot = (sl_distance / float(symbol_info.point)) * point_val
    min_lot_risk_1pos = symbol_info.volume_min * loss_per_1_lot
    min_lot_risk_3pos = (symbol_info.volume_min * 3) * loss_per_1_lot

    print("\n[ACCOUNT & BROKER CONSTRAINTS]")
    print(f" • Detected MT5 Account Mode  : {account.get('margin_mode_str')} ({'Hedging Multi-Position Supported' if account.get('is_hedging') else 'Netting Aggregated'})")
    print(f" • Account Balance             : ${account.get('balance', 0):,.2f}")
    print(f" • Broker Symbol               : {connector.resolved_symbol}")
    print(f" • Broker Volume Min           : {symbol_info.volume_min} lots")
    print(f" • Broker Volume Step          : {symbol_info.volume_step} lots")
    print(f" • Broker Contract Size        : {symbol_info.trade_contract_size}")

    print("\n[MARKET & STRATEGY CONTEXT]")
    print(f" • Current Ask Price           : {tick['ask']:.3f}")
    print(f" • Live ATR(14)                : ${atr:.3f}")
    print(f" • SL Distance (1.5 x ATR)     : ${sl_distance:.3f}")
    print(f" • Monetary Loss per 1.0 Lot   : ${loss_per_1_lot:,.2f}")

    print("\n[RISK INVARIANT & VOLUME SIZING BREAKDOWN]")
    print(f" • Maximum Allowed Group Risk  : ${res.risk_amount_currency:.2f}")
    print(f" • Configured Group Target     : ${config.group_profit_target:.2f} TOTAL")
    print(f" • Preferred Positions/Group   : {config.positions_per_group}")
    print(f" • Theoretical Volume (Raw)    : {res.raw_lot_size:.5f} lots (${res.risk_amount_currency:.2f} / ${loss_per_1_lot:.2f})")
    print(f" • Broker-Quantized Volume     : {res.quantized_lot_size:.2f} lots")
    print(f" • Executable Volume           : {res.total_lot_size if res.approved else 0.00} lots")
    print(f" • 3 Positions (3 x {symbol_info.volume_min} lot) : {'FEASIBLE' if min_lot_risk_3pos <= res.risk_amount_currency else f'IMPOSSIBLE (~${min_lot_risk_3pos:.2f} risk exceeds ${res.risk_amount_currency:.2f} budget)'}")
    print(f" • 1 Position  (1 x {symbol_info.volume_min} lot) : {'FEASIBLE' if min_lot_risk_1pos <= res.risk_amount_currency else f'IMPOSSIBLE (~${min_lot_risk_1pos:.2f} risk exceeds ${res.risk_amount_currency:.2f} budget)'}")

    print("\n[FINAL DECISION]")
    if res.approved:
        print(f" * DECISION: TRADE [APPROVED]")
        print(f" * Positions Selected          : {res.positions_count} position(s) -> {res.position_lots}")
        print(f" * Calculated Group SL Risk    : ~${res.theoretical_group_risk:.2f} <= Budget ${res.risk_amount_currency:.2f}")
    else:
        print(f" * DECISION: NO TRADE [REJECTED]")
        print(f" * Rejection Reason            : {res.rejection_reason}")

    connector.shutdown()
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
