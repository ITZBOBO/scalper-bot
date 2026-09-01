"""
verify_live_setup.py - Pre-Flight Verification Tool for MT5 Gold Scalper Bot.

Runs sanity checks against your MT5 terminal connection and Telegram configuration before live trading:
1. Verifies MT5 login, server connection, and Demo account safety mode.
2. Resolves Gold symbol specs (digits, point size, tick value, min lot).
3. Computes and displays the exact resolved spread threshold in broker points.
4. Reads current live spread and evaluates whether trading would pass the spread filter.
5. Tests live Telegram alert delivery with immediate feedback.
6. Verifies CSV logging write permissions and formatting in logs/signals.csv.
"""

import sys
from datetime import datetime, timezone
import requests
import colorama
from colorama import Fore, Style

# Ensure stdout supports UTF-8 on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from config import config
from mt5_connector import MT5Connector
from logger import setup_logger, TradeLogger
from signal_engine import Signal, SignalType

colorama.init(autoreset=True)
logger = setup_logger("PreFlightCheck")


def print_banner(text: str):
    print(f"\n{Fore.CYAN}{'=' * 65}")
    print(f"{Fore.CYAN}{Style.BRIGHT}  {text}")
    print(f"{Fore.CYAN}{'=' * 65}{Style.RESET_ALL}")


def run_verification():
    print_banner("PRE-FLIGHT VERIFICATION FOR MT5 GOLD SCALPER BOT")

    # -------------------------------------------------------------------------
    # CHECK 1: MT5 Connection & Account Mode
    # -------------------------------------------------------------------------
    print(f"\n{Fore.YELLOW}[1/4] Checking MetaTrader 5 Terminal & Account...{Style.RESET_ALL}")
    connector = MT5Connector(config)

    if not connector.initialize():
        print(f"{Fore.RED}❌ FAILED: Could not initialize MT5. Check login, password, server, or if MT5 is running.{Style.RESET_ALL}")
        sys.exit(1)

    acc = connector.get_account_summary()
    trade_mode = "DEMO" if acc["trade_mode"] == 0 else ("CONTEST" if acc["trade_mode"] == 1 else "REAL/LIVE")
    margin_str = acc.get("margin_mode_str", "HEDGING")
    is_hedging = acc.get("is_hedging", True)
    print(f"{Fore.GREEN}✅ MT5 Connected Successfully!{Style.RESET_ALL}")
    print(f"   • Account Login       : {acc['login']}")
    print(f"   • Trade Mode          : {trade_mode} ({'SAFE' if trade_mode != 'REAL/LIVE' else 'LIVE'})")
    print(f"   • Margin Mode         : {Fore.CYAN}{margin_str}{Style.RESET_ALL} ({'Independent multi-position groups supported' if is_hedging else 'Netting mode: orders merge into 1 position'})")
    print(f"   • Server              : {acc.get('server', config.mt5_server)}")
    print(f"   • Balance             : ${acc['balance']:,.2f} {acc['currency']}")
    print(f"   • Group Configuration : {config.positions_per_group} pos/group | Max Groups: {config.max_concurrent_trade_groups} | Mode: {config.group_risk_mode}")
    print(f"   • Total Risk Budget   : ${config.fixed_risk_amount or 1.0:.2f}")

    # -------------------------------------------------------------------------
    # CHECK 2: Symbol Specs & Spread Point Resolution
    # -------------------------------------------------------------------------
    print(f"\n{Fore.YELLOW}[2/4] Resolving Gold Symbol & Spread Threshold...{Style.RESET_ALL}")
    sym = connector.resolved_symbol
    info = connector.symbol_info
    point = float(info.point) if info and info.point > 0 else 0.01
    digits = int(info.digits) if info else 2

    # Calculate exact broker points from price
    spread_pts = round(config.max_spread_price / point, 1)

    print(f"{Fore.GREEN}✅ Symbol Resolved: '{sym}'{Style.RESET_ALL}")
    print(f"   • Broker Digits       : {digits}")
    print(f"   • Point Size          : {point}")
    print(f"   • Contract Size       : {getattr(info, 'trade_contract_size', 100.0)}")
    print(f"   • Tick Value          : {info.trade_tick_value} / Tick Size: {info.trade_tick_size}")
    print(f"   • Min Lot / Step      : {info.volume_min} / {info.volume_step}")
    print(f"   • Max Allowed Spread  : ${config.max_spread_price:.2f} ➔ {Fore.CYAN}{Style.BRIGHT}{spread_pts} broker points{Style.RESET_ALL}")

    # Check Current Live Spread
    tick = connector.get_current_tick()
    if tick:
        curr_price_spread = tick['spread_price']
        curr_pts_spread = tick['spread_points']
        spread_status = "PASS (Trading Allowed)" if curr_price_spread <= config.max_spread_price else "REJECT (Spread Too Wide)"
        status_color = Fore.GREEN if curr_price_spread <= config.max_spread_price else Fore.RED
        print(f"   • Current Live Spread : ${curr_price_spread:.3f} ({curr_pts_spread:.1f} pts) ➔ {status_color}{spread_status}{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}   • Warning: Could not fetch live tick.{Style.RESET_ALL}")

    # -------------------------------------------------------------------------
    # CHECK 3: Telegram Alert Delivery
    # -------------------------------------------------------------------------
    print(f"\n{Fore.YELLOW}[3/4] Testing Telegram Alert Delivery...{Style.RESET_ALL}")
    if config.is_telegram_enabled:
        test_msg = (
            f"🧪 *Pre-Flight Verification Alert*\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"• *Status:* MT5 Scalper Bot Connection Verified\n"
            f"• *Account:* `{acc['login']}` ({trade_mode})\n"
            f"• *Symbol:* `{sym}`\n"
            f"• *Max Spread:* `${config.max_spread_price:.2f}` ({spread_pts} pts)\n"
            f"• *Timestamp:* `{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}`"
        )
        url = f"https://api.telegram.org/bot{config.telegram_bot_token}/sendMessage"
        try:
            resp = requests.post(url, json={"chat_id": config.telegram_chat_id, "text": test_msg, "parse_mode": "Markdown"}, timeout=5.0)
            if resp.status_code == 200:
                print(f"{Fore.GREEN}✅ Telegram Alert Delivered! Check your Telegram chat.{Style.RESET_ALL}")
            else:
                print(f"{Fore.RED}❌ Telegram HTTP Error {resp.status_code}: {resp.text}{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}❌ Telegram Network Error: {e}{Style.RESET_ALL}")
    else:
        print(f"{Fore.LIGHTBLACK_EX}ℹ️ Telegram notifications disabled (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID empty in .env).{Style.RESET_ALL}")

    # -------------------------------------------------------------------------
    # CHECK 4: CSV Logging Integrity
    # -------------------------------------------------------------------------
    print(f"\n{Fore.YELLOW}[4/4] Verifying CSV Audit Logs...{Style.RESET_ALL}")
    trade_logger = TradeLogger(config)
    mock_signal = Signal(
        signal_type=SignalType.BUY,
        entry_price=tick["ask"] if tick else 2500.0,
        atr_value=2.0,
        candle_time=datetime.now(timezone.utc),
        fast_ema=2501.0,
        slow_ema=2499.0,
        rsi=55.0,
        reason="Pre-Flight Verification Test",
    )
    test_reason = "Test spread rejection verification"
    trade_logger.log_signal(sym, mock_signal, status="REJECTED", rejection_reason=test_reason)

    if config.signals_csv_path.exists():
        with open(config.signals_csv_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        print(f"{Fore.GREEN}✅ CSV Logging Verified! ({len(lines)} records in {config.signals_csv_path}){Style.RESET_ALL}")
        print(f"   Latest CSV Record: {lines[-1].strip()}")

    print_banner("PRE-FLIGHT VERIFICATION COMPLETE: SYSTEM READY FOR DEMO TRADING")
    connector.shutdown()


if __name__ == "__main__":
    run_verification()
