"""
main.py - Main Event Loop for MT5 Gold Scalper Bot.

Synchronizes execution with closed candle boundaries on M1, continuously monitors
active positions for SL/TP exits, and coordinates MT5 connector, signal engine,
risk manager, trade executor, kill switch, and trade logger.
"""

import logging
import signal
import sys
import time
from datetime import datetime, timezone
from typing import Optional

# Ensure stdout supports UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import MetaTrader5 as mt5

from config import BotConfig, config as default_config
from mt5_connector import MT5Connector
from signal_engine import SignalEngine, SignalType
from risk_manager import RiskManager
from executor import TradeExecutor
from kill_switch import KillSwitch
from logger import setup_logger, TradeLogger

# Setup primary application logger
logger = setup_logger("ScalperBot")


class ScalperBot:
    """Orchestrates all trading bot modules in an event-driven loop."""

    def __init__(self, bot_config: BotConfig = default_config):
        self.config = bot_config
        self.is_running: bool = False
        self.last_closed_candle_time: Optional[datetime] = None

        # Instantiate modules
        self.connector = MT5Connector(self.config)
        self.signal_engine = SignalEngine(self.config)
        self.risk_manager = RiskManager(self.config)
        self.executor = TradeExecutor(self.config)
        self.kill_switch = KillSwitch(self.config)
        self.trade_logger = TradeLogger(self.config)

    def start(self) -> None:
        """Initializes terminal connection and enters the main trading loop."""
        # 1. Connect to MT5
        if not self.connector.initialize():
            logger.critical("Initialization failed. Terminating bot.")
            sys.exit(1)

        symbol = self.connector.resolved_symbol
        account = self.connector.get_account_summary()
        server_dt = self.connector.get_server_time()
        sym_info = self.connector.symbol_info

        point = float(sym_info.point) if sym_info and sym_info.point > 0 else 0.01
        spread_pts = round(self.config.max_spread_price / point, 1)
        mode_str = "DEMO SAFE" if not self.config.allow_live_trading else "LIVE TRADING"
        margin_str = account.get("margin_mode_str", "HEDGING") if account else "HEDGING"

        # Recover any active groups on startup
        self.executor.recover_active_groups(symbol)

        # Prominent startup configuration banner
        logger.info("==================================================================")
        logger.info("             MT5 GOLD (XAUUSD) SCALPING BOT INITIALIZED           ")
        logger.info("==================================================================")
        logger.info(f" Account Login       : {account.get('login', 'N/A')} ({mode_str}) | Margin: {margin_str}")
        logger.info(f" Server & Currency   : {account.get('server', 'N/A')} | {account.get('balance', 0):,.2f} {account.get('currency', 'USD')}")
        logger.info(f" Resolved Symbol     : '{symbol}' (Digits: {sym_info.digits}, Point: {sym_info.point})")
        logger.info(f" Spread Threshold    : MAX_SPREAD_PRICE = ${self.config.max_spread_price:.2f} ➔ {spread_pts} broker points")
        logger.info(f" Strategy Settings   : Timeframe {self.config.timeframe_str} | Fast EMA({self.config.ema_fast_period}) | Slow EMA({self.config.ema_slow_period}) | RSI({self.config.rsi_period}) | ATR({self.config.atr_period})")
        logger.info(f" Group Execution     : Positions/Group: {self.config.positions_per_group} | Max Concurrent Groups: {self.config.max_concurrent_trade_groups} | Mode: {self.config.group_risk_mode}")
        logger.info(f" Risk Rules          : Fixed Risk: ${self.config.fixed_risk_amount or 1.0:.2f} | Max Daily Loss {self.config.max_daily_loss_pct}% | Max Consecutive Losses {self.config.max_consecutive_losses}")
        logger.info(f" Active Groups Open  : {self.executor.get_active_groups_count(symbol)}")
        logger.info(f" Telegram Alerts     : {'ENABLED' if self.config.is_telegram_enabled else 'DISABLED (No token/chat_id in .env)'}")
        logger.info("==================================================================")

        # 2. Sync Kill Switch with broker server date
        if account:
            self.kill_switch.sync_server_date(server_dt, account["balance"])

        # 3. Notify Startup via Telegram
        if account:
            self.trade_logger.notify_bot_started(account, symbol)

        self.is_running = True
        self._register_signal_handlers()

        # 4. Main Event Loop
        self._run_event_loop()

    def _register_signal_handlers(self) -> None:
        """Hooks SIGINT and SIGTERM for graceful exit."""
        def handle_exit(signum, frame):
            logger.warning(f"Received termination signal ({signum}). Initiating clean shutdown...")
            self.is_running = False

        signal.signal(signal.SIGINT, handle_exit)
        signal.signal(signal.SIGTERM, handle_exit)

    def _run_event_loop(self) -> None:
        """Core polling loop checking candle closes and managing active trade groups."""
        symbol = self.connector.resolved_symbol

        while self.is_running:
            try:
                # -------------------------------------------------------------
                # 1. CONTINUOUS LIFECYCLE CHECK: Monitor Open Trade Group Exits
                # -------------------------------------------------------------
                closed_groups = self.executor.detect_closed_groups(symbol)
                if closed_groups:
                    server_dt = self.connector.get_server_time()
                    account = self.connector.get_account_summary()
                    curr_balance = account["balance"] if account else self.kill_switch.starting_daily_balance

                    for closed in closed_groups:
                        # Log to CSV
                        self.trade_logger.log_closed_group(closed)
                        # Notify Telegram
                        self.trade_logger.notify_group_closed(closed)

                        # Update Kill Switch (1 group = 1 trade unit)
                        prev_tripped = self.kill_switch.is_tripped
                        self.kill_switch.record_trade_result(closed.net_pnl, server_dt, curr_balance)

                        # If Kill Switch tripped as a result of this group close
                        if not prev_tripped and self.kill_switch.is_tripped:
                            self.trade_logger.notify_kill_switch(
                                self.kill_switch.trip_reason or "Risk Limit Exceeded",
                                self.kill_switch.daily_realized_pnl,
                                self.kill_switch.consecutive_losses,
                            )

                # -------------------------------------------------------------
                # 2. CANDLE SYNCHRONIZATION: Check for newly completed bar
                # -------------------------------------------------------------
                candle_df = self.connector.get_candles()
                if candle_df is None or len(candle_df) < 5:
                    time.sleep(self.config.loop_poll_interval_sec)
                    continue

                # The latest completed candle is at index -2 (index -1 is active/in-progress)
                latest_closed_row = candle_df.iloc[-2]
                candle_dt = latest_closed_row["time"]
                if hasattr(candle_dt, "to_pydatetime"):
                    candle_dt = candle_dt.to_pydatetime()

                # If we have not processed this closed candle yet:
                if self.last_closed_candle_time is None or candle_dt > self.last_closed_candle_time:
                    self._process_candle_close(candle_df, candle_dt)
                    self.last_closed_candle_time = candle_dt

                # Sleep brief interval before next iteration
                time.sleep(self.config.loop_poll_interval_sec)

            except Exception as e:
                logger.error(f"Unexpected exception in trading loop: {e}", exc_info=True)
                time.sleep(2.0)

        # Clean shutdown after loop exit
        self._shutdown()

    def _process_candle_close(self, candle_df, candle_dt: datetime) -> None:
        """Evaluates signal and executes trade group upon candle close."""
        symbol = self.connector.resolved_symbol

        # A. Evaluate Technical Signal
        signal = self.signal_engine.evaluate_signals(candle_df)
        
        # Live Heartbeat on every closed candle
        time_str = candle_dt.strftime("%H:%M:%S") if hasattr(candle_dt, "strftime") else str(candle_dt)
        status_desc = "WAITING FOR CROSSOVER" if signal.signal_type == SignalType.NONE else f"TRIGGERED {signal.signal_type.value}"
        logger.info(
            f"📊 Candle [{time_str}] Closed | Close: {signal.entry_price:.2f} | "
            f"Fast EMA: {signal.fast_ema:.2f} | Slow EMA: {signal.slow_ema:.2f} | "
            f"RSI: {signal.rsi:.1f} | ATR: ${signal.atr_value:.2f} | {status_desc}"
        )

        if not signal.is_valid:
            # No entry signal on this bar
            return

        logger.info(f"Signal Generated: {signal.signal_type.value} on {symbol} (Reason: {signal.reason})")

        # B. Check Circuit Breaker (Kill Switch)
        server_dt = self.connector.get_server_time()
        account = self.connector.get_account_summary()
        if not account:
            logger.error("Cannot evaluate signal: failed to get account summary.")
            return

        can_trade, halt_reason = self.kill_switch.can_trade(account["balance"], server_dt)
        if not can_trade:
            rejection_reason = f"Kill Switch Halted: {halt_reason}"
            logger.warning(f"Trade entry BLOCKED: {rejection_reason}")
            self.trade_logger.log_signal(symbol, signal, status="REJECTED", rejection_reason=rejection_reason)
            return

        # C. Retrieve Live Market Tick & Symbol Info
        tick = self.connector.get_current_tick()
        symbol_info = self.connector.refresh_symbol_info()
        if tick is None or symbol_info is None:
            rejection_reason = "Market tick or symbol information unavailable"
            logger.error(f"Trade entry BLOCKED: {rejection_reason}")
            self.trade_logger.log_signal(symbol, signal, status="REJECTED", rejection_reason=rejection_reason)
            return

        # D. Risk Assessment & Sizing
        active_groups_count = self.executor.get_active_groups_count(symbol)
        risk_result = self.risk_manager.evaluate_signal(
            signal=signal,
            tick=tick,
            account_summary=account,
            symbol_info=symbol_info,
            open_groups_count=active_groups_count,
        )

        if not risk_result.approved:
            logger.warning(f"Trade REJECTED by Risk Manager: {risk_result.rejection_reason}")
            self.trade_logger.log_signal(
                symbol,
                signal,
                status="REJECTED",
                rejection_reason=risk_result.rejection_reason,
            )
            return

        # E. Grouped Order Execution
        group_res = self.executor.execute_trade_group(
            symbol=symbol,
            signal_type=signal.signal_type,
            risk_result=risk_result,
            symbol_info=symbol_info,
        )

        if group_res.success and group_res.group:
            # Log Approved Signal
            self.trade_logger.log_signal(symbol, signal, status="APPROVED", rejection_reason="Executed Group")
            # Dispatch Telegram Notification
            self.trade_logger.notify_group_opened(group_res.group, risk_result)
        else:
            # Order placement failed at MT5 level
            self.trade_logger.log_signal(
                symbol,
                signal,
                status="REJECTED",
                rejection_reason=f"Execution Failed: {group_res.rejection_reason}",
            )

    def _shutdown(self) -> None:
        """Performs clean shutdown and releases MT5 link."""
        logger.info("Executing clean shutdown procedure...")
        self.trade_logger.notify_bot_stopped()
        self.connector.shutdown()
        logger.info("Scalper bot terminated.")


def main():
    """Application entry point."""
    bot = ScalperBot(default_config)
    bot.start()


if __name__ == "__main__":
    main()
