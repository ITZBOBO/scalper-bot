"""
logger.py - Comprehensive Logging and Non-Blocking Telegram Alerting.

Maintains dual CSV records (logs/signals.csv and logs/trades.csv),
provides color-coded console logs, and dispatches background Telegram notifications.
"""

import csv
import logging
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any

# Ensure stdout supports UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import colorama
from colorama import Fore, Style
import requests

from config import BotConfig, config as default_config
from signal_engine import Signal
from risk_manager import RiskAssessmentResult
from executor import ExecutionResult, ClosedPositionInfo, TradeGroup, ClosedGroupInfo, GroupExecutionResult

# Initialize colorama
colorama.init(autoreset=True)


class ColoredFormatter(logging.Formatter):
    """Custom logging formatter that adds color codes to console output."""

    LEVEL_COLORS = {
        logging.DEBUG: Fore.CYAN,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.RED + Style.BRIGHT,
    }

    def format(self, record):
        color = self.LEVEL_COLORS.get(record.levelno, Fore.WHITE)
        timestamp = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S")
        message = super().format(record)
        return f"{Fore.LIGHTBLACK_EX}[{timestamp}]{Style.RESET_ALL} {color}[{record.levelname:<8}]{Style.RESET_ALL} {Fore.WHITE}{record.name}:{Style.RESET_ALL} {message}"


def setup_logger(name: str = "ScalperBot", log_file: Optional[Path] = None, level: int = logging.INFO) -> logging.Logger:
    """Configures and returns the central logger instance."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        # Console Handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(ColoredFormatter("%(message)s"))
        logger.addHandler(console_handler)

        # Optional File Handler
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(level)
            file_formatter = logging.Formatter("[%(asctime)s] [%(levelname)-8s] %(name)s: %(message)s")
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

    return logger


class TradeLogger:
    """Handles CSV audit trails for signals and trades, and asynchronous Telegram messaging."""

    def __init__(self, bot_config: BotConfig = default_config):
        self.config = bot_config
        self.signals_csv_path = self.config.signals_csv_path
        self.trades_csv_path = self.config.trades_csv_path
        self.logger = logging.getLogger("ScalperBot.TradeLogger")

        self._init_csv_headers()

    def _init_csv_headers(self) -> None:
        """Ensures CSV files exist with proper header rows."""
        self.signals_csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.trades_csv_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.signals_csv_path.exists():
            with open(self.signals_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp_utc",
                    "symbol",
                    "candle_time",
                    "signal_type",
                    "entry_price",
                    "atr",
                    "fast_ema",
                    "slow_ema",
                    "rsi",
                    "status",
                    "rejection_reason",
                ])

        if not self.trades_csv_path.exists():
            with open(self.trades_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "ticket",
                    "deal_id",
                    "symbol",
                    "order_type",
                    "lot_size",
                    "open_time",
                    "close_time",
                    "open_price",
                    "sl_price",
                    "tp_price",
                    "close_price",
                    "gross_profit",
                    "commission",
                    "swap",
                    "net_pnl",
                    "exit_reason",
                ])

    def log_signal(
        self,
        symbol: str,
        signal: Signal,
        status: str,  # "APPROVED", "REJECTED", "NO_SIGNAL"
        rejection_reason: Optional[str] = None,
    ) -> None:
        """Appends a signal evaluation event to signals.csv."""
        try:
            with open(self.signals_csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now(timezone.utc).isoformat(),
                    symbol,
                    signal.candle_time.isoformat() if signal.candle_time else "",
                    signal.signal_type.value,
                    f"{signal.entry_price:.3f}",
                    f"{signal.atr_value:.3f}",
                    f"{signal.fast_ema:.3f}",
                    f"{signal.slow_ema:.3f}",
                    f"{signal.rsi:.2f}",
                    status,
                    rejection_reason or signal.reason,
                ])
        except Exception as e:
            self.logger.error(f"Failed to write to signals.csv: {e}")

    def log_closed_trade(self, closed_info: ClosedPositionInfo) -> None:
        """Appends a completed trade lifecycle record to trades.csv."""
        try:
            with open(self.trades_csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    closed_info.ticket,
                    getattr(closed_info, "deal_id", 0),
                    closed_info.symbol,
                    closed_info.order_type,
                    f"{closed_info.lot_size:.2f}",
                    closed_info.open_time.isoformat(),
                    closed_info.close_time.isoformat(),
                    f"{closed_info.open_price:.3f}",
                    "",  # SL price
                    "",  # TP price
                    f"{closed_info.close_price:.3f}",
                    f"{closed_info.profit:.2f}",
                    f"{closed_info.commission:.2f}",
                    f"{closed_info.swap:.2f}",
                    f"{closed_info.net_pnl:.2f}",
                    closed_info.exit_reason,
                ])
        except Exception as e:
            self.logger.error(f"Failed to write to trades.csv: {e}")

    def log_closed_group(self, closed_group: ClosedGroupInfo) -> None:
        """Appends an aggregated completed trade group record to trades.csv."""
        try:
            with open(self.trades_csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                tickets_str = ";".join(str(t) for t in closed_group.tickets)
                writer.writerow([
                    closed_group.group_id,
                    tickets_str,
                    closed_group.symbol,
                    closed_group.order_type,
                    f"{closed_group.total_volume:.2f}",
                    closed_group.open_time.isoformat(),
                    closed_group.close_time.isoformat(),
                    f"{closed_group.avg_open_price:.3f}",
                    "",  # SL price
                    "",  # TP price
                    f"{closed_group.avg_close_price:.3f}",
                    f"{closed_group.gross_profit:.2f}",
                    f"{closed_group.commission:.2f}",
                    f"{closed_group.swap:.2f}",
                    f"{closed_group.net_pnl:.2f}",
                    closed_group.exit_reason,
                ])
        except Exception as e:
            self.logger.error(f"Failed to write group to trades.csv: {e}")

    def send_telegram_async(self, message: str) -> None:
        """
        Dispatches a Telegram notification in a separate daemon thread
        so network latency or API drops NEVER stall trading logic.
        """
        if not self.config.is_telegram_enabled:
            return

        threading.Thread(
            target=self._send_telegram_worker,
            args=(message,),
            daemon=True,
        ).start()

    def _send_telegram_worker(self, text: str) -> None:
        """HTTP POST worker sending message to Telegram Bot API."""
        try:
            url = f"https://api.telegram.org/bot{self.config.telegram_bot_token}/sendMessage"
            payload = {
                "chat_id": self.config.telegram_chat_id,
                "text": text,
                "parse_mode": "Markdown",
            }
            response = requests.post(url, json=payload, timeout=4.0)
            if response.status_code != 200:
                self.logger.warning(
                    f"Telegram alert failed with status {response.status_code}: {response.text}"
                )
        except Exception as e:
            self.logger.warning(f"Telegram notification network error: {e}")

    # -------------------------------------------------------------------------
    # Structured Telegram Event Formatters
    # -------------------------------------------------------------------------
    def notify_bot_started(self, account_summary: Dict[str, Any], symbol: str) -> None:
        margin_str = account_summary.get("margin_mode_str", "HEDGING")
        msg = (
            f"🚀 *MT5 Gold Scalper Bot Started*\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"• *Symbol:* `{symbol}` (M1)\n"
            f"• *Account:* `{account_summary.get('login', 'N/A')}`\n"
            f"• *Margin Mode:* `{margin_str}`\n"
            f"• *Balance:* `${account_summary.get('balance', 0):,.2f}`\n"
            f"• *Positions/Group:* `{self.config.positions_per_group}`\n"
            f"• *Risk Mode:* `{self.config.group_risk_mode}`\n"
            f"• *Risk Target:* `${self.config.fixed_risk_amount or 1.0:.2f}`\n"
            f"• *Max Daily Loss:* `{self.config.max_daily_loss_pct}%`\n"
            f"• *Max Consecutive Losses:* `{self.config.max_consecutive_losses}`\n"
            f"• *Max Spread:* `${self.config.max_spread_price:.2f}`\n"
            f"• *Mode:* `{'LIVE' if self.config.allow_live_trading else 'DEMO SAFE'}`"
        )
        self.send_telegram_async(msg)

    def notify_group_opened(self, group: TradeGroup, risk_result: RiskAssessmentResult) -> None:
        direction_emoji = "🟢 BUY" if group.order_type == "BUY" else "🔴 SELL"
        positions_str = ", ".join([f"#{t} ({v} lots)" for t, v in group.position_lots.items()])
        msg = (
            f"⚡ *TRADE GROUP OPENED* `{group.group_id}`\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"• *Action:* {direction_emoji}\n"
            f"• *Symbol:* `{group.symbol}`\n"
            f"• *Total Volume:* `{group.total_volume:.2f} lots` ({len(group.tickets)} positions)\n"
            f"• *Positions:* `{positions_str}`\n"
            f"• *Stop Loss:* `{risk_result.sl_price:.3f}`\n"
            f"• *Take Profit:* `{risk_result.tp_price:.3f}`\n"
            f"• *Total Group Risk Target:* `${risk_result.risk_amount_currency:.2f}`\n"
            f"• *Theoretical SL Risk:* `~${risk_result.theoretical_group_risk:.2f}`\n"
            f"• *Time (UTC):* `{group.created_at.strftime('%H:%M:%S')}`"
        )
        self.send_telegram_async(msg)

    def notify_group_closed(self, closed_group: ClosedGroupInfo) -> None:
        pnl_emoji = "💰" if closed_group.net_pnl >= 0 else "🔻"
        outcome_str = "PROFIT" if closed_group.net_pnl >= 0 else "LOSS"
        msg = (
            f"{pnl_emoji} *TRADE GROUP CLOSED* `{closed_group.group_id}`\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"• *Outcome:* *{outcome_str}*\n"
            f"• *Symbol:* `{closed_group.symbol}` ({closed_group.order_type})\n"
            f"• *Positions Closed:* `{closed_group.positions_closed}/{closed_group.total_positions}` ({closed_group.total_volume:.2f} lots)\n"
            f"• *Entry Avg:* `{closed_group.avg_open_price:.3f}` ➔ *Exit Avg:* `{closed_group.avg_close_price:.3f}`\n"
            f"• *Net P&L:* *${closed_group.net_pnl:+,.2f}*\n"
            f"• *Exit Reason:* `{closed_group.exit_reason}`\n"
            f"• *Close Time:* `{closed_group.close_time.strftime('%Y-%m-%d %H:%M:%S')}`"
        )
        self.send_telegram_async(msg)

    def notify_trade_opened(self, exec_result: ExecutionResult, risk_result: RiskAssessmentResult) -> None:
        direction_emoji = "🟢 BUY" if exec_result.order_type == "BUY" else "🔴 SELL"
        msg = (
            f"⚡ *TRADE OPENED* #{exec_result.ticket}\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"• *Action:* {direction_emoji}\n"
            f"• *Symbol:* `{exec_result.symbol}`\n"
            f"• *Volume:* `{exec_result.lot_size} lots`\n"
            f"• *Entry Price:* `{exec_result.price:.3f}`\n"
            f"• *Stop Loss:* `{risk_result.sl_price:.3f}`\n"
            f"• *Take Profit:* `{risk_result.tp_price:.3f}`\n"
            f"• *Risk Amount:* `${risk_result.risk_amount_currency:.2f}` ({self.config.risk_per_trade_pct}%)\n"
            f"• *Time (UTC):* `{exec_result.timestamp.strftime('%H:%M:%S') if exec_result.timestamp else ''}`"
        )
        self.send_telegram_async(msg)

    def notify_trade_closed(self, closed_info: ClosedPositionInfo) -> None:
        pnl_emoji = "💰" if closed_info.net_pnl >= 0 else "🔻"
        outcome_str = "PROFIT" if closed_info.net_pnl >= 0 else "LOSS"
        msg = (
            f"{pnl_emoji} *TRADE CLOSED* #{closed_info.ticket}\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"• *Outcome:* *{outcome_str}*\n"
            f"• *Symbol:* `{closed_info.symbol}` ({closed_info.order_type})\n"
            f"• *Volume:* `{closed_info.lot_size} lots`\n"
            f"• *Entry:* `{closed_info.open_price:.3f}` ➔ *Exit:* `{closed_info.close_price:.3f}`\n"
            f"• *Net P&L:* *${closed_info.net_pnl:+,.2f}*\n"
            f"• *Exit Reason:* `{closed_info.exit_reason}`\n"
            f"• *Close Time:* `{closed_info.close_time.strftime('%Y-%m-%d %H:%M:%S')}`"
        )
        self.send_telegram_async(msg)

    def notify_kill_switch(self, reason: str, daily_pnl: float, consecutive_losses: int) -> None:
        msg = (
            f"🚨 *KILL SWITCH TRIGGERED*\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"• *Reason:* {reason}\n"
            f"• *Daily Realized P&L:* `${daily_pnl:+,.2f}`\n"
            f"• *Consecutive Losses:* `{consecutive_losses}`\n"
            f"• *Action:* *NEW ENTRIES SUSPENDED*\n"
            f"Existing positions will continue to be monitored."
        )
        self.send_telegram_async(msg)

    def notify_bot_stopped(self) -> None:
        msg = "🛑 *MT5 Gold Scalper Bot Stopped Gracefully.*"
        self.send_telegram_async(msg)
