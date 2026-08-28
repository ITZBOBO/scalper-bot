"""
kill_switch.py - Circuit Breaker and Loss Limit Safeguards for MT5 Scalper.

Tracks realized daily P&L and consecutive loss counts in memory and persists to disk.
Automatically resets at broker server midnight (aligned with MT5 candles and broker rollover).
Enforces maximum daily drawdown %, consecutive loss halts, and manual emergency stop files.
"""

import json
import logging
from datetime import datetime, date, timezone
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

from config import BotConfig, config as default_config

logger = logging.getLogger("ScalperBot.KillSwitch")


class KillSwitch:
    """Safeguards trading capital by halting new entries upon risk breaches."""

    def __init__(self, bot_config: BotConfig = default_config):
        self.config = bot_config
        self.state_file = self.config.kill_switch_state_path
        self.manual_stop_file = self.config.manual_stop_file_path

        # State Variables
        self.server_date: str = ""
        self.starting_daily_balance: float = 0.0
        self.daily_realized_pnl: float = 0.0
        self.daily_trade_count: int = 0
        self.consecutive_losses: int = 0
        self.is_tripped: bool = False
        self.trip_reason: Optional[str] = None
        self.last_updated: str = ""

        # Load persisted state or initialize fresh
        self._load_state()

    def _load_state(self) -> None:
        """Loads previous circuit breaker state from disk if available."""
        if self.state_file.exists():
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.server_date = data.get("server_date", "")
                self.starting_daily_balance = float(data.get("starting_daily_balance", 0.0))
                self.daily_realized_pnl = float(data.get("daily_realized_pnl", 0.0))
                self.daily_trade_count = int(data.get("daily_trade_count", 0))
                self.consecutive_losses = int(data.get("consecutive_losses", 0))
                self.is_tripped = bool(data.get("is_tripped", False))
                self.trip_reason = data.get("trip_reason")
                self.last_updated = data.get("last_updated", "")
                logger.info(
                    f"Loaded KillSwitch State from disk: Server Date: {self.server_date} | "
                    f"Daily PnL: ${self.daily_realized_pnl:.2f} | Consecutive Losses: {self.consecutive_losses} | "
                    f"Tripped: {self.is_tripped} ({self.trip_reason or 'None'})"
                )
            except Exception as e:
                logger.error(f"Failed to read kill switch state file: {e}. Initializing fresh state.")
                self._save_state()
        else:
            self._save_state()

    def _save_state(self) -> None:
        """Persists current state to JSON file."""
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "server_date": self.server_date,
                "starting_daily_balance": self.starting_daily_balance,
                "daily_realized_pnl": self.daily_realized_pnl,
                "daily_trade_count": self.daily_trade_count,
                "consecutive_losses": self.consecutive_losses,
                "is_tripped": self.is_tripped,
                "trip_reason": self.trip_reason,
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save kill switch state: {e}")

    def sync_server_date(self, server_dt: datetime, current_balance: float) -> bool:
        """
        Synchronizes with broker server date.
        If server date changed (broker midnight rollover), resets daily metrics.
        Returns True if a daily reset occurred.
        """
        date_str = server_dt.strftime("%Y-%m-%d")

        if not self.server_date:
            # First initialization
            self.server_date = date_str
            if self.starting_daily_balance <= 0 and current_balance > 0:
                self.starting_daily_balance = current_balance
            self._save_state()
            return False

        if date_str != self.server_date:
            # Broker server midnight reached: Reset daily stats
            logger.info(
                f"Broker Server Day Rollover detected ({self.server_date} -> {date_str}). "
                f"Resetting daily PnL (Previous Daily PnL: ${self.daily_realized_pnl:.2f})."
            )
            self.server_date = date_str
            self.daily_realized_pnl = 0.0
            self.daily_trade_count = 0
            self.consecutive_losses = 0
            self.starting_daily_balance = current_balance if current_balance > 0 else self.starting_daily_balance
            self.is_tripped = False
            self.trip_reason = None
            self._save_state()
            return True

        if self.starting_daily_balance <= 0 and current_balance > 0:
            self.starting_daily_balance = current_balance
            self._save_state()

        return False

    def record_trade_result(self, profit: float, server_dt: datetime, current_balance: float) -> None:
        """
        Updates daily realized P&L and consecutive loss tracking when a position closes.
        """
        self.sync_server_date(server_dt, current_balance)

        self.daily_realized_pnl += profit
        self.daily_trade_count += 1

        if profit < 0:
            self.consecutive_losses += 1
            logger.warning(
                f"Trade closed with LOSS: ${profit:.2f}. Consecutive losses: {self.consecutive_losses}"
            )
        else:
            if self.consecutive_losses > 0:
                logger.info(f"Loss streak broken by WIN: ${profit:.2f}. Consecutive losses reset to 0.")
            self.consecutive_losses = 0

        # Check if consecutive loss cap reached
        if self.consecutive_losses >= self.config.max_consecutive_losses:
            self.trip(
                f"Consecutive loss cap hit ({self.consecutive_losses}/{self.config.max_consecutive_losses} losses)"
            )

        # Check if daily loss cap reached
        if self.starting_daily_balance > 0:
            daily_loss_pct = (-self.daily_realized_pnl / self.starting_daily_balance) * 100.0
            if daily_loss_pct >= self.config.max_daily_loss_pct:
                self.trip(
                    f"Daily loss cap hit: Drawdown -{daily_loss_pct:.2f}% >= limit {self.config.max_daily_loss_pct:.2f}% "
                    f"(${abs(self.daily_realized_pnl):.2f} / ${self.starting_daily_balance:.2f})"
                )

        self._save_state()

    def can_trade(self, current_balance: float, server_dt: datetime) -> Tuple[bool, Optional[str]]:
        """
        Evaluates whether new trades are permitted.
        Returns (can_trade, refusal_reason).
        """
        self.sync_server_date(server_dt, current_balance)

        # 1. Check Manual Stop File
        if self.manual_stop_file.exists():
            msg = f"Manual stop trigger active (file exists: {self.manual_stop_file})"
            if not self.is_tripped:
                self.trip(msg)
            return False, msg

        # 2. Check If Already Tripped
        if self.is_tripped:
            return False, f"Kill Switch is tripped: {self.trip_reason}"

        # 3. Check Consecutive Losses
        if self.consecutive_losses >= self.config.max_consecutive_losses:
            msg = f"Max consecutive losses reached ({self.consecutive_losses}/{self.config.max_consecutive_losses})"
            self.trip(msg)
            return False, msg

        # 4. Check Daily Loss Limit %
        if self.starting_daily_balance > 0:
            if self.daily_realized_pnl < 0:
                loss_pct = (-self.daily_realized_pnl / self.starting_daily_balance) * 100.0
                if loss_pct >= self.config.max_daily_loss_pct:
                    msg = (
                        f"Max daily loss reached: -{loss_pct:.2f}% >= {self.config.max_daily_loss_pct:.2f}% "
                        f"(${abs(self.daily_realized_pnl):.2f} loss)"
                    )
                    self.trip(msg)
                    return False, msg

        return True, None

    def trip(self, reason: str) -> None:
        """Trips the circuit breaker, halting new trades until next day or manual reset."""
        if not self.is_tripped:
            self.is_tripped = True
            self.trip_reason = reason
            self._save_state()
            logger.critical(f"🚨 KILL SWITCH TRIPPED: {reason}. New trades are suspended.")

    def reset_manual(self, current_balance: float) -> None:
        """Manually un-trips the kill switch and resets starting balance."""
        if self.manual_stop_file.exists():
            try:
                self.manual_stop_file.unlink()
            except Exception as e:
                logger.warning(f"Could not remove manual stop file: {e}")

        self.is_tripped = False
        self.trip_reason = None
        self.consecutive_losses = 0
        self.daily_realized_pnl = 0.0
        self.starting_daily_balance = current_balance
        self._save_state()
        logger.info("Kill switch manually reset. Trading resumed.")
