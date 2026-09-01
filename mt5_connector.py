"""
mt5_connector.py - MetaTrader 5 Connection & Data Interface.

Handles safe login, demo account validation, broker symbol alias resolution,
tick data, candle DataFrame retrieval, and server-time synchronization.
"""

import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Tuple
import pandas as pd
import MetaTrader5 as mt5

from config import BotConfig, config as default_config

logger = logging.getLogger("ScalperBot.MT5Connector")

# Mapping timeframe strings to MT5 constants
TIMEFRAME_MAP = {
    "M1": mt5.TIMEFRAME_M1,
    "M2": mt5.TIMEFRAME_M2,
    "M3": mt5.TIMEFRAME_M3,
    "M4": mt5.TIMEFRAME_M4,
    "M5": mt5.TIMEFRAME_M5,
    "M6": mt5.TIMEFRAME_M6,
    "M10": mt5.TIMEFRAME_M10,
    "M12": mt5.TIMEFRAME_M12,
    "M15": mt5.TIMEFRAME_M15,
    "M20": mt5.TIMEFRAME_M20,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H2": mt5.TIMEFRAME_H2,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
}


class MT5Connector:
    """Encapsulates all MetaTrader 5 API interactions."""

    def __init__(self, bot_config: BotConfig = default_config):
        self.config = bot_config
        self.resolved_symbol: Optional[str] = None
        self.symbol_info: Optional[Any] = None
        self.is_connected: bool = False
        self.account_info: Optional[Any] = None

    def initialize(self) -> bool:
        """
        Initializes the MT5 terminal and establishes connection with credentials.
        Validates demo mode and resolves the trade symbol.
        """
        logger.info("Initializing MetaTrader 5 interface...")

        # 1. Connect to MT5 terminal
        init_kwargs: Dict[str, Any] = {"timeout": 10000}
        if self.config.mt5_path:
            init_kwargs["path"] = self.config.mt5_path

        initialized = mt5.initialize(**init_kwargs)
        if not initialized and (self.config.mt5_login > 0 and self.config.mt5_password):
            # Fallback with credentials in initialize call
            init_with_creds = dict(init_kwargs)
            init_with_creds["login"] = self.config.mt5_login
            init_with_creds["password"] = self.config.mt5_password
            init_with_creds["server"] = self.config.mt5_server
            initialized = mt5.initialize(**init_with_creds)

        if not initialized:
            err_code, err_msg = mt5.last_error()
            logger.error(f"MT5 initialization failed: [{err_code}] {err_msg}")
            self.is_connected = False
            return False

        # Verify terminal info
        term_info = mt5.terminal_info()
        if term_info is None:
            logger.error("Failed to retrieve terminal info from MT5.")
            self.is_connected = False
            return False

        # 2. Check current account or perform explicit login if needed
        current_acc = mt5.account_info()
        if current_acc is None or (self.config.mt5_login > 0 and current_acc.login != self.config.mt5_login):
            if self.config.mt5_login > 0 and self.config.mt5_password:
                logged_in = mt5.login(
                    login=self.config.mt5_login,
                    password=self.config.mt5_password,
                    server=self.config.mt5_server,
                    timeout=10000,
                )
                if not logged_in:
                    err_code, err_msg = mt5.last_error()
                    logger.error(f"MT5 login failed for account {self.config.mt5_login}: [{err_code}] {err_msg}")
                    self.is_connected = False
                    return False

        # Retrieve and validate account info
        acc_info = mt5.account_info()
        if acc_info is None:
            err_code, err_msg = mt5.last_error()
            logger.error(f"Failed to retrieve MT5 account info: [{err_code}] {err_msg}")
            self.is_connected = False
            return False

        self.account_info = acc_info
        trade_mode_str = "DEMO" if acc_info.trade_mode == mt5.ACCOUNT_TRADE_MODE_DEMO else (
            "CONTEST" if acc_info.trade_mode == mt5.ACCOUNT_TRADE_MODE_CONTEST else "REAL"
        )
        margin_mode_val = getattr(acc_info, "margin_mode", 2)
        hedging_const = getattr(mt5, "ACCOUNT_MARGIN_MODE_RETAIL_HEDGING", 2)
        is_hedging = (margin_mode_val == hedging_const)
        margin_mode_str = "HEDGING" if is_hedging else "NETTING"

        logger.info(
            f"Connected to Account: {acc_info.login} | Server: {acc_info.server} | "
            f"Trade Mode: {trade_mode_str} | Margin Mode: {margin_mode_str} | "
            f"Balance: {acc_info.balance} {acc_info.currency} | Leverage: 1:{acc_info.leverage}"
        )

        # DEMO SAFETY GUARD
        if acc_info.trade_mode == mt5.ACCOUNT_TRADE_MODE_REAL and not self.config.allow_live_trading:
            self.shutdown()
            error_msg = (
                "CRITICAL SAFETY LOCK: Live/Real trading account detected but ALLOW_LIVE_TRADING=False. "
                "Execution aborted to protect funds. Set ALLOW_LIVE_TRADING=True in .env only if intentional."
            )
            logger.critical(error_msg)
            raise RuntimeError(error_msg)

        # Resolve Symbol
        if not self._resolve_symbol():
            logger.error(f"Could not resolve symbol '{self.config.symbol}' or its aliases in MT5.")
            return False

        self.is_connected = True
        return True

    def _resolve_symbol(self) -> bool:
        """
        Attempts to find and select the configured symbol or known aliases in Market Watch.
        """
        candidates = [self.config.symbol] + [
            s for s in self.config.symbol_aliases if s != self.config.symbol
        ]

        for sym in candidates:
            # Check if symbol exists in broker's database
            info = mt5.symbol_info(sym)
            if info is not None:
                # Make sure it is selected in Market Watch
                if not info.visible:
                    selected = mt5.symbol_select(sym, True)
                    if not selected:
                        logger.warning(f"Could not select symbol '{sym}' in Market Watch.")
                        continue
                    info = mt5.symbol_info(sym)

                self.resolved_symbol = sym
                self.symbol_info = info
                
                point = float(info.point) if info.point > 0 else 0.01
                max_spread_pts = round(self.config.max_spread_price / point, 1)
                
                logger.info(
                    f"Resolved Trade Symbol: '{sym}' | Digits: {info.digits} | "
                    f"Point: {info.point} | Tick Value: {info.trade_tick_value} | "
                    f"Tick Size: {info.trade_tick_size} | Min Lot: {info.volume_min} | "
                    f"Step: {info.volume_step} | Stops Level: {info.trade_stops_level}"
                )
                logger.info(
                    f"Spread Filter Active: MAX_SPREAD_PRICE=${self.config.max_spread_price:.2f} ➔ "
                    f"Resolves to {max_spread_pts} broker points (Broker Digits: {info.digits}, Point: {info.point})"
                )
                return True

        return False

    def refresh_symbol_info(self) -> Optional[Any]:
        """Refreshes and returns the current symbol specification."""
        if not self.resolved_symbol:
            return None
        self.symbol_info = mt5.symbol_info(self.resolved_symbol)
        return self.symbol_info

    def get_server_time(self) -> datetime:
        """
        Returns the current broker server time based on the latest tick or server clock.
        This is crucial for keying daily loss resets to the broker's trading day calendar.
        """
        if self.resolved_symbol:
            tick = mt5.symbol_info_tick(self.resolved_symbol)
            if tick and tick.time > 0:
                return datetime.fromtimestamp(tick.time, tz=timezone.utc)

        # Fallback to local UTC if tick time unavailable
        return datetime.now(timezone.utc)

    def get_candles(self, count: Optional[int] = None) -> Optional[pd.DataFrame]:
        """
        Pulls the last N candles for the resolved symbol and configured timeframe.
        Returns a formatted pandas DataFrame with datetime index or None on error.
        """
        if not self.ensure_connection():
            return None

        n = count or self.config.candle_history_count
        tf = TIMEFRAME_MAP.get(self.config.timeframe_str, mt5.TIMEFRAME_M1)

        rates = mt5.copy_rates_from_pos(self.resolved_symbol, tf, 0, n)
        if rates is None or len(rates) == 0:
            err_code, err_msg = mt5.last_error()
            logger.warning(f"Failed to copy rates for '{self.resolved_symbol}': [{err_code}] {err_msg}")
            return None

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        return df

    def get_current_tick(self) -> Optional[Dict[str, float]]:
        """
        Returns the current Bid, Ask, Spread (points), Spread (price), and Tick Time.
        """
        if not self.ensure_connection():
            return None

        tick = mt5.symbol_info_tick(self.resolved_symbol)
        if tick is None:
            err_code, err_msg = mt5.last_error()
            logger.warning(f"Failed to get tick for '{self.resolved_symbol}': [{err_code}] {err_msg}")
            return None

        info = self.symbol_info or self.refresh_symbol_info()
        point = info.point if info else 0.01

        spread_price = round(tick.ask - tick.bid, 5)
        spread_points = round(spread_price / point, 2) if point > 0 else 0.0

        return {
            "bid": tick.bid,
            "ask": tick.ask,
            "last": tick.last,
            "spread_price": spread_price,
            "spread_points": spread_points,
            "time": tick.time,
        }

    def get_account_summary(self) -> Optional[Dict[str, Any]]:
        """Pulls updated balance, equity, margin, free margin, profit."""
        if not self.ensure_connection():
            return None

        acc = mt5.account_info()
        if acc is None:
            return None

        self.account_info = acc
        margin_mode_val = getattr(acc, "margin_mode", 2)
        hedging_const = getattr(mt5, "ACCOUNT_MARGIN_MODE_RETAIL_HEDGING", 2)
        is_hedging = (margin_mode_val == hedging_const)
        margin_mode_str = "HEDGING" if is_hedging else "NETTING"

        return {
            "login": acc.login,
            "balance": acc.balance,
            "equity": acc.equity,
            "margin": acc.margin,
            "free_margin": acc.margin_free,
            "profit": acc.profit,
            "currency": acc.currency,
            "trade_mode": acc.trade_mode,
            "margin_mode": margin_mode_val,
            "margin_mode_str": margin_mode_str,
            "is_hedging": is_hedging,
            "server": getattr(acc, "server", self.config.mt5_server),
        }

    def ensure_connection(self) -> bool:
        """Checks connection health and attempts automatic reconnection if lost."""
        term_info = mt5.terminal_info()
        if term_info is None or not term_info.connected:
            logger.warning("MT5 terminal connection dropped. Attempting reconnection...")
            return self.initialize()
        return True

    def shutdown(self) -> None:
        """Cleanly disconnects and releases MT5 terminal resources."""
        logger.info("Shutting down MT5 connection...")
        try:
            mt5.shutdown()
        except Exception as e:
            logger.error(f"Error during MT5 shutdown: {e}")
        self.is_connected = False
