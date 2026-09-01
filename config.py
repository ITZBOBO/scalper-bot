"""
config.py - Centralized configuration for MT5 Gold Scalper Bot.

All tunable parameters, credentials, risk limits, indicator settings,
and file paths are defined here. No logic files should contain hardcoded parameters.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()


def _get_bool_env(key: str, default: bool) -> bool:
    val = os.getenv(key)
    if val is None:
        return default
    return val.strip().lower() in ("true", "1", "yes", "t", "y")


def _get_float_env(key: str, default: float) -> float:
    val = os.getenv(key)
    if val is None or val.strip() == "":
        return default
    try:
        return float(val.strip())
    except ValueError:
        return default


def _get_int_env(key: str, default: int) -> int:
    val = os.getenv(key)
    if val is None or val.strip() == "":
        return default
    try:
        return int(val.strip())
    except ValueError:
        return default


@dataclass
class BotConfig:
    # -------------------------------------------------------------------------
    # MT5 Terminal & Account Settings
    # -------------------------------------------------------------------------
    mt5_login: int = field(default_factory=lambda: _get_int_env("MT5_LOGIN", 0))
    mt5_password: str = field(default_factory=lambda: os.getenv("MT5_PASSWORD", ""))
    mt5_server: str = field(default_factory=lambda: os.getenv("MT5_SERVER", "MetaQuotes-Demo"))
    mt5_path: Optional[str] = field(default_factory=lambda: os.getenv("MT5_PATH") or None)

    # SAFETY LOCK: Default to Demo only. Must explicitly set True in .env to allow live trading.
    allow_live_trading: bool = field(default_factory=lambda: _get_bool_env("ALLOW_LIVE_TRADING", False))

    # -------------------------------------------------------------------------
    # Asset & Timeframe
    # -------------------------------------------------------------------------
    symbol: str = field(default_factory=lambda: os.getenv("SYMBOL", "XAUUSD").strip().upper())
    # Symbol aliases to check if standard 'XAUUSD' is named differently by broker
    symbol_aliases: List[str] = field(default_factory=lambda: [
        "XAUUSD", "GOLD", "XAUUSD.m", "XAUUSDm", "XAUUSD_i", "XAUUSD.", "XAUUSDpro", "XAU_USD",
        "XAUUSDc", "XAUUSD_c", "XAUUSD.c"
    ])
    timeframe_str: str = field(default_factory=lambda: os.getenv("TIMEFRAME", "M1").strip().upper())
    candle_history_count: int = 150  # Number of candles pulled for indicator calculation

    # -------------------------------------------------------------------------
    # Strategy & Indicator Parameters
    # -------------------------------------------------------------------------
    ema_fast_period: int = field(default_factory=lambda: _get_int_env("EMA_FAST_PERIOD", 9))
    ema_slow_period: int = field(default_factory=lambda: _get_int_env("EMA_SLOW_PERIOD", 21))
    rsi_period: int = field(default_factory=lambda: _get_int_env("RSI_PERIOD", 14))
    rsi_overbought: float = field(default_factory=lambda: _get_float_env("RSI_OVERBOUGHT", 70.0))
    rsi_oversold: float = field(default_factory=lambda: _get_float_env("RSI_OVERSOLD", 30.0))
    atr_period: int = field(default_factory=lambda: _get_int_env("ATR_PERIOD", 14))
    sl_atr_multiplier: float = field(default_factory=lambda: _get_float_env("SL_ATR_MULTIPLIER", 1.5))
    tp_atr_multiplier: float = field(default_factory=lambda: _get_float_env("TP_ATR_MULTIPLIER", 1.0))
    fixed_tp_price_distance: Optional[float] = field(
        default_factory=lambda: _get_float_env("FIXED_TP_PRICE_DISTANCE", 0.0) if _get_float_env("FIXED_TP_PRICE_DISTANCE", 0.0) > 0 else None
    )

    # -------------------------------------------------------------------------
    # Risk Management & Spread Filter
    # -------------------------------------------------------------------------
    # Maximum allowable spread in dollar/price units (e.g. 0.40 = $0.40 on Gold)
    # This prevents broker pricing differences (2-digit vs 3-digit quotes) from distorting thresholds.
    max_spread_price: float = field(default_factory=lambda: _get_float_env("MAX_SPREAD_PRICE", 0.40))
    risk_per_trade_pct: float = field(default_factory=lambda: _get_float_env("RISK_PER_TRADE_PCT", 1.0))
    # Optional fixed cash risk (e.g. 2000 in NGN account or 2.0 in USD account). If set > 0, overrides pct.
    fixed_risk_amount: Optional[float] = field(
        default_factory=lambda: _get_float_env("FIXED_RISK_AMOUNT", 0.0) if _get_float_env("FIXED_RISK_AMOUNT", 0.0) > 0 else None
    )
    max_daily_loss_pct: float = field(default_factory=lambda: _get_float_env("MAX_DAILY_LOSS_PCT", 3.0))
    max_consecutive_losses: int = field(default_factory=lambda: _get_int_env("MAX_CONSECUTIVE_LOSSES", 3))
    max_concurrent_trades: int = field(default_factory=lambda: _get_int_env("MAX_CONCURRENT_TRADES", 1))
    max_concurrent_trade_groups: int = field(default_factory=lambda: _get_int_env("MAX_CONCURRENT_TRADE_GROUPS", 1))
    positions_per_group: int = field(default_factory=lambda: _get_int_env("POSITIONS_PER_GROUP", 3))
    group_risk_mode: str = field(default_factory=lambda: os.getenv("GROUP_RISK_MODE", "FIXED_TOTAL_RISK").strip())
    group_profit_target: float = field(default_factory=lambda: _get_float_env("GROUP_PROFIT_TARGET", 2.00))
    slippage_points: int = field(default_factory=lambda: _get_int_env("SLIPPAGE_POINTS", 20))

    # -------------------------------------------------------------------------
    # Trade Identification
    # -------------------------------------------------------------------------
    magic_number: int = 987654
    order_comment: str = "XAU Scalper"

    # -------------------------------------------------------------------------
    # Telegram Notifications
    # -------------------------------------------------------------------------
    telegram_bot_token: Optional[str] = field(
        default_factory=lambda: os.getenv("TELEGRAM_BOT_TOKEN") or None
    )
    telegram_chat_id: Optional[str] = field(
        default_factory=lambda: os.getenv("TELEGRAM_CHAT_ID") or None
    )

    # -------------------------------------------------------------------------
    # Paths and Storage
    # -------------------------------------------------------------------------
    logs_dir: Path = Path("logs")
    data_dir: Path = Path("data")
    trades_csv_path: Path = Path("logs/trades.csv")
    signals_csv_path: Path = Path("logs/signals.csv")
    kill_switch_state_path: Path = Path("data/kill_switch_state.json")
    manual_stop_file_path: Path = Path("data/STOP")

    # -------------------------------------------------------------------------
    # Loop & Polling Intervals (Seconds)
    # -------------------------------------------------------------------------
    loop_poll_interval_sec: float = 0.25      # High frequency check for open positions & candle transition
    max_order_retries: int = 1                 # Retry once on requote/timeout, then abort
    retry_delay_sec: float = 0.25             # Backoff before single retry

    def __post_init__(self):
        # Create directories if they do not exist
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    @property
    def is_telegram_enabled(self) -> bool:
        return bool(self.telegram_bot_token and self.telegram_chat_id)


# Global default configuration instance
config = BotConfig()
