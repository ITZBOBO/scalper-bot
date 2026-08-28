"""
signal_engine.py - Signal Generation Engine for MT5 Gold Scalper.

Computes Fast EMA (9), Slow EMA (21), RSI (14), and ATR (14) on closed candles.
Evaluates crossover entries with RSI exhaustion filters and strict ATR validation guards.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional, Tuple
import logging
import numpy as np
import pandas as pd

from config import BotConfig, config as default_config

logger = logging.getLogger("ScalperBot.SignalEngine")


class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    NONE = "NONE"


@dataclass
class Signal:
    """Represents a trading signal produced by the signal engine."""
    signal_type: SignalType
    entry_price: float
    atr_value: float
    candle_time: Optional[datetime]
    fast_ema: float
    slow_ema: float
    rsi: float
    reason: str

    @property
    def is_buy(self) -> bool:
        return self.signal_type == SignalType.BUY

    @property
    def is_sell(self) -> bool:
        return self.signal_type == SignalType.SELL

    @property
    def is_valid(self) -> bool:
        return self.signal_type in (SignalType.BUY, SignalType.SELL)


class SignalEngine:
    """Computes technical indicators and generates trading signals on completed candles."""

    def __init__(self, bot_config: BotConfig = default_config):
        self.config = bot_config

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates Fast EMA, Slow EMA, RSI, and ATR on the given candle DataFrame.
        """
        data = df.copy()

        # 1. Exponential Moving Averages (EMA)
        data["ema_fast"] = data["close"].ewm(span=self.config.ema_fast_period, adjust=False).mean()
        data["ema_slow"] = data["close"].ewm(span=self.config.ema_slow_period, adjust=False).mean()

        # 2. Relative Strength Index (RSI - Wilder's Smoothing)
        delta = data["close"].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        # Wilder's exponential smoothing: alpha = 1 / period
        avg_gain = gain.ewm(alpha=1.0 / self.config.rsi_period, min_periods=self.config.rsi_period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / self.config.rsi_period, min_periods=self.config.rsi_period, adjust=False).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        data["rsi"] = 100.0 - (100.0 / (1.0 + rs))
        # Where avg_loss was 0 and avg_gain > 0, RSI is 100; where both 0, RSI is 50
        data.loc[avg_loss == 0, "rsi"] = 100.0
        data.loc[(avg_loss == 0) & (avg_gain == 0), "rsi"] = 50.0

        # 3. Average True Range (ATR)
        prev_close = data["close"].shift(1)
        tr1 = data["high"] - data["low"]
        tr2 = (data["high"] - prev_close).abs()
        tr3 = (data["low"] - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        data["atr"] = tr.ewm(alpha=1.0 / self.config.atr_period, min_periods=self.config.atr_period, adjust=False).mean()

        return data

    def evaluate_signals(self, candle_df: pd.DataFrame) -> Signal:
        """
        Evaluates indicator crossover signals strictly on completed (closed) candles.
        
        Note: The incoming DataFrame is expected to contain at least (ema_slow_period + 2) bars.
        The last row in a live MT5 copy_rates call (index -1) represents the actively forming,
        incomplete bar. Therefore, we evaluate index -2 as the latest completed candle and
        index -3 as the prior completed candle for crossover confirmation.
        """
        min_required_bars = max(self.config.ema_slow_period, self.config.rsi_period, self.config.atr_period) + 5
        if candle_df is None or len(candle_df) < min_required_bars:
            return Signal(
                signal_type=SignalType.NONE,
                entry_price=0.0,
                atr_value=0.0,
                candle_time=None,
                fast_ema=0.0,
                slow_ema=0.0,
                rsi=0.0,
                reason=f"Insufficient candle history ({len(candle_df) if candle_df is not None else 0}/{min_required_bars} bars)",
            )

        df = self.compute_indicators(candle_df)

        # Row -2 is the most recently closed candle; Row -3 is the candle before it
        curr_bar = df.iloc[-2]
        prev_bar = df.iloc[-3]

        candle_time = curr_bar["time"]
        if hasattr(candle_time, "to_pydatetime"):
            candle_time = candle_time.to_pydatetime()

        curr_close = float(curr_bar["close"])
        fast_curr = float(curr_bar["ema_fast"])
        slow_curr = float(curr_bar["ema_slow"])
        fast_prev = float(prev_bar["ema_fast"])
        slow_prev = float(prev_bar["ema_slow"])
        rsi_curr = float(curr_bar["rsi"])
        atr_curr = float(curr_bar["atr"])

        # ---------------------------------------------------------------------
        # ATR VALIDATION GUARD: Reject immediately if ATR is zero, NaN, or negative
        # ---------------------------------------------------------------------
        if np.isnan(atr_curr) or np.isinf(atr_curr) or atr_curr <= 0.0:
            logger.warning(f"Signal rejected on bar {candle_time}: Invalid/Zero ATR ({atr_curr})")
            return Signal(
                signal_type=SignalType.NONE,
                entry_price=curr_close,
                atr_value=0.0,
                candle_time=candle_time,
                fast_ema=fast_curr,
                slow_ema=slow_curr,
                rsi=rsi_curr,
                reason=f"Invalid or Zero ATR ({atr_curr}) on closed candle",
            )

        # ---------------------------------------------------------------------
        # BULLISH CROSSOVER: Fast EMA crosses ABOVE Slow EMA + RSI < Overbought
        # ---------------------------------------------------------------------
        bullish_crossover = (fast_prev <= slow_prev) and (fast_curr > slow_curr)
        if bullish_crossover:
            if rsi_curr < self.config.rsi_overbought:
                logger.info(
                    f"BUY SIGNAL generated at {candle_time} | Close: {curr_close:.2f} | "
                    f"Fast EMA: {fast_curr:.2f} > Slow EMA: {slow_curr:.2f} | "
                    f"RSI: {rsi_curr:.2f} (< {self.config.rsi_overbought}) | ATR: {atr_curr:.2f}"
                )
                return Signal(
                    signal_type=SignalType.BUY,
                    entry_price=curr_close,
                    atr_value=atr_curr,
                    candle_time=candle_time,
                    fast_ema=fast_curr,
                    slow_ema=slow_curr,
                    rsi=rsi_curr,
                    reason=f"Bullish EMA crossover (Fast {fast_curr:.2f} > Slow {slow_curr:.2f}) & RSI {rsi_curr:.2f} < {self.config.rsi_overbought}",
                )
            else:
                logger.debug(
                    f"Buy crossover filtered out at {candle_time}: RSI {rsi_curr:.2f} >= overbought threshold {self.config.rsi_overbought}"
                )
                return Signal(
                    signal_type=SignalType.NONE,
                    entry_price=curr_close,
                    atr_value=atr_curr,
                    candle_time=candle_time,
                    fast_ema=fast_curr,
                    slow_ema=slow_curr,
                    rsi=rsi_curr,
                    reason=f"Bullish crossover filtered: RSI {rsi_curr:.2f} >= Overbought {self.config.rsi_overbought}",
                )

        # ---------------------------------------------------------------------
        # BEARISH CROSSOVER: Fast EMA crosses BELOW Slow EMA + RSI > Oversold
        # ---------------------------------------------------------------------
        bearish_crossover = (fast_prev >= slow_prev) and (fast_curr < slow_curr)
        if bearish_crossover:
            if rsi_curr > self.config.rsi_oversold:
                logger.info(
                    f"SELL SIGNAL generated at {candle_time} | Close: {curr_close:.2f} | "
                    f"Fast EMA: {fast_curr:.2f} < Slow EMA: {slow_curr:.2f} | "
                    f"RSI: {rsi_curr:.2f} (> {self.config.rsi_oversold}) | ATR: {atr_curr:.2f}"
                )
                return Signal(
                    signal_type=SignalType.SELL,
                    entry_price=curr_close,
                    atr_value=atr_curr,
                    candle_time=candle_time,
                    fast_ema=fast_curr,
                    slow_ema=slow_curr,
                    rsi=rsi_curr,
                    reason=f"Bearish EMA crossover (Fast {fast_curr:.2f} < Slow {slow_curr:.2f}) & RSI {rsi_curr:.2f} > {self.config.rsi_oversold}",
                )
            else:
                logger.debug(
                    f"Sell crossover filtered out at {candle_time}: RSI {rsi_curr:.2f} <= oversold threshold {self.config.rsi_oversold}"
                )
                return Signal(
                    signal_type=SignalType.NONE,
                    entry_price=curr_close,
                    atr_value=atr_curr,
                    candle_time=candle_time,
                    fast_ema=fast_curr,
                    slow_ema=slow_curr,
                    rsi=rsi_curr,
                    reason=f"Bearish crossover filtered: RSI {rsi_curr:.2f} <= Oversold {self.config.rsi_oversold}",
                )

        # No crossover
        return Signal(
            signal_type=SignalType.NONE,
            entry_price=curr_close,
            atr_value=atr_curr,
            candle_time=candle_time,
            fast_ema=fast_curr,
            slow_ema=slow_curr,
            rsi=rsi_curr,
            reason="No EMA crossover on completed candle",
        )
