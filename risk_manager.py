"""
risk_manager.py - Risk Management & Position Sizing for MT5 Gold Scalper.

Performs broker-normalized dynamic lot sizing, ATR-based SL/TP calculation,
broker-aware spread filtering, and concurrency validation.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import logging
import math

from config import BotConfig, config as default_config
from signal_engine import Signal, SignalType

logger = logging.getLogger("ScalperBot.RiskManager")


@dataclass
class RiskAssessmentResult:
    """Outcome of risk manager evaluation on a trading signal."""
    approved: bool
    lot_size: float = 0.0
    entry_price: float = 0.0
    sl_price: float = 0.0
    tp_price: float = 0.0
    stop_distance_price: float = 0.0
    tp_distance_price: float = 0.0
    risk_amount_currency: float = 0.0
    rejection_reason: Optional[str] = None


class RiskManager:
    """Enforces risk rules, dynamic lot calculations, and trade filters."""

    def __init__(self, bot_config: BotConfig = default_config):
        self.config = bot_config

    def evaluate_signal(
        self,
        signal: Signal,
        tick: Dict[str, float],
        account_summary: Dict[str, Any],
        symbol_info: Any,
        open_positions_count: int,
    ) -> RiskAssessmentResult:
        """
        Validates signal against spread limits, concurrency limits, and calculates
        broker-normalized lot size and ATR-based SL/TP price levels.
        """
        # 1. Basic Signal Validity Check
        if not signal.is_valid:
            return RiskAssessmentResult(
                approved=False,
                rejection_reason=f"Invalid signal type: {signal.signal_type.value} ({signal.reason})",
            )

        # 2. Concurrency Check
        if open_positions_count >= self.config.max_concurrent_trades:
            reason = (
                f"Max concurrent trades reached: {open_positions_count} open >= "
                f"limit of {self.config.max_concurrent_trades}"
            )
            logger.info(f"Risk Manager REJECT: {reason}")
            return RiskAssessmentResult(approved=False, rejection_reason=reason)

        # 3. Spread Filter Check (Broker & Digits Aware)
        # Gold broker pricing may be 2-digit (0.01) or 3-digit (0.001).
        # We compare spread in price ($) as well as normalized points.
        point = float(symbol_info.point) if symbol_info and symbol_info.point > 0 else 0.01
        digits = int(symbol_info.digits) if symbol_info else 2

        current_spread_price = tick["spread_price"]
        current_spread_points = tick["spread_points"]
        max_allowed_spread_price = self.config.max_spread_price
        max_allowed_spread_points = round(max_allowed_spread_price / point, 1)

        if current_spread_price > max_allowed_spread_price:
            reason = (
                f"Spread too wide: Current spread {current_spread_price:.3f} price "
                f"({current_spread_points:.1f} pts) > Max allowed {max_allowed_spread_price:.3f} price "
                f"({max_allowed_spread_points:.1f} pts)"
            )
            logger.warning(f"Risk Manager REJECT: {reason}")
            return RiskAssessmentResult(approved=False, rejection_reason=reason)

        # 4. Account Balance & Risk Capital
        balance = float(account_summary.get("balance", 0.0))
        if balance <= 0:
            reason = f"Invalid account balance: {balance}"
            logger.error(f"Risk Manager REJECT: {reason}")
            return RiskAssessmentResult(approved=False, rejection_reason=reason)

        if self.config.fixed_risk_amount and self.config.fixed_risk_amount > 0:
            risk_amount = self.config.fixed_risk_amount
        else:
            risk_amount = balance * (self.config.risk_per_trade_pct / 100.0)

        # 5. ATR Stop & Take Profit Distances
        atr = signal.atr_value
        if atr <= 0:
            reason = f"Non-positive ATR value for sizing: {atr}"
            logger.error(f"Risk Manager REJECT: {reason}")
            return RiskAssessmentResult(approved=False, rejection_reason=reason)

        sl_distance_price = atr * self.config.sl_atr_multiplier
        tp_distance_price = atr * self.config.tp_atr_multiplier
        sl_distance_points = sl_distance_price / point

        # 6. Dynamic Broker-Normalized Lot Sizing
        # Pull broker specifications
        contract_size = float(symbol_info.trade_contract_size) if hasattr(symbol_info, "trade_contract_size") and symbol_info.trade_contract_size > 0 else 100.0
        volume_min = float(symbol_info.volume_min) if symbol_info else 0.01
        volume_max = float(symbol_info.volume_max) if symbol_info else 100.0
        volume_step = float(symbol_info.volume_step) if symbol_info else 0.01
        sym_name = getattr(symbol_info, "name", self.config.symbol)

        entry_test = tick["ask"] if signal.is_buy else tick["bid"]
        exit_test = entry_test - sl_distance_price if signal.is_buy else entry_test + sl_distance_price
        order_type = 0 if signal.is_buy else 1  # 0 = ORDER_TYPE_BUY, 1 = ORDER_TYPE_SELL

        # Calculate exact monetary loss for 1.0 standard lot
        loss_per_1_lot = None
        try:
            import MetaTrader5 as mt5_calc
            p_calc = mt5_calc.order_calc_profit(order_type, sym_name, 1.0, entry_test, exit_test)
            if p_calc is not None and abs(p_calc) > 0:
                loss_per_1_lot = abs(float(p_calc))
        except Exception:
            loss_per_1_lot = None

        if loss_per_1_lot is None or loss_per_1_lot <= 0:
            # Fallback based on contract size: loss = stop_distance_price * contract_size
            loss_per_1_lot = sl_distance_price * contract_size

        if loss_per_1_lot <= 0:
            loss_per_1_lot = sl_distance_price * 100.0

        # Exact calculated lot size to risk target percentage
        raw_lot_size = risk_amount / loss_per_1_lot

        # Quantize to broker volume step
        precision = max(0, int(round(-math.log10(volume_step)))) if volume_step < 1 else 0
        steps = math.floor(raw_lot_size / volume_step)
        calculated_lot = round(steps * volume_step, precision)

        # Margin Affordability Guard
        free_margin = float(account_summary.get("free_margin", balance))
        try:
            import MetaTrader5 as mt5_margin
            req_margin = mt5_margin.order_calc_margin(order_type, sym_name, calculated_lot, entry_test)
            if req_margin is not None and req_margin > (free_margin * 0.90) and req_margin > 0:
                margin_per_lot = req_margin / calculated_lot
                max_affordable_lots = math.floor((free_margin * 0.80 / margin_per_lot) / volume_step) * volume_step
                new_lot = max(volume_min, round(max_affordable_lots, precision))
                logger.warning(
                    f"Calculated lot {calculated_lot} required ${req_margin:.2f} margin (Free Margin: ${free_margin:.2f}). "
                    f"Scaled down to {new_lot} lots for safety."
                )
                calculated_lot = new_lot
        except Exception:
            pass

        # Clamp between broker min and max lot size
        if calculated_lot < volume_min:
            min_lot_risk = volume_min * loss_per_1_lot
            max_tolerable_risk = risk_amount * 1.5
            if min_lot_risk <= max_tolerable_risk:
                logger.info(
                    f"Calculated lot {calculated_lot} below volume_min {volume_min}; clamping to volume_min {volume_min}"
                )
                calculated_lot = volume_min
            else:
                reason = (
                    f"Calculated lot {raw_lot_size:.3f} below broker min {volume_min}. "
                    f"Required risk ${min_lot_risk:.2f} exceeds target ${risk_amount:.2f}"
                )
                logger.warning(f"Risk Manager REJECT: {reason}")
                return RiskAssessmentResult(approved=False, rejection_reason=reason)

        if calculated_lot > volume_max:
            logger.warning(f"Calculated lot {calculated_lot} exceeded volume_max {volume_max}; clamping.")
            calculated_lot = volume_max

        # 7. Exact SL and TP Price Calculation
        stops_level = int(symbol_info.trade_stops_level) if symbol_info else 0
        min_stop_distance_price = stops_level * point

        if signal.is_buy:
            entry_price = round(tick["ask"], digits)
            sl_price = round(entry_price - sl_distance_price, digits)
            tp_price = round(entry_price + tp_distance_price, digits)

            # Ensure SL/TP are further than broker minimum stops level
            if (entry_price - sl_price) < min_stop_distance_price:
                sl_price = round(entry_price - min_stop_distance_price, digits)
            if (tp_price - entry_price) < min_stop_distance_price:
                tp_price = round(entry_price + min_stop_distance_price, digits)

        else:  # SELL
            entry_price = round(tick["bid"], digits)
            sl_price = round(entry_price + sl_distance_price, digits)
            tp_price = round(entry_price - tp_distance_price, digits)

            # Ensure SL/TP are further than broker minimum stops level
            if (sl_price - entry_price) < min_stop_distance_price:
                sl_price = round(entry_price + min_stop_distance_price, digits)
            if (entry_price - tp_price) < min_stop_distance_price:
                tp_price = round(entry_price - min_stop_distance_price, digits)

        logger.info(
            f"Risk Assessment APPROVED for {signal.signal_type.value}: "
            f"Lots: {calculated_lot} | Entry: {entry_price:.{digits}f} | "
            f"SL: {sl_price:.{digits}f} ({sl_distance_price:.2f}$) | "
            f"TP: {tp_price:.{digits}f} ({tp_distance_price:.2f}$) | "
            f"Risk: ${risk_amount:.2f} ({self.config.risk_per_trade_pct}% of ${balance:.2f})"
        )

        return RiskAssessmentResult(
            approved=True,
            lot_size=calculated_lot,
            entry_price=entry_price,
            sl_price=sl_price,
            tp_price=tp_price,
            stop_distance_price=sl_distance_price,
            tp_distance_price=tp_distance_price,
            risk_amount_currency=risk_amount,
            rejection_reason=None,
        )
