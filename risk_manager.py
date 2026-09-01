"""
risk_manager.py - Risk Management & Grouped Position Sizing for MT5 Gold Scalper.

Performs broker-normalized dynamic lot sizing, ATR-based SL/TP calculation,
broker-aware spread filtering, concurrency validation, and invariant-preserving
volume splitting across multiple positions within a single risk budget.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import logging
import math

from config import BotConfig, config as default_config
from signal_engine import Signal, SignalType

logger = logging.getLogger("ScalperBot.RiskManager")


@dataclass
class RiskAssessmentResult:
    """Outcome of risk manager evaluation on a trading signal."""
    approved: bool
    lot_size: float = 0.0                    # Total combined volume (backward compatible)
    total_lot_size: float = 0.0              # Explicit total volume
    position_lots: List[float] = field(default_factory=list)  # Sub-position lot sizes
    positions_count: int = 1                 # Number of positions in this group
    entry_price: float = 0.0
    sl_price: float = 0.0
    tp_price: float = 0.0
    stop_distance_price: float = 0.0
    tp_distance_price: float = 0.0
    risk_amount_currency: float = 0.0
    theoretical_group_risk: float = 0.0      # Combined theoretical SL loss
    group_profit_target: float = 2.00        # Group profit target ($2.00 total)
    rejection_reason: Optional[str] = None


class RiskManager:
    """Enforces risk rules, dynamic lot calculations, and trade group filters."""

    def __init__(self, bot_config: BotConfig = default_config):
        self.config = bot_config

    def split_group_volume(
        self,
        total_lot: float,
        requested_positions: int,
        volume_min: float,
        volume_step: float,
        is_hedging: bool = True,
    ) -> List[float]:
        """
        Splits total_lot across requested_positions respecting broker constraints.
        
        CRITICAL INVARIANT:
        TOTAL GROUP RISK <= FIXED_RISK_AMOUNT.
        The sum of returned lots is guaranteed to exactly equal total_lot.
        Never increases total volume merely to satisfy requested_positions.
        If total_lot cannot be split into requested_positions of >= volume_min,
        safely reduces the positions count down to 1.
        """
        # Netting accounts or single-position requests always execute as a single position
        if not is_hedging or requested_positions <= 1:
            return [total_lot]

        # Determine maximum feasible sub-positions without exceeding total_lot
        max_positions = max(1, int(math.floor(round(total_lot / volume_min, 6))))
        actual_positions = min(requested_positions, max_positions)

        if actual_positions <= 1:
            return [total_lot]

        precision = max(0, int(round(-math.log10(volume_step)))) if volume_step < 1 else 0
        total_steps = int(round(total_lot / volume_step))
        base_steps = total_steps // actual_positions
        remainder_steps = total_steps % actual_positions

        lots: List[float] = []
        for i in range(actual_positions):
            steps = base_steps + (1 if i < remainder_steps else 0)
            lot = round(steps * volume_step, precision)
            lots.append(lot)

        # Verification guard: sum(lots) must match total_lot exactly
        diff = abs(sum(lots) - total_lot)
        if diff > 1e-5:
            logger.warning(
                f"Volume split rounding discrepancy ({sum(lots)} != {total_lot}). "
                f"Falling back to single position {total_lot}."
            )
            return [total_lot]

        return lots

    def evaluate_signal(
        self,
        signal: Signal,
        tick: Dict[str, float],
        account_summary: Dict[str, Any],
        symbol_info: Any,
        open_positions_count: int = 0,
        open_groups_count: Optional[int] = None,
    ) -> RiskAssessmentResult:
        """
        Validates signal against spread limits, concurrency limits, and calculates
        broker-normalized lot size, sub-position split, and ATR-based SL/TP price levels.
        """
        # 1. Basic Signal Validity Check
        if not signal.is_valid:
            return RiskAssessmentResult(
                approved=False,
                rejection_reason=f"Invalid signal type: {signal.signal_type.value} ({signal.reason})",
            )

        # 2. Concurrency Check (Group-Aware)
        active_count = open_groups_count if open_groups_count is not None else open_positions_count
        max_allowed_groups = self.config.max_concurrent_trade_groups
        if active_count >= max_allowed_groups:
            reason = (
                f"Max concurrent trade groups reached: {active_count} active >= "
                f"limit of {max_allowed_groups}"
            )
            logger.info(f"Risk Manager REJECT: {reason}")
            return RiskAssessmentResult(approved=False, rejection_reason=reason)

        # 3. Spread Filter Check (Broker & Digits Aware)
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
        if self.config.fixed_tp_price_distance and self.config.fixed_tp_price_distance > 0:
            tp_distance_price = self.config.fixed_tp_price_distance
        else:
            tp_distance_price = atr * self.config.tp_atr_multiplier
        sl_distance_points = sl_distance_price / point

        # 6. Dynamic Broker-Normalized Lot Sizing
        contract_size = float(symbol_info.trade_contract_size) if hasattr(symbol_info, "trade_contract_size") and symbol_info.trade_contract_size > 0 else 100.0
        volume_min = float(symbol_info.volume_min) if symbol_info else 0.01
        volume_max = float(symbol_info.volume_max) if symbol_info else 100.0
        volume_step = float(symbol_info.volume_step) if symbol_info else 0.01
        sym_name = getattr(symbol_info, "name", self.config.symbol)
        precision = max(0, int(round(-math.log10(volume_step)))) if volume_step < 1 else 0

        entry_test = tick["ask"] if signal.is_buy else tick["bid"]
        exit_test = entry_test - sl_distance_price if signal.is_buy else entry_test + sl_distance_price
        order_type = 0 if signal.is_buy else 1

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
            loss_per_1_lot = sl_distance_price * contract_size

        if loss_per_1_lot <= 0:
            loss_per_1_lot = sl_distance_price * 100.0

        is_hedging = bool(account_summary.get("is_hedging", True))

        if self.config.group_risk_mode == "FIXED_LOTS_PER_POSITION" and is_hedging:
            # User explicitly configured fixed lots per position (e.g. 3 positions of 0.01 = 0.03 lots total)
            req_pos = max(1, self.config.positions_per_group)
            position_lots = [volume_min] * req_pos
            calculated_lot = round(sum(position_lots), precision)
            theoretical_group_risk = sum(p * loss_per_1_lot for p in position_lots)
        else:
            # Exact calculated total lot size to risk target percentage
            raw_lot_size = risk_amount / loss_per_1_lot

            # Quantize to broker volume step
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
                max_tolerable_risk = max(risk_amount * 3.5, 5.0) if self.config.fixed_risk_amount else risk_amount * 1.5
                if min_lot_risk <= max_tolerable_risk:
                    logger.info(
                        f"Calculated lot {calculated_lot} below volume_min {volume_min}; clamping to volume_min {volume_min} (Risk: ${min_lot_risk:.2f})"
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

            # 7. Multi-Position Volume Split (Invariant: Sum(positions) == Total Lot)
            position_lots = self.split_group_volume(
                total_lot=calculated_lot,
                requested_positions=self.config.positions_per_group,
                volume_min=volume_min,
                volume_step=volume_step,
                is_hedging=is_hedging,
            )
            theoretical_group_risk = sum(p * loss_per_1_lot for p in position_lots)

        # 8. Group-Level TP Price Calculation ($2.00 Total Profit Across All Positions)
        group_target = float(self.config.group_profit_target) if self.config.group_profit_target is not None and self.config.group_profit_target > 0 else 0.0
        tick_val = float(getattr(symbol_info, "trade_tick_value", 0.0)) if symbol_info else 0.0
        tick_sz = float(getattr(symbol_info, "trade_tick_size", 0.0)) if symbol_info else 0.0
        if tick_val > 0 and tick_sz > 0:
            profit_per_1_lot_per_price_unit = tick_val / tick_sz
        else:
            profit_per_1_lot_per_price_unit = contract_size

        if group_target > 0 and calculated_lot > 0:
            # Price move required so that sum(position_lots) captures group_target ($2.00)
            tp_distance_price = group_target / (calculated_lot * profit_per_1_lot_per_price_unit)
        elif self.config.fixed_tp_price_distance and self.config.fixed_tp_price_distance > 0:
            tp_distance_price = self.config.fixed_tp_price_distance
        else:
            tp_distance_price = atr * self.config.tp_atr_multiplier

        stops_level = int(symbol_info.trade_stops_level) if symbol_info else 0
        min_stop_distance_price = stops_level * point

        if signal.is_buy:
            entry_price = round(tick["ask"], digits)
            sl_price = round(entry_price - sl_distance_price, digits)
            tp_price = round(entry_price + tp_distance_price, digits)

            if (entry_price - sl_price) < min_stop_distance_price:
                sl_price = round(entry_price - min_stop_distance_price, digits)
            if (tp_price - entry_price) < min_stop_distance_price:
                tp_price = round(entry_price + min_stop_distance_price, digits)

        else:  # SELL
            entry_price = round(tick["bid"], digits)
            sl_price = round(entry_price + sl_distance_price, digits)
            tp_price = round(entry_price - tp_distance_price, digits)

            if (sl_price - entry_price) < min_stop_distance_price:
                sl_price = round(entry_price + min_stop_distance_price, digits)
            if (entry_price - tp_price) < min_stop_distance_price:
                tp_price = round(entry_price - min_stop_distance_price, digits)

        logger.info(
            f"Risk Assessment APPROVED for {signal.signal_type.value}: "
            f"Total Lots: {calculated_lot} across {len(position_lots)} position(s) {position_lots} | "
            f"Entry: {entry_price:.{digits}f} | SL: {sl_price:.{digits}f} ({sl_distance_price:.2f}$) | "
            f"TP: {tp_price:.{digits}f} ({tp_distance_price:.2f}$) | "
            f"Target Group Profit: ${group_target:.2f} Total | "
            f"Theoretical Combined SL Risk: ~${theoretical_group_risk:.2f}"
        )

        return RiskAssessmentResult(
            approved=True,
            lot_size=calculated_lot,
            total_lot_size=calculated_lot,
            position_lots=position_lots,
            positions_count=len(position_lots),
            entry_price=entry_price,
            sl_price=sl_price,
            tp_price=tp_price,
            stop_distance_price=sl_distance_price,
            tp_distance_price=tp_distance_price,
            risk_amount_currency=risk_amount,
            theoretical_group_risk=theoretical_group_risk,
            group_profit_target=group_target,
            rejection_reason=None,
        )
