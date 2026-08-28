"""
executor.py - Order Execution and Position Lifecycle Tracking for MT5.

Handles market order routing, filling mode negotiation, single-retry requote resilience,
fill confirmation, and position exit detection with realized P&L accounting.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
import logging
import time
import MetaTrader5 as mt5

from config import BotConfig, config as default_config
from signal_engine import SignalType
from risk_manager import RiskAssessmentResult

logger = logging.getLogger("ScalperBot.Executor")

# MT5 Return Codes
RETCODE_DESCRIPTIONS = {
    mt5.TRADE_RETCODE_REQUOTE: "Requote",
    mt5.TRADE_RETCODE_REJECT: "Request rejected",
    mt5.TRADE_RETCODE_CANCEL: "Request canceled by trader",
    mt5.TRADE_RETCODE_PLACED: "Order placed",
    mt5.TRADE_RETCODE_DONE: "Request completed",
    mt5.TRADE_RETCODE_DONE_PARTIAL: "Only part of request was completed",
    mt5.TRADE_RETCODE_ERROR: "Request processing error",
    mt5.TRADE_RETCODE_TIMEOUT: "Request canceled by timeout",
    mt5.TRADE_RETCODE_INVALID: "Invalid request",
    mt5.TRADE_RETCODE_INVALID_VOLUME: "Invalid volume in request",
    mt5.TRADE_RETCODE_INVALID_PRICE: "Invalid price in request",
    mt5.TRADE_RETCODE_INVALID_STOPS: "Invalid stops in request",
    mt5.TRADE_RETCODE_TRADE_DISABLED: "Trade is disabled",
    mt5.TRADE_RETCODE_MARKET_CLOSED: "Market is closed",
    mt5.TRADE_RETCODE_NO_MONEY: "Not enough money to complete request",
    mt5.TRADE_RETCODE_PRICE_CHANGED: "Prices changed",
    mt5.TRADE_RETCODE_PRICE_OFF: "Off quotes",
    mt5.TRADE_RETCODE_INVALID_EXPIRATION: "Invalid order expiration date in request",
    mt5.TRADE_RETCODE_ORDER_CHANGED: "Order state changed",
    mt5.TRADE_RETCODE_TOO_MANY_REQUESTS: "Too frequent requests",
    mt5.TRADE_RETCODE_NO_CHANGES: "No changes in request",
    mt5.TRADE_RETCODE_SERVER_DISABLES_AT: "Autotrading disabled by server",
    mt5.TRADE_RETCODE_CLIENT_DISABLES_AT: "Autotrading disabled by client terminal",
    mt5.TRADE_RETCODE_LOCKED: "Request locked for processing",
    mt5.TRADE_RETCODE_FROZEN: "Order or position frozen",
    mt5.TRADE_RETCODE_INVALID_FILL: "Invalid order filling type",
    mt5.TRADE_RETCODE_CONNECTION: "No connection with trade server",
    mt5.TRADE_RETCODE_ONLY_REAL: "Operation allowed only for live accounts",
    mt5.TRADE_RETCODE_LIMIT_ORDERS: "The number of open orders has reached the limit",
    mt5.TRADE_RETCODE_LIMIT_VOLUME: "The volume of orders has reached the limit",
    mt5.TRADE_RETCODE_POSITION_CLOSED: "Position with the specified identifier is closed",
}


@dataclass
class ExecutionResult:
    """Outcome of an order placement attempt."""
    success: bool
    ticket: int = 0
    deal_id: int = 0
    symbol: str = ""
    order_type: str = ""
    lot_size: float = 0.0
    price: float = 0.0
    sl: float = 0.0
    tp: float = 0.0
    retcode: int = 0
    comment: str = ""
    timestamp: Optional[datetime] = None


@dataclass
class ClosedPositionInfo:
    """Summary of a closed position extracted from trade history."""
    ticket: int
    symbol: str
    order_type: str
    lot_size: float
    open_price: float
    close_price: float
    open_time: datetime
    close_time: datetime
    profit: float
    commission: float
    swap: float
    net_pnl: float
    exit_reason: str  # "SL", "TP", "MANUAL", "KILL_SWITCH"


class TradeExecutor:
    """Executes trades and tracks open/closed positions in MetaTrader 5."""

    def __init__(self, bot_config: BotConfig = default_config):
        self.config = bot_config
        # In-memory tracking of active positions: {ticket: position_info}
        self.tracked_positions: Dict[int, Dict[str, Any]] = {}

    def get_filling_mode(self, symbol_info: Any) -> int:
        """
        Determines the optimal MT5 order filling mode based on symbol properties.
        """
        filling_flags = getattr(symbol_info, "filling_mode", 0)
        # Check IOC (2) -> FOK (1) -> RETURN (0)
        if filling_flags & 2:
            return mt5.ORDER_FILLING_IOC
        elif filling_flags & 1:
            return mt5.ORDER_FILLING_FOK
        else:
            return mt5.ORDER_FILLING_RETURN

    def execute_market_order(
        self,
        symbol: str,
        signal_type: SignalType,
        risk_result: RiskAssessmentResult,
        symbol_info: Any,
        retry_count: int = 0,
    ) -> ExecutionResult:
        """
        Submits a market order to MT5 with calculated lot, SL, and TP.
        Includes a single retry on requote, timeout, or off-quote states.
        """
        if not risk_result.approved:
            return ExecutionResult(
                success=False,
                comment=f"Order rejected before execution: {risk_result.rejection_reason}",
            )

        order_type = mt5.ORDER_TYPE_BUY if signal_type == SignalType.BUY else mt5.ORDER_TYPE_SELL
        type_str = "BUY" if signal_type == SignalType.BUY else "SELL"
        filling_mode = self.get_filling_mode(symbol_info)

        # Build order request structure
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": risk_result.lot_size,
            "type": order_type,
            "price": risk_result.entry_price,
            "sl": risk_result.sl_price,
            "tp": risk_result.tp_price,
            "deviation": self.config.slippage_points,
            "magic": self.config.magic_number,
            "comment": self.config.order_comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling_mode,
        }

        logger.info(
            f"Sending MT5 Order: {type_str} {risk_result.lot_size} lots on {symbol} @ {risk_result.entry_price} "
            f"(SL: {risk_result.sl_price}, TP: {risk_result.tp_price}, Fill: {filling_mode})"
        )

        result = mt5.order_send(request)

        # Handle None response
        if result is None:
            err_code, err_msg = mt5.last_error()
            logger.error(f"mt5.order_send returned None. Error: [{err_code}] {err_msg}")
            return ExecutionResult(
                success=False,
                retcode=err_code,
                comment=f"API Error: {err_msg}",
            )

        # Evaluate execution outcome
        if result.retcode in (mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_PLACED):
            ticket = result.order or result.deal
            executed_price = result.price if result.price > 0 else risk_result.entry_price
            logger.info(
                f"✅ ORDER EXECUTED: Ticket #{ticket} | Deal #{result.deal} | {type_str} {result.volume} lots "
                f"on {symbol} @ {executed_price} (SL: {risk_result.sl_price}, TP: {risk_result.tp_price})"
            )

            # Record in active tracking table
            self.tracked_positions[ticket] = {
                "ticket": ticket,
                "deal_id": result.deal,
                "symbol": symbol,
                "order_type": type_str,
                "lot_size": result.volume or risk_result.lot_size,
                "entry_price": executed_price,
                "sl": risk_result.sl_price,
                "tp": risk_result.tp_price,
                "open_time": datetime.now(timezone.utc),
            }

            return ExecutionResult(
                success=True,
                ticket=ticket,
                deal_id=result.deal,
                symbol=symbol,
                order_type=type_str,
                lot_size=result.volume or risk_result.lot_size,
                price=executed_price,
                sl=risk_result.sl_price,
                tp=risk_result.tp_price,
                retcode=result.retcode,
                comment=result.comment or "Order Done",
                timestamp=datetime.now(timezone.utc),
            )

        # Requote / Locked / Timeout / Price Changed Retry Handling
        retryable_codes = (
            mt5.TRADE_RETCODE_REQUOTE,
            mt5.TRADE_RETCODE_PRICE_OFF,
            mt5.TRADE_RETCODE_TIMEOUT,
            mt5.TRADE_RETCODE_PRICE_CHANGED,
            mt5.TRADE_RETCODE_TOO_MANY_REQUESTS,
            mt5.TRADE_RETCODE_LOCKED,
        )

        retcode_desc = RETCODE_DESCRIPTIONS.get(result.retcode, result.comment)

        if result.retcode in retryable_codes and retry_count < self.config.max_order_retries:
            logger.warning(
                f"Order failed with retryable status [{result.retcode}: {retcode_desc}]. "
                f"Waiting {self.config.retry_delay_sec}s for retry {retry_count + 1}/{self.config.max_order_retries}..."
            )
            time.sleep(self.config.retry_delay_sec)

            # Refresh tick price for retry
            tick = mt5.symbol_info_tick(symbol)
            if tick:
                updated_entry = tick.ask if signal_type == SignalType.BUY else tick.bid
                digits = int(symbol_info.digits) if symbol_info else 2
                point = float(symbol_info.point) if symbol_info else 0.01

                # Recompute SL/TP relative to fresh entry
                if signal_type == SignalType.BUY:
                    updated_sl = round(updated_entry - risk_result.stop_distance_price, digits)
                    updated_tp = round(updated_entry + risk_result.tp_distance_price, digits)
                else:
                    updated_sl = round(updated_entry + risk_result.stop_distance_price, digits)
                    updated_tp = round(updated_entry - risk_result.tp_distance_price, digits)

                risk_result.entry_price = updated_entry
                risk_result.sl_price = updated_sl
                risk_result.tp_price = updated_tp

            return self.execute_market_order(
                symbol=symbol,
                signal_type=signal_type,
                risk_result=risk_result,
                symbol_info=symbol_info,
                retry_count=retry_count + 1,
            )

        # Permanent failure or retries exhausted
        logger.error(
            f"❌ Order placement failed permanently: [{result.retcode}] {retcode_desc}. "
            f"Request parameters: {request}"
        )
        return ExecutionResult(
            success=False,
            ticket=0,
            symbol=symbol,
            order_type=type_str,
            retcode=result.retcode,
            comment=f"[{result.retcode}] {retcode_desc}",
            timestamp=datetime.now(timezone.utc),
        )

    def get_open_positions(self, symbol: str) -> List[Any]:
        """
        Retrieves all currently open positions for the given symbol matching our Magic Number.
        """
        positions = mt5.positions_get(symbol=symbol)
        if positions is None:
            return []

        # Filter by bot's magic number
        our_positions = [pos for pos in positions if pos.magic == self.config.magic_number]
        return our_positions

    def get_open_positions_count(self, symbol: str) -> int:
        """Returns the number of concurrent open positions for our magic number."""
        return len(self.get_open_positions(symbol))

    def detect_closed_positions(self, symbol: str) -> List[ClosedPositionInfo]:
        """
        Compares tracked open positions against current MT5 open positions.
        For positions that have closed, inspects history deals to extract realized P&L,
        exit prices, exit reasons, and cleans up tracking.
        """
        current_open_positions = self.get_open_positions(symbol)
        current_open_tickets = {pos.ticket for pos in current_open_positions}

        closed_events: List[ClosedPositionInfo] = []
        tracked_tickets = list(self.tracked_positions.keys())

        for ticket in tracked_tickets:
            if ticket not in current_open_tickets:
                # Position is no longer in open list -> it has closed!
                pos_meta = self.tracked_positions.pop(ticket)

                # Query history deals for this position
                from_time = datetime.fromtimestamp(0, tz=timezone.utc)
                to_time = datetime.now(timezone.utc)
                history_deals = mt5.history_deals_get(from_time, to_time, position=ticket)

                net_profit = 0.0
                commission = 0.0
                swap = 0.0
                close_price = pos_meta.get("entry_price", 0.0)
                close_time = datetime.now(timezone.utc)
                exit_reason = "CLOSED"

                if history_deals and len(history_deals) > 0:
                    for deal in history_deals:
                        # Deal entry OUT means closing deal
                        if deal.entry == mt5.DEAL_ENTRY_OUT:
                            close_price = deal.price
                            close_time = datetime.fromtimestamp(deal.time, tz=timezone.utc)
                            net_profit += deal.profit
                            commission += deal.commission
                            swap += deal.swap

                            # Determine exit reason (SL, TP, or normal deal)
                            if deal.reason == mt5.DEAL_REASON_SL:
                                exit_reason = "SL"
                            elif deal.reason == mt5.DEAL_REASON_TP:
                                exit_reason = "TP"
                            else:
                                exit_reason = "MANUAL_OR_CALL"
                        elif deal.entry == mt5.DEAL_ENTRY_IN:
                            commission += deal.commission

                total_net_pnl = net_profit + commission + swap

                closed_info = ClosedPositionInfo(
                    ticket=ticket,
                    symbol=pos_meta.get("symbol", symbol),
                    order_type=pos_meta.get("order_type", "UNKNOWN"),
                    lot_size=pos_meta.get("lot_size", 0.0),
                    open_price=pos_meta.get("entry_price", 0.0),
                    close_price=close_price,
                    open_time=pos_meta.get("open_time", close_time),
                    close_time=close_time,
                    profit=net_profit,
                    commission=commission,
                    swap=swap,
                    net_pnl=total_net_pnl,
                    exit_reason=exit_reason,
                )

                logger.info(
                    f"🔔 Position #{ticket} CLOSED | Type: {closed_info.order_type} | "
                    f"Entry: {closed_info.open_price} -> Exit: {closed_info.close_price} | "
                    f"Net PnL: ${closed_info.net_pnl:.2f} | Reason: {closed_info.exit_reason}"
                )
                closed_events.append(closed_info)

        return closed_events
