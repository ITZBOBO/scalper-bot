"""
executor.py - Order Execution and Grouped Position Lifecycle Tracking for MT5.

Handles market order routing, filling mode negotiation, single-retry requote resilience,
grouped multi-position trade execution, orphan position prevention, group recovery,
and group-level realized P&L accounting from history deals.
"""

from dataclasses import dataclass, field
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
    """Outcome of an individual order placement attempt."""
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
class TradeGroup:
    """Represents a logical trade group comprising one or more sub-positions."""
    group_id: str
    symbol: str
    order_type: str                            # "BUY" or "SELL"
    tickets: List[int] = field(default_factory=list)
    position_lots: Dict[int, float] = field(default_factory=dict)     # ticket -> lots
    entry_prices: Dict[int, float] = field(default_factory=dict)      # ticket -> price
    sl_price: float = 0.0
    tp_price: float = 0.0
    target_risk: float = 0.0                   # Intended group loss budget from config
    profit_target: float = 0.0                 # Intended group profit target from config
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    status: str = "OPEN"                       # "OPEN", "PARTIAL", "CLOSED"

    @property
    def total_volume(self) -> float:
        return round(sum(self.position_lots.values()), 4)

    @property
    def positions_count(self) -> int:
        return len(self.tickets)

    @property
    def weighted_avg_entry(self) -> float:
        total_vol = self.total_volume
        if total_vol <= 0 or not self.entry_prices:
            return 0.0
        return sum(self.entry_prices.get(t, 0.0) * self.position_lots.get(t, 0.0) for t in self.tickets) / total_vol


@dataclass
class GroupExecutionResult:
    """Outcome of executing a grouped multi-position order."""
    success: bool
    group: Optional[TradeGroup] = None
    results: List[ExecutionResult] = field(default_factory=list)
    rejection_reason: Optional[str] = None


@dataclass
class ClosedPositionInfo:
    """Summary of an individual closed position."""
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
    exit_reason: str


@dataclass
class ClosedGroupInfo:
    """Aggregated summary of a closed trade group."""
    group_id: str
    symbol: str
    order_type: str
    positions_closed: int
    total_positions: int
    total_volume: float
    avg_open_price: float
    avg_close_price: float
    gross_profit: float
    commission: float
    swap: float
    net_pnl: float
    exit_reason: str
    tickets: List[int]
    open_time: datetime
    close_time: datetime


class TradeExecutor:
    """Executes trades and tracks open/closed trade groups in MetaTrader 5."""

    def __init__(self, bot_config: BotConfig = default_config):
        self.config = bot_config
        self.active_groups: Dict[str, TradeGroup] = {}
        self.ticket_to_group: Dict[int, str] = {}
        self.tracked_positions: Dict[int, Dict[str, Any]] = {}
        self._group_counter: int = 0

    def get_filling_mode(self, symbol_info: Any) -> int:
        """Determines the optimal MT5 order filling mode based on symbol properties."""
        filling_flags = getattr(symbol_info, "filling_mode", 0)
        if filling_flags & 2:
            return mt5.ORDER_FILLING_IOC
        elif filling_flags & 1:
            return mt5.ORDER_FILLING_FOK
        else:
            return mt5.ORDER_FILLING_RETURN

    def _generate_group_id(self) -> str:
        """Generates a unique group ID (e.g. GRP-20260901-174711-001)."""
        now = datetime.now(timezone.utc)
        self._group_counter += 1
        return f"GRP-{now.strftime('%Y%m%d-%H%M%S')}-{self._group_counter:03d}"

    def execute_market_order(
        self,
        symbol: str,
        signal_type: SignalType,
        risk_result: RiskAssessmentResult,
        symbol_info: Any,
        retry_count: int = 0,
    ) -> ExecutionResult:
        """
        Submits a single market order to MT5.
        Included for backward compatibility and sub-position order placement.
        """
        if not risk_result.approved:
            return ExecutionResult(
                success=False,
                comment=f"Order rejected before execution: {risk_result.rejection_reason}",
            )

        order_type = mt5.ORDER_TYPE_BUY if signal_type == SignalType.BUY else mt5.ORDER_TYPE_SELL
        type_str = "BUY" if signal_type == SignalType.BUY else "SELL"
        filling_mode = self.get_filling_mode(symbol_info)
        lot = risk_result.lot_size

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
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
            f"Sending MT5 Order: {type_str} {lot} lots on {symbol} @ {risk_result.entry_price} "
            f"(SL: {risk_result.sl_price}, TP: {risk_result.tp_price}, Fill: {filling_mode})"
        )

        result = mt5.order_send(request)

        if result is None:
            err_code, err_msg = mt5.last_error()
            logger.error(f"mt5.order_send returned None. Error: [{err_code}] {err_msg}")
            return ExecutionResult(
                success=False,
                retcode=err_code,
                comment=f"API Error: {err_msg}",
            )

        if result.retcode in (mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_PLACED):
            ticket = result.order or result.deal
            executed_price = result.price if result.price > 0 else risk_result.entry_price
            logger.info(
                f"✅ ORDER EXECUTED: Ticket #{ticket} | Deal #{result.deal} | {type_str} {result.volume} lots "
                f"on {symbol} @ {executed_price} (SL: {risk_result.sl_price}, TP: {risk_result.tp_price})"
            )

            self.tracked_positions[ticket] = {
                "ticket": ticket,
                "deal_id": result.deal,
                "symbol": symbol,
                "order_type": type_str,
                "lot_size": result.volume or lot,
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
                lot_size=result.volume or lot,
                price=executed_price,
                sl=risk_result.sl_price,
                tp=risk_result.tp_price,
                retcode=result.retcode,
                comment=result.comment or "Order Done",
                timestamp=datetime.now(timezone.utc),
            )

        # Retry Handling for Requote/Timeout
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

            tick = mt5.symbol_info_tick(symbol)
            if tick:
                updated_entry = tick.ask if signal_type == SignalType.BUY else tick.bid
                digits = int(symbol_info.digits) if symbol_info else 2
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

    def execute_trade_group(
        self,
        symbol: str,
        signal_type: SignalType,
        risk_result: RiskAssessmentResult,
        symbol_info: Any,
    ) -> GroupExecutionResult:
        """
        Executes a single strategy signal as a group of multiple smaller positions.
        All sub-positions share identical direction, signal time, SL, and TP.
        Enforces group atomicity and strict risk invariants.
        """
        if not risk_result.approved:
            return GroupExecutionResult(
                success=False,
                rejection_reason=f"Group execution rejected: {risk_result.rejection_reason}",
            )

        group_id = self._generate_group_id()
        type_str = "BUY" if signal_type == SignalType.BUY else "SELL"
        order_type_mt5 = mt5.ORDER_TYPE_BUY if signal_type == SignalType.BUY else mt5.ORDER_TYPE_SELL
        filling_mode = self.get_filling_mode(symbol_info)

        sub_lots = risk_result.position_lots if risk_result.position_lots else [risk_result.lot_size]
        num_pos = len(sub_lots)

        logger.info("------------------------------------------------------------------")
        logger.info(f"[GROUP] Creating {group_id}")
        logger.info(f"[GROUP] Total risk target: ${risk_result.risk_amount_currency:.2f}")
        logger.info(f"[GROUP] Positions: {num_pos}")
        logger.info(f"[GROUP] Total calculated volume: {sum(sub_lots):.2f} lots")
        for i, lot in enumerate(sub_lots, 1):
            logger.info(f"[GROUP] Position {i}: {lot:.2f} lots")
        logger.info(f"[GROUP] Combined theoretical SL risk: ~${risk_result.theoretical_group_risk:.2f}")
        logger.info(f"[GROUP] SL: {risk_result.sl_price} | TP: {risk_result.tp_price}")
        logger.info("------------------------------------------------------------------")

        group = TradeGroup(
            group_id=group_id,
            symbol=symbol,
            order_type=type_str,
            sl_price=risk_result.sl_price,
            tp_price=risk_result.tp_price,
            target_risk=risk_result.risk_amount_currency,
            profit_target=getattr(risk_result, "group_profit_target", 2.00),
        )

        exec_results: List[ExecutionResult] = []

        # Execute each sub-position order
        for idx, lot in enumerate(sub_lots, 1):
            comment = f"G:{group_id[-8:]}:{idx}/{num_pos}"

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": lot,
                "type": order_type_mt5,
                "price": risk_result.entry_price,
                "sl": risk_result.sl_price,
                "tp": risk_result.tp_price,
                "deviation": self.config.slippage_points,
                "magic": self.config.magic_number,
                "comment": comment,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": filling_mode,
            }

            logger.info(
                f"[GROUP ORDER {idx}/{num_pos}] {type_str} {lot} lots on {symbol} @ {risk_result.entry_price} "
                f"(Comment: {comment})"
            )

            result = mt5.order_send(request)

            # Retry once if requote/timeout
            if result and result.retcode in (mt5.TRADE_RETCODE_REQUOTE, mt5.TRADE_RETCODE_TIMEOUT, mt5.TRADE_RETCODE_PRICE_CHANGED):
                time.sleep(self.config.retry_delay_sec)
                tick = mt5.symbol_info_tick(symbol)
                if tick:
                    request["price"] = tick.ask if signal_type == SignalType.BUY else tick.bid
                result = mt5.order_send(request)

            if result and result.retcode in (mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_PLACED):
                ticket = result.order or result.deal
                executed_price = result.price if result.price > 0 else risk_result.entry_price
                
                group.tickets.append(ticket)
                group.position_lots[ticket] = result.volume or lot
                group.entry_prices[ticket] = executed_price
                self.ticket_to_group[ticket] = group_id

                self.tracked_positions[ticket] = {
                    "ticket": ticket,
                    "group_id": group_id,
                    "deal_id": result.deal,
                    "symbol": symbol,
                    "order_type": type_str,
                    "lot_size": result.volume or lot,
                    "entry_price": executed_price,
                    "sl": risk_result.sl_price,
                    "tp": risk_result.tp_price,
                    "open_time": datetime.now(timezone.utc),
                }

                res = ExecutionResult(
                    success=True,
                    ticket=ticket,
                    deal_id=result.deal,
                    symbol=symbol,
                    order_type=type_str,
                    lot_size=result.volume or lot,
                    price=executed_price,
                    sl=risk_result.sl_price,
                    tp=risk_result.tp_price,
                    retcode=result.retcode,
                    comment=comment,
                    timestamp=datetime.now(timezone.utc),
                )
                exec_results.append(res)
                logger.info(f"✅ Position {idx}/{num_pos} Filled: Ticket #{ticket} @ {executed_price}")
            else:
                retcode = result.retcode if result else -1
                desc = RETCODE_DESCRIPTIONS.get(retcode, "Unknown Error")
                logger.error(f"[GROUP ERROR] Position {idx}/{num_pos} failed to open: [{retcode}] {desc}")
                exec_results.append(
                    ExecutionResult(
                        success=False,
                        symbol=symbol,
                        order_type=type_str,
                        retcode=retcode,
                        comment=f"[{retcode}] {desc}",
                    )
                )

        if len(group.tickets) > 0:
            self.active_groups[group_id] = group
            logger.info(
                f"🎉 [GROUP ACTIVE] {group_id} successfully opened with {len(group.tickets)}/{num_pos} "
                f"position(s) totaling {group.total_volume:.2f} lots."
            )
            return GroupExecutionResult(success=True, group=group, results=exec_results)
        else:
            logger.error(f"❌ [GROUP FAILED] {group_id} failed completely. 0 positions opened.")
            return GroupExecutionResult(
                success=False,
                group=None,
                results=exec_results,
                rejection_reason="All sub-orders in group failed to execute",
            )

    def get_open_positions(self, symbol: str) -> List[Any]:
        """Retrieves all currently open positions matching our Magic Number."""
        positions = mt5.positions_get(symbol=symbol)
        if positions is None:
            return []
        return [pos for pos in positions if pos.magic == self.config.magic_number]

    def get_active_groups_count(self, symbol: Optional[str] = None) -> int:
        """Returns the number of currently active trade groups."""
        if symbol is None:
            return len(self.active_groups)
        return sum(1 for g in self.active_groups.values() if g.symbol == symbol)

    def get_open_positions_count(self, symbol: str) -> int:
        """Returns the total number of open positions for our magic number."""
        return len(self.get_open_positions(symbol))

    def close_position_by_ticket(self, ticket: int, symbol: str) -> bool:
        """Sends a market closing order for an individual ticket."""
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            return False

        pos = positions[0]
        close_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            return False
        price = tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask

        sym_info = mt5.symbol_info(symbol)
        filling_mode = self.get_filling_mode(sym_info)

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": pos.volume,
            "type": close_type,
            "position": ticket,
            "price": price,
            "deviation": self.config.slippage_points,
            "magic": self.config.magic_number,
            "comment": f"Close #{ticket}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling_mode,
        }

        result = mt5.order_send(request)
        if result and result.retcode in (mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_PLACED):
            logger.info(f"Closed orphan/sync position #{ticket} on {symbol}")
            return True
        return False

    def close_group_positions(self, group_id: str, symbol: str) -> None:
        """Closes all open positions belonging to a specific group."""
        group = self.active_groups.get(group_id)
        if not group:
            return

        for ticket in list(group.tickets):
            self.close_position_by_ticket(ticket, symbol)

    def detect_closed_groups(self, symbol: str) -> List[ClosedGroupInfo]:
        """
        Monitors active groups and detects when positions close.
        Enforces GROUP ATOMICITY: if one position closes prematurely, remaining
        positions in the group are immediately closed so no orphaned exposure remains.
        Aggregates deal history across all sub-positions into a single ClosedGroupInfo.
        """
        current_open_positions = self.get_open_positions(symbol)
        current_open_tickets = {pos.ticket for pos in current_open_positions}

        closed_group_events: List[ClosedGroupInfo] = []
        active_group_ids = list(self.active_groups.keys())

        for gid in active_group_ids:
            group = self.active_groups.get(gid)
            if not group:
                continue

            open_sub_tickets = [t for t in group.tickets if t in current_open_tickets]
            closed_sub_tickets = [t for t in group.tickets if t not in current_open_tickets]

            # 1. REAL-TIME AUTHORITATIVE AGGREGATE GROUP P&L EVALUATION ($2.00 Total Target Across Group)
            open_pos_map = {pos.ticket: pos for pos in current_open_positions if pos.ticket in open_sub_tickets}
            if len(open_pos_map) > 0 and getattr(group, "profit_target", 0.0) > 0:
                group_floating_profit = sum(getattr(pos, 'profit', 0.0) for pos in open_pos_map.values())
                group_floating_swap = sum(getattr(pos, 'swap', 0.0) for pos in open_pos_map.values())
                group_floating_commission = sum(getattr(pos, 'commission', 0.0) for pos in open_pos_map.values())
                group_floating_pnl = group_floating_profit + group_floating_swap + group_floating_commission

                if group_floating_pnl >= group.profit_target:
                    logger.info(
                        f"🎯 [AUTHORITATIVE GROUP PROFIT TARGET REACHED] {gid} combined floating PnL ${group_floating_pnl:+.2f} >= target ${group.profit_target:.2f}. "
                        f"Closing all {len(open_pos_map)} position(s) to secure total group profit!"
                    )
                    for t in list(open_pos_map.keys()):
                        self.close_position_by_ticket(t, symbol)
                    # Re-query open positions
                    current_open_positions = self.get_open_positions(symbol)
                    current_open_tickets = {pos.ticket for pos in current_open_positions}
                    open_sub_tickets = [t for t in group.tickets if t in current_open_tickets]
                    closed_sub_tickets = [t for t in group.tickets if t not in current_open_tickets]

            # 2. ORPHAN PREVENTION & UNEXPECTED DISAPPEARANCE RECONCILIATION
            if len(closed_sub_tickets) > 0 and len(open_sub_tickets) > 0:
                logger.warning(
                    f"⚠️ [GROUP RECONCILIATION / ORPHAN PREVENTION] Position in {gid} closed/disappeared ({len(closed_sub_tickets)}/{len(group.tickets)} closed). "
                    f"Executing market close on remaining {len(open_sub_tickets)} position(s) immediately to maintain group atomicity."
                )
                for t in open_sub_tickets:
                    self.close_position_by_ticket(t, symbol)
                # Re-query open positions
                current_open_positions = self.get_open_positions(symbol)
                current_open_tickets = {pos.ticket for pos in current_open_positions}
                open_sub_tickets = [t for t in group.tickets if t in current_open_tickets]

            # 3. COMPLETE GROUP EXIT: All positions closed
            if len(open_sub_tickets) == 0 and len(group.tickets) > 0:
                self.active_groups.pop(gid, None)

                # Query history deals for all tickets in group
                from_time = datetime.fromtimestamp(0, tz=timezone.utc)
                to_time = datetime.now(timezone.utc)

                gross_profit = 0.0
                commission = 0.0
                swap = 0.0
                close_prices = []
                exit_reasons = []
                last_close_time = datetime.now(timezone.utc)

                for ticket in group.tickets:
                    self.ticket_to_group.pop(ticket, None)
                    self.tracked_positions.pop(ticket, None)

                    deals = mt5.history_deals_get(from_time, to_time, position=ticket)
                    if deals:
                        for deal in deals:
                            if deal.entry == mt5.DEAL_ENTRY_OUT:
                                gross_profit += deal.profit
                                commission += deal.commission
                                swap += deal.swap
                                close_prices.append(deal.price)
                                last_close_time = datetime.fromtimestamp(deal.time, tz=timezone.utc)
                                if deal.reason == mt5.DEAL_REASON_SL:
                                    exit_reasons.append("SL")
                                elif deal.reason == mt5.DEAL_REASON_TP:
                                    exit_reasons.append("TP")
                                else:
                                    exit_reasons.append("MANUAL_OR_CALL")
                            elif deal.entry == mt5.DEAL_ENTRY_IN:
                                commission += deal.commission

                total_net_pnl = gross_profit + commission + swap
                avg_open = sum(group.entry_prices.values()) / len(group.entry_prices) if group.entry_prices else 0.0
                avg_close = sum(close_prices) / len(close_prices) if close_prices else avg_open
                primary_reason = exit_reasons[0] if exit_reasons else "CLOSED"

                closed_group_info = ClosedGroupInfo(
                    group_id=gid,
                    symbol=group.symbol,
                    order_type=group.order_type,
                    positions_closed=len(group.tickets),
                    total_positions=len(group.tickets),
                    total_volume=group.total_volume,
                    avg_open_price=avg_open,
                    avg_close_price=avg_close,
                    gross_profit=gross_profit,
                    commission=commission,
                    swap=swap,
                    net_pnl=total_net_pnl,
                    exit_reason=primary_reason,
                    tickets=list(group.tickets),
                    open_time=group.created_at,
                    close_time=last_close_time,
                )

                logger.info("------------------------------------------------------------------")
                logger.info(f"[GROUP CLOSED] {gid}")
                logger.info(f"[GROUP RESULT] PnL: ${total_net_pnl:+,.2f} | Reason: {primary_reason}")
                logger.info(f"[GROUP RESULT] Positions closed: {len(group.tickets)}/{len(group.tickets)}")
                logger.info(f"[GROUP RESULT] Volume: {group.total_volume:.2f} lots | Entry Avg: {avg_open:.3f} -> Exit Avg: {avg_close:.3f}")
                logger.info("------------------------------------------------------------------")

                closed_group_events.append(closed_group_info)

        return closed_group_events

    def detect_closed_positions(self, symbol: str) -> List[ClosedPositionInfo]:
        """
        Maintains backward compatibility by returning individual closed positions.
        """
        closed_groups = self.detect_closed_groups(symbol)
        legacy_closed: List[ClosedPositionInfo] = []
        for g in closed_groups:
            legacy_closed.append(
                ClosedPositionInfo(
                    ticket=g.tickets[0] if g.tickets else 0,
                    symbol=g.symbol,
                    order_type=g.order_type,
                    lot_size=g.total_volume,
                    open_price=g.avg_open_price,
                    close_price=g.avg_close_price,
                    open_time=g.open_time,
                    close_time=g.close_time,
                    profit=g.gross_profit,
                    commission=g.commission,
                    swap=g.swap,
                    net_pnl=g.net_pnl,
                    exit_reason=g.exit_reason,
                )
            )

        # Handle standalone tracked positions not mapped to a group
        current_open_positions = self.get_open_positions(symbol)
        current_open_tickets = {pos.ticket for pos in current_open_positions}
        for ticket in list(self.tracked_positions.keys()):
            if ticket not in self.ticket_to_group and ticket not in current_open_tickets:
                pos_meta = self.tracked_positions.pop(ticket)
                from_time = datetime.fromtimestamp(0, tz=timezone.utc)
                to_time = datetime.now(timezone.utc)
                history_deals = mt5.history_deals_get(from_time, to_time, position=ticket)
                net_profit = 0.0
                commission = 0.0
                swap = 0.0
                close_price = pos_meta.get("entry_price", 0.0)
                close_time = datetime.now(timezone.utc)
                exit_reason = "CLOSED"
                if history_deals:
                    for deal in history_deals:
                        if deal.entry == mt5.DEAL_ENTRY_OUT:
                            close_price = deal.price
                            close_time = datetime.fromtimestamp(deal.time, tz=timezone.utc)
                            net_profit += deal.profit
                            commission += deal.commission
                            swap += deal.swap
                            if deal.reason == mt5.DEAL_REASON_SL:
                                exit_reason = "SL"
                            elif deal.reason == mt5.DEAL_REASON_TP:
                                exit_reason = "TP"
                        elif deal.entry == mt5.DEAL_ENTRY_IN:
                            commission += deal.commission

                total_net_pnl = net_profit + commission + swap
                legacy_closed.append(
                    ClosedPositionInfo(
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
                )
        return legacy_closed

    def recover_active_groups(self, symbol: str) -> None:
        """
        On startup or restart, inspects current open positions with our Magic Number
        and reconstructs active TradeGroup tracking objects.
        """
        open_positions = self.get_open_positions(symbol)
        if not open_positions:
            logger.info("No open positions found in MT5 on startup.")
            return

        logger.info(f"🔄 [RECOVERY] Found {len(open_positions)} open position(s) in MT5. Reconstructing trade groups...")

        # Group positions by comment tag or fallback to a single recovered group
        groups_by_tag: Dict[str, List[Any]] = {}
        for pos in open_positions:
            comment = getattr(pos, "comment", "")
            if comment and "G:" in comment:
                tag = comment.split(":")[1] if len(comment.split(":")) > 1 else "RECOVERED"
            else:
                tag = "RECOVERED"
            groups_by_tag.setdefault(tag, []).append(pos)

        for tag, pos_list in groups_by_tag.items():
            gid = f"GRP-REC-{tag}-{datetime.now(timezone.utc).strftime('%H%M%S')}"
            first_pos = pos_list[0]
            type_str = "BUY" if first_pos.type == mt5.ORDER_TYPE_BUY else "SELL"

            rec_group = TradeGroup(
                group_id=gid,
                symbol=symbol,
                order_type=type_str,
                sl_price=first_pos.sl,
                tp_price=first_pos.tp,
                target_risk=self.config.fixed_risk_amount or 1.0,
            )

            for pos in pos_list:
                rec_group.tickets.append(pos.ticket)
                rec_group.position_lots[pos.ticket] = pos.volume
                rec_group.entry_prices[pos.ticket] = pos.price_open
                self.ticket_to_group[pos.ticket] = gid
                self.tracked_positions[pos.ticket] = {
                    "ticket": pos.ticket,
                    "group_id": gid,
                    "symbol": symbol,
                    "order_type": type_str,
                    "lot_size": pos.volume,
                    "entry_price": pos.price_open,
                    "sl": pos.sl,
                    "tp": pos.tp,
                    "open_time": datetime.fromtimestamp(pos.time, tz=timezone.utc),
                }

            self.active_groups[gid] = rec_group
            logger.info(
                f"✅ [RECOVERY] Reconstructed active group '{gid}' with {len(pos_list)} position(s) "
                f"totalling {rec_group.total_volume:.2f} lots (Tickets: {rec_group.tickets})."
            )

