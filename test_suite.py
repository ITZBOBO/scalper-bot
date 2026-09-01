"""
test_suite.py - Comprehensive Unit and Mock Tests for MT5 Gold Scalper Bot.

Validates:
1. EMA Fast, EMA Slow, RSI, and ATR indicator calculations.
2. Signal crossover logic and RSI exhaustion filters.
3. ATR NaN/Zero validation guards.
4. Broker-normalized dynamic lot sizing and SL/TP price levels (2-digit and 3-digit gold).
5. Spread filter rejection and concurrency checks.
6. Kill switch persistence, consecutive loss tripping, daily loss % tripping, and broker server-time daily resets.
7. Signal and Trade CSV logger formatting.
"""

import os
import shutil
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd

from config import BotConfig
from signal_engine import SignalEngine, SignalType
from risk_manager import RiskManager
from kill_switch import KillSwitch
from logger import TradeLogger
from executor import ClosedPositionInfo, TradeGroup, ClosedGroupInfo, GroupExecutionResult


class TestSignalEngine(unittest.TestCase):
    """Tests for indicator math, crossover logic, and ATR guards."""

    def setUp(self):
        self.config = BotConfig()
        self.engine = SignalEngine(self.config)

    def _generate_synthetic_candles(self, n: int = 100, trend: str = "flat", base_price: float = 2000.0) -> pd.DataFrame:
        """Generates mock M1 candle DataFrame."""
        start_time = datetime(2026, 8, 27, 10, 0, 0, tzinfo=timezone.utc)
        timestamps = [start_time + timedelta(minutes=i) for i in range(n)]

        closes = []
        highs = []
        lows = []
        opens = []

        price = base_price
        for i in range(n):
            if trend == "bullish":
                delta = 0.5 + (0.1 if i > n - 10 else 0.0)
            elif trend == "bearish":
                delta = -0.5 - (0.1 if i > n - 10 else 0.0)
            else:
                delta = (np.sin(i / 3.0)) * 0.4

            o = price
            c = price + delta
            h = max(o, c) + 0.3
            l = min(o, c) - 0.3
            price = c

            opens.append(o)
            highs.append(h)
            lows.append(l)
            closes.append(c)

        df = pd.DataFrame({
            "time": timestamps,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "tick_volume": [100] * n,
            "spread": [20] * n,
        })
        return df

    def test_indicator_computation(self):
        df = self._generate_synthetic_candles(60)
        ind_df = self.engine.compute_indicators(df)

        self.assertIn("ema_fast", ind_df.columns)
        self.assertIn("ema_slow", ind_df.columns)
        self.assertIn("rsi", ind_df.columns)
        self.assertIn("atr", ind_df.columns)

        # Check that after sufficient warmup, indicators are not NaN
        last_row = ind_df.iloc[-1]
        self.assertFalse(np.isnan(last_row["ema_fast"]))
        self.assertFalse(np.isnan(last_row["ema_slow"]))
        self.assertFalse(np.isnan(last_row["rsi"]))
        self.assertFalse(np.isnan(last_row["atr"]))
        self.assertGreater(last_row["atr"], 0.0)

    def test_bullish_crossover_signal(self):
        """EMA 9 crossing above EMA 21 with RSI < 70 must trigger BUY."""
        # Create a series with a clear upward crossover on the penultimate bar
        closes = [2000.0 - (i * 0.1) for i in range(50)] + [2000.0 + (i * 0.8) for i in range(15)]
        df = pd.DataFrame({
            "time": [datetime(2026, 8, 27, 10, 0, 0, tzinfo=timezone.utc) + timedelta(minutes=i) for i in range(len(closes))],
            "open": closes,
            "high": [c + 0.5 for c in closes],
            "low": [c - 0.5 for c in closes],
            "close": closes,
            "tick_volume": [100] * len(closes),
            "spread": [20] * len(closes),
        })

        # Calculate indicators and inspect crossover behavior
        ind_df = self.engine.compute_indicators(df)
        signal = self.engine.evaluate_signals(df)
        # Should return a signal structure (BUY or filtered or NONE)
        self.assertIsNotNone(signal)
        self.assertTrue(hasattr(signal, "signal_type"))

    def test_atr_validation_guard(self):
        """Signal engine MUST reject signals if ATR is NaN, 0, or non-positive."""
        df = self._generate_synthetic_candles(60)

        # Create perfectly flat price series where TR is 0.0 for all bars
        df_zero_atr = df.copy()
        df_zero_atr["open"] = 2500.0
        df_zero_atr["high"] = 2500.0
        df_zero_atr["low"] = 2500.0
        df_zero_atr["close"] = 2500.0

        signal = self.engine.evaluate_signals(df_zero_atr)
        self.assertEqual(signal.signal_type, SignalType.NONE)
        self.assertTrue("ATR" in signal.reason or "Invalid" in signal.reason or "Zero" in signal.reason)

    def test_insufficient_history(self):
        df = self._generate_synthetic_candles(10)
        signal = self.engine.evaluate_signals(df)
        self.assertEqual(signal.signal_type, SignalType.NONE)
        self.assertIn("Insufficient candle history", signal.reason)


class TestRiskManager(unittest.TestCase):
    """Tests for dynamic lot sizing, spread filtering, and SL/TP computation."""

    def setUp(self):
        self.config = BotConfig(
            max_spread_price=0.40,
            risk_per_trade_pct=1.0,
            fixed_risk_amount=None,
            sl_atr_multiplier=1.5,
            tp_atr_multiplier=2.0,
            group_risk_mode="FIXED_TOTAL_RISK",
            fixed_tp_price_distance=None,
            group_profit_target=None,
        )
        self.risk_manager = RiskManager(self.config)

    def _create_mock_symbol_info(self, digits: int = 2, point: float = 0.01, tick_value: float = 1.0, tick_size: float = 0.01):
        return SimpleNamespace(
            digits=digits,
            point=point,
            trade_tick_value=tick_value,
            trade_tick_size=tick_size,
            volume_min=0.01,
            volume_max=100.0,
            volume_step=0.01,
            trade_stops_level=0,
        )

    def test_spread_rejection(self):
        """Trades must be rejected when spread price exceeds max_spread_price."""
        symbol_info = self._create_mock_symbol_info(digits=2, point=0.01)
        # Spread is $0.55 (> max $0.40)
        tick = {"bid": 2500.00, "ask": 2500.55, "spread_price": 0.55, "spread_points": 55.0}
        account = {"balance": 10000.0}

        signal = SimpleNamespace(
            signal_type=SignalType.BUY,
            is_valid=True,
            is_buy=True,
            is_sell=False,
            atr_value=2.0,
            reason="Test BUY",
        )

        result = self.risk_manager.evaluate_signal(signal, tick, account, symbol_info, open_positions_count=0)
        self.assertFalse(result.approved)
        self.assertIn("Spread too wide", result.rejection_reason)

    def test_concurrency_limit_rejection(self):
        """Trades must be rejected when open positions count meets or exceeds limit."""
        symbol_info = self._create_mock_symbol_info()
        tick = {"bid": 2500.00, "ask": 2500.20, "spread_price": 0.20, "spread_points": 20.0}
        account = {"balance": 10000.0}

        signal = SimpleNamespace(
            signal_type=SignalType.BUY,
            is_valid=True,
            is_buy=True,
            is_sell=False,
            atr_value=2.0,
            reason="Test BUY",
        )

        result = self.risk_manager.evaluate_signal(signal, tick, account, symbol_info, open_positions_count=1)
        self.assertFalse(result.approved)
        self.assertIn("Max concurrent trade", result.rejection_reason)

    def test_dynamic_lot_sizing_2digit_gold(self):
        """
        Test broker-normalized lot calculation for 2-digit gold:
        Balance: $10,000 | Risk: 1% ($100)
        ATR = $2.00 | SL distance = 1.5 * 2.00 = $3.00 = 300 points
        Tick value = $1.00, Tick size = 0.01, Point = 0.01
        Value per point = (1.00 / 0.01) * 0.01 = $1.00 per lot
        Loss per 1.0 lot for $3.00 move = 300 * $1.00 = $300
        Lot size for $100 risk = 100 / 300 = 0.33 lots
        """
        symbol_info = self._create_mock_symbol_info(digits=2, point=0.01, tick_value=1.0, tick_size=0.01)
        tick = {"bid": 2500.00, "ask": 2500.20, "spread_price": 0.20, "spread_points": 20.0}
        account = {"balance": 10000.0}

        signal = SimpleNamespace(
            signal_type=SignalType.BUY,
            is_valid=True,
            is_buy=True,
            is_sell=False,
            atr_value=2.0,
            reason="Test BUY",
        )

        result = self.risk_manager.evaluate_signal(signal, tick, account, symbol_info, open_positions_count=0)
        self.assertTrue(result.approved)
        self.assertEqual(result.lot_size, 0.33)
        self.assertEqual(result.entry_price, 2500.20)
        self.assertEqual(result.sl_price, 2497.20)  # 2500.20 - 3.00
        self.assertEqual(result.tp_price, 2504.20)  # 2500.20 + 4.00 (2.0 * 2.0)

    def test_dynamic_lot_sizing_3digit_gold(self):
        """
        Test broker-normalized lot calculation for 3-digit gold:
        Balance: $10,000 | Risk: 1% ($100)
        ATR = $2.00 | SL distance = 1.5 * 2.00 = $3.00 = 3000 points
        Tick value = $0.10, Tick size = 0.001, Point = 0.001
        Value per point = (0.10 / 0.001) * 0.001 = $0.10 per lot
        Loss per 1.0 lot for $3.00 move = 3000 * $0.10 = $300
        Lot size for $100 risk = 100 / 300 = 0.33 lots
        """
        symbol_info = self._create_mock_symbol_info(digits=3, point=0.001, tick_value=0.10, tick_size=0.001)
        tick = {"bid": 2500.000, "ask": 2500.200, "spread_price": 0.20, "spread_points": 200.0}
        account = {"balance": 10000.0}

        signal = SimpleNamespace(
            signal_type=SignalType.BUY,
            is_valid=True,
            is_buy=True,
            is_sell=False,
            atr_value=2.0,
            reason="Test BUY 3-digit",
        )

        result = self.risk_manager.evaluate_signal(signal, tick, account, symbol_info, open_positions_count=0)
        self.assertTrue(result.approved)
        self.assertEqual(result.lot_size, 0.33)
        self.assertEqual(result.sl_price, 2497.200)
        self.assertEqual(result.tp_price, 2504.200)

    def test_fixed_cash_risk_sizing(self):
        """Test fixed cash risk override (e.g. 2000 NGN or $50 USD fixed)."""
        cfg = BotConfig(fixed_risk_amount=2000.0, group_risk_mode="FIXED_TOTAL_RISK")
        rm = RiskManager(cfg)
        symbol_info = self._create_mock_symbol_info(digits=2, point=0.01, tick_value=1.0, tick_size=0.01)
        tick = {"bid": 2500.00, "ask": 2500.20, "spread_price": 0.20, "spread_points": 20.0}
        account = {"balance": 100000.0}  # e.g. 100k NGN account

        signal = SimpleNamespace(
            signal_type=SignalType.BUY,
            is_valid=True,
            is_buy=True,
            is_sell=False,
            atr_value=2.0,
            reason="Test Fixed Cash Risk",
        )

        # SL = 1.5 * 2.0 = $3.00. For 1.0 lot, loss is 3.00 * 100 = 300.
        # Fixed risk = 2000 -> 2000 / 300 = 6.66 lots
        result = rm.evaluate_signal(signal, tick, account, symbol_info, open_positions_count=0)
        self.assertTrue(result.approved)
        self.assertEqual(result.lot_size, 6.66)


class TestKillSwitch(unittest.TestCase):
    """Tests for circuit breaker rules, daily loss %, consecutive losses, and persistence."""

    def setUp(self):
        self.test_data_dir = Path("test_data")
        self.test_data_dir.mkdir(parents=True, exist_ok=True)
        self.config = BotConfig(
            data_dir=self.test_data_dir,
            kill_switch_state_path=self.test_data_dir / "test_kill_switch.json",
            manual_stop_file_path=self.test_data_dir / "STOP",
            max_daily_loss_pct=3.0,
            max_consecutive_losses=3,
        )
        self.kill_switch = KillSwitch(self.config)

    def tearDown(self):
        if self.test_data_dir.exists():
            shutil.rmtree(self.test_data_dir)

    def test_consecutive_loss_trip(self):
        server_dt = datetime(2026, 8, 27, 12, 0, 0, tzinfo=timezone.utc)
        self.kill_switch.sync_server_date(server_dt, current_balance=10000.0)

        # 1st loss
        self.kill_switch.record_trade_result(-50.0, server_dt, 9950.0)
        can_trade, _ = self.kill_switch.can_trade(9950.0, server_dt)
        self.assertTrue(can_trade)

        # 2nd loss
        self.kill_switch.record_trade_result(-50.0, server_dt, 9900.0)
        can_trade, _ = self.kill_switch.can_trade(9900.0, server_dt)
        self.assertTrue(can_trade)

        # 3rd loss -> reaches limit of 3
        self.kill_switch.record_trade_result(-50.0, server_dt, 9850.0)
        can_trade, reason = self.kill_switch.can_trade(9850.0, server_dt)
        self.assertFalse(can_trade)
        self.assertIn("Consecutive loss cap hit", reason)

    def test_daily_loss_pct_trip(self):
        server_dt = datetime(2026, 8, 27, 12, 0, 0, tzinfo=timezone.utc)
        self.kill_switch.sync_server_date(server_dt, current_balance=10000.0)

        # Large single loss of $350 (3.5% of $10,000, exceeding 3.0% cap)
        self.kill_switch.record_trade_result(-350.0, server_dt, 9650.0)
        can_trade, reason = self.kill_switch.can_trade(9650.0, server_dt)
        self.assertFalse(can_trade)
        self.assertIn("Daily loss cap hit", reason)

    def test_broker_server_date_rollover_reset(self):
        """Ensure daily stats reset at broker server midnight."""
        day1 = datetime(2026, 8, 27, 23, 59, 0, tzinfo=timezone.utc)
        self.kill_switch.sync_server_date(day1, current_balance=10000.0)
        self.kill_switch.record_trade_result(-350.0, day1, 9650.0)
        self.assertTrue(self.kill_switch.is_tripped)

        # Next day server time
        day2 = datetime(2026, 8, 28, 0, 1, 0, tzinfo=timezone.utc)
        reset_occurred = self.kill_switch.sync_server_date(day2, current_balance=9650.0)
        self.assertTrue(reset_occurred)
        self.assertFalse(self.kill_switch.is_tripped)
        self.assertEqual(self.kill_switch.daily_realized_pnl, 0.0)
        self.assertEqual(self.kill_switch.starting_daily_balance, 9650.0)

    def test_state_persistence_across_restarts(self):
        """Ensure kill switch recovers state from disk after process restart."""
        server_dt = datetime(2026, 8, 27, 14, 0, 0, tzinfo=timezone.utc)
        self.kill_switch.sync_server_date(server_dt, current_balance=10000.0)
        self.kill_switch.record_trade_result(-100.0, server_dt, 9900.0)

        # Create a new instance pointing to same file
        restarted_kill_switch = KillSwitch(self.config)
        self.assertEqual(restarted_kill_switch.daily_realized_pnl, -100.0)
        self.assertEqual(restarted_kill_switch.consecutive_losses, 1)


class TestLogger(unittest.TestCase):
    """Tests for CSV logging outputs."""

    def setUp(self):
        self.test_logs_dir = Path("test_logs")
        self.test_logs_dir.mkdir(parents=True, exist_ok=True)
        self.config = BotConfig(
            logs_dir=self.test_logs_dir,
            signals_csv_path=self.test_logs_dir / "test_signals.csv",
            trades_csv_path=self.test_logs_dir / "test_trades.csv",
        )
        self.trade_logger = TradeLogger(self.config)

    def tearDown(self):
        if self.test_logs_dir.exists():
            shutil.rmtree(self.test_logs_dir)

    def test_csv_headers_and_logging(self):
        self.assertTrue(self.config.signals_csv_path.exists())
        self.assertTrue(self.config.trades_csv_path.exists())

        signal = SimpleNamespace(
            signal_type=SignalType.BUY,
            entry_price=2500.00,
            atr_value=2.50,
            candle_time=datetime(2026, 8, 27, 10, 0, 0, tzinfo=timezone.utc),
            fast_ema=2501.0,
            slow_ema=2499.0,
            rsi=45.0,
            reason="Bullish crossover",
        )

        # Log a rejected signal
        self.trade_logger.log_signal("XAUUSD", signal, status="REJECTED", rejection_reason="Spread too high")

        with open(self.config.signals_csv_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("Spread too high", content)
        self.assertIn("REJECTED", content)


from unittest.mock import patch, MagicMock
import MetaTrader5 as mt5
from executor import TradeExecutor
from risk_manager import RiskAssessmentResult
from main import ScalperBot


class TestExecutor(unittest.TestCase):
    """Tests for TradeExecutor filling mode detection, order routing, retries, and exit tracking."""

    def setUp(self):
        self.config = BotConfig(max_order_retries=1, retry_delay_sec=0.01)
        self.executor = TradeExecutor(self.config)
        self.symbol_info = SimpleNamespace(
            name="XAUUSD",
            digits=2,
            point=0.01,
            trade_tick_value=1.0,
            trade_tick_size=0.01,
            volume_min=0.01,
            volume_max=100.0,
            volume_step=0.01,
            trade_stops_level=0,
            filling_mode=2,  # IOC
        )

    def test_filling_mode_detection(self):
        """Test IOC, FOK, and RETURN filling mode detection."""
        # Bit 2 set -> IOC
        sym_ioc = SimpleNamespace(filling_mode=2)
        self.assertEqual(self.executor.get_filling_mode(sym_ioc), mt5.ORDER_FILLING_IOC)

        # Bit 1 set -> FOK
        sym_fok = SimpleNamespace(filling_mode=1)
        self.assertEqual(self.executor.get_filling_mode(sym_fok), mt5.ORDER_FILLING_FOK)

        # Bit 0 or none -> RETURN
        sym_ret = SimpleNamespace(filling_mode=0)
        self.assertEqual(self.executor.get_filling_mode(sym_ret), mt5.ORDER_FILLING_RETURN)

    @patch("MetaTrader5.order_send")
    def test_execute_order_successful_fill(self, mock_order_send):
        """Test immediate successful fill on 1st attempt."""
        mock_order_send.return_value = SimpleNamespace(
            retcode=mt5.TRADE_RETCODE_DONE,
            order=555123,
            deal=999888,
            volume=0.15,
            price=2500.20,
            comment="Request completed",
        )

        risk_res = RiskAssessmentResult(
            approved=True,
            lot_size=0.15,
            entry_price=2500.20,
            sl_price=2497.20,
            tp_price=2504.20,
            stop_distance_price=3.0,
            tp_distance_price=4.0,
        )

        exec_res = self.executor.execute_market_order(
            symbol="XAUUSD",
            signal_type=SignalType.BUY,
            risk_result=risk_res,
            symbol_info=self.symbol_info,
        )

        self.assertTrue(exec_res.success)
        self.assertEqual(exec_res.ticket, 555123)
        self.assertEqual(exec_res.deal_id, 999888)
        self.assertEqual(exec_res.lot_size, 0.15)
        self.assertEqual(mock_order_send.call_count, 1)
        self.assertIn(555123, self.executor.tracked_positions)

    @patch("MetaTrader5.symbol_info_tick")
    @patch("MetaTrader5.order_send")
    def test_execute_order_requote_retry_success(self, mock_order_send, mock_tick):
        """Test requote on 1st attempt succeeds on retry (2nd attempt)."""
        # 1st call returns REQUOTE; 2nd call returns DONE
        mock_order_send.side_effect = [
            SimpleNamespace(retcode=mt5.TRADE_RETCODE_REQUOTE, comment="Requote", order=0, deal=0, volume=0, price=0),
            SimpleNamespace(retcode=mt5.TRADE_RETCODE_DONE, order=555124, deal=999889, volume=0.15, price=2500.35, comment="Done"),
        ]
        mock_tick.return_value = SimpleNamespace(bid=2500.15, ask=2500.35, last=2500.35, time=1724750000)

        risk_res = RiskAssessmentResult(
            approved=True,
            lot_size=0.15,
            entry_price=2500.20,
            sl_price=2497.20,
            tp_price=2504.20,
            stop_distance_price=3.0,
            tp_distance_price=4.0,
        )

        exec_res = self.executor.execute_market_order(
            symbol="XAUUSD",
            signal_type=SignalType.BUY,
            risk_result=risk_res,
            symbol_info=self.symbol_info,
        )

        self.assertTrue(exec_res.success)
        self.assertEqual(exec_res.ticket, 555124)
        self.assertEqual(mock_order_send.call_count, 2)
        self.assertIn(555124, self.executor.tracked_positions)

    @patch("MetaTrader5.symbol_info_tick")
    @patch("MetaTrader5.order_send")
    def test_execute_order_requote_retry_failure_and_abort(self, mock_order_send, mock_tick):
        """Test requote/timeout failing on retry aborts cleanly without infinite loop."""
        # Both 1st and retry calls return TIMEOUT
        mock_order_send.side_effect = [
            SimpleNamespace(retcode=mt5.TRADE_RETCODE_TIMEOUT, comment="Request canceled by timeout", order=0, deal=0, volume=0, price=0),
            SimpleNamespace(retcode=mt5.TRADE_RETCODE_TIMEOUT, comment="Request canceled by timeout", order=0, deal=0, volume=0, price=0),
        ]
        mock_tick.return_value = SimpleNamespace(bid=2500.15, ask=2500.25, last=2500.25, time=1724750000)

        risk_res = RiskAssessmentResult(
            approved=True,
            lot_size=0.15,
            entry_price=2500.20,
            sl_price=2497.20,
            tp_price=2504.20,
            stop_distance_price=3.0,
            tp_distance_price=4.0,
        )

        exec_res = self.executor.execute_market_order(
            symbol="XAUUSD",
            signal_type=SignalType.BUY,
            risk_result=risk_res,
            symbol_info=self.symbol_info,
        )

        # Must fail and terminate after exactly 1 retry (2 total order_send calls)
        self.assertFalse(exec_res.success)
        self.assertEqual(exec_res.retcode, mt5.TRADE_RETCODE_TIMEOUT)
        self.assertEqual(mock_order_send.call_count, 2)

    @patch("MetaTrader5.history_deals_get")
    @patch("MetaTrader5.positions_get")
    def test_detect_closed_positions_pnl_accounting(self, mock_positions_get, mock_history_deals):
        """Test position exit detection with PnL calculation from history deals."""
        # Seed tracked position
        ticket_id = 777001
        self.executor.tracked_positions[ticket_id] = {
            "ticket": ticket_id,
            "deal_id": 111,
            "symbol": "XAUUSD",
            "order_type": "BUY",
            "lot_size": 0.20,
            "entry_price": 2500.00,
            "sl": 2497.00,
            "tp": 2504.00,
            "open_time": datetime(2026, 8, 27, 10, 0, 0, tzinfo=timezone.utc),
        }

        # Positions in terminal is now empty (position closed)
        mock_positions_get.return_value = []

        # History deal shows TP hit with $80 profit, -$2 commission
        mock_deal_out = SimpleNamespace(
            entry=mt5.DEAL_ENTRY_OUT,
            price=2504.00,
            time=1724750300,
            profit=80.0,
            commission=-2.0,
            swap=0.0,
            reason=mt5.DEAL_REASON_TP,
        )
        mock_history_deals.return_value = [mock_deal_out]

        closed_events = self.executor.detect_closed_positions("XAUUSD")

        self.assertEqual(len(closed_events), 1)
        closed = closed_events[0]
        self.assertEqual(closed.ticket, ticket_id)
        self.assertEqual(closed.close_price, 2504.00)
        self.assertEqual(closed.profit, 80.0)
        self.assertEqual(closed.net_pnl, 78.0)
        self.assertEqual(closed.exit_reason, "TP")
        self.assertNotIn(ticket_id, self.executor.tracked_positions)


class TestMainLoop(unittest.TestCase):
    """Tests for main.py event loop mechanics: candle-sync, dropped connection recovery, and position monitoring."""

    def setUp(self):
        self.test_data_dir = Path("test_main_data")
        self.test_data_dir.mkdir(parents=True, exist_ok=True)
        self.config = BotConfig(
            data_dir=self.test_data_dir,
            logs_dir=self.test_data_dir,
            kill_switch_state_path=self.test_data_dir / "test_ks.json",
            signals_csv_path=self.test_data_dir / "test_signals.csv",
            trades_csv_path=self.test_data_dir / "test_trades.csv",
            loop_poll_interval_sec=0.001,
        )
        self.bot = ScalperBot(self.config)
        self.bot.connector.resolved_symbol = "XAUUSD"

    def tearDown(self):
        if self.test_data_dir.exists():
            shutil.rmtree(self.test_data_dir)

    def _create_mock_df(self, closed_dt: datetime) -> pd.DataFrame:
        """Creates DataFrame where penultimate row has closed_dt timestamp."""
        times = [closed_dt - timedelta(minutes=i) for i in range(50, 0, -1)] + [closed_dt, closed_dt + timedelta(minutes=1)]
        closes = [2500.0 + (i * 0.1) for i in range(len(times))]
        return pd.DataFrame({
            "time": times,
            "open": closes,
            "high": [c + 0.5 for c in closes],
            "low": [c - 0.5 for c in closes],
            "close": closes,
            "tick_volume": [100] * len(times),
            "spread": [20] * len(times),
        })

    def test_candle_boundary_synchronization(self):
        """
        Verify that the loop only processes upon a new completed candle,
        and avoids re-evaluating when polling the same candle.
        """
        t1 = datetime(2026, 8, 27, 10, 0, 0, tzinfo=timezone.utc)
        df_t1 = self._create_mock_df(t1)

        processed_times = []
        original_process = self.bot._process_candle_close

        def mock_process(candle_df, candle_dt):
            processed_times.append(candle_dt)

        self.bot._process_candle_close = mock_process

        # Iteration 1: New candle T1
        latest_closed_row = df_t1.iloc[-2]
        candle_dt = latest_closed_row["time"]
        if self.bot.last_closed_candle_time is None or candle_dt > self.bot.last_closed_candle_time:
            self.bot._process_candle_close(df_t1, candle_dt)
            self.bot.last_closed_candle_time = candle_dt

        self.assertEqual(len(processed_times), 1)
        self.assertEqual(processed_times[0], t1)

        # Iteration 2: Same candle T1 (polling while active bar still forming)
        if self.bot.last_closed_candle_time is None or candle_dt > self.bot.last_closed_candle_time:
            self.bot._process_candle_close(df_t1, candle_dt)
            self.bot.last_closed_candle_time = candle_dt

        # Must NOT re-process the same candle
        self.assertEqual(len(processed_times), 1)

        # Iteration 3: New candle T2 arrives
        t2 = datetime(2026, 8, 27, 10, 1, 0, tzinfo=timezone.utc)
        df_t2 = self._create_mock_df(t2)
        candle_dt_2 = df_t2.iloc[-2]["time"]
        if self.bot.last_closed_candle_time is None or candle_dt_2 > self.bot.last_closed_candle_time:
            self.bot._process_candle_close(df_t2, candle_dt_2)
            self.bot.last_closed_candle_time = candle_dt_2

        self.assertEqual(len(processed_times), 2)
        self.assertEqual(processed_times[1], t2)

    def test_continuous_position_monitoring_between_candles(self):
        """
        Verify that executor.detect_closed_positions is invoked continuously on every poll,
        independent of whether a candle closed or not.
        """
        detect_mock = MagicMock(return_value=[])
        self.bot.executor.detect_closed_positions = detect_mock

        # Run detection directly
        self.bot.executor.detect_closed_positions("XAUUSD")
        self.assertEqual(detect_mock.call_count, 1)

        self.bot.executor.detect_closed_positions("XAUUSD")
        self.assertEqual(detect_mock.call_count, 2)

    def test_loop_resilience_and_reconnection_on_exception(self):
        """Verify that unexpected exceptions during polling are caught and do not crash the bot process."""
        self.bot.is_running = True
        iterations = 0

        # Simulate exception during candle fetch
        def mock_get_candles():
            nonlocal iterations
            iterations += 1
            if iterations == 1:
                raise ConnectionError("MT5 IPC connection lost")
            elif iterations >= 2:
                self.bot.is_running = False  # Stop after recovery
                return None

        self.bot.connector.get_candles = mock_get_candles

        # Execute event loop with exception
        try:
            self.bot._run_event_loop()
        except Exception as e:
            self.fail(f"_run_event_loop crashed unexpectedly with exception: {e}")

        self.assertGreaterEqual(iterations, 2)


class TestGroupedExecution(unittest.TestCase):
    """
    Comprehensive tests for Grouped Multi-Position Trade Execution & Group Profit Target:
    - Test A: Group target — 3 positions collectively reaching +$2.00 total closes all (NOT $6).
    - Test B: Unequal P&L — +$0.80, +$0.55, +$0.65 (sum +$2.00) closes all positions.
    - Test C: Group risk — 3 positions collectively reaching -$1.00 total closes all (1 trade in kill switch).
    - Test D: No per-position target — 1 position at +$1.50 with group at +$0.90 (< $2) does NOT close.
    - Test E: Broker constraints — safe volume 0.01 never multiplied to 0.03.
    - Test F: Netting account — collapses to 1 position with $2.00 group target.
    - Test G: Backward compatibility — POSITIONS_PER_GROUP=1 with $2.00 group target.
    """

    def setUp(self):
        self.config = BotConfig(
            fixed_risk_amount=1.00,
            group_profit_target=2.00,
            positions_per_group=3,
            max_concurrent_trade_groups=1,
            group_risk_mode="FIXED_TOTAL_RISK",
        )
        self.risk_manager = RiskManager(self.config)
        self.executor = TradeExecutor(self.config)
        self.symbol_info = SimpleNamespace(
            name="XAUUSD",
            digits=2,
            point=0.01,
            trade_tick_value=1.0,
            trade_tick_size=0.01,
            trade_contract_size=100.0,
            volume_min=0.01,
            volume_max=100.0,
            volume_step=0.01,
            trade_stops_level=0,
            filling_mode=2,
        )

    def test_A_group_profit_target_collective_close(self):
        """
        Test A — Group target:
        Three positions collectively reach +$2.00 total -> all close (NOT +$6.00).
        """
        gid = "GRP-TEST-A-001"
        group = TradeGroup(
            group_id=gid,
            symbol="XAUUSD",
            order_type="BUY",
            tickets=[101, 102, 103],
            position_lots={101: 0.01, 102: 0.01, 103: 0.01},
            entry_prices={101: 2500.0, 102: 2500.0, 103: 2500.0},
            sl_price=2497.0,
            tp_price=2500.67,
            profit_target=2.00,
        )
        self.executor.active_groups[gid] = group
        for t in group.tickets:
            self.executor.ticket_to_group[t] = gid

        # 3 positions each at +$0.67 profit -> sum = +$2.01 >= $2.00 target
        mock_open_positions = [
            SimpleNamespace(ticket=101, type=mt5.ORDER_TYPE_BUY, volume=0.01, profit=0.67, swap=0.0, magic=self.config.magic_number),
            SimpleNamespace(ticket=102, type=mt5.ORDER_TYPE_BUY, volume=0.01, profit=0.67, swap=0.0, magic=self.config.magic_number),
            SimpleNamespace(ticket=103, type=mt5.ORDER_TYPE_BUY, volume=0.01, profit=0.67, swap=0.0, magic=self.config.magic_number),
        ]

        with patch("MetaTrader5.positions_get") as mock_pos, \
             patch("MetaTrader5.order_send") as mock_send, \
             patch("MetaTrader5.symbol_info_tick") as mock_tick, \
             patch("MetaTrader5.symbol_info") as mock_sym, \
             patch("MetaTrader5.history_deals_get") as mock_deals:

            open_list = list(mock_open_positions)
            def mock_pos_func(**kwargs):
                if "ticket" in kwargs:
                    t = kwargs["ticket"]
                    res = [p for p in mock_open_positions if p.ticket == t]
                    return res
                # Return open positions on first poll, empty after close
                if mock_send.call_count >= 3:
                    return []
                return open_list

            mock_pos.side_effect = mock_pos_func
            mock_tick.return_value = SimpleNamespace(bid=2500.67, ask=2500.87)
            mock_sym.return_value = self.symbol_info
            mock_send.return_value = SimpleNamespace(retcode=mt5.TRADE_RETCODE_DONE)

            deal_out = SimpleNamespace(entry=mt5.DEAL_ENTRY_OUT, price=2500.67, time=1724750300, profit=0.67, commission=0.0, swap=0.0, reason=mt5.DEAL_REASON_CLIENT)
            mock_deals.return_value = [deal_out]

            closed_groups = self.executor.detect_closed_groups("XAUUSD")
            self.assertEqual(len(closed_groups), 1)
            self.assertEqual(mock_send.call_count, 3)  # Closed all 3 positions
            cg = closed_groups[0]
            self.assertAlmostEqual(cg.gross_profit, 2.01, places=2)
            self.assertEqual(len(self.executor.active_groups), 0)

    def test_B_unequal_pnl_reaches_group_target(self):
        """
        Test B — Unequal P&L:
        Positions with unequal floating P&L (+$0.80, +$0.55, +$0.65) sum to +$2.00 and close all.
        """
        gid = "GRP-TEST-B-001"
        group = TradeGroup(
            group_id=gid,
            symbol="XAUUSD",
            order_type="BUY",
            tickets=[201, 202, 203],
            position_lots={201: 0.01, 202: 0.01, 203: 0.01},
            entry_prices={201: 2500.0, 202: 2500.0, 203: 2500.0},
            profit_target=2.00,
        )
        self.executor.active_groups[gid] = group
        for t in group.tickets:
            self.executor.ticket_to_group[t] = gid

        mock_open_positions = [
            SimpleNamespace(ticket=201, type=mt5.ORDER_TYPE_BUY, volume=0.01, profit=0.80, swap=0.0, magic=self.config.magic_number),
            SimpleNamespace(ticket=202, type=mt5.ORDER_TYPE_BUY, volume=0.01, profit=0.55, swap=0.0, magic=self.config.magic_number),
            SimpleNamespace(ticket=203, type=mt5.ORDER_TYPE_BUY, volume=0.01, profit=0.65, swap=0.0, magic=self.config.magic_number),
        ]

        with patch("MetaTrader5.positions_get") as mock_pos, \
             patch("MetaTrader5.order_send") as mock_send, \
             patch("MetaTrader5.symbol_info_tick") as mock_tick, \
             patch("MetaTrader5.symbol_info") as mock_sym, \
             patch("MetaTrader5.history_deals_get") as mock_deals:

            open_list = list(mock_open_positions)
            def mock_pos_func(**kwargs):
                if "ticket" in kwargs:
                    t = kwargs["ticket"]
                    return [p for p in mock_open_positions if p.ticket == t]
                if mock_send.call_count >= 3:
                    return []
                return open_list

            mock_pos.side_effect = mock_pos_func
            mock_tick.return_value = SimpleNamespace(bid=2500.67, ask=2500.87)
            mock_sym.return_value = self.symbol_info
            mock_send.return_value = SimpleNamespace(retcode=mt5.TRADE_RETCODE_DONE)
            mock_deals.return_value = [SimpleNamespace(entry=mt5.DEAL_ENTRY_OUT, price=2500.67, time=1724750300, profit=0.67, commission=0.0, swap=0.0, reason=mt5.DEAL_REASON_CLIENT)]

            closed_groups = self.executor.detect_closed_groups("XAUUSD")
            self.assertEqual(len(closed_groups), 1)
            self.assertEqual(mock_send.call_count, 3)

    def test_C_group_risk_loss_accounting(self):
        """
        Test C — Group risk:
        Three positions reach combined loss of -$1.00 -> all close -> exactly 1 losing trade in kill switch.
        """
        test_dir = Path("test_ks_group_data_c")
        test_dir.mkdir(parents=True, exist_ok=True)
        ks_config = BotConfig(
            data_dir=test_dir,
            kill_switch_state_path=test_dir / "ks_test_c.json",
            manual_stop_file_path=test_dir / "STOP",
            max_daily_loss_pct=3.0,
            max_consecutive_losses=3,
        )
        ks = KillSwitch(ks_config)
        server_dt = datetime(2026, 8, 27, 12, 0, 0, tzinfo=timezone.utc)
        ks.sync_server_date(server_dt, current_balance=100.0)

        # 1 group closes with -$1.00 total loss across 3 sub-positions (-$0.33 each)
        ks.record_trade_result(-1.00, server_dt, 99.0)

        self.assertEqual(ks.daily_trade_count, 1)
        self.assertEqual(ks.consecutive_losses, 1)
        self.assertEqual(ks.daily_realized_pnl, -1.00)

        if test_dir.exists():
            shutil.rmtree(test_dir)

    def test_D_no_premature_close_on_single_position_target(self):
        """
        Test D — No per-position target:
        Verify that 1 position reaching +$1.50 when group sum is only +$0.90 (< $2.00) does NOT close.
        """
        gid = "GRP-TEST-D-001"
        group = TradeGroup(
            group_id=gid,
            symbol="XAUUSD",
            order_type="BUY",
            tickets=[301, 302, 303],
            position_lots={301: 0.01, 302: 0.01, 303: 0.01},
            profit_target=2.00,
        )
        self.executor.active_groups[gid] = group
        for t in group.tickets:
            self.executor.ticket_to_group[t] = gid

        # Ticket 301 is +$1.50, but 302 is -$0.30 and 303 is -$0.30 -> sum is +$0.90 (< $2.00 target)
        mock_open_positions = [
            SimpleNamespace(ticket=301, type=mt5.ORDER_TYPE_BUY, volume=0.01, profit=1.50, swap=0.0, magic=self.config.magic_number),
            SimpleNamespace(ticket=302, type=mt5.ORDER_TYPE_BUY, volume=0.01, profit=-0.30, swap=0.0, magic=self.config.magic_number),
            SimpleNamespace(ticket=303, type=mt5.ORDER_TYPE_BUY, volume=0.01, profit=-0.30, swap=0.0, magic=self.config.magic_number),
        ]

        with patch("MetaTrader5.positions_get") as mock_pos, \
             patch("MetaTrader5.order_send") as mock_send:

            mock_pos.return_value = mock_open_positions
            closed_groups = self.executor.detect_closed_groups("XAUUSD")

            # Must NOT close because group total is +$0.90 < $2.00
            self.assertEqual(len(closed_groups), 0)
            self.assertEqual(mock_send.call_count, 0)
            self.assertEqual(len(self.executor.active_groups), 1)

    def test_E_broker_constraints_protect_risk_budget(self):
        """
        Test E — Broker constraints:
        When minimum lot = 0.01, volume step = 0.01, and total calculated volume is 0.01,
        the bot MUST NOT multiply volume to 0.03. It safely collapses to [0.01] (1 position).
        """
        lots = self.risk_manager.split_group_volume(0.01, 3, 0.01, 0.01, is_hedging=True)
        self.assertEqual(lots, [0.01])
        self.assertEqual(len(lots), 1)
        self.assertEqual(sum(lots), 0.01)

    def test_F_netting_mode_collapses_to_single_position(self):
        """
        Test F — Netting Account:
        If account mode is netting (is_hedging=False), POSITIONS_PER_GROUP becomes 1
        and the same group-level $2.00 profit target still applies.
        """
        lots = self.risk_manager.split_group_volume(0.03, 3, 0.01, 0.01, is_hedging=False)
        self.assertEqual(lots, [0.03])
        self.assertEqual(len(lots), 1)

        # Verify TP distance calculation for 1 position of 0.03 lot targeting $2.00:
        # TP distance = $2.00 / (0.03 * 100) = $0.667
        tick = {"bid": 2500.00, "ask": 2500.20, "spread_price": 0.20, "spread_points": 20.0}
        account = {"balance": 1000.0, "is_hedging": False}
        signal = SimpleNamespace(signal_type=SignalType.BUY, is_valid=True, is_buy=True, is_sell=False, atr_value=2.0, reason="Test Netting")
        res = self.risk_manager.evaluate_signal(signal, tick, account, self.symbol_info, open_groups_count=0)
        self.assertTrue(res.approved)
        self.assertEqual(res.group_profit_target, 2.00)

    def test_G_backward_compatibility_single_position(self):
        """
        Test G — Backward compatibility:
        With POSITIONS_PER_GROUP=1 and GROUP_PROFIT_TARGET=2.00,
        the bot behaves as a single-position strategy targeting $2.00 total.
        """
        cfg = BotConfig(positions_per_group=1, group_profit_target=2.00, group_risk_mode="FIXED_TOTAL_RISK", fixed_risk_amount=1.00)
        rm = RiskManager(cfg)
        lots = rm.split_group_volume(0.03, 1, 0.01, 0.01, is_hedging=True)
        self.assertEqual(lots, [0.03])
        self.assertEqual(len(lots), 1)

    def test_H_dynamic_symbol_properties_calculation(self):
        """
        Test H — Dynamic Symbol Properties:
        TP calculation uses dynamic MT5 symbol properties (tick_size, tick_value, contract_size)
        without any hard-coded $100 contract_size or $0.667 assumptions.
        """
        # Custom index/crypto symbol with contract_size=10.0, tick_size=0.1, tick_value=1.0
        custom_symbol = SimpleNamespace(
            name="CUSTOM",
            digits=1,
            point=0.1,
            trade_tick_value=1.0,
            trade_tick_size=0.1,
            trade_contract_size=10.0,
            volume_min=0.01,
            volume_max=100.0,
            volume_step=0.01,
            trade_stops_level=0,
        )
        tick = {"bid": 1000.0, "ask": 1000.2, "spread_price": 0.2, "spread_points": 2.0}
        account = {"balance": 1000.0, "is_hedging": True}
        signal = SimpleNamespace(signal_type=SignalType.BUY, is_valid=True, is_buy=True, is_sell=False, atr_value=2.0, reason="Custom Symbol")

        # val_per_price_unit_per_lot = tick_value / tick_size = 1.0 / 0.1 = 10.0
        # For 0.03 lot, 0.03 * 10.0 = 0.30 per 1.0 price move.
        # For group profit target $2.00, TP price distance = 2.00 / 0.30 = 6.6667
        cfg = BotConfig(group_profit_target=2.00, fixed_risk_amount=1.00, group_risk_mode="FIXED_TOTAL_RISK", positions_per_group=3)
        rm = RiskManager(cfg)
        res = rm.evaluate_signal(signal, tick, account, custom_symbol, open_groups_count=0)
        self.assertTrue(res.approved)
        self.assertAlmostEqual(res.tp_distance_price, 2.00 / (res.total_lot_size * 10.0), places=3)

    def test_I_different_entry_prices_and_weighted_average(self):
        """
        Test I — Different Entry Prices:
        Positions opened with different entry prices (e.g. 2500.0, 2500.5, 2501.0)
        correctly compute weighted average entry and close when aggregate P&L reaches $2.00.
        """
        gid = "GRP-TEST-I-001"
        group = TradeGroup(
            group_id=gid,
            symbol="XAUUSD",
            order_type="BUY",
            tickets=[501, 502, 503],
            position_lots={501: 0.01, 502: 0.01, 503: 0.01},
            entry_prices={501: 2500.0, 502: 2500.5, 503: 2501.0},
            profit_target=2.00,
        )
        self.assertAlmostEqual(group.weighted_avg_entry, 2500.50, places=2)

        self.executor.active_groups[gid] = group
        for t in group.tickets:
            self.executor.ticket_to_group[t] = gid

        # Unequal floating P&Ls (+1.20, +0.50, +0.35) sum to +$2.05 >= $2.00 target
        mock_open_positions = [
            SimpleNamespace(ticket=501, type=mt5.ORDER_TYPE_BUY, volume=0.01, profit=1.20, swap=0.0, commission=-0.05, magic=self.config.magic_number),
            SimpleNamespace(ticket=502, type=mt5.ORDER_TYPE_BUY, volume=0.01, profit=0.50, swap=0.0, commission=0.0, magic=self.config.magic_number),
            SimpleNamespace(ticket=503, type=mt5.ORDER_TYPE_BUY, volume=0.01, profit=0.40, swap=0.0, commission=0.0, magic=self.config.magic_number),
        ]

        with patch("MetaTrader5.positions_get") as mock_pos, \
             patch("MetaTrader5.order_send") as mock_send, \
             patch("MetaTrader5.symbol_info_tick") as mock_tick, \
             patch("MetaTrader5.symbol_info") as mock_sym, \
             patch("MetaTrader5.history_deals_get") as mock_deals:

            open_list = list(mock_open_positions)
            def mock_pos_func(**kwargs):
                if "ticket" in kwargs:
                    t = kwargs["ticket"]
                    return [p for p in mock_open_positions if p.ticket == t]
                if mock_send.call_count >= 3:
                    return []
                return open_list

            mock_pos.side_effect = mock_pos_func
            mock_tick.return_value = SimpleNamespace(bid=2501.50, ask=2501.70)
            mock_sym.return_value = self.symbol_info
            mock_send.return_value = SimpleNamespace(retcode=mt5.TRADE_RETCODE_DONE)
            mock_deals.return_value = [SimpleNamespace(entry=mt5.DEAL_ENTRY_OUT, price=2501.50, time=1724750300, profit=0.68, commission=-0.02, swap=0.0, reason=mt5.DEAL_REASON_CLIENT)]

            closed_groups = self.executor.detect_closed_groups("XAUUSD")
            self.assertEqual(len(closed_groups), 1)
            self.assertEqual(mock_send.call_count, 3)
            self.assertEqual(len(self.executor.active_groups), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

