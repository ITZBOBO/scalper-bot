---
name: mt5-scalper-bot
description: Operational runbook and troubleshooting skill for the MT5 Gold (XAUUSD) Scalper Bot. Use when inspecting trade history, checking open positions, evaluating indicators, debugging spread filters, managing kill switches, or running the trading engine.
---

# MT5 Gold Scalper Bot Runbook

## Overview
This skill provides standardized procedures for operating, testing, and debugging the MT5 Gold (XAUUSD) M1 scalping bot.

## Key Files
- `main.py`: Main event loop with 1-minute candle close synchronization and continuous position exit monitoring.
- `check_chart.py`: Live CLI status checker (balance, equity, spread, latest indicators, open positions, recent deals).
- `config.py`: Environment-driven configuration and safety limits.
- `signal_engine.py`: 9/21 EMA crossover + RSI filter + ATR calculation.
- `risk_manager.py`: Lot calculation based on broker point/tick value, max spread guard, max concurrent trade limits.
- `executor.py`: MT5 order execution with IOC/FOK/Return fallback and slippage control.
- `kill_switch.py`: Broker server date rollover tracking, daily drawdown cap (3%), and consecutive loss threshold (3).
- `test_suite.py`: Comprehensive 22-test automated unit and integration suite.

## Common Operations

### 1. Check Live Chart, Account, and Trades
```powershell
.\.venv\Scripts\python.exe check_chart.py
```

### 2. Run the Full Test Suite
```powershell
.\.venv\Scripts\python.exe test_suite.py
```

### 3. Launch the Bot Daemon
```powershell
.\.venv\Scripts\python.exe -u main.py
```

### 4. Verify Live Environment & Configuration
```powershell
.\.venv\Scripts\python.exe verify_live_setup.py
```

## Safety Constraints
- Live trading is hard-locked by default (`ALLOW_LIVE_TRADING=False`).
- Spread filter rejects trades exceeding `MAX_SPREAD_PRICE` ($0.40 / 40 pts).
- Daily loss exceeding 3.0% trips the kill switch until the broker server day rolls over.
- 3 consecutive losses immediately suspends new order generation.
