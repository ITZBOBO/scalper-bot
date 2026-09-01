# Agent Guidelines & Repository Architecture

## Workspace Context
- **Primary Engine**: MT5 Gold (XAUUSD) Automated M1 Scalper with EMA(9/21), RSI(14), and ATR(14) filters.
- **Companion Architecture**: BOBO AI Personal Assistant & Distributed Agent Orchestrator.

## Standard Commands
- Check live chart & trades: `.\.venv\Scripts\python.exe check_chart.py`
- Run test suite: `.\.venv\Scripts\python.exe test_suite.py`
- Run trading daemon: `.\.venv\Scripts\python.exe -u main.py`
- Validate live setup: `.\.venv\Scripts\python.exe verify_live_setup.py`
