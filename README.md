# MetaTrader 5 Gold (XAUUSD) Scalping Bot

A production-grade, modular Python scalping bot designed for trading Gold (`XAUUSD`) on MetaTrader 5 (MT5). 

---

## 🏛 Architecture & Project Structure

```
scalper-bot/
├── config.py           # Centralized configuration (all parameters & env parsing)
├── mt5_connector.py     # Safe MT5 login, symbol alias resolution, candle & tick feed
├── signal_engine.py      # Fast EMA (9), Slow EMA (21), RSI (14), ATR (14) & ATR guards
├── risk_manager.py       # Broker-normalized lot sizing, ATR SL/TP, spread filter
├── executor.py            # Order routing, filling mode negotiation, requote retries
├── logger.py               # Dual CSV audits (signals & trades) + async Telegram alerts
├── kill_switch.py           # Daily loss % & consecutive loss circuit breaker (server-time reset)
├── main.py                    # Candle-boundary event loop & active position monitoring
├── test_suite.py              # Offline test suite (100% test pass rate)
├── requirements.txt           # Pinned dependencies
├── .env.example              # Credentials & parameter template
└── README.md
```

---

## ⚡ Key Trading Logic & Safeguards

### 1. Strategy & Entry Rules (M1 Closed Candles)
- **Fast EMA (9) & Slow EMA (21)**: Evaluates exponential moving average crossovers strictly on **completed/closed candles** (disregards the actively ticking incomplete candle).
- **RSI (14) Exhaustion Filter**:
  - **BUY**: Fast EMA crosses above Slow EMA **AND** RSI < 70 (avoids buying into overbought exhaustion).
  - **SELL**: Fast EMA crosses below Slow EMA **AND** RSI > 30 (avoids selling into oversold exhaustion).
- **ATR Validation Guard**: If ATR on the closed candle is 0, NaN, or non-positive, the signal engine **immediately rejects** the signal, preventing zero-width or malformed SL/TP levels downstream.

### 2. Risk Management & Dynamic Sizing
- **Broker-Normalized Lot Sizing**:
  Does **not** assume flat tick values. Dynamically queries broker `trade_tick_value`, `trade_tick_size`, and `point` from `symbol_info`:
  $$\text{Value Per Point} = \left(\frac{\text{trade\_tick\_value}}{\text{trade\_tick\_size}}\right) \times \text{point}$$
  $$\text{Lot Size} = \frac{\text{Account Balance} \times (\text{Risk \%} / 100)}{\text{Stop Distance in Points} \times \text{Value Per Point}}$$
  *Seamlessly handles both 2-digit (point=0.01) and 3-digit (point=0.001) Gold pricing across brokers.*
- **Broker-Aware Spread Filter**:
  Trades are rejected if the current spread exceeds `MAX_SPREAD_PRICE` ($0.40 default). This is the single most critical guard for gold scalping to prevent spread widening from degrading performance.
- **ATR Stop Loss & Take Profit**:
  - $\text{SL Distance} = 1.5 \times \text{ATR}$
  - $\text{TP Distance} = 2.0 \times \text{ATR}$
  - Enforces broker `trade_stops_level` clearance.

### 3. Circuit Breaker (Kill Switch)
- **Daily Drawdown Limit**: Halts new trade entries if daily realized loss reaches `MAX_DAILY_LOSS_PCT` (default: 3.0%).
- **Consecutive Loss Limit**: Halts new trade entries if consecutive losses reach `MAX_CONSECUTIVE_LOSSES` (default: 3).
- **Broker Server Midnight Reset**: Daily loss windows and P&L stats are keyed to **broker server time** (`TimeCurrent()`), perfectly aligning with MT5 candle generation and broker daily rollover.
- **Persistence Across Restarts**: State is continuously synced to `data/kill_switch_state.json`. If the bot restarts mid-day, loss totals and streak counters are preserved.
- **Manual Halt**: Place a file at `data/STOP` or invoke manual halt to freeze new trade entries immediately.

### 4. Demo-First Safety Lock
- `ALLOW_LIVE_TRADING=False` by default. Connecting to a Real account without explicitly enabling this setting will immediately abort execution.

---

## 🛠 Setup & Installation

### Prerequisites
1. Windows OS with **MetaTrader 5 Terminal** installed.
2. Python 3.10, 3.11, or 3.12 (CPython x64).

### Step 1: MT5 Terminal Configuration
1. Open your MetaTrader 5 terminal.
2. Log into your **Demo Account**.
3. Enable automated trading:
   - Navigate to **Tools ➔ Options ➔ Expert Advisors**.
   - Check **"Allow Algorithmic Trading"**.
   - (Optional) If using custom network paths, check **"Allow WebRequest for listed URL"** and add `https://api.telegram.org`.

### Step 2: Bot Setup
Clone or navigate to the bot workspace and set up your virtual environment:

```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Configure Environment Variables
Copy `.env.example` to `.env`:

```powershell
copy .env.example .env
```

Edit `.env` with your MT5 Demo credentials and optional Telegram keys:

```ini
# MT5 Account Credentials
MT5_LOGIN=12345678
MT5_PASSWORD=YourDemoPasswordHere
MT5_SERVER=MetaQuotes-Demo
MT5_PATH=

# Safety Lock
ALLOW_LIVE_TRADING=False

# Strategy & Risk Parameters
SYMBOL=XAUUSD
TIMEFRAME=M1
MAX_SPREAD_PRICE=0.40
RISK_PER_TRADE_PCT=1.0
MAX_DAILY_LOSS_PCT=3.0
MAX_CONSECUTIVE_LOSSES=3
MAX_CONCURRENT_TRADES=1
SLIPPAGE_POINTS=20

# Telegram Notifications (Optional)
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
```

---

## 🚀 Running the Bot

### 1. Run Offline Tests
Before connecting to the terminal, verify all indicator formulas, sizing mechanics, and kill switch logic:

```powershell
.venv\Scripts\python.exe test_suite.py
# Expected: Ran 21 tests (100% OK)
```

### 2. Pre-Flight Verification Tool (`verify_live_setup.py`)
Run the automated pre-flight tool to verify live terminal connectivity, spread threshold resolution, Telegram delivery, and CSV writing:

```powershell
.venv\Scripts\python.exe verify_live_setup.py
```

This will print diagnostics and confirm:
1. MT5 authentication & Demo mode status.
2. Resolved Gold symbol specs and exact spread points resolution (e.g. `MAX_SPREAD_PRICE = $0.40 ➔ 40.0 broker points`).
3. Current live spread vs max allowed threshold (`PASS` / `REJECT`).
4. Live Telegram alert delivery (fires a test push notification to your phone/desktop).
5. Writing and reading `logs/signals.csv`.

### 3. Start the Live Scalper Bot
Run the main event loop:

```powershell
.venv\Scripts\python.exe main.py
```

To stop the bot gracefully, press `Ctrl + C`.

---

## 🔍 Manual Verification Procedures

### 1. Confirming Spread Threshold at Startup
When you launch `main.py` or `verify_live_setup.py`, check the startup console banner:
```text
==================================================================
             MT5 GOLD (XAUUSD) SCALPING BOT INITIALIZED           
==================================================================
 Account Login       : 12345678 (DEMO SAFE)
 Server & Currency   : MetaQuotes-Demo | 10,000.00 USD
 Resolved Symbol     : 'XAUUSD' (Digits: 2, Point: 0.01)
 Spread Threshold    : MAX_SPREAD_PRICE = $0.40 ➔ 40.0 broker points
 Strategy Settings   : Timeframe M1 | Fast EMA(9) | Slow EMA(21) | RSI(14) | ATR(14)
 Risk Rules          : Risk/Trade 1.0% | Max Daily Loss 3.0% | Max Consecutive Losses 3
 Telegram Alerts     : ENABLED
==================================================================
```
*Verify that the printed broker points threshold matches your broker's digit convention (e.g. 40 points on 2-digit gold vs 400 points on 3-digit gold).*

### 2. Confirming Signal Rejection Logging in `logs/signals.csv`
During trading or volatile market periods (such as news or session opens), open `logs/signals.csv` to ensure rejected signals are recorded with explicit reasons rather than silently dropped:
- **Spread Rejection**: `Spread too wide: Current spread 0.550 price (55.0 pts) > Max allowed 0.400 price (40.0 pts)`
- **Concurrency Rejection**: `Max concurrent trades reached: 1 open >= limit of 1`
- **Kill Switch Rejection**: `Kill Switch Halted: Max consecutive losses reached (3/3)`
- **ATR Validation Rejection**: `Invalid or Zero ATR (0.0) on closed candle`

### 3. Confirming Live Telegram Alerts
1. Run `verify_live_setup.py` to confirm the test alert reaches your Telegram chat.
2. When the bot triggers a live trade, verify that:
   - **Trade Open Alert** fires with ticket number, direction, lots, entry, SL, and TP.
   - **Trade Close Alert** fires when the position hits SL or TP with the realized net P&L.

---

## 📊 CSV Logging & Audit Trail

All bot actions are persistently logged to the `logs/` directory:

1. **`logs/signals.csv`**:
   Records every signal generated on closed candles, capturing whether it was **APPROVED** or **REJECTED**, with exact indicators and rejection reasons (e.g. `Spread too wide: 0.52 > max 0.40`, `Max concurrent trades reached`, `Kill Switch Halted`).

2. **`logs/trades.csv`**:
   Records complete trade lifecycles upon position closure:
   `ticket`, `symbol`, `order_type`, `lot_size`, `open_time`, `close_time`, `open_price`, `close_price`, `gross_profit`, `commission`, `swap`, `net_pnl`, and `exit_reason` (`SL`, `TP`, `MANUAL`).

---

## 📱 Telegram Alerts

When `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` are configured, the bot dispatches non-blocking notifications for:
- 🚀 **Bot Startup**: Account summary, leverage, symbol specs, and risk parameters.
- ⚡ **Trade Opened**: Ticket, direction, volume, entry, calculated SL/TP, and risk amount.
- 💰 **Trade Closed**: Realized P&L breakdown, exit price, and trigger reason (SL/TP).
- 🚨 **Kill Switch Tripped**: Reason, daily P&L, consecutive losses, and trading freeze notification.
- 🛑 **Bot Shutdown**: Clean disconnect alert.
