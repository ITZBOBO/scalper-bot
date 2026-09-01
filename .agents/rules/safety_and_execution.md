# Scalper Bot & BOBO AI Operational Rules

## Safety & Risk Policy
1. **Safety Lock**: Never enable live trading (`ALLOW_LIVE_TRADING=True`) without explicit confirmation.
2. **Demo Verification**: All test executions, backtests, and development must run against MT5 Demo accounts (`ACCOUNT_TRADE_MODE_DEMO`).
3. **Fail-Safe Isolation**: When running browser or system operations requiring sandbox isolation, strictly enforce Docker boundaries; never silently downgrade.
4. **Approval Gates**: Form submissions, application dispatches, and financial orders must always pass through an approval layer or predefined risk checks.
