---
name: bobo-ai-gateway
description: Architecture, safety protocols, and implementation workflows for BOBO AI personal assistant, Playwright browser automation, Telegram gateway, security sandboxes, and distributed agent fleets.
---

# BOBO AI Personal Assistant & Gateway Skill

## Core Architecture
```text
                 USER
                  │
             Telegram
                  │
                  ▼
          ┌───────────────┐
          │ BOBO Interface │
          │   / Gateway    │
          └───────┬───────┘
                  │
                  ▼
            AI Orchestrator
                  │
      ┌───────────┼────────────┐
      ▼           ▼            ▼
   Research     Career       Coding
   Money        Business     Browser
      │           │            │
      └───────────┼────────────┘
                  ▼
          Objective / Work OS
                  │
                  ▼
        Distributed Worker Fleet
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
    Internet             Execution
        │                   │
        └─────────┬─────────┘
                  ▼
             USER NOTIFIED
        Telegram notification
```

## Security & Reliability Standards

### 1. Browser Automation (Playwright)
- Playwright-driven headless and headed browsers.
- Explicit approval barriers for real-world form submission and financial transactions.
- Automated pauses for CAPTCHA / 2FA.

### 2. Sandbox Boundary Enforcement
- Docker is a hard security boundary.
- If Docker is unavailable, return `SANDBOX_UNAVAILABLE` rather than silently downgrading to an unisolated child process.

### 3. Telegram Thin Gateway
- Telegram is an authenticated frontend client for the Control Plane, PermissionService, and Kill Switch.
- Interactive inline approval buttons for sensitive actions.
- Asynchronous task notification and progress streaming.

### 4. Agent Capabilities
- Specialized worker agents: Career, Business, Money/Research, Coding, Browser.
- Explicit capability signaling (never stub without marking unavailable).
