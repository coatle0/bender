# Bender

## Language Convention

All code, comments, commit messages, documentation, variable names, and technical writing in this project MUST be in English. No exceptions.

---

## Overview

Bender is an open-source framework that connects Slack to Claude Code in headless mode. It provides a thin orchestration layer: receives messages from Slack, invokes Claude Code, maintains conversational context across messages using session management, and exposes an HTTP API for external triggers.

Bender is completely agnostic — it has no business logic, no predefined agent roles, and no integrations beyond Slack and Claude Code. Users inject their own behavior through CLAUDE.md files, skills, and agent team definitions. What Claude Code does once invoked is entirely up to the user's configuration.

The name "Bender" is inspired by the robot from Futurama.

---

## What Bender Does

1. **Slack Gateway** — Listens for messages via slack-bolt (Socket Mode), detects @mentions and thread replies
2. **Claude Code Invocation** — Executes Claude Code CLI in headless mode via subprocess, with full support for user-defined agent teams
3. **Session Management** — Maps each Slack thread to a Claude Code session (`--resume`), enabling multi-turn conversations (e.g., plan → authorization → execution)
4. **HTTP API** — Exposes a FastAPI endpoint so external scripts (cron jobs, webhooks, polling systems) can trigger Bender programmatically

## What Bender Does NOT Do

- Does not define agent roles (leader, watchdog, executor, etc.) — users create their own
- Does not integrate with task management systems (Leantime, Jira, etc.) — users inject via skills
- Does not implement authorization workflows — users define them in their CLAUDE.md and skills
- Does not contain client-specific logic — users provide their own context
- Does not manage infrastructure tools — Claude Code handles that based on user configuration

---

## Technology Stack

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| Slack integration | Python + slack-bolt (Socket Mode) | Mature SDK, no public endpoint needed, outbound WebSocket |
| HTTP API | FastAPI | Async support, lightweight, allows external triggers |
| Claude Code | CLI headless mode (subprocess) | Official tool, supports agent teams, session resume |
| Python SDK (alternative) | claude-agent-sdk | Official Python SDK for programmatic invocation |

### Why Python

Claude Code runs in headless mode — all AI interaction is handled by the CLI. What Bender builds around it is glue code, and Python is the best fit because:

- Claude Code SDK has official Python support (claude-agent-sdk)
- slack-bolt for Python is mature with native Socket Mode
- FastAPI provides async performance for concurrent message handling
- Minimal dependencies, simple deployment

---

## Architecture

### Core flow

```
                    Slack (Socket Mode - WebSocket)
                              ↕
┌──────────────────────────────────────────────────────────┐
│                        Bender                            │
│                                                          │
│  ┌─────────────┐    ┌──────────────┐    ┌────────────┐   │
│  │  slack-bolt  │    │   Session    │    │  FastAPI   │   │
│  │             │    │   Manager    │    │  HTTP API  │   │
│  │  @mention ──┼───►│             │◄───┼── POST /   │   │
│  │  thread   ──┼───►│  thread_ts  │    │   invoke   │   │
│  │  reply    ──┼───►│  ↔ session  │    │            │   │
│  │             │    │             │    │            │   │
│  └─────────────┘    └──────┬───────┘    └────────────┘   │
│                            │                             │
│                     ┌──────▼───────┐                     │
│                     │  Claude Code │                     │
│                     │  (headless)  │                     │
│                     │              │                     │
│                     │  --print     │                     │
│                     │  --resume    │                     │
│                     │  --session-id│                     │
│                     │              │                     │
│                     │  Uses:       │                     │
│                     │  - CLAUDE.md │                     │
│                     │  - skills    │                     │
│                     │  - agent     │                     │
│                     │    teams     │                     │
│                     └──────────────┘                     │
│                                                          │
│  CONFIGURATION:                                          │
│  - SLACK_BOT_TOKEN        (Slack bot token)              │
│  - SLACK_APP_TOKEN        (Slack app token, Socket Mode) │
│  - ANTHROPIC_API_KEY      (API key) OR                   │
│    CLAUDE_CODE_OAUTH_TOKEN (Max subscription token)      │
│  - BENDER_WORKSPACE       (working directory for Claude) │
│  - BENDER_API_KEY         (Bearer token for HTTP API)    │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### Session model: Slack thread = Claude Code session

Each Slack thread maps to exactly one Claude Code session. This enables multi-turn conversations where context is preserved across messages.

```
Channel: #ops
│
├─ Thread 1: "The app is down"
│    ├─ User: "@Bender the app for client X is down"
│    │    → New Claude Code session (session_id: abc-123)
│    ├─ Bender: "I've investigated. Here's the plan... Authorized?"
│    ├─ User: "Go ahead"
│    │    → Resume session abc-123 (claude --resume abc-123)
│    └─ Bender: "Done. Deployed fix and verified."
│
├─ Thread 2: "Create VPN user"           (independent session: def-456)
│
└─ Thread 3: (created by external API)   (independent session: ghi-789)
```

**Key behaviors:**
- New @Bender mention → creates new Slack thread + new Claude Code session
- Reply in a Bender thread → resumes the existing session with `--resume`
- External API call → creates new Slack thread + new Claude Code session
- Sessions persist on disk (`~/.claude/projects/`) and survive process restarts

---

## Entry Points

### 1. Slack (direct interaction)

Someone mentions `@Bender` in any channel where the bot is present:

1. Bender receives the `app_mention` event via Socket Mode
2. Creates a new thread (replies under the original message)
3. Invokes Claude Code headless with the message as prompt
4. Posts Claude's response in the thread
5. Stores `thread_ts → session_id` mapping
6. Subsequent replies in the thread resume the same session

### 2. HTTP API (external scripts)

An external system (cron job, webhook, CI/CD) sends a POST request:

```
POST /api/invoke
{
  "channel": "C0XXXXXXX01",
  "message": "New task detected: ..."
}
```

1. Bender receives the HTTP request
2. Posts the initial message in the specified channel (creates thread)
3. Invokes Claude Code headless with the message
4. Posts response in the thread
5. Stores mapping — thread continues naturally via Slack from there

Both entry points converge into the same flow: load workspace → invoke Claude Code → manage session → respond in thread.

---

## Configuration

### Environment variables

```bash
# Required: Slack
SLACK_BOT_TOKEN="xoxb-..."           # Bot User OAuth Token
SLACK_APP_TOKEN="xapp-..."           # App-Level Token (Socket Mode)

# Required: Claude Code authentication (one of these)
ANTHROPIC_API_KEY="sk-ant-..."       # API key (pay-per-use)
# OR
CLAUDE_CODE_OAUTH_TOKEN="..."        # Max subscription token (via `claude setup-token`)

# Optional
BENDER_WORKSPACE="/home/agent"       # Working directory for Claude Code (default: cwd)
BENDER_API_PORT="8080"               # FastAPI port (default: 8080)
BENDER_API_KEY="your-secret-key"     # Bearer token for HTTP API authentication
LOG_LEVEL="info"                     # Logging level (default: info)
```

> **Note:** The `/api/invoke` endpoint requires a Bearer token (`BENDER_API_KEY`). If `BENDER_API_KEY` is not configured, the endpoint returns HTTP 503 (fail-closed behavior).

### Workspace directory

The workspace is where Claude Code runs. It should contain the user's CLAUDE.md, skills, and any agent team configuration. Bender does not manage this content — it simply invokes Claude Code in this directory.

```
$BENDER_WORKSPACE/
├── CLAUDE.md                        # User's instructions for Claude Code
├── .claude/
│   ├── commands/                    # User-defined slash commands (skills)
│   │   ├── my-skill.md
│   │   └── another-skill.md
│   ├── settings.json                # Claude Code settings (tool permissions)
│   └── teams/                       # Agent team definitions (if using teams)
│       └── my-team/
│           └── config.json
└── (any other files the user needs)
```

The repository includes an example workspace in `workspace/` that you can use as a starting point. The root `docker-compose.yaml` mounts this directory by default. To customize, replace its contents or point `BENDER_WORKSPACE` to your own directory.

### Slack App setup

Required scopes:
- `app_mentions:read` — Listen for @Bender mentions
- `chat:write` — Post messages and thread replies
- `channels:history` — Read messages in public channels
- `groups:history` — Read messages in private channels (if needed)

Required event subscriptions:
- `app_mention` — Trigger on @mentions
- `message.channels` — Listen for thread replies

Socket Mode must be enabled in the Slack app configuration.

---

## Deployment

### Single instance

Bender runs as a single application managing all channels. It does not need one instance per channel or per client. The Slack bot listens to all channels where it has been invited, and Claude Code handles the context based on the message content and the workspace configuration.

### Docker

Two Dockerfiles are provided:

- **`Dockerfile`** (root) — Base image with Python + Node.js + Claude Code CLI. Suitable for general-purpose agents.
- **`docker/Dockerfile`** — Infrastructure variant that adds kubectl, vault, and argocd CLIs on top of the base image. Suitable for SRE/DevOps agents that need to interact with Kubernetes clusters, Vault, and ArgoCD.

Base Dockerfile:

```dockerfile
FROM python:3.12-slim

# Node.js + Claude Code CLI
RUN apt-get update && apt-get install -y curl && \
    curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y nodejs && \
    npm install -g @anthropic-ai/claude-code

# Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir .

COPY src/ /app/src/

WORKDIR /app
CMD ["python", "-m", "bender"]
```

Docker Compose (root) — mounts `./workspace` with the example agent configuration:

```yaml
services:
  bender:
    build: .
    container_name: bender
    env_file:
      - .env
    ports:
      - "${BENDER_API_PORT:-8080}:8080"
    volumes:
      - ./workspace:/workspace
      - claude-data:/root/.claude
    restart: unless-stopped

volumes:
  claude-data:
```

The infra variant (`docker/docker-compose.yaml`) builds from `docker/Dockerfile` and can be customized to mount a different workspace with infrastructure-specific skills and tools.

### Kubernetes (example)

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: bender
spec:
  replicas: 1
  selector:
    matchLabels:
      app: bender
  template:
    metadata:
      labels:
        app: bender
    spec:
      containers:
        - name: bender
          image: bender:latest
          env:
            - name: SLACK_BOT_TOKEN
              valueFrom:
                secretKeyRef:
                  name: bender-secrets
                  key: slack-bot-token
            - name: SLACK_APP_TOKEN
              valueFrom:
                secretKeyRef:
                  name: bender-secrets
                  key: slack-app-token
            - name: ANTHROPIC_API_KEY
              valueFrom:
                secretKeyRef:
                  name: bender-secrets
                  key: anthropic-api-key
            - name: BENDER_WORKSPACE
              value: "/workspace"
          ports:
            - containerPort: 8080
          volumeMounts:
            - name: workspace
              mountPath: /workspace
            - name: claude-data
              mountPath: /root/.claude
      volumes:
        - name: workspace
          configMap:
            name: bender-workspace    # CLAUDE.md, skills, etc.
        - name: claude-data
          emptyDir: {}                # Session persistence
```

---

## Persistence

| State | Where it lives | Notes |
|-------|---------------|-------|
| Claude Code sessions | `~/.claude/projects/` (local filesystem) | Survives process restarts. Lost if pod dies (acceptable for MVP). |
| Thread → Session mapping | SQLite file at `BENDER_DB_PATH` (default: `bender_sessions.sqlite3` in `bender_workspace`) | Survives process restarts — a fresh `SessionManager` pointed at the same file recovers every thread→session mapping. Uses WAL mode. Defaults to an in-memory database (`:memory:`) when constructed with no `db_path`, which is what the test suite uses. |
| Conversation history | Slack threads | Messages stay in Slack. No separate storage needed. |

---

## Security Considerations

Bender itself is a thin layer, but it invokes Claude Code which can execute arbitrary commands. Security is the responsibility of whoever configures the workspace:

- **Claude Code permissions**: Use `--permission-mode` and `--allowed-tools` to restrict what Claude can do
- **Agent teams**: Define watchdog/supervisor agents in team configuration to enforce workflows
- **Workspace isolation**: Mount workspace as read-only where possible
- **Secrets**: Use Kubernetes secrets, Vault, or environment variables — never hardcode in CLAUDE.md
- **Network**: Socket Mode requires no inbound connections (outbound WebSocket only)

---

## Project Structure

```
bender/
├── src/
│   └── bender/
│       ├── __init__.py              # Package metadata
│       ├── __main__.py              # Entry point
│       ├── app.py                   # FastAPI app + slack-bolt wiring
│       ├── api.py                   # HTTP API endpoints (/api/invoke, /health)
│       ├── claude_code.py           # Claude Code CLI subprocess wrapper
│       ├── config.py                # Environment variable loading (pydantic-settings)
│       ├── session_manager.py       # Thread ↔ Session mapping
│       ├── slack_handler.py         # Slack event handlers (@mention, thread replies)
│       └── slack_utils.py           # Message splitting utilities (Slack 4000-char limit)
├── tests/
│   ├── conftest.py                  # Shared fixtures
│   ├── test_api.py                  # API endpoint tests
│   ├── test_app.py                  # App wiring tests
│   ├── test_claude_code.py          # CLI invocation tests
│   ├── test_config.py               # Config loading tests
│   ├── test_session_manager.py      # Session mapping tests
│   ├── test_slack_handler.py        # Slack handler tests
│   └── test_slack_utils.py          # Message splitting tests
├── workspace/                       # Example agent configuration
│   ├── CLAUDE.md                    # Agent instructions (identity, behavior, rules)
│   └── .claude/
│       ├── commands/
│       │   └── hello.md             # Example skill
│       └── settings.json            # Tool permissions (pre-approved commands)
├── docker/                          # Infrastructure-oriented Docker variant
│   ├── Dockerfile                   # Base + kubectl, vault, argocd
│   └── docker-compose.yaml          # Compose for infra agent
├── Dockerfile                       # Base Docker image
├── docker-compose.yaml              # Compose with example workspace
├── pyproject.toml                   # Project metadata and dependencies
├── .env.example                     # Environment variable template
├── .gitignore
├── CLAUDE.md                        # Development instructions (this file)
├── README.md                        # User-facing documentation
└── LICENSE                          # MIT License
```

---

## Development Team (Agent Teams)

Agent teams are enabled at project level via `.claude/settings.json`. To start the development team, tell Claude Code:

```
Create an agent team for developing Bender with these roles: Dev, Quality, and QA.
```

### Team roles

**Dev (Python Developer)**
- Develops the application code
- Implements features following the project structure and architecture defined in this CLAUDE.md
- Writes clean, efficient Python code
- Follows the project conventions (English only, type hints, async patterns)
- Responds to feedback from Quality and QA by fixing issues

**Quality (Code Reviewer)**
- Reviews all code generated by Dev
- Checks for: syntax errors, type issues, code smells, inefficient patterns, security concerns
- Verifies code follows Python best practices (PEP 8, proper async/await usage, error handling)
- When issues are found, creates a detailed improvement plan and sends it to Dev
- Does NOT modify code directly — only reviews and provides feedback

**QA (Test Engineer)**
- Writes tests using pytest to verify the application works correctly
- Covers: unit tests, integration tests, edge cases
- Runs the test suite and reports results
- When tests fail, notifies Dev with details about the failure (what failed, expected vs actual, stack trace)
- Ensures test coverage for all core modules (slack_handler, claude_code, session_manager, api)

### Workflow

1. Dev implements a feature or module
2. Quality reviews the code and sends feedback (if any) to Dev
3. QA writes tests for the implemented code and runs them
4. If Quality finds issues → Dev fixes them
5. If QA tests fail → Dev fixes the code
6. Cycle repeats until Quality approves and all tests pass

---

## Development Status

All core features are implemented and tested:

- [x] **Slack connection** — slack-bolt Socket Mode, @mentions and thread replies
- [x] **Claude Code invocation** — Subprocess wrapper, headless mode, JSON output
- [x] **Session management** — Thread ↔ Session mapping, `--resume` support
- [x] **HTTP API** — FastAPI endpoint for external triggers with Bearer token auth
- [x] **Docker images** — Base image + infrastructure variant (kubectl, vault, argocd)
- [x] **Test suite** — 96 tests across all modules (pytest)
- [x] **Documentation** — README, CLAUDE.md, example workspace

### Future improvements

- [x] Persist thread → session mapping to SQLite for crash recovery
- Store session data on a PVC for pod persistence in Kubernetes
- Python SDK integration (claude-agent-sdk) as alternative to CLI subprocess
