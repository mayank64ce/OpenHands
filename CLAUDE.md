# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

OpenHands is a general-purpose AI agent platform, adapted here as a git submodule of `FHE-agent-bench` for solving Fully Homomorphic Encryption (FHE) challenges. The codebase is a full-stack application: Python backend (FastAPI), React/TypeScript frontend, with Docker-sandboxed agent execution.

## Build & Development Commands

**Prerequisites:** Python 3.11+, Node.js 18.17.1+, Docker, Poetry 1.8+

```bash
# Full build (installs Python deps, frontend deps, pre-commit hooks, builds frontend)
make build

# Configure LLM credentials interactively
make setup-config

# Run full app (backend on :3000, frontend on :3001)
make run

# Backend only (hot reload)
make start-backend

# Frontend only (hot reload)
make start-frontend
```

### Linting

```bash
make lint                    # Both frontend + backend
make lint-backend            # ruff, mypy, pre-commit (config: dev_config/python/.pre-commit-config.yaml)
make lint-frontend           # npm run lint in frontend/
```

### Testing

```bash
poetry run pytest tests/unit/                        # All unit tests
poetry run pytest tests/unit/test_llm.py -v          # Single test file
poetry run pytest tests/unit/test_llm.py::test_name  # Single test
cd frontend && npm run test                          # Frontend tests
```

### Running the CLI Agent

```bash
# General-purpose agent
poetry run python openhands/core/main.py \
    -t "your task" -d ./workspace/ -c CodeActAgent

# FHE challenge runner (the main use case for this fork)
conda run -n openhands python run_fhe.py \
    --model gpt-4o \
    --challenge-dir ../fhe_challenge/black_box/challenge_relu \
    --max-steps 30 \
    --log-dir logs/fhe/relu_test \
    --build-timeout 600 --run-timeout 600
```

## Architecture

### Event-Driven Core Loop

The system is built around a central **EventStream** (pub/sub) that connects all components:

```
Agent.step(state) → Action → Runtime (Docker sandbox) → Observation → EventStream → next iteration
```

Key classes and their roles:
- **`openhands/events/stream.py`** — EventStream: central message bus, persists events to FileStore
- **`openhands/controller/agent.py`** — Agent base class with `step(state) -> Action` interface
- **`openhands/controller/state/state.py`** — State: tracks history, metrics (tokens/cost), delegation
- **`openhands/llm/llm.py`** — LLM: wraps LiteLLM with retries, cost tracking, prompt caching
- **`openhands/runtime/`** — Runtime backends (Docker, E2B, local) for sandboxed execution
- **`openhands/server/listen.py`** — FastAPI app with WebSocket for real-time UI updates

### Action/Observation Types

Actions (`openhands/events/action/`): `CmdRunAction`, `FileReadAction`, `FileWriteAction`, `IPythonRunCellAction`, `BrowseURLAction`, `MessageAction`, `AgentFinishAction`

Observations (`openhands/events/observation/`): `CmdOutputObservation`, `FileReadObservation`, `ErrorObservation`, `SuccessObservation`

### Agent Implementations (`agenthub/`)

- **`codeact_agent/`** — Primary general-purpose code agent
- **`fhe_agent/`** — FHE challenge solver (custom addition for this fork)
- **`browsing_agent/`** — Web browsing agent
- **`delegator_agent/`** — Multi-agent orchestration

### FHE Adaptation (`fhe/` and `run_fhe.py`)

The FHE-specific layer bypasses the normal Runtime sandbox and uses its own Docker-based judge:

- **`run_fhe.py`** — Orchestrator: parses challenge → agent generates code → inject into Docker → build/run → parse results → feed back to agent
- **`fhe/challenge_parser.py`** — Parses challenge specs (type, scheme, constraints, template code)
- **`fhe/interpreters/`** — Docker-based executors per challenge type: `black_box.py`, `white_box.py`, `ml_inference.py`, `non_openfhe.py`
- **`agenthub/fhe_agent/fhe_agent.py`** — Agent that generates `eval()` function body (C++ or Python) from challenge context

FHE challenge types: `BLACK_BOX` (pre-encrypted), `WHITE_BOX_OPENFHE`, `ML_INFERENCE`, `NON_OPENFHE`

## Configuration

Primary config: `config.toml` (copy from `config.template.toml`). Key sections:

- `[core]` — `workspace_base`, `max_iterations`, `default_agent`
- `[llm]` — `model`, `api_key`, `base_url`, `temperature`, `embedding_model`
- `[agent]` — Per-agent LLM config overrides (e.g., `[agent.RepoExplorerAgent] llm_config = 'gpt3'`)
- `[sandbox]` — `timeout`, `base_container_image`, `use_host_network`

Debug mode: set `DEBUG=1` env var → LLM calls logged to `logs/llm/`

## FHE Run Output Structure

```
logs/fhe/<run>/
├── steps/step_N/{code.cpp, feedback.txt, result.json}
├── trajectory.json        # Full agent history
├── best_solution.cpp      # Highest-scoring solution
├── summary.json
└── llm_metrics.json       # Token usage and cost
```
