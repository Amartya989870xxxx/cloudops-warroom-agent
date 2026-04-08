# 🔥 CloudOpsWarRoomEnv

> **A production-grade OpenEnv RL environment simulating real-world SRE/DevOps incident response.**

Built for the **Scaler × Meta OpenEnv Hackathon**.

---

## 🎯 Overview

CloudOpsWarRoomEnv drops an AI agent into a simulated cloud operations war room. The agent acts as an on-call SRE engineer who must:

1. **Investigate** system state (logs, metrics, traces)
2. **Diagnose** the root cause of a production incident
3. **Fix** the issue (restart, rollback, feature flag, scaling)
4. **Communicate** with stakeholders
5. **Optimize** infrastructure costs

The agent learns through **reinforcement learning** — rewards, not labels.

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────┐
│                    AI Agent (LLM)                    │
│  observe → reason → act → learn → repeat             │
└────────────┬─────────────────────────────┬───────────┘
             │ POST /step                  │ POST /reset
┌────────────▼─────────────────────────────▼───────────┐
│              FastAPI Server (OpenEnv)                 │
│  ┌─────────────┐  ┌──────────┐  ┌────────────────┐  │
│  │ Environment  │  │  Tasks   │  │ Reward Engine  │  │
│  │  (env.py)    │  │ Registry │  │  (graders/)    │  │
│  └──────┬──────┘  └────┬─────┘  └───────┬────────┘  │
│         │              │                │            │
│  ┌──────▼──────────────▼────────────────▼────────┐   │
│  │           Simulation Engine                    │   │
│  │  • Microservice dependency graph              │   │
│  │  • Fault propagation & cascading              │   │
│  │  • Dynamic metrics & log generation           │   │
│  │  • Action processing & state updates          │   │
│  └───────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────┘
```

## 📦 Project Structure

```
cloudops_env/
├── __init__.py          # Package exports
├── models.py            # Pydantic models (Action, Observation, State, StepResult)
├── env.py               # Core environment logic
├── client.py            # EnvClient for remote usage
├── tasks/
│   ├── easy.py          # Noisy Alert (easy)
│   ├── medium.py        # Bad Deploy (medium)
│   ├── hard.py          # Cascade Failure, Cost vs Perf, Fog of War
│   └── registry.py      # Task registry
├── graders/
│   └── reward.py        # Dense reward calculation engine
├── server/
│   ├── app.py           # FastAPI server
│   ├── requirements.txt # Server dependencies
│   └── Dockerfile       # Container definition
├── inference.py         # Baseline LLM agent
├── openenv.yaml         # OpenEnv manifest
├── pyproject.toml       # Package config
├── requirements.txt     # Top-level dependencies
└── README.md            # This file
```

## 🚀 Quick Start

### Install Dependencies

```bash
cd cloudops_env
pip install -r requirements.txt
```

### Run Locally (No Server)

```bash
# Random agent — no API key needed
python inference.py --local --random --task noisy_alert

# Run all tasks
for task in noisy_alert bad_deploy cascade_failure cost_vs_performance fog_of_war; do
  python inference.py --local --random --task $task
done
```

### Run with FastAPI Server

```bash
# Terminal 1: Start server
cd cloudops_env
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: Run agent
python inference.py --task bad_deploy --random
```

### Run with Docker

```bash
cd cloudops_env
docker build -t cloudops-warroom -f server/Dockerfile .
docker run -p 8000:8000 cloudops-warroom

# Then run agent against the container
python inference.py --url http://localhost:8000 --task cascade_failure --random
```

### Use the LLM Agent

```bash
export OPENAI_API_KEY="sk-..."
python inference.py --local --task fog_of_war
```

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/reset` | Start new episode (optional `task_id`) |
| `POST` | `/step` | Execute an action |
| `GET` | `/state` | Get episode metadata |
| `POST` | `/validate` | Validate action without executing |
| `GET` | `/tasks` | List available scenarios |

### Example: Reset

```bash
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "noisy_alert"}'
```

### Example: Step

```bash
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "check_metrics", "parameters": {"service": "payment-service"}}'
```

## 🎮 Tasks (5 Scenarios)

| # | Task | Difficulty | Root Cause | Fix |
|---|------|-----------|------------|-----|
| 1 | **Noisy Alert** | Easy | Memory leak in payment-service | Restart |
| 2 | **Bad Deploy** | Medium | DB regression in order-service deploy | Rollback |
| 3 | **Cascade Failure** | Hard | Redis cluster (session-store) OOM | Restart |
| 4 | **Cost vs Performance** | Hard | Buggy feature flag + overprovisioned search | Feature Flag + Right-size |
| 5 | **Fog of War** | Expert | Bad deploy with 8 alerts (half fake) | Rollback |

## ⚡ Action Space (15 Actions)

### Investigate
- `query_logs(service)` — View recent logs
- `check_metrics(service)` — Check CPU, memory, error rate, latency
- `trace_request(service)` — Trace through dependency graph

### Diagnose
- `diagnose(root_cause_service)` — Declare the root cause

### Fix
- `restart_service(service)` — Restart a service
- `rollback_deploy(service)` — Rollback to previous version
- `scale_service(service, direction)` — Scale up/down
- `toggle_feature_flag(flag_name)` — Toggle a feature flag
- `apply_rate_limit(service)` — Apply rate limiting

### Communicate
- `update_status_page(message)` — Update public status page
- `reply_stakeholder(message)` — Reply to waiting stakeholder
- `page_oncall(service)` — Page on-call engineer

### Optimize
- `adjust_autoscaling(service)` — Adjust autoscaling policy
- `right_size_service(service)` — Right-size overprovisioned service

## 📊 Reward Function

| Signal | Reward |
|--------|--------|
| Correct diagnosis | +0.25 |
| Correct restart | +0.20 |
| Correct rollback | +0.30 |
| Correct feature flag fix | +0.35 |
| Status page update | +0.05 |
| Stakeholder reply | +0.08 |
| Episode completion | +0.30 |
| Wrong restart | -0.10 |
| Wrong rollback | -0.15 |
| Wrong diagnosis | -0.10 |
| Useless scaling | -0.05 |
| Wasted step | -0.02 |
| Per unhealthy service/step | -0.01 |
| Timeout | -0.20 |

## 🧠 Design Principles

- **No trivial solutions** — Multi-step reasoning required
- **Noisy observations** — Some alerts are fake/misleading
- **Dense rewards** — Every step returns a reward signal
- **Realistic simulation** — Faults propagate through dependency graphs
- **Investigation → Diagnosis → Fix → Communicate** workflow
- **Root cause is hidden** — Never directly visible in observations
- **Time pressure** — Urgency increases each step

## 📋 OpenEnv Compatibility

This environment follows the [OpenEnv spec](https://github.com/meta-pytorch/OpenEnv):

- Gymnasium-style API: `reset()`, `step()`, `state()`
- Typed models via Pydantic
- FastAPI server with standard endpoints
- Docker containerization
- Compatible with Hugging Face Spaces deployment

## 📄 License

MIT
