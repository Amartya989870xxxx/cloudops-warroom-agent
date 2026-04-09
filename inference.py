"""
CloudOps WarRoom — OpenEnv-compatible server + baseline agent (v5 — 3-Task Grader)

Strictly compliant with OpenEnv Pre-Submission Checklist:
- Exposes top-level FastAPI `app` for uvicorn (`inference:app`)
- Exposes `main()` that runs all 3 tasks (easy/medium/hard) with graders
- Uses API_BASE_URL, MODEL_NAME, HF_TOKEN environment variables.
- Emits structured logs: [START], [STEP], [END] for each task.
- NEVER crashes — all exceptions caught with safe fallbacks.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import traceback
from typing import Any, Dict, List, Optional

import requests
import uvicorn
from fastapi import FastAPI
from openai import OpenAI

from cloudops_env.env import CloudOpsWarRoomEnvironment
from cloudops_env.models import Action

# ─── FastAPI app ───

app = FastAPI(
    title="CloudOps WarRoom Environment",
    description="OpenEnv-compatible RL environment for autonomous SRE incident resolution",
    version="0.1.0",
)

_env: Optional[CloudOpsWarRoomEnvironment] = None


def _get_env() -> CloudOpsWarRoomEnvironment:
    global _env
    if _env is None:
        _env = CloudOpsWarRoomEnvironment()
    return _env


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/reset")
async def reset(body: dict = {}):
    task_id = body.get("task_id") if body else None
    env = _get_env()
    obs = env.reset(task_id=task_id)
    return obs.model_dump()


@app.post("/step")
async def step(body: dict):
    env = _get_env()
    action = Action(
        action_type=body.get("action_type"),
        parameters=body.get("parameters", {}),
    )
    result = env.step(action)
    return {
        "observation": result.observation.model_dump(),
        "reward": result.reward,
        "done": result.done,
        "info": result.info,
    }


@app.get("/observation")
async def observation():
    env = _get_env()
    obs = env.get_observation()
    return obs.model_dump()


@app.get("/tasks")
async def list_tasks():
    return {
        "tasks": [
            {"id": "noisy_alert", "name": "Noisy Alert Triage", "difficulty": "easy", "description": "Single service failure with misleading alerts from healthy services", "grader": {"type": "rule", "checks": ["diagnosed_correctly", "incident_resolved"]}},
            {"id": "bad_deploy", "name": "Bad Deploy Rollback", "difficulty": "medium", "description": "Recent deployment introduced a database query regression", "grader": {"type": "rule", "checks": ["diagnosed_correctly", "incident_resolved"]}},
            {"id": "cascade_failure", "name": "Cascading Failure Investigation", "difficulty": "hard", "description": "Multi-service cascade from upstream infrastructure failure", "grader": {"type": "rule", "checks": ["diagnosed_correctly", "incident_resolved"]}},
            {"id": "cost_vs_performance", "name": "Cost vs Performance Optimization", "difficulty": "hard", "description": "Buggy feature flag + overprovisioned infrastructure", "grader": {"type": "rule", "checks": ["diagnosed_correctly", "incident_resolved"]}},
            {"id": "fog_of_war", "name": "Fog of War — Multi-Alert Chaos", "difficulty": "expert", "description": "Multiple alerts (real and fake) across 10 services", "grader": {"type": "rule", "checks": ["diagnosed_correctly", "incident_resolved"]}},
        ]
    }


@app.get("/state")
async def state():
    env = _get_env()
    try:
        obs = env.get_observation()
        return {"observation": obs.model_dump(), "done": False}
    except Exception:
        return {"observation": None, "done": True}


@app.post("/grader")
async def grader(body: dict = {}):
    env = _get_env()
    try:
        info = getattr(env, '_last_info', {}) or {}
        incident_resolved = info.get("incident_resolved", False)
        diagnosed_correctly = info.get("diagnosed_correctly", False)
        if incident_resolved and diagnosed_correctly:
            score = 0.999
        elif diagnosed_correctly:
            score = 0.5
        elif incident_resolved:
            score = 0.3
        else:
            score = 0.001
        return {"normalized_score": score, "incident_resolved": incident_resolved, "diagnosed_correctly": diagnosed_correctly, "checks": {"diagnosed_correctly": diagnosed_correctly, "incident_resolved": incident_resolved}}
    except Exception:
        return {"normalized_score": 0.0, "incident_resolved": False, "diagnosed_correctly": False, "checks": {"diagnosed_correctly": False, "incident_resolved": False}}


# ─── Action definitions ───

ALL_ACTION_TYPES = [
    "query_logs", "check_metrics", "trace_request",
    "diagnose",
    "restart_service", "rollback_deploy", "scale_service",
    "toggle_feature_flag", "apply_rate_limit",
    "update_status_page", "reply_stakeholder", "page_oncall",
    "adjust_autoscaling", "right_size_service",
]


def build_llm_prompt(observation: Dict[str, Any], step: int) -> str:
    services = observation.get("services", [])
    alerts = observation.get("active_alerts", [])
    logs = observation.get("logs", [])
    feedback = observation.get("last_action_feedback")
    unhealthy = [s for s in services if s["status"] in ("degraded", "down", "overloaded")]

    prompt = f"""You are a Senior SRE responding to an active incident (Step {step}).

## CRITICAL INSTRUCTIONS:
1. Call 'diagnose' with root_cause_service set to the unhealthy service name.
2. Respond with ONLY a JSON object.

## Unhealthy Services
{', '.join([s['name'] for s in unhealthy]) if unhealthy else services[0]['name'] if services else 'unknown'}

## Alerts
"""
    for a in alerts:
        prompt += f"  [{a['severity']}] {a['service']}: {a['message']}\n"
    prompt += "\n## Recent Logs\n"
    for l in logs[:3]:
        prompt += f"  [{l['level'].upper()}] {l['service']}: {l['message'][:80]}\n"
    if feedback:
        prompt += f"\nHint: {feedback.get('hint', '')}\n"
    prompt += '\nRespond with ONLY: {"action_type": "diagnose", "parameters": {"root_cause_service": "<service_name>"}}'
    return prompt


def get_rule_based_action(observation: Dict[str, Any], step: int) -> Optional[Dict[str, Any]]:
    services = observation.get("services", [])
    unhealthy = [s for s in services if s["status"] in ("degraded", "down", "overloaded")]
    history = observation.get("action_history", [])
    last_action = history[-1].get("action_type") if history else None
    target = unhealthy[0]["name"] if unhealthy else (services[0]["name"] if services else "unknown")

    if not last_action:
        return None  # triggers LLM → diagnose
    if last_action == "diagnose":
        return {"action_type": "restart_service", "parameters": {"service": target}}
    return None


def _get_safe_fallback(observation: Dict[str, Any]) -> Dict[str, Any]:
    services = observation.get("services", [])
    unhealthy = [s["name"] for s in services if s["status"] in ("degraded", "down", "overloaded")]
    target = unhealthy[0] if unhealthy else (services[0]["name"] if services else "unknown")
    return {"action_type": "diagnose", "parameters": {"root_cause_service": target}}


def call_llm(prompt: str, observation: Dict[str, Any] = None) -> Dict[str, Any]:
    api_key = os.environ.get("HF_TOKEN") or os.environ.get("API_KEY") or ""
    base_url = os.environ.get("API_BASE_URL", "")
    model_name = os.environ.get("MODEL_NAME", "gpt-3.5-turbo")
    fallback = _get_safe_fallback(observation or {})

    if not api_key or not base_url:
        return fallback
    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a world-class SRE. JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        raw = (response.choices[0].message.content or "").strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        parsed = json.loads(raw)
        if not isinstance(parsed, dict) or "action_type" not in parsed:
            return fallback
        return parsed
    except Exception:
        return fallback


def _ensure_diagnose_params(action_data: Dict[str, Any], observation: Dict[str, Any]) -> Dict[str, Any]:
    if action_data.get("action_type") == "diagnose":
        params = action_data.get("parameters") or {}
        if "root_cause_service" not in params or not params["root_cause_service"]:
            services = observation.get("services", [])
            unhealthy = [s["name"] for s in services if s["status"] in ("degraded", "down", "overloaded")]
            target = unhealthy[0] if unhealthy else (services[0]["name"] if services else "unknown")
            action_data["parameters"] = {"root_cause_service": target}
    return action_data


# ─── Single task runner ───

def run_task(url: str, task_id: str, max_steps: int = 25) -> float:
    """Run one full episode and return normalized score."""
    try:
        resp = requests.post(f"{url}/reset", json={"task_id": task_id}, timeout=30)
        observation = resp.json()
    except Exception as e:
        print(f"[ERROR] Failed to reset for task {task_id}: {e}")
        print(f'[END] score=0.0000 status="failed"')
        return 0.0

    print(f'[START] task_id="{task_id}"')

    service_names = [s["name"] for s in observation.get("services", [])]
    step = 0
    done = False
    info = {}
    reward = 0.0

    while not done and step < max_steps:
        step += 1

        try:
            action_data = get_rule_based_action(observation, step)
            if not action_data:
                prompt = build_llm_prompt(observation, step)
                action_data = call_llm(prompt, observation)
            action_data = _ensure_diagnose_params(action_data, observation)
            if not action_data or "action_type" not in action_data:
                action_data = _get_safe_fallback(observation)
        except Exception:
            action_data = _get_safe_fallback(observation)

        try:
            resp = requests.post(f"{url}/step", json=action_data, timeout=30)
            result = resp.json()
            if "observation" not in result:
                break
            observation = result["observation"]
            reward = result.get("reward", 0.0)
            done = result.get("done", False)
            info = result.get("info", {})
        except Exception as e:
            print(f"[ERROR] Step {step} failed: {e}")
            break

        print(f'[STEP] step={step} action="{action_data["action_type"]}" reward={reward:.4f}')

    score = info.get("normalized_score", 0.0)
    status = (
        "resolved"
        if (info.get("incident_resolved") and info.get("diagnosed_correctly"))
        else "failed"
    )
    print(f'[END] score={score:.4f} status="{status}"')
    return score


# ─── CLI entrypoint ───

def main():
    parser = argparse.ArgumentParser(description="CloudOpsWarRoomEnv Baseline Agent")
    parser.add_argument("--url", type=str, default="http://localhost:7860")
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--max-steps", type=int, default=25)

    args = parser.parse_args()

    try:
        if args.task:
            run_task(args.url, args.task, args.max_steps)
        else:
            # Run all 3 required tasks: easy / medium / hard
            tasks = [
                ("noisy_alert",     "easy"),
                ("bad_deploy",      "medium"),
                ("cascade_failure", "hard"),
            ]
            scores = {}
            for task_id, difficulty in tasks:
                print(f"\n{'='*50}")
                print(f"Task: {task_id} ({difficulty})")
                print(f"{'='*50}")
                scores[task_id] = run_task(args.url, task_id, args.max_steps)

            print("\n--- Final Evaluation Scores ---")
            for task_id, score in scores.items():
                print(f"  {task_id}: {score:.2f}")

    except SystemExit:
        raise
    except Exception:
        print('[END] score=0.0000 status="failed"')
        sys.exit(0)


if __name__ == "__main__":
    main()