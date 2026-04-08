"""
CloudOps WarRoom — OpenEnv-compatible server + baseline agent (v2 — Enhanced)

Strictly compliant with OpenEnv Pre-Submission Checklist:
- Exposes top-level FastAPI `app` for uvicorn (`inference:app`)
- Exposes `main()` for pyproject.toml [project.scripts] entrypoint
- Uses API_BASE_URL, MODEL_NAME, HF_TOKEN environment variables.
- Emits structured logs: [START], [STEP], [END].
- Supports local/random execution without API keys.
- Optimal for vcpu=2, memory=8gb machines.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional

import requests
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from cloudops_env.env import CloudOpsWarRoomEnvironment
from cloudops_env.models import Action

# ─── FastAPI app (top-level — required by uvicorn `inference:app`) ───

app = FastAPI(
    title="CloudOps WarRoom Environment",
    description="OpenEnv-compatible RL environment for autonomous SRE incident resolution",
    version="0.1.0",
)

# Singleton env instance used by API routes
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
    """Convert an observation dict into a natural language prompt for the LLM (#5)."""
    services = observation.get("services", [])
    alerts = observation.get("active_alerts", [])
    deploys = observation.get("recent_deploys", [])
    logs = observation.get("logs", [])
    deps = observation.get("dependencies", [])
    action_history = observation.get("action_history", [])
    feedback = observation.get("last_action_feedback")

    unhealthy = [s for s in services if s["status"] in ("degraded", "down", "overloaded")]

    prompt = f"""You are a Senior SRE responding to an active incident (Step {step}).
Your goal is to resolve the incident with MAXIMUM efficiency (minimal steps) and full communication.

## CRITICAL INSTRUCTIONS:
1. Investigation: Start with check_metrics/query_logs/trace_request on suspected services.
2. Diagnosis: You MUST call 'diagnose' with the correct service BEFORE applying any fix.
3. Fix: Apply rollback_deploy, restart_service, or toggle_feature_flag ONLY after diagnosis.
4. Communication: After fixing, call 'update_status_page' and 'reply_stakeholder' to finish.
5. No Redundancy: Do NOT repeat the same action on the same service.

## System State
Unhealthy Services: {', '.join([s['name'] for s in unhealthy]) if unhealthy else 'None'}

## Alerts
"""
    for a in alerts:
        prompt += f"  [{a['severity']}] {a['service']}: {a['message']}\n"

    prompt += "\n## Recent Logs\n"
    for l in logs[:5]:
        prompt += f"  [{l['level'].upper()}] {l['service']}: {l['message'][:100]}\n"

    if feedback:
        prompt += f"\n## Last Action Feedback\nImpact: {feedback.get('impact', 'N/A')}\nHint: {feedback.get('hint', 'N/A')}\n"

    prompt += f"\n## Action History (Last 5 steps)\n"
    for h in action_history[-5:]:
        prompt += f" - Step {h.get('step')}: {h.get('action_type')} {h.get('parameters')}\n"

    prompt += "\nRespond with ONLY a JSON object: {\"action_type\": \"...\", \"parameters\": {...}}"
    return prompt


def get_rule_based_action(observation: Dict[str, Any], step: int) -> Optional[Dict[str, Any]]:
    services = observation.get("services", [])
    unhealthy = [s for s in services if s["status"] in ("degraded", "down", "overloaded")]
    history = observation.get("action_history", [])

    last_action = history[-1].get("action_type") if history else None
    target = unhealthy[0]["name"] if unhealthy else services[0]["name"]

    # Step 1: No history → return None to force LLM call
    # LLM will pick "diagnose" based on the prompt instructions
    if not last_action:
        return None

    # Step 2: After LLM diagnoses → rule fires restart_service
    if last_action == "diagnose":
        return {"action_type": "restart_service", "parameters": {"service": target}}

    return None


def call_llm(prompt: str) -> Dict[str, Any]:
    from openai import OpenAI

    api_key = os.environ.get("API_KEY")
    base_url = os.environ.get("API_BASE_URL")
    model_name = os.environ.get("MODEL_NAME", "mistralai/mistral-7b-instruct")

    if not api_key or not base_url:
        raise ValueError("Missing HF_TOKEN or API_BASE_URL environment variables.")

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        default_headers={
            "HTTP-Referer": "https://cloudops-agent.local",
            "X-Title": "CloudOpsWarRoom",
        },
    )

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a world-class SRE. JSON only."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
    )

    return json.loads(response.choices[0].message.content.strip())


def random_action(service_names: List[str], observation: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a random valid action for testing."""
    action_type = random.choice(ALL_ACTION_TYPES)
    params = {}

    if action_type in (
        "query_logs", "check_metrics", "trace_request", "restart_service",
        "apply_rate_limit", "page_oncall", "adjust_autoscaling", "right_size_service",
    ):
        params = {"service": random.choice(service_names)}
    elif action_type == "diagnose":
        params = {"root_cause_service": random.choice(service_names)}
    elif action_type == "rollback_deploy":
        deploys = [d["service"] for d in observation.get("recent_deploys", [])]
        params = {"service": random.choice(deploys) if deploys else random.choice(service_names)}
    elif action_type == "scale_service":
        params = {"service": random.choice(service_names), "direction": random.choice(["up", "down"])}
    elif action_type == "toggle_feature_flag":
        params = {"flag_name": "new_product_page_v2"}
    elif action_type in ("update_status_page", "reply_stakeholder"):
        params = {"message": "Investigating incident."}

    return {"action_type": action_type, "parameters": params}


def run_agent(args):
    """Main agent loop supporting local and remote execution."""
    if args.local:
        env = CloudOpsWarRoomEnvironment(debug=args.debug)
        obs_obj = env.reset(task_id=args.task)
        observation = obs_obj.model_dump()
        task_id = args.task or "default"
    else:
        resp = requests.post(f"{args.url}/reset", json={"task_id": args.task})
        observation = resp.json()
        task_id = args.task or observation.get("task_id", "default")

    # [START] Mandatory Tag (#4)
    print(f"[START] task_id=\"{task_id}\"")

    service_names = [s["name"] for s in observation.get("services", [])]
    step = 0
    done = False
    total_reward = 0.0
    info = {}

    while not done and step < args.max_steps:
        step += 1

        if args.random:
            action_data = random_action(service_names, observation)
        else:
            action_data = get_rule_based_action(observation, step)
            if not action_data:
                prompt = build_llm_prompt(observation, step)
                action_data = call_llm(prompt)

        reward = 0.0
        if args.local:
            action = Action(
                action_type=action_data["action_type"],
                parameters=action_data.get("parameters", {}),
            )
            result = env.step(action)
            observation = result.observation.model_dump()
            reward = result.reward
            done = result.done
            info = result.info
        else:
            if action_data.get("action_type") == "diagnose":
                if "root_cause_service" not in action_data.get("parameters", {}):
                    services = [s["name"] for s in observation.get("services", [])]
                    if services:
                        action_data["parameters"] = {"root_cause_service": services[0]}

            resp = requests.post(f"{args.url}/step", json=action_data)
            result = resp.json()

            if "observation" not in result:
                print("[ERROR] Invalid response from environment:", result)
                break

            observation = result["observation"]
            reward = result.get("reward", 0.0)
            done = result.get("done", False)
            info = result.get("info", {})

        # [STEP] Mandatory Tag
        print(f"[STEP] step={step} action=\"{action_data['action_type']}\" reward={reward:.4f}")

    # [END] Mandatory Tag (#4)
    score = info.get("normalized_score", 0.0)
    status = (
        "resolved"
        if (info.get("incident_resolved") and info.get("diagnosed_correctly"))
        else "failed"
    )
    print(f"[END] score={score:.4f} status=\"{status}\"")


# ─── CLI entrypoint (required by pyproject.toml [project.scripts]) ───

def main():
    parser = argparse.ArgumentParser(description="CloudOpsWarRoomEnv Baseline Agent")
    # Default URL points to the running server in the same container
    parser.add_argument("--url", type=str, default="http://localhost:7860")
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--random", action="store_true", default=True)
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--max-steps", type=int, default=25)

    args = parser.parse_args()
    # Always run the agent loop — server is started by Dockerfile CMD (uvicorn inference:app)
    # Never call uvicorn.run() here; port 7860 is already bound by the time validator calls main()
    run_agent(args)


if __name__ == "__main__":
    main()