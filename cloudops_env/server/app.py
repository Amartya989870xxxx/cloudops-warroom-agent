"""
FastAPI Server for CloudOpsWarRoomEnv (v2 — Enhanced)

Exposes the OpenEnv-compliant API:
  POST /reset   — Start a new episode (supports debug mode)
  POST /step    — Execute an action (with validation)
  GET  /state   — Get episode metadata
  GET  /health  — Health check
  POST /validate — Validate an action without executing
  GET  /tasks   — List available tasks
"""

from __future__ import annotations

import sys
import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cloudops_env.env import CloudOpsWarRoomEnvironment
from cloudops_env.models import Action, Observation, State, StepResult


# ─── Request/Response Models ───

class ResetRequest(BaseModel):
    task_id: Optional[str] = None
    debug: bool = False   # Enable debug mode (#9)


class StepRequest(BaseModel):
    action_type: str
    parameters: Dict[str, Any] = {}


class HealthResponse(BaseModel):
    status: str = "healthy"
    environment: str = "CloudOpsWarRoomEnv"
    version: str = "2.0.0"
    episode_active: bool = False


class ValidateRequest(BaseModel):
    action_type: str
    parameters: Dict[str, Any] = {}


# ─── App Setup ───

app = FastAPI(
    title="CloudOpsWarRoomEnv",
    description=(
        "🔥 An OpenEnv RL environment simulating a real-world SRE/DevOps "
        "incident response war room. v2 with diagnosis-gated rewards, "
        "structured feedback, and deterministic execution."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singleton environment instance — debug=False by default
env = CloudOpsWarRoomEnvironment(debug=False)


# ─── Endpoints ───

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        episode_active=not env._done,
    )


@app.post("/reset", response_model=Observation)
async def reset_environment(request: ResetRequest = ResetRequest()):
    """
    Reset the environment and start a new episode.

    Optionally specify:
      - task_id: noisy_alert, bad_deploy, cascade_failure, cost_vs_performance, fog_of_war
      - debug: true to include hidden root cause in /state response
    """
    global env
    try:
        # Re-create environment with debug flag if needed
        if request.debug != env._debug:
            env = CloudOpsWarRoomEnvironment(debug=request.debug)

        observation = env.reset(task_id=request.task_id)
        return observation
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step", response_model=StepResult)
async def step_environment(request: StepRequest):
    """
    Execute one action in the environment.

    Returns observation with structured feedback, reward, done flag, and info.
    Actions are validated before execution — missing parameters return 400.
    """
    try:
        action = Action(
            action_type=request.action_type,
            parameters=request.parameters,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid action: {str(e)}",
        )

    try:
        result = env.step(action)
        return result
    except ValueError as e:
        # Action validation error (#8)
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state", response_model=State)
async def get_state():
    """Get current episode metadata. Debug fields included if debug=True."""
    return env.state()


@app.post("/validate")
async def validate_action(request: ValidateRequest):
    """Validate an action without executing it."""
    try:
        action = Action(
            action_type=request.action_type,
            parameters=request.parameters,
        )
        result = env.validate_action(action)
        return result
    except ValueError as e:
        return {
            "valid": False,
            "errors": [str(e)],
            "action_type": request.action_type,
            "parameters": request.parameters,
        }


@app.get("/tasks")
async def list_tasks():
    """List all available incident scenarios."""
    return {
        "tasks": env.get_available_tasks(),
        "total": len(env.get_available_tasks()),
    }


# ─── Main ───

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=port,
        reload=True,
    )
