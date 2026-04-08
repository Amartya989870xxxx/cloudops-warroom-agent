"""
EnvClient for CloudOpsWarRoomEnv

Provides a typed Python client for interacting with the
CloudOpsWarRoomEnv server (remote or local).

Usage:
    from cloudops_env.client import CloudOpsClient

    client = CloudOpsClient("http://localhost:8000")
    obs = client.reset(task_id="noisy_alert")
    result = client.step(Action(action_type="check_metrics", parameters={"service": "api-gateway"}))
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import requests

from cloudops_env.models import Action, Observation, State, StepResult


class CloudOpsClient:
    """
    Synchronous HTTP client for the CloudOpsWarRoomEnv server.

    Compatible with the OpenEnv client interface pattern.
    """

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()

    def reset(self, task_id: Optional[str] = None, debug: bool = False) -> Observation:
        """Reset the environment and start a new episode."""
        payload = {"debug": debug}
        if task_id:
            payload["task_id"] = task_id
        resp = self._session.post(
            f"{self.base_url}/reset", json=payload
        )
        resp.raise_for_status()
        return Observation.model_validate(resp.json())

    def step(self, action: Action) -> StepResult:
        """Execute an action in the environment."""
        payload = {
            "action_type": action.action_type.value,
            "parameters": action.parameters,
        }
        resp = self._session.post(
            f"{self.base_url}/step", json=payload
        )
        resp.raise_for_status()
        return StepResult.model_validate(resp.json())

    def state(self) -> State:
        """Get current episode metadata."""
        resp = self._session.get(f"{self.base_url}/state")
        resp.raise_for_status()
        return State.model_validate(resp.json())

    def health(self) -> Dict[str, Any]:
        """Check server health."""
        resp = self._session.get(f"{self.base_url}/health")
        resp.raise_for_status()
        return resp.json()

    def validate(self, action: Action) -> Dict[str, Any]:
        """Validate an action without executing it."""
        payload = {
            "action_type": action.action_type.value,
            "parameters": action.parameters,
        }
        resp = self._session.post(
            f"{self.base_url}/validate", json=payload
        )
        resp.raise_for_status()
        return resp.json()

    def list_tasks(self) -> List[Dict[str, str]]:
        """List all available tasks."""
        resp = self._session.get(f"{self.base_url}/tasks")
        resp.raise_for_status()
        data = resp.json()
        return data.get("tasks", [])

    def close(self):
        """Close the HTTP session."""
        self._session.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
