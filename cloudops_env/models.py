"""
CloudOpsWarRoomEnv — Pydantic Models (v2 — Enhanced)

Defines all typed data structures for the RL environment:
- Observation space (what the agent sees)
- Action space (what the agent can do)
- State (episode metadata)
- StepResult (return from env.step())
- ActionFeedback (structured feedback from each action)

v2 Changes:
  - ActionFeedback model for structured last_action_result
  - action_history in Observation (last N actions)
  - Debug fields in State (hidden root cause when debug=True)
  - Action validation via PARAM_REQUIREMENTS
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────

class ServiceStatus(str, Enum):
    """Health status of a microservice."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    DOWN = "down"
    OVERLOADED = "overloaded"


class AlertSeverity(str, Enum):
    """Severity levels for alerts."""
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


class LogLevel(str, Enum):
    """Log severity levels."""
    ERROR = "error"
    WARN = "warn"
    INFO = "info"
    DEBUG = "debug"


class ActionType(str, Enum):
    """All possible agent actions — one per step."""
    # Investigate
    QUERY_LOGS = "query_logs"
    CHECK_METRICS = "check_metrics"
    TRACE_REQUEST = "trace_request"
    # Diagnose
    DIAGNOSE = "diagnose"
    # Fix
    RESTART_SERVICE = "restart_service"
    ROLLBACK_DEPLOY = "rollback_deploy"
    SCALE_SERVICE = "scale_service"
    TOGGLE_FEATURE_FLAG = "toggle_feature_flag"
    APPLY_RATE_LIMIT = "apply_rate_limit"
    # Communicate
    UPDATE_STATUS_PAGE = "update_status_page"
    REPLY_STAKEHOLDER = "reply_stakeholder"
    PAGE_ONCALL = "page_oncall"
    # Optimize
    ADJUST_AUTOSCALING = "adjust_autoscaling"
    RIGHT_SIZE_SERVICE = "right_size_service"


class FixType(str, Enum):
    """The type of fix required for the root cause."""
    RESTART = "restart"
    ROLLBACK = "rollback"
    FEATURE_FLAG = "feature_flag"
    SCALE = "scale"
    RATE_LIMIT = "rate_limit"


# ─────────────────────────────────────────────
# Action Parameter Validation
# ─────────────────────────────────────────────

# Required parameters per action type — used for validation
PARAM_REQUIREMENTS: Dict[str, List[str]] = {
    "query_logs": ["service"],
    "check_metrics": ["service"],
    "trace_request": ["service"],
    "diagnose": ["root_cause_service"],
    "restart_service": ["service"],
    "rollback_deploy": ["service"],
    "scale_service": ["service", "direction"],
    "toggle_feature_flag": ["flag_name"],
    "apply_rate_limit": ["service"],
    "update_status_page": ["message"],
    "reply_stakeholder": ["message"],
    "page_oncall": ["service"],
    "adjust_autoscaling": ["service"],
    "right_size_service": ["service"],
}

# Fix action types — used to detect blind fixing behavior
FIX_ACTION_TYPES = {
    ActionType.RESTART_SERVICE,
    ActionType.ROLLBACK_DEPLOY,
    ActionType.SCALE_SERVICE,
    ActionType.TOGGLE_FEATURE_FLAG,
    ActionType.APPLY_RATE_LIMIT,
}


# ─────────────────────────────────────────────
# Observation Components
# ─────────────────────────────────────────────

class ServiceInfo(BaseModel):
    """A single microservice node in the system."""
    name: str
    status: ServiceStatus = ServiceStatus.HEALTHY
    cpu_percent: float = Field(default=25.0, ge=0, le=100)
    memory_percent: float = Field(default=40.0, ge=0, le=100)
    error_rate: float = Field(default=0.01, ge=0, le=1.0)
    latency_p99_ms: float = Field(default=50.0, ge=0)
    request_rate: float = Field(default=100.0, ge=0)


class ServiceDependency(BaseModel):
    """Directed edge in the microservice dependency graph."""
    source: str
    target: str


class Alert(BaseModel):
    """An active alert — may be real or noisy."""
    alert_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    severity: AlertSeverity
    service: str
    message: str
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat()
    )
    # Hidden from agent — used internally for grading
    is_noise: bool = Field(default=False, exclude=True)


class DeployInfo(BaseModel):
    """A recent deployment record."""
    deploy_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    service: str
    timestamp: str
    change_summary: str
    version: str = "v1.0.0"
    # Hidden from agent — used internally
    is_buggy: bool = Field(default=False, exclude=True)


class LogEntry(BaseModel):
    """A single log line from a service."""
    timestamp: str
    service: str
    level: LogLevel
    message: str


# ─────────────────────────────────────────────
# Structured Action Feedback (Enhancement #5)
# ─────────────────────────────────────────────

class ActionFeedback(BaseModel):
    """
    Structured feedback from the last action taken.

    Replaces the old plain string `last_action_result`.
    Provides success/impact/hint fields to help LLM reasoning.
    """
    success: bool = False
    impact: str = "no effect"
    hint: str = ""
    details: str = ""


# ─────────────────────────────────────────────
# Action History Entry (Enhancement #11)
# ─────────────────────────────────────────────

class ActionHistoryEntry(BaseModel):
    """A record of a previously taken action and its outcome."""
    step: int
    action_type: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    reward: float = 0.0
    success: bool = False


# ─────────────────────────────────────────────
# Observation (what the agent sees each step)
# ─────────────────────────────────────────────

class Observation(BaseModel):
    """
    Full observation returned to the agent at each step.

    IMPORTANT: Root cause is NEVER directly visible here.
    Some alerts may be noisy/misleading.
    Agent must infer the real issue from signals.
    """
    services: List[ServiceInfo] = Field(
        default_factory=list,
        description="Current state of all microservices"
    )
    dependencies: List[ServiceDependency] = Field(
        default_factory=list,
        description="Service dependency graph edges"
    )
    active_alerts: List[Alert] = Field(
        default_factory=list,
        description="Currently firing alerts (may include noise)"
    )
    recent_deploys: List[DeployInfo] = Field(
        default_factory=list,
        description="Recent deployment records"
    )
    logs: List[LogEntry] = Field(
        default_factory=list,
        description="Recent log lines across services"
    )
    time_pressure: float = Field(
        default=1.0, ge=0, le=1.0,
        description="Urgency — decays each step (1.0=fresh, 0.0=expired)"
    )
    cost_rate: float = Field(
        default=1.0, ge=0,
        description="Current infrastructure cost rate ($/min)"
    )
    slo_budget: float = Field(
        default=1.0, ge=0, le=1.0,
        description="Remaining SLO error budget (1.0=full, 0.0=breached)"
    )
    stakeholder_flag: bool = Field(
        default=False,
        description="True if an executive is asking for an update"
    )
    last_action_result: str = Field(
        default="No actions taken yet.",
        description="Human-readable feedback from the previous action"
    )
    last_action_feedback: Optional[ActionFeedback] = Field(
        default=None,
        description="Structured feedback from the previous action"
    )
    investigated_services: List[str] = Field(
        default_factory=list,
        description="Services the agent has already investigated"
    )
    action_history: List[ActionHistoryEntry] = Field(
        default_factory=list,
        description="Last N actions taken (for avoiding repetition)"
    )


# ─────────────────────────────────────────────
# Action (what the agent does each step)
# ─────────────────────────────────────────────

class Action(BaseModel):
    """
    A single agent action per timestep.
    action_type determines which parameters are required.
    """
    action_type: ActionType
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Parameters depend on action_type. Examples:\n"
            "  query_logs: {service: str}\n"
            "  check_metrics: {service: str}\n"
            "  trace_request: {service: str}\n"
            "  diagnose: {root_cause_service: str}\n"
            "  restart_service: {service: str}\n"
            "  rollback_deploy: {service: str}\n"
            "  scale_service: {service: str, direction: 'up'|'down'}\n"
            "  toggle_feature_flag: {flag_name: str}\n"
            "  apply_rate_limit: {service: str}\n"
            "  update_status_page: {message: str}\n"
            "  reply_stakeholder: {message: str}\n"
            "  page_oncall: {service: str}\n"
            "  adjust_autoscaling: {service: str}\n"
            "  right_size_service: {service: str}\n"
        )
    )

    def validate_params(self) -> List[str]:
        """Validate that required parameters are present. Returns list of errors."""
        errors = []
        required = PARAM_REQUIREMENTS.get(self.action_type.value, [])
        for param in required:
            if param not in self.parameters or self.parameters[param] == "":
                errors.append(
                    f"Missing required parameter '{param}' for action '{self.action_type.value}'"
                )
        # Validate direction enum for scale_service
        if self.action_type == ActionType.SCALE_SERVICE:
            direction = self.parameters.get("direction", "")
            if direction not in ("up", "down"):
                errors.append(
                    f"Invalid direction '{direction}' — must be 'up' or 'down'"
                )
        return errors


# ─────────────────────────────────────────────
# State (episode metadata)
# ─────────────────────────────────────────────

class State(BaseModel):
    """Episode-level metadata, accessible via env.state()."""
    episode_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str = ""
    task_name: str = ""
    difficulty: str = ""
    step_count: int = 0
    max_steps: int = 20
    done: bool = False
    total_reward: float = 0.0
    normalized_score: float = 0.0
    actions_taken: List[str] = Field(default_factory=list)
    diagnosed_correctly: bool = False
    incident_resolved: bool = False
    status_page_updated: bool = False
    stakeholder_replied: bool = False
    wrong_fix_count: int = 0
    # Debug fields — only populated when debug=True
    debug_root_cause: Optional[str] = Field(
        default=None,
        description="Hidden root cause service (only in debug mode)"
    )
    debug_required_fix: Optional[str] = Field(
        default=None,
        description="Required fix type (only in debug mode)"
    )
    debug_feature_flag: Optional[str] = Field(
        default=None,
        description="Feature flag name if relevant (only in debug mode)"
    )


# ─────────────────────────────────────────────
# StepResult (return from env.step())
# ─────────────────────────────────────────────

class StepResult(BaseModel):
    """Result returned by env.step(action)."""
    observation: Observation
    reward: float = 0.0
    done: bool = False
    info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (step_count, total_reward, etc.)"
    )


# ─────────────────────────────────────────────
# Task Definition (used by task registry)
# ─────────────────────────────────────────────

class TaskConfig(BaseModel):
    """Configuration for a single incident scenario."""
    task_id: str
    task_name: str
    difficulty: str
    description: str
    max_steps: int = 20
    services: List[ServiceInfo] = Field(default_factory=list)
    dependencies: List[ServiceDependency] = Field(default_factory=list)
    initial_alerts: List[Alert] = Field(default_factory=list)
    recent_deploys: List[DeployInfo] = Field(default_factory=list)
    root_cause_service: str = ""
    root_cause_description: str = ""
    required_fix: FixType = FixType.RESTART
    feature_flag_name: Optional[str] = None
    initial_cost_rate: float = 1.0
    initial_slo_budget: float = 1.0
    stakeholder_asks_at_step: Optional[int] = None
    overprovisioned_service: Optional[str] = None
    optimal_steps: int = 5
    max_reward: float = 1.0  # Theoretical max for optimal path
    min_reward: float = -0.5 # Theoretical min for worst-case path
    # Phantom incident for Fog of War (Enhancement #7)
    phantom_root_cause: Optional[str] = Field(
        default=None,
        description="A fake root cause service that produces realistic but misleading signals"
    )
