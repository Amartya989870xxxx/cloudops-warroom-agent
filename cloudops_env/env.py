"""
CloudOpsWarRoomEnvironment — Core RL Environment (v2 — Enhanced)

CRITICAL FIXES:
  1. Diagnose-before-fix reward dependency
  2. Escalating wrong fix penalties
  3. Stronger step efficiency signal
  4. Completion bonus requires diagnosis
  5. Structured action feedback (success/impact/hint)
  6. Partial progress rewards for investigation
  7. Deeper task difficulty (cascade/fog of war)
  8. Action validation layer
  9. Debug mode (shows hidden state)
  10. Full determinism (seeded RNG)
  11. Action history in observation
  12. Confidence signal on diagnosis
  13. Realistic log generation with stack traces

Follows OpenEnv spec: reset() → step(action) → state()
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set

from cloudops_env.graders.reward import RewardCalculator
from cloudops_env.models import (
    Action,
    ActionFeedback,
    ActionHistoryEntry,
    ActionType,
    Alert,
    AlertSeverity,
    DeployInfo,
    FixType,
    FIX_ACTION_TYPES,
    LogEntry,
    LogLevel,
    Observation,
    PARAM_REQUIREMENTS,
    ServiceDependency,
    ServiceInfo,
    ServiceStatus,
    State,
    StepResult,
    TaskConfig,
)
from cloudops_env.tasks.registry import TaskRegistry


# ─── Log message templates for realistic log generation (#13) ───

_ERROR_TEMPLATES = {
    "high_error_rate": [
        "ERROR: Request processing failed — {error}",
        "ERROR: Unhandled exception in request handler: {error}",
        "ERROR: Service returned HTTP 500: internal server error",
        "FATAL: Connection pool exhausted — cannot serve requests",
    ],
    "high_latency": [
        "WARN: Request latency exceeded SLA threshold (p99={latency}ms)",
        "WARN: Slow query detected — query took {latency}ms",
        "WARN: Upstream dependency timeout after {latency}ms",
    ],
    "memory_issue": [
        "WARN: Memory usage at {mem}% — approaching OOM threshold",
        "ERROR: GC pressure — full GC cycle took 2.3s",
        "WARN: Heap allocation rate abnormally high",
        "ERROR: java.lang.OutOfMemoryError: Java heap space\n"
        "    at com.service.core.RequestHandler.process(RequestHandler.java:142)\n"
        "    at com.service.core.WorkerThread.run(WorkerThread.java:89)\n"
        "    at java.lang.Thread.run(Thread.java:829)",
    ],
    "deploy_bug": [
        "ERROR: NullPointerException in OrderValidator.validate()\n"
        "    at com.orders.validation.OrderValidator.validate(OrderValidator.java:87)\n"
        "    at com.orders.api.OrderController.createOrder(OrderController.java:45)\n"
        "    at org.springframework.web.servlet.FrameworkServlet.service(FrameworkServlet.java:897)",
        "ERROR: SQL query returned unexpected column count — schema mismatch\n"
        "    at com.orders.db.QueryExecutor.execute(QueryExecutor.java:203)\n"
        "    at com.orders.repository.OrderRepository.findByStatus(OrderRepository.java:67)",
        "ERROR: Feature flag evaluation failed — falling back to default\n"
        "    at com.platform.flags.FlagEvaluator.evaluate(FlagEvaluator.java:34)\n"
        "    at com.products.api.ProductController.getProduct(ProductController.java:112)",
        "WARN: New validation rule rejecting 60% of requests — check deploy v{version}",
    ],
    "cascade": [
        "ERROR: Upstream service '{upstream}' returned 503 — circuit breaker tripped\n"
        "    at com.infra.client.ServiceClient.call(ServiceClient.java:78)\n"
        "    at com.infra.client.RetryPolicy.execute(RetryPolicy.java:45)",
        "WARN: Circuit breaker OPEN for dependency '{upstream}' — all requests failing",
        "ERROR: Timeout waiting for response from '{upstream}' (30000ms exceeded)\n"
        "    at com.infra.client.HttpClient.send(HttpClient.java:156)\n"
        "    at com.service.core.DependencyResolver.resolve(DependencyResolver.java:92)",
    ],
    "healthy": [
        "INFO: Health check passed — all systems nominal",
        "INFO: Request processed successfully in {latency}ms",
        "DEBUG: Cache hit ratio: 94.2%",
        "INFO: Background job completed — 0 errors",
    ],
    "phantom": [
        "ERROR: Unexpected connection reset from upstream — possible network flap\n"
        "    at com.infra.net.ConnectionPool.acquire(ConnectionPool.java:201)",
        "WARN: DNS resolution spike — TTL expired, re-resolving endpoints",
        "ERROR: TLS handshake failure with upstream — certificate chain incomplete\n"
        "    at com.infra.tls.CertValidator.validate(CertValidator.java:89)",
    ],
}


class CloudOpsWarRoomEnvironment:
    """
    Core RL environment simulating a cloud operations war room.

    v2 Enhancements:
      - debug mode for development
      - deterministic execution (seeded RNG)
      - structured action feedback
      - action history tracking
      - action validation
      - diagnosis-gated rewards
      - wrong fix escalation tracking

    Usage:
        env = CloudOpsWarRoomEnvironment(debug=False, seed=42)
        obs = env.reset(task_id="noisy_alert")
        while True:
            action = agent.act(obs)
            result = env.step(action)
            obs = result.observation
            if result.done:
                break
    """

    def __init__(self, debug: bool = False, seed: Optional[int] = 42):
        """
        Args:
            debug: If True, state() includes hidden root cause info (#9)
            seed: Random seed for deterministic execution (#10). None for random.
        """
        self._registry = TaskRegistry()
        self._reward_calc = RewardCalculator()
        self._debug = debug
        self._seed = seed

        # Episode state — set on reset()
        self._task_config: Optional[TaskConfig] = None
        self._episode_id: str = ""
        self._step_count: int = 0
        self._done: bool = True
        self._total_reward: float = 0.0

        # World state
        self._services: Dict[str, ServiceInfo] = {}
        self._dependencies: List[ServiceDependency] = []
        self._alerts: List[Alert] = []
        self._deploys: List[DeployInfo] = []
        self._actions_taken: List[str] = []

        # Internal tracking (hidden from agent)
        self._investigated_services: Set[str] = set()
        self._diagnosed_correctly: bool = False
        self._incident_resolved: bool = False
        self._status_page_updated: bool = False
        self._stakeholder_replied: bool = False
        self._stakeholder_waiting: bool = False
        self._right_sized: bool = False
        self._wrong_fix_count: int = 0  # (#2) tracks escalating penalties
        self._last_action_result: str = "No actions taken yet."
        self._last_action_feedback: Optional[ActionFeedback] = None
        self._time_pressure: float = 1.0
        self._cost_rate: float = 1.0
        self._slo_budget: float = 1.0
        self._base_time: datetime = datetime(2024, 1, 15, 10, 0, 0)  # Fixed for determinism
        self._action_history: List[ActionHistoryEntry] = []  # (#11)
        self._last_step_reward: float = 0.0

    # ─────────────────────────────────────────
    # PUBLIC API — OpenEnv spec
    # ─────────────────────────────────────────

    def reset(self, task_id: Optional[str] = None) -> Observation:
        """
        Reset the environment to a new episode.

        Args:
            task_id: Specific task to run. If None, picks first task (deterministic).

        Returns:
            Initial observation for the new episode.
        """
        # Select task — deterministic if no task_id
        if task_id:
            self._task_config = self._registry.get_task(task_id)
        else:
            # Deterministic: always pick first task instead of random
            task_ids = self._registry.get_task_ids()
            self._task_config = self._registry.get_task(task_ids[0])

        task = self._task_config

        # Deterministic episode ID from seed + task
        if self._seed is not None:
            seed_str = f"{self._seed}_{task.task_id}"
            self._episode_id = hashlib.md5(seed_str.encode()).hexdigest()
        else:
            self._episode_id = str(uuid.uuid4())

        self._step_count = 0
        self._done = False
        self._total_reward = 0.0
        self._actions_taken = []

        # Reset world state from task config
        self._services = {s.name: s.model_copy() for s in task.services}
        self._dependencies = [d.model_copy() for d in task.dependencies]
        self._alerts = [a.model_copy() for a in task.initial_alerts]
        self._deploys = [d.model_copy() for d in task.recent_deploys]

        # Reset internal tracking
        self._investigated_services = set()
        self._diagnosed_correctly = False
        self._incident_resolved = False
        self._status_page_updated = False
        self._stakeholder_replied = False
        self._stakeholder_waiting = False
        self._right_sized = False
        self._wrong_fix_count = 0
        self._last_action_result = "Incident detected. You are now on-call. Investigate and resolve."
        self._last_action_feedback = ActionFeedback(
            success=True,
            impact="incident detected",
            hint="Start by investigating unhealthy services — check metrics and logs.",
            details="You have been paged. An active incident requires your attention.",
        )
        self._time_pressure = 1.0
        self._cost_rate = task.initial_cost_rate
        self._slo_budget = task.initial_slo_budget
        self._base_time = datetime(2024, 1, 15, 10, 0, 0)  # Fixed
        self._action_history = []
        self._last_step_reward = 0.0

        return self._build_observation()

    def step(self, action: Action) -> StepResult:
        """
        Execute one action and advance the environment by one step.

        Args:
            action: The agent's chosen action for this timestep.

        Returns:
            StepResult with observation, reward, done flag, and info.

        Raises:
            RuntimeError: If episode is done or not started.
            ValueError: If action has invalid parameters (#8).
        """
        if self._done:
            raise RuntimeError(
                "Episode is done. Call reset() to start a new episode."
            )
        if self._task_config is None:
            raise RuntimeError("No task loaded. Call reset() first.")

        # ─── Action Validation (#8) ───
        validation_errors = action.validate_params()
        if validation_errors:
            raise ValueError(
                f"Invalid action parameters: {'; '.join(validation_errors)}"
            )

        task = self._task_config
        self._step_count += 1

        # Record action
        action_str = f"{action.action_type.value}({action.parameters})"
        self._actions_taken.append(action_str)

        # Process the action — updates world state + sets feedback
        self._process_action(action)

        # Calculate reward
        reward = self._reward_calc.calculate_reward(action, self, self._step_count)
        self._total_reward += reward
        self._last_step_reward = reward

        # Record in action history (#11)
        self._action_history.append(ActionHistoryEntry(
            step=self._step_count,
            action_type=action.action_type.value,
            parameters=action.parameters,
            reward=round(reward, 4),
            success=self._last_action_feedback.success if self._last_action_feedback else False,
        ))

        # Update time pressure (decays each step)
        self._time_pressure = max(
            0.0,
            1.0 - (self._step_count / task.max_steps)
        )

        # Update SLO budget based on unhealthy services
        unhealthy = sum(
            1 for s in self._services.values()
            if s.status in (ServiceStatus.DEGRADED, ServiceStatus.DOWN)
        )
        self._slo_budget = max(0.0, self._slo_budget - 0.02 * unhealthy)

        # Check if stakeholder should start asking
        if (
            task.stakeholder_asks_at_step is not None
            and self._step_count >= task.stakeholder_asks_at_step
            and not self._stakeholder_replied
        ):
            self._stakeholder_waiting = True

        # Check termination
        self._check_done()

        # Apply completion bonus or timeout penalty (#4)
        if self._done:
            if self._incident_resolved:
                bonus = self._reward_calc.completion_bonus(
                    diagnosed=self._diagnosed_correctly
                )
                reward += bonus
                self._total_reward += bonus
            elif self._step_count >= task.max_steps:
                penalty = self._reward_calc.timeout_penalty()
                reward += penalty
                self._total_reward += penalty

        observation = self._build_observation()
        
        # Calculate normalized score for submission (#1, #2)
        # We always calculate it to keep it current in info
        normalized_score = self._reward_calc.normalize_final_score(
            self._total_reward, task
        )

        return StepResult(
            observation=observation,
            reward=round(reward, 4),
            done=self._done,
            info={
                "step_count": self._step_count,
                "total_reward": round(self._total_reward, 4),
                "normalized_score": round(normalized_score, 4),
                "incident_resolved": self._incident_resolved,
                "diagnosed_correctly": self._diagnosed_correctly,
                "task_id": task.task_id,
                "wrong_fix_count": self._wrong_fix_count,
            },
        )

    def state(self) -> State:
        """
        Return current episode metadata.
        If debug=True, includes hidden root cause info (#9).
        """
        task = self._task_config
        state = State(
            episode_id=self._episode_id,
            task_id=task.task_id if task else "",
            task_name=task.task_name if task else "",
            difficulty=task.difficulty if task else "",
            step_count=self._step_count,
            max_steps=task.max_steps if task else 0,
            done=self._done,
            total_reward=round(self._total_reward, 4),
            normalized_score=round(
                self._reward_calc.normalize_final_score(self._total_reward, task),
                4
            ) if task else 0.0,
            actions_taken=self._actions_taken.copy(),
            diagnosed_correctly=self._diagnosed_correctly,
            incident_resolved=self._incident_resolved,
            status_page_updated=self._status_page_updated,
            stakeholder_replied=self._stakeholder_replied,
            wrong_fix_count=self._wrong_fix_count,
        )

        # Debug mode (#9) — include hidden state
        if self._debug and task:
            state.debug_root_cause = task.root_cause_service
            state.debug_required_fix = task.required_fix.value
            state.debug_feature_flag = task.feature_flag_name

        return state

    # ─────────────────────────────────────────
    # ACTION PROCESSING
    # ─────────────────────────────────────────

    def _process_action(self, action: Action):
        """Process an action and update world state accordingly."""
        atype = action.action_type
        params = action.parameters

        if atype == ActionType.QUERY_LOGS:
            self._handle_query_logs(params)
        elif atype == ActionType.CHECK_METRICS:
            self._handle_check_metrics(params)
        elif atype == ActionType.TRACE_REQUEST:
            self._handle_trace_request(params)
        elif atype == ActionType.DIAGNOSE:
            self._handle_diagnose(params)
        elif atype == ActionType.RESTART_SERVICE:
            self._handle_restart(params)
        elif atype == ActionType.ROLLBACK_DEPLOY:
            self._handle_rollback(params)
        elif atype == ActionType.SCALE_SERVICE:
            self._handle_scale(params)
        elif atype == ActionType.TOGGLE_FEATURE_FLAG:
            self._handle_feature_flag(params)
        elif atype == ActionType.APPLY_RATE_LIMIT:
            self._handle_rate_limit(params)
        elif atype == ActionType.UPDATE_STATUS_PAGE:
            self._handle_status_page(params)
        elif atype == ActionType.REPLY_STAKEHOLDER:
            self._handle_stakeholder_reply(params)
        elif atype == ActionType.PAGE_ONCALL:
            self._handle_page_oncall(params)
        elif atype == ActionType.ADJUST_AUTOSCALING:
            self._handle_autoscaling(params)
        elif atype == ActionType.RIGHT_SIZE_SERVICE:
            self._handle_right_size(params)

    # ─── Investigate Handlers ───

    def _handle_query_logs(self, params: Dict[str, Any]):
        service = params.get("service", "")
        if service not in self._services:
            self._set_feedback(
                success=False,
                impact="no effect",
                hint=f"Service '{service}' not found. Check available service names.",
                details=f"Available: {', '.join(self._services.keys())}",
            )
            return

        self._investigated_services.add(service)
        svc = self._services[service]
        logs = self._generate_logs(service, svc)

        log_text = "\n".join(
            f"  [{l.level.value.upper()}] {l.message}" for l in logs[-6:]
        )

        is_root = service == self._task_config.root_cause_service
        self._set_feedback(
            success=True,
            impact="information gathered" if not is_root else "issue isolated",
            hint=self._investigation_hint(service, "logs"),
            details=f"=== Logs for {service} (last 6 entries) ===\n{log_text}",
        )

    def _handle_check_metrics(self, params: Dict[str, Any]):
        service = params.get("service", "")
        if service not in self._services:
            self._set_feedback(
                success=False,
                impact="no effect",
                hint=f"Service '{service}' not found.",
                details=f"Available: {', '.join(self._services.keys())}",
            )
            return

        self._investigated_services.add(service)
        svc = self._services[service]

        metrics_text = (
            f"=== Metrics for {service} ===\n"
            f"  Status:       {svc.status.value}\n"
            f"  CPU:          {svc.cpu_percent:.1f}%\n"
            f"  Memory:       {svc.memory_percent:.1f}%\n"
            f"  Error Rate:   {svc.error_rate:.2%}\n"
            f"  Latency p99:  {svc.latency_p99_ms:.0f}ms\n"
            f"  Request Rate: {svc.request_rate:.0f} req/s"
        )

        is_root = service == self._task_config.root_cause_service
        self._set_feedback(
            success=True,
            impact="information gathered" if not is_root else "issue isolated",
            hint=self._investigation_hint(service, "metrics"),
            details=metrics_text,
        )

    def _handle_trace_request(self, params: Dict[str, Any]):
        service = params.get("service", "")
        if service not in self._services:
            self._set_feedback(
                success=False,
                impact="no effect",
                hint=f"Service '{service}' not found.",
                details=f"Available: {', '.join(self._services.keys())}",
            )
            return

        self._investigated_services.add(service)
        task = self._task_config

        upstream = [d.source for d in self._dependencies if d.target == service]
        downstream = [d.target for d in self._dependencies if d.source == service]

        trace_lines = [f"=== Request Trace for {service} ==="]
        trace_lines.append(f"  Upstream callers: {', '.join(upstream) or 'none (entry point)'}")
        trace_lines.append(f"  Downstream deps: {', '.join(downstream) or 'none (leaf service)'}")

        for dep_name in downstream:
            if dep_name in self._services:
                dep = self._services[dep_name]
                status_icon = "✓" if dep.status == ServiceStatus.HEALTHY else "✗"
                trace_lines.append(
                    f"  {status_icon} {dep_name}: {dep.status.value} "
                    f"(err={dep.error_rate:.2%}, lat={dep.latency_p99_ms:.0f}ms)"
                )

        svc = self._services[service]
        hint = "Consider checking other services."
        if svc.status in (ServiceStatus.DEGRADED, ServiceStatus.DOWN):
            if service == task.root_cause_service:
                trace_lines.append(
                    f"  ⚠ Trace shows errors originating FROM {service} — not from upstream"
                )
                hint = f"Errors originate from {service} itself — this could be the root cause."
            else:
                bad_upstream = [
                    d.source for d in self._dependencies
                    if d.target == service
                    and self._services.get(d.source, ServiceInfo(name="")).status
                    in (ServiceStatus.DEGRADED, ServiceStatus.DOWN)
                ]
                if bad_upstream:
                    trace_lines.append(
                        f"  ⚠ Errors appear to cascade FROM upstream: {', '.join(bad_upstream)}"
                    )
                    hint = f"This service's issues likely cascade from: {', '.join(bad_upstream)}"

        self._set_feedback(
            success=True,
            impact="dependency map revealed",
            hint=hint,
            details="\n".join(trace_lines),
        )

    # ─── Diagnose Handler (#12 — confidence signal) ───

    def _handle_diagnose(self, params: Dict[str, Any]):
        diagnosed = params.get("root_cause_service", "")
        task = self._task_config

        if diagnosed == task.root_cause_service:
            self._diagnosed_correctly = True
            self._set_feedback(
                success=True,
                impact="root cause confirmed",
                hint=f"Correct! Apply the fix: {task.required_fix.value}",
                details=(
                    f"✓ DIAGNOSIS CONFIRMED: {diagnosed} identified as root cause. "
                    f"Recommended action: {task.required_fix.value}"
                ),
            )
        else:
            # Wrong diagnosis — give a directional hint (#12)
            # Check if diagnosed service is related to root cause
            is_downstream = any(
                d.source == task.root_cause_service and d.target == diagnosed
                for d in self._dependencies
            )
            if is_downstream:
                hint = f"{diagnosed} is affected, but the issue originates upstream. Trace the dependency chain."
            elif diagnosed in self._services:
                hint = f"{diagnosed} is not the root cause. Look at services with the highest error rates or most dependencies."
            else:
                hint = "That service doesn't exist. Check the service list."

            self._set_feedback(
                success=False,
                impact="incorrect diagnosis",
                hint=hint,
                details=f"✗ Diagnosis '{diagnosed}' does not match the root cause. Continue investigating.",
            )

    # ─── Fix Handlers (with diagnosis gate tracking) ───

    def _handle_restart(self, params: Dict[str, Any]):
        service = params.get("service", "")
        task = self._task_config

        if service not in self._services:
            self._set_feedback(
                success=False, impact="no effect",
                hint=f"Service '{service}' not found.",
                details=f"Cannot restart — service '{service}' not found.",
            )
            return

        if service == task.root_cause_service and task.required_fix == FixType.RESTART:
            self._fix_root_cause()
            self._set_feedback(
                success=True, impact="issue resolved",
                hint="Service recovered. Consider updating the status page.",
                details=f"✓ {service} restarted successfully. Service recovering — error rates dropping.",
            )
        else:
            self._wrong_fix_count += 1
            self._set_feedback(
                success=False, impact="no effect",
                hint="This service may not be the root cause, or a restart isn't the right fix.",
                details=f"⚠ {service} restarted, but the issue persists.",
            )

    def _handle_rollback(self, params: Dict[str, Any]):
        service = params.get("service", "")
        task = self._task_config

        if service not in self._services:
            self._set_feedback(
                success=False, impact="no effect",
                hint=f"Service '{service}' not found.",
                details=f"Cannot rollback — service '{service}' not found.",
            )
            return

        has_deploy = any(d.service == service for d in self._deploys)
        if not has_deploy:
            self._set_feedback(
                success=False, impact="no effect",
                hint="Check recent_deploys for services that were recently deployed.",
                details=f"⚠ No recent deploy found for {service}. Nothing to rollback.",
            )
            return

        if service == task.root_cause_service and task.required_fix == FixType.ROLLBACK:
            self._fix_root_cause()
            self._set_feedback(
                success=True, impact="issue resolved",
                hint="Bad deploy reverted. Consider updating the status page.",
                details=f"✓ {service} rolled back to previous version. System stabilizing.",
            )
        else:
            self._wrong_fix_count += 1
            self._set_feedback(
                success=False, impact="no effect",
                hint="The deploy may not be the cause. Check if the timeline correlates.",
                details=f"⚠ {service} rolled back, but the incident continues.",
            )

    def _handle_scale(self, params: Dict[str, Any]):
        service = params.get("service", "")
        direction = params.get("direction", "up")

        if service not in self._services:
            self._set_feedback(
                success=False, impact="no effect",
                hint=f"Service '{service}' not found.",
                details=f"Cannot scale — service '{service}' not found.",
            )
            return

        task = self._task_config
        svc = self._services[service]

        if service == task.root_cause_service and task.required_fix == FixType.SCALE:
            self._fix_root_cause()
            self._set_feedback(
                success=True, impact="issue resolved",
                hint="Scaling resolved the performance issue.",
                details=f"✓ {service} scaled {direction}. Issue resolved.",
            )
        else:
            if direction == "up":
                svc.cpu_percent = max(10.0, svc.cpu_percent * 0.7)
                svc.memory_percent = max(10.0, svc.memory_percent * 0.7)
                svc.latency_p99_ms = max(20.0, svc.latency_p99_ms * 0.8)
                self._cost_rate += 0.3
                self._set_feedback(
                    success=False, impact="partial — cost increased",
                    hint="Scaling may mask symptoms but won't fix the root cause.",
                    details=f"↑ {service} scaled up. Cost increased to ${self._cost_rate:.2f}/min.",
                )
            else:
                svc.cpu_percent = min(95.0, svc.cpu_percent * 1.3)
                svc.memory_percent = min(95.0, svc.memory_percent * 1.2)
                self._cost_rate = max(0.5, self._cost_rate - 0.3)
                self._set_feedback(
                    success=False, impact="partial — cost reduced",
                    hint="Scaling down during an incident is risky.",
                    details=f"↓ {service} scaled down. Cost reduced to ${self._cost_rate:.2f}/min.",
                )
            self._wrong_fix_count += 1

    def _handle_feature_flag(self, params: Dict[str, Any]):
        flag = params.get("flag_name", "")
        task = self._task_config

        if task.required_fix == FixType.FEATURE_FLAG and flag == task.feature_flag_name:
            self._fix_root_cause()
            self._set_feedback(
                success=True, impact="issue resolved",
                hint="Buggy feature disabled. Consider updating the status page.",
                details=f"✓ Feature flag '{flag}' toggled OFF. Errors clearing.",
            )
        else:
            self._wrong_fix_count += 1
            self._set_feedback(
                success=False, impact="no effect",
                hint="This feature flag doesn't affect the incident.",
                details=f"⚠ Feature flag '{flag}' toggled — no effect on the incident.",
            )

    def _handle_rate_limit(self, params: Dict[str, Any]):
        service = params.get("service", "")
        task = self._task_config

        if service not in self._services:
            self._set_feedback(
                success=False, impact="no effect",
                hint=f"Service '{service}' not found.",
                details=f"Cannot rate-limit — service '{service}' not found.",
            )
            return

        svc = self._services[service]
        svc.request_rate = max(10.0, svc.request_rate * 0.5)
        svc.error_rate = max(0.01, svc.error_rate * 0.7)

        if service == task.root_cause_service and task.required_fix == FixType.RATE_LIMIT:
            self._fix_root_cause()
            self._set_feedback(
                success=True, impact="issue resolved",
                hint="Rate limiting resolved the issue.",
                details=f"✓ Rate limiting applied to {service}. Errors stabilizing.",
            )
        else:
            self._wrong_fix_count += 1
            self._set_feedback(
                success=False, impact="partial — traffic reduced",
                hint="Rate limiting won't fix the root cause.",
                details=f"⚠ Rate limiting applied to {service}. Traffic reduced but root cause not addressed.",
            )

    # ─── Communicate Handlers ───

    def _handle_status_page(self, params: Dict[str, Any]):
        message = params.get("message", "Investigating an ongoing incident.")
        self._status_page_updated = True
        self._set_feedback(
            success=True, impact="stakeholders notified",
            hint="Good communication. Now focus on fixing the issue.",
            details=f"✓ Status page updated: \"{message}\"\n  Customers notified.",
        )

    def _handle_stakeholder_reply(self, params: Dict[str, Any]):
        message = params.get("message", "We are investigating the issue.")
        if self._stakeholder_waiting:
            self._stakeholder_replied = True
            self._stakeholder_waiting = False
            self._set_feedback(
                success=True, impact="executive updated",
                hint="Stakeholder acknowledged. Focus on the technical fix.",
                details=f"✓ Stakeholder update sent: \"{message}\"",
            )
        else:
            self._set_feedback(
                success=False, impact="no effect",
                hint="No stakeholder is currently waiting. Focus on investigation.",
                details="⚠ No stakeholder is currently waiting for an update.",
            )

    def _handle_page_oncall(self, params: Dict[str, Any]):
        service = params.get("service", "")
        self._set_feedback(
            success=True, impact="help requested",
            hint="On-call engineer will assist. Continue investigating.",
            details=f"📟 Paged on-call engineer for {service}.",
        )

    # ─── Optimize Handlers ───

    def _handle_autoscaling(self, params: Dict[str, Any]):
        service = params.get("service", "")
        if service not in self._services:
            self._set_feedback(
                success=False, impact="no effect",
                hint=f"Service '{service}' not found.",
                details=f"Cannot adjust — service '{service}' not found.",
            )
            return
        self._set_feedback(
            success=True, impact="policy updated (delayed)",
            hint="Autoscaling changes take time. Focus on immediate fixes.",
            details=f"⚙ Autoscaling policy updated for {service}.",
        )

    def _handle_right_size(self, params: Dict[str, Any]):
        service = params.get("service", "")
        task = self._task_config

        if service not in self._services:
            self._set_feedback(
                success=False, impact="no effect",
                hint=f"Service '{service}' not found.",
                details=f"Cannot right-size — service '{service}' not found.",
            )
            return

        if service == task.overprovisioned_service:
            self._right_sized = True
            svc = self._services[service]
            svc.cpu_percent = min(50.0, svc.cpu_percent + 20.0)
            svc.memory_percent = min(60.0, svc.memory_percent + 15.0)
            self._cost_rate = max(0.5, self._cost_rate * 0.6)
            self._set_feedback(
                success=True, impact="cost optimized",
                hint="Good cost optimization. Now fix the reliability issue.",
                details=f"✓ {service} right-sized. Cost dropped to ${self._cost_rate:.2f}/min.",
            )
        else:
            self._set_feedback(
                success=False, impact="no effect",
                hint=f"{service} is already appropriately sized. Look for overprovisioned services.",
                details=f"⚠ {service} is already appropriately sized.",
            )

    # ─────────────────────────────────────────
    # INTERNAL HELPERS
    # ─────────────────────────────────────────

    def _set_feedback(self, success: bool, impact: str, hint: str, details: str):
        """Set structured action feedback (#5)."""
        self._last_action_feedback = ActionFeedback(
            success=success,
            impact=impact,
            hint=hint,
            details=details,
        )
        self._last_action_result = details

    def _investigation_hint(self, service: str, method: str) -> str:
        """Generate a contextual hint after investigation (#5)."""
        task = self._task_config
        svc = self._services.get(service)
        if not svc:
            return "Service not found."

        if service == task.root_cause_service:
            if svc.error_rate > 0.3:
                return f"Errors are isolated to {service} — this is highly likely the root cause."
            if svc.memory_percent > 85:
                return f"Internal resource exhaustion detected on {service}; investigate local processes."
            if svc.latency_p99_ms > 1000:
                return f"Extreme latency originates from {service} itself; check its resource limits."
            return f"{service} is the bottleneck — investigation confirms local failure."
        elif svc.status in (ServiceStatus.DEGRADED, ServiceStatus.DOWN):
            # Check if cascade victim (Directional Hint — UPSTREAM)
            bad_upstream = [
                d.source for d in self._dependencies
                if d.target == service
                and self._services.get(d.source, ServiceInfo(name="")).status
                in (ServiceStatus.DEGRADED, ServiceStatus.DOWN)
            ]
            if bad_upstream:
                return f"{service} is affected by upstream failures from: {', '.join(bad_upstream)}. Investigate upstream dependencies."
            
            # Check if affecting downstream (Directional Hint — DOWNSTREAM)
            bad_downstream = [
                d.target for d in self._dependencies
                if d.source == service
                and self._services.get(d.target, ServiceInfo(name="")).status
                in (ServiceStatus.DEGRADED, ServiceStatus.DOWN)
            ]
            if bad_downstream:
                return f"{service} is unhealthy and affecting downstream: {', '.join(bad_downstream)}. Focus here or check its own dependencies."
                
            return f"{service} is unhealthy — trace its dependencies to find the failure origin."
        else:
            return f"{service} appears healthy and is not propagating errors. Focus on services with high error rates."

    def _fix_root_cause(self):
        """Apply the correct fix and propagate healing."""
        task = self._task_config
        rc_service = task.root_cause_service

        if rc_service in self._services:
            svc = self._services[rc_service]
            svc.status = ServiceStatus.HEALTHY
            svc.error_rate = 0.02
            svc.latency_p99_ms = max(30.0, svc.latency_p99_ms * 0.1)
            svc.cpu_percent = max(15.0, svc.cpu_percent * 0.4)
            svc.memory_percent = max(20.0, svc.memory_percent * 0.5)

        self._propagate_healing(rc_service, visited=set())
        self._incident_resolved = True

    def _propagate_healing(self, service: str, visited: Set[str]):
        """Recursively heal services that depend on a fixed service."""
        if service in visited:
            return
        visited.add(service)

        dependents = set()
        for dep in self._dependencies:
            if dep.target == service:
                dependents.add(dep.source)
            if dep.source == service:
                dependents.add(dep.target)

        for dep_name in dependents:
            if dep_name in self._services and dep_name not in visited:
                dep_svc = self._services[dep_name]
                if dep_svc.status in (ServiceStatus.DEGRADED, ServiceStatus.DOWN):
                    dep_svc.status = ServiceStatus.HEALTHY
                    dep_svc.error_rate = max(0.01, dep_svc.error_rate * 0.2)
                    dep_svc.latency_p99_ms = max(30.0, dep_svc.latency_p99_ms * 0.3)
                    dep_svc.cpu_percent = max(15.0, dep_svc.cpu_percent * 0.5)
                    self._propagate_healing(dep_name, visited)

    def _check_done(self):
        """Check if the episode should end."""
        task = self._task_config
        if self._incident_resolved:
            self._done = True
            return
        if self._step_count >= task.max_steps:
            self._done = True
            return

    def _build_observation(self) -> Observation:
        """Construct the current observation for the agent."""
        task = self._task_config

        # Generate deterministic logs (#10)
        all_logs = []
        for name in sorted(self._services.keys()):  # sorted for determinism
            svc = self._services[name]
            all_logs.extend(self._generate_logs(name, svc))

        all_logs.sort(key=lambda l: l.timestamp, reverse=True)
        recent_logs = all_logs[:20]

        # Action history — last 10 actions (#11)
        history = self._action_history[-10:]

        return Observation(
            services=list(self._services.values()),
            dependencies=self._dependencies.copy(),
            active_alerts=self._alerts.copy(),
            recent_deploys=self._deploys.copy(),
            logs=recent_logs,
            time_pressure=round(self._time_pressure, 3),
            cost_rate=round(self._cost_rate, 2),
            slo_budget=round(self._slo_budget, 3),
            stakeholder_flag=self._stakeholder_waiting,
            last_action_result=self._last_action_result,
            last_action_feedback=self._last_action_feedback,
            investigated_services=sorted(list(self._investigated_services)),
            action_history=history,
        )

    def _generate_logs(
        self, service: str, svc: ServiceInfo
    ) -> List[LogEntry]:
        """Generate deterministic, realistic log entries (#10, #13)."""
        task = self._task_config
        logs: List[LogEntry] = []
        base = self._base_time + timedelta(seconds=self._step_count * 30)

        if svc.status == ServiceStatus.DOWN:
            templates = _ERROR_TEMPLATES["high_error_rate"]
            for i, tmpl in enumerate(templates[:3]):
                logs.append(LogEntry(
                    timestamp=(base + timedelta(seconds=i)).isoformat(),
                    service=service,
                    level=LogLevel.ERROR,
                    message=tmpl.format(
                        error="connection refused",
                        latency=int(svc.latency_p99_ms),
                    ),
                ))

        elif svc.status == ServiceStatus.DEGRADED:
            is_cascade_victim = service != task.root_cause_service and any(
                d.target == service
                and self._services.get(d.source, ServiceInfo(name="")).status
                in (ServiceStatus.DEGRADED, ServiceStatus.DOWN)
                for d in self._dependencies
            )
            # Check phantom root cause (#7)
            is_phantom = (
                task.phantom_root_cause is not None
                and service == task.phantom_root_cause
            )

            if is_phantom:
                # Phantom incident — realistic but misleading logs (#7)
                templates = _ERROR_TEMPLATES["phantom"]
                for tmpl in templates[:2]:
                    logs.append(LogEntry(
                        timestamp=base.isoformat(),
                        service=service,
                        level=LogLevel.ERROR,
                        message=tmpl,
                    ))
            elif is_cascade_victim:
                upstream_bad = [
                    d.source for d in self._dependencies
                    if d.target == service
                    and self._services.get(d.source, ServiceInfo(name="")).status
                    in (ServiceStatus.DEGRADED, ServiceStatus.DOWN)
                ]
                templates = _ERROR_TEMPLATES["cascade"]
                for tmpl in templates[:2]:
                    logs.append(LogEntry(
                        timestamp=base.isoformat(),
                        service=service,
                        level=LogLevel.ERROR,
                        message=tmpl.format(
                            upstream=upstream_bad[0] if upstream_bad else "unknown"
                        ),
                    ))
            elif service == task.root_cause_service:
                # Root cause — show specific error patterns with stack traces (#13)
                if task.required_fix == FixType.ROLLBACK:
                    templates = _ERROR_TEMPLATES["deploy_bug"]
                elif task.required_fix == FixType.RESTART:
                    templates = _ERROR_TEMPLATES["memory_issue"]
                elif task.required_fix == FixType.FEATURE_FLAG:
                    templates = _ERROR_TEMPLATES["deploy_bug"]
                else:
                    templates = _ERROR_TEMPLATES["high_error_rate"]

                for tmpl in templates[:3]:
                    logs.append(LogEntry(
                        timestamp=base.isoformat(),
                        service=service,
                        level=LogLevel.ERROR,
                        message=tmpl.format(
                            error="internal error",
                            latency=int(svc.latency_p99_ms),
                            mem=int(svc.memory_percent),
                            version=task.recent_deploys[0].version if task.recent_deploys else "unknown",
                        ),
                    ))

            # Add latency warning if high
            if svc.latency_p99_ms > 200:
                templates = _ERROR_TEMPLATES["high_latency"]
                logs.append(LogEntry(
                    timestamp=base.isoformat(),
                    service=service,
                    level=LogLevel.WARN,
                    message=templates[0].format(latency=int(svc.latency_p99_ms)),
                ))

        else:
            # Healthy — deterministic healthy log
            # Use step_count to deterministically select log template (#10)
            templates = _ERROR_TEMPLATES["healthy"]
            idx = (self._step_count + hash(service)) % len(templates)
            logs.append(LogEntry(
                timestamp=base.isoformat(),
                service=service,
                level=LogLevel.INFO,
                message=templates[idx].format(latency=int(svc.latency_p99_ms)),
            ))

        return logs

    # ─────────────────────────────────────────
    # UTILITY
    # ─────────────────────────────────────────

    def get_available_tasks(self) -> List[Dict[str, str]]:
        """List all available tasks."""
        return self._registry.list_tasks()

    def validate_action(self, action: Action) -> Dict[str, Any]:
        """Validate an action without executing it (#8)."""
        errors = action.validate_params()

        # Check service exists if specified
        service = action.parameters.get("service") or action.parameters.get(
            "root_cause_service"
        )
        if service and service not in self._services and not self._done:
            errors.append(
                f"Unknown service: '{service}'. "
                f"Available: {sorted(list(self._services.keys()))}"
            )

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "action_type": action.action_type.value,
            "parameters": action.parameters,
        }
