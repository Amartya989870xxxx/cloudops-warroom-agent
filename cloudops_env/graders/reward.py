"""
Dense Reward Calculator for CloudOpsWarRoomEnv (v2 — Enhanced)

CRITICAL FIXES implemented:
  1. Diagnose-before-fix dependency — correct fix without diagnosis → tiny reward
  2. Escalating penalties for repeated wrong fixes
  3. Stronger per-step efficiency penalty
  4. Completion bonus requires BOTH resolved + diagnosed
  5. Partial progress rewards for useful investigation
  6. Confidence signal for diagnosis (correct → big reward, wrong → penalty + hint)

Reward table (with diagnosis gate):
  +0.25  correct diagnosis
  +0.20  correct restart     (ONLY if diagnosed_correctly, else +0.02)
  +0.30  correct rollback    (ONLY if diagnosed_correctly, else +0.02)
  +0.35  correct feature flag (ONLY if diagnosed_correctly, else +0.02)
  +0.05  status page update
  +0.08  stakeholder reply (when flagged)
  +0.30  completion bonus (ONLY if diagnosed AND resolved)
  +0.15  partial completion (resolved without diagnosis)
  +0.03  useful investigation (new service, first time)
  +0.06  investigating root cause service
  +0.05  identifying correct service via trace
  +0.10  right-sizing overprovisioned service

  -0.10  wrong restart (1st time)
  -0.15  wrong rollback (1st time)
  -0.10  wrong diagnosis
  -0.05  useless scaling
  -0.02  per-step base penalty (step tax)
  -0.015 × unhealthy services per step
  -0.20  timeout (max steps)
  -0.03  redundant action
  Escalating: wrong fix #2 → -0.15, #3+ → -0.20
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cloudops_env.models import (
    Action,
    ActionType,
    FixType,
    ServiceStatus,
    FIX_ACTION_TYPES,
)

if TYPE_CHECKING:
    from cloudops_env.env import CloudOpsWarRoomEnvironment


class RewardCalculator:
    """Calculates dense rewards with diagnosis-gated fix rewards."""

    # ─── Positive rewards ───
    CORRECT_DIAGNOSIS = 0.30        # Increased from 0.25
    CORRECT_RESTART = 0.20
    CORRECT_ROLLBACK = 0.30
    CORRECT_FEATURE_FLAG = 0.35
    CORRECT_RATE_LIMIT = 0.15
    CORRECT_SCALE_FIX = 0.20
    STATUS_PAGE_UPDATE = 0.05
    STAKEHOLDER_REPLY = 0.08
    FULL_COMPLETION_BONUS = 0.40    # Increased from 0.30 (resolved + diagnosed)
    PARTIAL_COMPLETION_BONUS = 0.15  # resolved without diagnosis
    USEFUL_INVESTIGATION = 0.03
    ROOT_CAUSE_INVESTIGATION = 0.05  # standardized to +0.05
    PARTIAL_IDENTIFICATION_BONUS = 0.10 # [NEW] Identifying root cause service
    TRACE_IDENTIFIES_ROOT = 0.05     # trace points to root cause
    RIGHT_SIZE_BONUS = 0.10

    # ─── Negative rewards ───
    WRONG_RESTART_BASE = -0.15      # Increased penalty from -0.10
    WRONG_ROLLBACK_BASE = -0.20     # Increased penalty from -0.15
    WRONG_DIAGNOSIS = -0.10
    USELESS_SCALING = -0.05
    STEP_TAX = -0.02               # standard efficiency tax
    GRACE_TAX = -0.005             # [NEW] reduced tax for early exploration
    GRACE_PERIOD_STEPS = 4         # [NEW] Set to 4 as a compromise
    PER_UNHEALTHY_PENALTY = -0.015
    TIMEOUT_PENALTY = -0.20
    REDUNDANT_ACTION = -0.03
    NOOP_PENALTY = -0.02

    # ─── Diagnosis gate ───
    # If agent applies correct fix WITHOUT prior diagnosis, reward is tiny
    UNDIAGNOSED_FIX_REWARD = 0.02

    # ─── Escalating wrong fix penalties ───
    WRONG_FIX_ESCALATION = -0.05   # added per repeated wrong fix

    def calculate_reward(
        self,
        action: Action,
        env: "CloudOpsWarRoomEnvironment",
        step_count: int,
    ) -> float:
        """
        Calculate the reward for a given action in the current env state.
        Now includes a grace period for early exploration.
        """
        reward = 0.0
        task = env._task_config
        is_grace = step_count <= self.GRACE_PERIOD_STEPS

        # ─── Step tax (efficiency signal — #3) ───
        reward += self.GRACE_TAX if is_grace else self.STEP_TAX

        # ─── Per-step penalty for unhealthy services (Reduced during grace) ───
        unhealthy_count = sum(
            1 for s in env._services.values()
            if s.status in (ServiceStatus.DEGRADED, ServiceStatus.DOWN)
        )
        unhealthy_tax = self.PER_UNHEALTHY_PENALTY * unhealthy_count
        reward += (unhealthy_tax * 0.1) if is_grace else unhealthy_tax

        # ─── Action-specific rewards ───
        action_type = action.action_type
        params = action.parameters

        # === INVESTIGATE actions (with partial progress — #6) ===
        if action_type in (
            ActionType.QUERY_LOGS,
            ActionType.CHECK_METRICS,
            ActionType.TRACE_REQUEST,
        ):
            target_service = params.get("service", "")
            if target_service in env._services:
                if target_service not in env._investigated_services:
                    # First investigation of this service — useful
                    if target_service == task.root_cause_service:
                        # PARTIAL PROGRESS - Identifying correct service early
                        reward += self.ROOT_CAUSE_INVESTIGATION
                        reward += self.PARTIAL_IDENTIFICATION_BONUS
                    else:
                        reward += self.USEFUL_INVESTIGATION
                    # Extra for trace that reveals root cause in deps
                    if action_type == ActionType.TRACE_REQUEST:
                        reward += self._trace_bonus(target_service, env)
                else:
                    # Already investigated — redundant
                    reward += self.REDUNDANT_ACTION
            else:
                reward += self.NOOP_PENALTY

        # === DIAGNOSE (with confidence signal — #12) ===
        elif action_type == ActionType.DIAGNOSE:
            diagnosed_service = params.get("root_cause_service", "")
            if diagnosed_service == task.root_cause_service:
                if not env._diagnosed_correctly:
                    reward += self.CORRECT_DIAGNOSIS
                else:
                    reward += self.REDUNDANT_ACTION  # Already diagnosed
            else:
                reward += self.WRONG_DIAGNOSIS

        # === FIX actions (with diagnosis gate — #1, #2) ===
        elif action_type in FIX_ACTION_TYPES:
            reward += self._calculate_fix_reward(action, env)

        # === COMMUNICATE: STATUS PAGE ===
        elif action_type == ActionType.UPDATE_STATUS_PAGE:
            if not env._status_page_updated:
                reward += self.STATUS_PAGE_UPDATE
            else:
                reward += self.REDUNDANT_ACTION

        # === COMMUNICATE: STAKEHOLDER ===
        elif action_type == ActionType.REPLY_STAKEHOLDER:
            if env._stakeholder_waiting and not env._stakeholder_replied:
                reward += self.STAKEHOLDER_REPLY
            else:
                reward += self.NOOP_PENALTY

        # === COMMUNICATE: PAGE ONCALL ===
        elif action_type == ActionType.PAGE_ONCALL:
            reward += 0.01

        # === OPTIMIZE: AUTOSCALING ===
        elif action_type == ActionType.ADJUST_AUTOSCALING:
            reward += self.NOOP_PENALTY

        # === OPTIMIZE: RIGHT SIZE ===
        elif action_type == ActionType.RIGHT_SIZE_SERVICE:
            target = params.get("service", "")
            if target == task.overprovisioned_service:
                if not env._right_sized:
                    reward += self.RIGHT_SIZE_BONUS
                else:
                    reward += self.REDUNDANT_ACTION
            else:
                reward += self.USELESS_SCALING

        return reward

    def _calculate_fix_reward(
        self,
        action: Action,
        env: "CloudOpsWarRoomEnvironment",
    ) -> float:
        """
        Calculate reward for fix actions with diagnosis gate.

        Rules:
          - Correct fix + diagnosed → full reward
          - Correct fix + NOT diagnosed → tiny reward (UNDIAGNOSED_FIX_REWARD)
          - Wrong fix → base penalty + escalation for repeats
        """
        task = env._task_config
        params = action.parameters
        action_type = action.action_type
        is_diagnosed = env._diagnosed_correctly

        # Determine if this is the correct fix
        is_correct_fix = False
        full_reward = 0.0

        if action_type == ActionType.RESTART_SERVICE:
            target = params.get("service", "")
            if target == task.root_cause_service and task.required_fix == FixType.RESTART:
                is_correct_fix = True
                full_reward = self.CORRECT_RESTART

        elif action_type == ActionType.ROLLBACK_DEPLOY:
            target = params.get("service", "")
            if target == task.root_cause_service and task.required_fix == FixType.ROLLBACK:
                is_correct_fix = True
                full_reward = self.CORRECT_ROLLBACK

        elif action_type == ActionType.TOGGLE_FEATURE_FLAG:
            flag = params.get("flag_name", "")
            if task.required_fix == FixType.FEATURE_FLAG and flag == task.feature_flag_name:
                is_correct_fix = True
                full_reward = self.CORRECT_FEATURE_FLAG

        elif action_type == ActionType.SCALE_SERVICE:
            target = params.get("service", "")
            if target == task.root_cause_service and task.required_fix == FixType.SCALE:
                is_correct_fix = True
                full_reward = self.CORRECT_SCALE_FIX

        elif action_type == ActionType.APPLY_RATE_LIMIT:
            target = params.get("service", "")
            if target == task.root_cause_service and task.required_fix == FixType.RATE_LIMIT:
                is_correct_fix = True
                full_reward = self.CORRECT_RATE_LIMIT

        if is_correct_fix:
            # DIAGNOSIS GATE (#1)
            if is_diagnosed:
                return full_reward
            else:
                return self.UNDIAGNOSED_FIX_REWARD
        else:
            # Wrong fix — apply escalating penalty (#2)
            wrong_count = env._wrong_fix_count
            if action_type == ActionType.ROLLBACK_DEPLOY:
                base = self.WRONG_ROLLBACK_BASE
            elif action_type == ActionType.RESTART_SERVICE:
                base = self.WRONG_RESTART_BASE
            else:
                base = self.USELESS_SCALING

            escalation = self.WRONG_FIX_ESCALATION * wrong_count
            return base + escalation

    def _trace_bonus(
        self,
        service: str,
        env: "CloudOpsWarRoomEnvironment",
    ) -> float:
        """
        Extra reward if tracing reveals the root cause in dependencies.
        """
        task = env._task_config
        # Check if any downstream dependency of this service is the root cause
        for dep in env._dependencies:
            if dep.source == service and dep.target == task.root_cause_service:
                return self.TRACE_IDENTIFIES_ROOT
            if dep.target == service and dep.source == task.root_cause_service:
                return self.TRACE_IDENTIFIES_ROOT
        return 0.0

    def completion_bonus(self, diagnosed: bool) -> float:
        """
        Return completion bonus — REQUIRES diagnosis (#4).
        Full bonus only if diagnosed AND resolved.
        Partial bonus if resolved without diagnosis.
        """
        if diagnosed:
            return self.FULL_COMPLETION_BONUS
        else:
            return self.PARTIAL_COMPLETION_BONUS

    def timeout_penalty(self) -> float:
        """Return the penalty for running out of steps."""
        return self.TIMEOUT_PENALTY

    def normalize_final_score(self, total_reward: float, task: "TaskConfig") -> float:
        """
        Normalize the final total_reward to a [0.0, 1.0] range using hard-coded bounds.
        
        Formula: (reward - min) / (max - min)
        Clipped to [0.0, 1.0] to ensure stability.
        """
        max_r = task.max_reward
        min_r = task.min_reward
        
        # Stability check: handle zero denominator
        if max_r <= min_r:
            return 1.0 if total_reward >= max_r else 0.0
            
        normalized = (total_reward - min_r) / (max_r - min_r)
        
        # Clipping (#2)
        return max(0.0, min(1.0, normalized))
