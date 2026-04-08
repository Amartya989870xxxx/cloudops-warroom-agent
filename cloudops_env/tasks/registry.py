"""
Task Registry — Central hub for all incident scenarios.

Provides:
- get_task(task_id) → TaskConfig
- list_tasks() → list of available task IDs with metadata
- random_task() → random task selection
"""

import random
from typing import Dict, List, Optional

from cloudops_env.models import TaskConfig
from cloudops_env.tasks.easy import create_noisy_alert_task
from cloudops_env.tasks.medium import create_bad_deploy_task
from cloudops_env.tasks.hard import (
    create_cascade_failure_task,
    create_cost_vs_performance_task,
    create_fog_of_war_task,
)


class TaskRegistry:
    """Registry of all available incident response scenarios."""

    def __init__(self):
        self._tasks: Dict[str, TaskConfig] = {}
        self._register_all()

    def _register_all(self):
        """Register all built-in tasks."""
        tasks = [
            create_noisy_alert_task(),
            create_bad_deploy_task(),
            create_cascade_failure_task(),
            create_cost_vs_performance_task(),
            create_fog_of_war_task(),
        ]
        for task in tasks:
            self._tasks[task.task_id] = task

    def get_task(self, task_id: str) -> TaskConfig:
        """Get a specific task by ID."""
        if task_id not in self._tasks:
            available = ", ".join(self._tasks.keys())
            raise ValueError(
                f"Unknown task_id: '{task_id}'. Available tasks: {available}"
            )
        return self._tasks[task_id].model_copy(deep=True)

    def random_task(self) -> TaskConfig:
        """Select a random task."""
        task_id = random.choice(list(self._tasks.keys()))
        return self.get_task(task_id)

    def list_tasks(self) -> List[Dict[str, str]]:
        """List all available tasks with metadata."""
        return [
            {
                "task_id": t.task_id,
                "task_name": t.task_name,
                "difficulty": t.difficulty,
                "description": t.description,
                "max_steps": str(t.max_steps),
                "optimal_steps": str(t.optimal_steps),
            }
            for t in self._tasks.values()
        ]

    def get_task_ids(self) -> List[str]:
        """Get all registered task IDs."""
        return list(self._tasks.keys())

    def get_tasks_by_difficulty(self, difficulty: str) -> List[TaskConfig]:
        """Get tasks filtered by difficulty level."""
        return [
            t.model_copy(deep=True)
            for t in self._tasks.values()
            if t.difficulty == difficulty
        ]
