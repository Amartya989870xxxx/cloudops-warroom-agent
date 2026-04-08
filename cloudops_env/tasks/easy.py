"""
Task 1: Noisy Alert (Easy)

Scenario:
  A single service (payment-service) has high error rates and is degraded.
  However, 3 misleading alerts fire on OTHER healthy services.
  The agent must cut through the noise, identify the real broken service,
  diagnose it, fix it, and communicate.

Root Cause: payment-service is overloaded due to a memory leak.
Fix: restart_service(payment-service)
Optimal Steps: ~5 (check_metrics → query_logs → diagnose → restart → update_status_page)
"""

from cloudops_env.models import (
    Alert,
    AlertSeverity,
    DeployInfo,
    FixType,
    ServiceDependency,
    ServiceInfo,
    ServiceStatus,
    TaskConfig,
)


def create_noisy_alert_task() -> TaskConfig:
    """Create the Noisy Alert scenario."""
    services = [
        ServiceInfo(
            name="api-gateway",
            status=ServiceStatus.HEALTHY,
            cpu_percent=35.0,
            memory_percent=42.0,
            error_rate=0.02,
            latency_p99_ms=85.0,
            request_rate=500.0,
        ),
        ServiceInfo(
            name="auth-service",
            status=ServiceStatus.HEALTHY,
            cpu_percent=20.0,
            memory_percent=35.0,
            error_rate=0.01,
            latency_p99_ms=30.0,
            request_rate=200.0,
        ),
        ServiceInfo(
            name="payment-service",
            status=ServiceStatus.DEGRADED,
            cpu_percent=45.0,
            memory_percent=92.0,  # Memory leak!
            error_rate=0.35,       # High error rate
            latency_p99_ms=1200.0, # Very slow
            request_rate=150.0,
        ),
        ServiceInfo(
            name="notification-service",
            status=ServiceStatus.HEALTHY,
            cpu_percent=15.0,
            memory_percent=30.0,
            error_rate=0.005,
            latency_p99_ms=25.0,
            request_rate=80.0,
        ),
        ServiceInfo(
            name="inventory-service",
            status=ServiceStatus.HEALTHY,
            cpu_percent=28.0,
            memory_percent=45.0,
            error_rate=0.01,
            latency_p99_ms=55.0,
            request_rate=120.0,
        ),
    ]

    dependencies = [
        ServiceDependency(source="api-gateway", target="auth-service"),
        ServiceDependency(source="api-gateway", target="payment-service"),
        ServiceDependency(source="api-gateway", target="inventory-service"),
        ServiceDependency(source="payment-service", target="notification-service"),
    ]

    # 1 real alert + 1 noisy alert (Reduced from 3 noise alerts)
    alerts = [
        Alert(
            severity=AlertSeverity.CRITICAL,
            service="payment-service",
            message="High error rate detected: 35% of requests failing",
            is_noise=False,
        ),
        Alert(
            severity=AlertSeverity.WARNING,
            service="notification-service",
            message="Email queue depth increasing — possible backlog",
            is_noise=True,
        ),
    ]

    deploys = [
        DeployInfo(
            service="auth-service",
            timestamp="2024-01-15T08:30:00Z",
            change_summary="Updated OAuth token refresh logic",
            version="v2.3.1",
            is_buggy=False,
        ),
    ]

    return TaskConfig(
        task_id="noisy_alert",
        task_name="Noisy Alert Triage",
        difficulty="easy",
        description=(
            "A single service is failing with high error rates, but misleading "
            "alerts from other healthy services create noise. Cut through the "
            "noise, find the real issue, fix it, and update stakeholders."
        ),
        max_steps=15,
        services=services,
        dependencies=dependencies,
        initial_alerts=alerts,
        recent_deploys=deploys,
        root_cause_service="payment-service",
        root_cause_description="Memory leak causing OOM errors and request failures",
        required_fix=FixType.RESTART,
        initial_cost_rate=1.2,
        initial_slo_budget=0.85,
        stakeholder_asks_at_step=5,
        optimal_steps=5,
        max_reward=0.65,
        min_reward=-0.30,
    )
