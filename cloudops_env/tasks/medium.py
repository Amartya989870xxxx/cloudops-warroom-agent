"""
Task 2: Bad Deploy (Medium)

Scenario:
  A recent deploy to order-service introduced a critical bug
  that causes intermittent 500 errors. Multiple downstream services
  show symptoms (elevated error rates, increased latency), but the
  root cause is the bad deploy.

Root Cause: Bad deploy on order-service (v2.5.0 introduced a DB query regression)
Fix: rollback_deploy(order-service)
Optimal Steps: ~6 (check_metrics → query_logs → trace_request → diagnose → rollback → update_status_page)
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


def create_bad_deploy_task() -> TaskConfig:
    """Create the Bad Deploy scenario."""
    services = [
        ServiceInfo(
            name="api-gateway",
            status=ServiceStatus.DEGRADED,
            cpu_percent=55.0,
            memory_percent=50.0,
            error_rate=0.18,       # Elevated due to failures downstream
            latency_p99_ms=450.0,
            request_rate=480.0,
        ),
        ServiceInfo(
            name="auth-service",
            status=ServiceStatus.HEALTHY,
            cpu_percent=22.0,
            memory_percent=38.0,
            error_rate=0.01,
            latency_p99_ms=35.0,
            request_rate=200.0,
        ),
        ServiceInfo(
            name="order-service",
            status=ServiceStatus.DEGRADED,
            cpu_percent=72.0,      # High CPU from bad query
            memory_percent=65.0,
            error_rate=0.42,       # Very high error rate
            latency_p99_ms=2800.0, # Extremely slow
            request_rate=100.0,
        ),
        ServiceInfo(
            name="payment-service",
            status=ServiceStatus.DEGRADED,
            cpu_percent=40.0,
            memory_percent=48.0,
            error_rate=0.15,       # Errors from order-service timeouts
            latency_p99_ms=600.0,
            request_rate=80.0,
        ),
        ServiceInfo(
            name="inventory-service",
            status=ServiceStatus.DEGRADED,
            cpu_percent=38.0,
            memory_percent=44.0,
            error_rate=0.12,       # Cascading from order-service
            latency_p99_ms=350.0,
            request_rate=110.0,
        ),
        ServiceInfo(
            name="notification-service",
            status=ServiceStatus.HEALTHY,
            cpu_percent=18.0,
            memory_percent=32.0,
            error_rate=0.03,
            latency_p99_ms=40.0,
            request_rate=60.0,
        ),
        ServiceInfo(
            name="recommendation-service",
            status=ServiceStatus.HEALTHY,
            cpu_percent=25.0,
            memory_percent=40.0,
            error_rate=0.02,
            latency_p99_ms=65.0,
            request_rate=90.0,
        ),
    ]

    dependencies = [
        ServiceDependency(source="api-gateway", target="auth-service"),
        ServiceDependency(source="api-gateway", target="order-service"),
        ServiceDependency(source="api-gateway", target="recommendation-service"),
        ServiceDependency(source="order-service", target="payment-service"),
        ServiceDependency(source="order-service", target="inventory-service"),
        ServiceDependency(source="order-service", target="notification-service"),
    ]

    alerts = [
        Alert(
            severity=AlertSeverity.CRITICAL,
            service="order-service",
            message="HTTP 500 error rate exceeds 40% — orders failing",
            is_noise=False,
        ),
        Alert(
            severity=AlertSeverity.CRITICAL,
            service="api-gateway",
            message="Gateway error rate elevated — upstream failures detected",
            is_noise=False,
        ),
        Alert(
            severity=AlertSeverity.WARNING,
            service="payment-service",
            message="Payment processing timeouts increasing",
            is_noise=False,
        ),
        Alert(
            severity=AlertSeverity.WARNING,
            service="inventory-service",
            message="Inventory sync failures — stock levels may be stale",
            is_noise=False,
        ),
    ]

    deploys = [
        DeployInfo(
            service="order-service",
            timestamp="2024-01-15T09:15:00Z",
            change_summary="Optimized order query with new JOIN strategy (v2.5.0)",
            version="v2.5.0",
            is_buggy=True,  # THIS IS THE ROOT CAUSE
        ),
        DeployInfo(
            service="recommendation-service",
            timestamp="2024-01-15T07:00:00Z",
            change_summary="Added personalized recommendations model v3",
            version="v1.8.0",
            is_buggy=False,
        ),
        DeployInfo(
            service="auth-service",
            timestamp="2024-01-14T22:00:00Z",
            change_summary="Security patch for token validation",
            version="v2.3.2",
            is_buggy=False,
        ),
    ]

    return TaskConfig(
        task_id="bad_deploy",
        task_name="Bad Deploy Rollback",
        difficulty="medium",
        description=(
            "A recent deployment to order-service introduced a database query "
            "regression causing cascading failures. Multiple services show elevated "
            "error rates, but the root cause is the bad deploy. Roll it back."
        ),
        max_steps=18,
        services=services,
        dependencies=dependencies,
        initial_alerts=alerts,
        recent_deploys=deploys,
        root_cause_service="order-service",
        root_cause_description="Deploy v2.5.0 introduced a DB query regression with N+1 problem",
        required_fix=FixType.ROLLBACK,
        initial_cost_rate=1.8,
        initial_slo_budget=0.65,
        stakeholder_asks_at_step=4,
        optimal_steps=6,
        max_reward=0.75,
        min_reward=-0.40,
    )
