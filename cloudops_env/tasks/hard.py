"""
Hard Tasks: Cascade Failure + Cost vs Performance + Fog of War (v2 — Enhanced)

v2 Enhancements (#7):
  - Cascade Failure: Deeper 4-level dependency chain with misleading downstream alerts
  - Fog of War: Added phantom incident (fake root cause with realistic logs/alerts)
  - Cost vs Performance: Unchanged (already well-designed)
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


def create_cascade_failure_task() -> TaskConfig:
    """
    Create the Cascade Failure scenario (Task 3) — Enhanced.

    4-level deep dependency chain:
      cdn-edge → api-gateway → user-service → session-store (ROOT CAUSE)
                              → order-service → payment-service → fraud-detector

    Root cause (session-store) is buried 3 levels deep.
    Alerts fire loudest on downstream/leaf services to mislead.
    """
    services = [
        ServiceInfo(
            name="cdn-edge",
            status=ServiceStatus.DEGRADED,
            cpu_percent=60.0,
            memory_percent=50.0,
            error_rate=0.25,
            latency_p99_ms=800.0,
            request_rate=1000.0,
        ),
        ServiceInfo(
            name="api-gateway",
            status=ServiceStatus.DEGRADED,
            cpu_percent=65.0,
            memory_percent=55.0,
            error_rate=0.30,
            latency_p99_ms=1200.0,
            request_rate=800.0,
        ),
        ServiceInfo(
            name="user-service",
            status=ServiceStatus.DEGRADED,
            cpu_percent=48.0,
            memory_percent=52.0,
            error_rate=0.20,
            latency_p99_ms=600.0,
            request_rate=300.0,
        ),
        ServiceInfo(
            name="session-store",
            status=ServiceStatus.DOWN,  # ROOT CAUSE — Redis cluster failure (Level 4)
            cpu_percent=95.0,
            memory_percent=98.0,
            error_rate=0.85,
            latency_p99_ms=5000.0,
            request_rate=20.0,
        ),
        ServiceInfo(
            name="order-service",
            status=ServiceStatus.DEGRADED,
            cpu_percent=55.0,
            memory_percent=50.0,
            error_rate=0.28,
            latency_p99_ms=900.0,
            request_rate=150.0,
        ),
        ServiceInfo(
            name="payment-service",
            status=ServiceStatus.DEGRADED,
            cpu_percent=42.0,
            memory_percent=48.0,
            error_rate=0.22,
            latency_p99_ms=700.0,
            request_rate=100.0,
        ),
        # NEW: Deeper 4th level — fraud-detector depends on payment-service
        ServiceInfo(
            name="fraud-detector",
            status=ServiceStatus.DEGRADED,
            cpu_percent=50.0,
            memory_percent=45.0,
            error_rate=0.18,
            latency_p99_ms=550.0,
            request_rate=80.0,
        ),
        # NEW: config-service depends on session-store too (lateral dependency)
        ServiceInfo(
            name="config-service",
            status=ServiceStatus.DEGRADED,
            cpu_percent=38.0,
            memory_percent=42.0,
            error_rate=0.15,
            latency_p99_ms=400.0,
            request_rate=500.0,
        ),
        ServiceInfo(
            name="search-service",
            status=ServiceStatus.HEALTHY,
            cpu_percent=30.0,
            memory_percent=40.0,
            error_rate=0.02,
            latency_p99_ms=45.0,
            request_rate=200.0,
        ),
        ServiceInfo(
            name="analytics-service",
            status=ServiceStatus.HEALTHY,
            cpu_percent=25.0,
            memory_percent=38.0,
            error_rate=0.01,
            latency_p99_ms=100.0,
            request_rate=50.0,
        ),
    ]

    # 4-level deep chain: cdn → gateway → user → session-store
    # Lateral paths: gateway → order → payment → fraud-detector
    # config-service → session-store (explains config-service degradation)
    dependencies = [
        # Level 1 → 2
        ServiceDependency(source="cdn-edge", target="api-gateway"),
        # Level 2 → 3
        ServiceDependency(source="api-gateway", target="user-service"),
        ServiceDependency(source="api-gateway", target="order-service"),
        ServiceDependency(source="api-gateway", target="search-service"),
        ServiceDependency(source="api-gateway", target="config-service"),
        # Level 3 → 4 (root cause)
        ServiceDependency(source="user-service", target="session-store"),
        ServiceDependency(source="config-service", target="session-store"),
        # Lateral chains
        ServiceDependency(source="order-service", target="payment-service"),
        ServiceDependency(source="order-service", target="user-service"),
        ServiceDependency(source="payment-service", target="fraud-detector"),
        ServiceDependency(source="analytics-service", target="session-store"),
    ]

    # Alerts fire mostly on DOWNSTREAM/LEAF services — misleading!
    alerts = [
        Alert(
            severity=AlertSeverity.CRITICAL,
            service="cdn-edge",
            message="CDN cache miss rate 90% — origin errors escalating",
            is_noise=False,
        ),
        Alert(
            severity=AlertSeverity.CRITICAL,
            service="api-gateway",
            message="Gateway 502 errors exceeding threshold — multiple backends failing",
            is_noise=False,
        ),
        Alert(
            severity=AlertSeverity.CRITICAL,
            service="fraud-detector",
            message="CRITICAL: Fraud detection pipeline stalled — transaction queue backing up",
            is_noise=False,
        ),
        Alert(
            severity=AlertSeverity.CRITICAL,
            service="order-service",
            message="Order creation failures spiking — DB timeout",
            is_noise=False,
        ),
        Alert(
            severity=AlertSeverity.WARNING,
            service="payment-service",
            message="Payment authorization timeouts increasing",
            is_noise=False,
        ),
        Alert(
            severity=AlertSeverity.WARNING,
            service="user-service",
            message="User session validation errors — possible auth issue",
            is_noise=False,
        ),
        Alert(
            severity=AlertSeverity.WARNING,
            service="config-service",
            message="Dynamic configuration reload failures — cached values stale",
            is_noise=False,
        ),
        # Root cause alert is buried and less severe
        Alert(
            severity=AlertSeverity.WARNING,
            service="session-store",
            message="Redis connection pool exhausted",
            is_noise=False,
        ),
    ]

    return TaskConfig(
        task_id="cascade_failure",
        task_name="Cascading Failure Investigation",
        difficulty="hard",
        description=(
            "Multiple services failing across a 4-level dependency chain. "
            "The root cause is session-store (Redis cluster OOM) buried 3 levels deep. "
            "Loudest alerts fire on downstream/leaf services. "
            "Agent must trace upstream through dependency graph to find the source."
        ),
        max_steps=22,
        services=services,
        dependencies=dependencies,
        initial_alerts=alerts,
        recent_deploys=[],
        root_cause_service="session-store",
        root_cause_description="Redis cluster OOM — memory exhausted, all connections dropped",
        required_fix=FixType.RESTART,
        initial_cost_rate=2.5,
        initial_slo_budget=0.40,
        stakeholder_asks_at_step=3,
        optimal_steps=7,
        max_reward=0.60,
        min_reward=-0.50,
    )


def create_cost_vs_performance_task() -> TaskConfig:
    """Create the Cost vs Performance scenario (Task 4)."""
    services = [
        ServiceInfo(
            name="api-gateway",
            status=ServiceStatus.HEALTHY,
            cpu_percent=40.0,
            memory_percent=45.0,
            error_rate=0.05,
            latency_p99_ms=120.0,
            request_rate=600.0,
        ),
        ServiceInfo(
            name="product-service",
            status=ServiceStatus.DEGRADED,
            cpu_percent=35.0,
            memory_percent=40.0,
            error_rate=0.18,
            latency_p99_ms=350.0,
            request_rate=300.0,
        ),
        ServiceInfo(
            name="search-service",
            status=ServiceStatus.OVERLOADED,
            cpu_percent=15.0,
            memory_percent=12.0,
            error_rate=0.01,
            latency_p99_ms=30.0,
            request_rate=200.0,
        ),
        ServiceInfo(
            name="cache-layer",
            status=ServiceStatus.HEALTHY,
            cpu_percent=30.0,
            memory_percent=55.0,
            error_rate=0.02,
            latency_p99_ms=15.0,
            request_rate=800.0,
        ),
        ServiceInfo(
            name="recommendation-engine",
            status=ServiceStatus.HEALTHY,
            cpu_percent=50.0,
            memory_percent=60.0,
            error_rate=0.03,
            latency_p99_ms=200.0,
            request_rate=150.0,
        ),
        ServiceInfo(
            name="analytics-pipeline",
            status=ServiceStatus.HEALTHY,
            cpu_percent=22.0,
            memory_percent=35.0,
            error_rate=0.01,
            latency_p99_ms=500.0,
            request_rate=40.0,
        ),
    ]

    dependencies = [
        ServiceDependency(source="api-gateway", target="product-service"),
        ServiceDependency(source="api-gateway", target="search-service"),
        ServiceDependency(source="product-service", target="cache-layer"),
        ServiceDependency(source="product-service", target="recommendation-engine"),
        ServiceDependency(source="search-service", target="cache-layer"),
        ServiceDependency(source="recommendation-engine", target="analytics-pipeline"),
    ]

    alerts = [
        Alert(
            severity=AlertSeverity.WARNING,
            service="product-service",
            message="Intermittent 500 errors on /api/products/v2 endpoint",
            is_noise=False,
        ),
        Alert(
            severity=AlertSeverity.WARNING,
            service="search-service",
            message="Infrastructure cost anomaly — search-service running 5x overprovisioned",
            is_noise=False,
        ),
        Alert(
            severity=AlertSeverity.INFO,
            service="recommendation-engine",
            message="Model inference latency slightly elevated",
            is_noise=True,
        ),
    ]

    deploys = [
        DeployInfo(
            service="product-service",
            timestamp="2024-01-15T06:00:00Z",
            change_summary="Enabled new_product_page_v2 feature flag for A/B test",
            version="v3.1.0",
            is_buggy=False,
        ),
    ]

    return TaskConfig(
        task_id="cost_vs_performance",
        task_name="Cost vs Performance Optimization",
        difficulty="hard",
        description=(
            "Two issues: (1) product-service has a buggy feature flag causing "
            "intermittent errors, and (2) search-service is massively overprovisioned "
            "wasting cloud budget. Fix the feature flag and right-size the infrastructure."
        ),
        max_steps=20,
        services=services,
        dependencies=dependencies,
        initial_alerts=alerts,
        recent_deploys=deploys,
        root_cause_service="product-service",
        root_cause_description="Feature flag 'new_product_page_v2' has a bug causing null pointer errors",
        required_fix=FixType.FEATURE_FLAG,
        feature_flag_name="new_product_page_v2",
        initial_cost_rate=4.5,
        initial_slo_budget=0.70,
        overprovisioned_service="search-service",
        stakeholder_asks_at_step=5,
        optimal_steps=7,
        max_reward=0.85,
        min_reward=-0.40,
    )


def create_fog_of_war_task() -> TaskConfig:
    """
    Create the Fog of War scenario (Task 5 — Expert) — Enhanced.

    v2 Enhancements (#7):
      - 2 simultaneous incidents: 1 real (order-service bad deploy) + 1 phantom (auth-service)
      - Phantom incident produces realistic logs/alerts but is fake
      - Agent must identify and ignore the phantom, solving only the real incident
      - 4 fake alerts out of 10 total
    """
    services = [
        ServiceInfo(
            name="load-balancer",
            status=ServiceStatus.HEALTHY,
            cpu_percent=30.0,
            memory_percent=25.0,
            error_rate=0.08,
            latency_p99_ms=50.0,
            request_rate=2000.0,
        ),
        ServiceInfo(
            name="api-gateway",
            status=ServiceStatus.DEGRADED,
            cpu_percent=70.0,
            memory_percent=65.0,
            error_rate=0.20,
            latency_p99_ms=500.0,
            request_rate=1500.0,
        ),
        # PHANTOM: auth-service appears degraded but is a fake incident
        ServiceInfo(
            name="auth-service",
            status=ServiceStatus.DEGRADED,
            cpu_percent=65.0,
            memory_percent=58.0,
            error_rate=0.12,
            latency_p99_ms=280.0,
            request_rate=400.0,
        ),
        ServiceInfo(
            name="user-service",
            status=ServiceStatus.DEGRADED,
            cpu_percent=58.0,
            memory_percent=55.0,
            error_rate=0.15,
            latency_p99_ms=380.0,
            request_rate=250.0,
        ),
        ServiceInfo(
            name="order-service",
            status=ServiceStatus.DOWN,  # REAL root cause
            cpu_percent=92.0,
            memory_percent=88.0,
            error_rate=0.65,
            latency_p99_ms=4500.0,
            request_rate=50.0,
        ),
        ServiceInfo(
            name="payment-service",
            status=ServiceStatus.DEGRADED,
            cpu_percent=45.0,
            memory_percent=50.0,
            error_rate=0.25,
            latency_p99_ms=800.0,
            request_rate=80.0,
        ),
        ServiceInfo(
            name="inventory-service",
            status=ServiceStatus.HEALTHY,
            cpu_percent=28.0,
            memory_percent=42.0,
            error_rate=0.03,
            latency_p99_ms=60.0,
            request_rate=120.0,
        ),
        ServiceInfo(
            name="notification-service",
            status=ServiceStatus.HEALTHY,
            cpu_percent=12.0,
            memory_percent=20.0,
            error_rate=0.01,
            latency_p99_ms=20.0,
            request_rate=100.0,
        ),
        ServiceInfo(
            name="cache-cluster",
            status=ServiceStatus.HEALTHY,
            cpu_percent=40.0,
            memory_percent=70.0,
            error_rate=0.01,
            latency_p99_ms=5.0,
            request_rate=5000.0,
        ),
        ServiceInfo(
            name="message-queue",
            status=ServiceStatus.HEALTHY,
            cpu_percent=20.0,
            memory_percent=30.0,
            error_rate=0.005,
            latency_p99_ms=10.0,
            request_rate=3000.0,
        ),
    ]

    dependencies = [
        ServiceDependency(source="load-balancer", target="api-gateway"),
        ServiceDependency(source="api-gateway", target="auth-service"),
        ServiceDependency(source="api-gateway", target="user-service"),
        ServiceDependency(source="api-gateway", target="order-service"),
        ServiceDependency(source="order-service", target="payment-service"),
        ServiceDependency(source="order-service", target="inventory-service"),
        ServiceDependency(source="order-service", target="notification-service"),
        ServiceDependency(source="user-service", target="cache-cluster"),
        ServiceDependency(source="order-service", target="message-queue"),
        ServiceDependency(source="payment-service", target="notification-service"),
        # auth-service dependencies (phantom path)
        ServiceDependency(source="auth-service", target="cache-cluster"),
        ServiceDependency(source="auth-service", target="user-service"),
    ]

    # Mix of real alerts, phantom alerts, and pure noise
    alerts = [
        # REAL incident alerts
        Alert(
            severity=AlertSeverity.CRITICAL,
            service="order-service",
            message="CRITICAL: Order processing completely halted — 65% error rate",
            is_noise=False,
        ),
        Alert(
            severity=AlertSeverity.CRITICAL,
            service="api-gateway",
            message="Gateway circuit breaker OPEN for /api/orders/*",
            is_noise=False,
        ),
        Alert(
            severity=AlertSeverity.CRITICAL,
            service="payment-service",
            message="Payment gateway timeout — transactions failing",
            is_noise=False,
        ),
        # PHANTOM incident alerts (look critical but are fake)
        Alert(
            severity=AlertSeverity.CRITICAL,
            service="auth-service",
            message="CRITICAL: Authentication failures spiking — TLS certificate issue detected",
            is_noise=True,
        ),
        Alert(
            severity=AlertSeverity.WARNING,
            service="auth-service",
            message="Token validation error rate elevated — possible key rotation failure",
            is_noise=True,
        ),
        # Pure noise alerts
        Alert(
            severity=AlertSeverity.CRITICAL,
            service="cache-cluster",
            message="CRITICAL: Cache eviction rate abnormally high — possible attack",
            is_noise=True,
        ),
        Alert(
            severity=AlertSeverity.WARNING,
            service="message-queue",
            message="Consumer lag increasing on order-events topic",
            is_noise=True,
        ),
        # Real secondary symptom
        Alert(
            severity=AlertSeverity.WARNING,
            service="user-service",
            message="User profile cache miss rate elevated",
            is_noise=False,
        ),
        Alert(
            severity=AlertSeverity.INFO,
            service="notification-service",
            message="Email delivery rate dropped — check SMTP config",
            is_noise=True,
        ),
    ]

    deploys = [
        DeployInfo(
            service="order-service",
            timestamp="2024-01-15T10:30:00Z",
            change_summary="Added new order validation rules and rate limiting",
            version="v4.2.0",
            is_buggy=True,  # REAL ROOT CAUSE
        ),
        # Phantom deploy — makes auth-service look suspicious
        DeployInfo(
            service="auth-service",
            timestamp="2024-01-15T09:45:00Z",
            change_summary="Updated TLS certificate rotation and key management",
            version="v2.4.0",
            is_buggy=False,  # NOT the real cause despite looking suspicious
        ),
        DeployInfo(
            service="cache-cluster",
            timestamp="2024-01-15T08:00:00Z",
            change_summary="Increased max memory to 16GB per node",
            version="v1.5.0",
            is_buggy=False,
        ),
    ]

    return TaskConfig(
        task_id="fog_of_war",
        task_name="Fog of War — Multi-Alert Chaos",
        difficulty="expert",
        description=(
            "Chaotic incident with 9 alerts across 10 services. "
            "TWO incidents appear active: (1) order-service bad deploy (REAL), "
            "(2) auth-service TLS issue (PHANTOM). "
            "The auth-service incident produces realistic error logs and alerts "
            "but fixing it has no effect. Agent must identify the real root cause "
            "through investigation, ignoring the phantom."
        ),
        max_steps=25,
        services=services,
        dependencies=dependencies,
        initial_alerts=alerts,
        recent_deploys=deploys,
        root_cause_service="order-service",
        root_cause_description="Deploy v4.2.0 introduced broken order validation that rejects all orders",
        required_fix=FixType.ROLLBACK,
        initial_cost_rate=3.2,
        initial_slo_budget=0.30,
        stakeholder_asks_at_step=2,
        optimal_steps=8,
        phantom_root_cause="auth-service",  # NEW: phantom incident (#7)
        max_reward=0.75,
        min_reward=-0.60,
    )
