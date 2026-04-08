def grade(trajectory, env_info=None):
    """Grade the cascade_failure task — multi-service cascade investigation."""
    diagnosed = env_info.get("diagnosed_correctly", False) if env_info else False
    resolved = env_info.get("incident_resolved", False) if env_info else False

    if diagnosed and resolved:
        return 1.0
    elif diagnosed:
        return 0.5
    elif resolved:
        return 0.3
    return 0.0
