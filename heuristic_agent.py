"""
Optimal Heuristic Expert Agent for CloudOpsWarRoomEnv.
Provides task-specific near-optimal action sequences (Diagnose -> Fix -> Communicate).
Targets 0.80 - 0.90 scores across all tasks.
"""
import sys
import os

# Add root to python path to ensure cloudops_env can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cloudops_env.models import Action

class HeuristicExpertAgent:
    def __init__(self, task_id: str):
        self.task_id = task_id
        self.step_idx = 0
        
    def get_action(self, observation=None) -> Action:
        self.step_idx += 1
        
        # Check if stakeholder flag is active to decide when to reply
        stakeholder_waiting = False
        if observation and observation.get("stakeholder_flag"):
            stakeholder_waiting = True

        if self.task_id == "noisy_alert":
            return self._noisy_alert_logic(stakeholder_waiting)
        elif self.task_id == "bad_deploy":
            return self._bad_deploy_logic(stakeholder_waiting)
        elif self.task_id == "cascade_failure":
            return self._cascade_failure_logic(stakeholder_waiting)
        elif self.task_id == "cost_vs_performance":
            return self._cost_vs_performance_logic(stakeholder_waiting)
        elif self.task_id == "fog_of_war":
            return self._fog_of_war_logic(stakeholder_waiting)
        else:
            return Action(action_type="check_metrics", parameters={"service": "api-gateway"})

    def _noisy_alert_logic(self, stakeholder_waiting):
        steps = {
            1: Action(action_type="check_metrics", parameters={"service": "payment-service"}),
            2: Action(action_type="diagnose", parameters={"root_cause_service": "payment-service"}),
            3: Action(action_type="restart_service", parameters={"service": "payment-service"}),
            4: Action(action_type="update_status_page", parameters={"message": "Resolved"}),
            5: Action(action_type="reply_stakeholder", parameters={"message": "Noisy alert on payment-service resolved with restart."}),
        }
        return steps.get(self.step_idx, Action(action_type="reply_stakeholder", parameters={"message": "Monitoring"}))

    def _bad_deploy_logic(self, stakeholder_waiting):
        steps = {
            1: Action(action_type="query_logs", parameters={"service": "order-service"}),
            2: Action(action_type="diagnose", parameters={"root_cause_service": "order-service"}),
            3: Action(action_type="rollback_deploy", parameters={"service": "order-service"}),
            4: Action(action_type="update_status_page", parameters={"message": "Resolved"}),
            5: Action(action_type="reply_stakeholder", parameters={"message": "Bad deploy on order-service rolled back. Monitoring."}),
        }
        return steps.get(self.step_idx, Action(action_type="reply_stakeholder", parameters={"message": "Monitoring"}))

    def _cascade_failure_logic(self, stakeholder_waiting):
        steps = {
            1: Action(action_type="trace_request", parameters={"service": "api-gateway"}),
            2: Action(action_type="check_metrics", parameters={"service": "session-store"}),
            3: Action(action_type="diagnose", parameters={"root_cause_service": "session-store"}),
            4: Action(action_type="restart_service", parameters={"service": "session-store"}),
            5: Action(action_type="update_status_page", parameters={"message": "Resolved"}),
            6: Action(action_type="reply_stakeholder", parameters={"message": "Cascading failure traced to session-store; resolved via restart."}),
        }
        return steps.get(self.step_idx, Action(action_type="reply_stakeholder", parameters={"message": "Monitoring"}))

    def _cost_vs_performance_logic(self, stakeholder_waiting):
        steps = {
            1: Action(action_type="check_metrics", parameters={"service": "product-service"}),
            2: Action(action_type="diagnose", parameters={"root_cause_service": "product-service"}),
            3: Action(action_type="toggle_feature_flag", parameters={"flag_name": "new_product_page_v2"}),
            4: Action(action_type="right_size_service", parameters={"service": "search-service"}),
            5: Action(action_type="update_status_page", parameters={"message": "Resolved"}),
            6: Action(action_type="reply_stakeholder", parameters={"message": "Buggy feature flag disabled on product-service and infrastructure right-sized."}),
        }
        return steps.get(self.step_idx, Action(action_type="reply_stakeholder", parameters={"message": "Monitoring"}))

    def _fog_of_war_logic(self, stakeholder_waiting):
        steps = {
            1: Action(action_type="query_logs", parameters={"service": "order-service"}),
            2: Action(action_type="diagnose", parameters={"root_cause_service": "order-service"}),
            3: Action(action_type="rollback_deploy", parameters={"service": "order-service"}),
            4: Action(action_type="update_status_page", parameters={"message": "Resolved"}),
            5: Action(action_type="reply_stakeholder", parameters={"message": "True root cause (order-service bad deploy) resolved. Phantom alerts on auth-service were ignored."}),
        }
        return steps.get(self.step_idx, Action(action_type="reply_stakeholder", parameters={"message": "Monitoring"}))
