from app.core.schemas import WorkflowTrace
from app.nodes.draft_node import DraftNode
from app.nodes.intent_node import IntentNode
from app.nodes.policy_node import PolicyNode
from app.nodes.priority_node import PriorityNode
from app.nodes.router_node import RouterNode
from app.nodes.validation_node import ValidationNode


class Orchestrator:
    def __init__(self):
        self.intent_node = IntentNode()
        self.priority_node = PriorityNode()
        self.policy_node = PolicyNode()
        self.draft_node = DraftNode()
        self.validation_node = ValidationNode()
        self.router_node = RouterNode()

    def run_workflow(self, message: str):
        intent = self.intent_node.process(message)
        priority = self.priority_node.process(message, intent)
        policy = self.policy_node.process(intent)
        draft = self.draft_node.process(message, intent, priority, policy)
        validation = self.validation_node.process(draft, intent, policy)
        routing = self.router_node.process(priority, validation, draft)

        if intent.source == "grpc_intent_service":
            engine_mode = "grpc_intent_service"
        elif intent.source.startswith("fallback_rules_after_grpc_error"):
            engine_mode = "grpc_unavailable_local_fallback"
        else:
            engine_mode = intent.source
        trace = WorkflowTrace(
            intent=intent,
            priority=priority,
            policy=policy,
            draft=draft,
            validation=validation,
            routing=routing,
            engine_mode=engine_mode,
        )
        return routing, trace
