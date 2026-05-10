from app.nodes.intent_node import IntentNode
from app.nodes.priority_node import PriorityNode
from app.nodes.policy_node import PolicyNode
from app.nodes.draft_node import DraftNode
from app.nodes.router_node import RouterNode

class Orchestrator:
    def __init__(self):
        self.intent_node = IntentNode()
        self.priority_node = PriorityNode()
        self.policy_node = PolicyNode()
        self.draft_node = DraftNode()
        self.router_node = RouterNode()

    def run_workflow(self, message: str):
        # Thực hiện luồng theo trình tự yêu cầu
        intent = self.intent_node.process(message)
        priority = self.priority_node.process(message, intent)
        policy = self.policy_node.process(intent)
        draft = self.draft_node.process(message, intent, priority, policy)
        
        # Validation đơn giản (ví dụ: câu trả lời không được quá ngắn)
        is_valid = len(draft) > 50 
        decision = self.router_node.process(priority, is_valid)

        trace = {
            "intent": intent,
            "priority": priority,
            "policy": policy,
            "draft": draft,
            "decision": decision
        }
        return decision, trace