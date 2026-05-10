from app.nodes.intent_node import IntentNode
from app.nodes.priority_node import PriorityNode
from app.nodes.policy_node import PolicyNode
from app.nodes.draft_node import DraftNode
from app.nodes.validation_node import ValidationNode
from app.nodes.router_node import RouterNode

class Orchestrator:
    def __init__(self):
        # Khởi tạo tất cả các node [cite: 181-191]
        self.intent_node = IntentNode()
        self.priority_node = PriorityNode()
        self.policy_node = PolicyNode()
        self.draft_node = DraftNode()
        self.validation_node = ValidationNode()
        self.router_node = RouterNode()

    def run_workflow(self, message: str):
        # Luồng chạy tuần tự [cite: 30]
        intent = self.intent_node.process(message)
        priority = self.priority_node.process(message, intent)
        policy = self.policy_node.process(intent)
        
        # Tạo câu trả lời nháp từ LLM
        draft = self.draft_node.process(message, intent, priority, policy)
        
        # Kiểm tra chất lượng câu trả lời
        is_valid = self.validation_node.process(draft, intent)
        
        # Ra quyết định cuối cùng
        decision = self.router_node.process(priority, is_valid)

        # Trả về kết quả kèm trace để giảng viên chấm điểm [cite: 192]
        return decision, {
            "intent": intent,
            "priority": priority,
            "policy": policy,
            "draft": draft,
            "validation": "Pass" if is_valid else "Fail",
            "decision": decision
        }