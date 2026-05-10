from app.data.policies import BANKING_POLICIES

class PolicyNode:
    def process(self, intent: str) -> str:
        # Truy xuất chính sách dựa trên Intent đã dự đoán
        return BANKING_POLICIES.get(intent, "Liên hệ nhân viên để được hỗ trợ chi tiết.")