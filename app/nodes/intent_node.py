from app.core.schemas import NodeOutput

class IntentNode:
    def process(self, message: str) -> str:

        message_lower = message.lower()
        if "mất thẻ" in message_lower or "khóa thẻ" in message_lower:
            return "card_lost"
        if "chuyển tiền" in message_lower or "không nhận được tiền" in message_lower:
            return "transfer_issue"
        return "general_inquiry"