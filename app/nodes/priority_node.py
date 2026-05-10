class PriorityNode:
    def process(self, message: str, intent: str) -> str:
        high_priority_intents = ["card_lost", "account_blocked"]
        urgent_keywords = ["khẩn cấp", "bị lừa", "mất tiền"]
        
        if intent in high_priority_intents or any(k in message.lower() for k in urgent_keywords):
            return "High"
        return "Medium" if intent != "general_inquiry" else "Low"