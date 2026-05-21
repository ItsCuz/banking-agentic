from app.core.schemas import IntentResult, PriorityResult


class PriorityNode:
    def process(self, message: str, intent: IntentResult) -> PriorityResult:
        msg = message.lower()
        high_keywords = ["fraud", "scam", "otp", "lost money", "lừa đảo", "mất tiền", "khẩn cấp"]
        medium_keywords = ["failed", "blocked", "refund", "bị trừ", "khóa", "hoàn tiền"]

        if intent.label in {"fraud_alert", "card_lost", "account_blocked"}:
            return PriorityResult(level="High", reason=f"{intent.label} requires immediate control or verification.")
        if any(keyword in msg for keyword in high_keywords):
            return PriorityResult(level="High", reason="The message contains urgent risk keywords.")
        if intent.label in {"transfer_issue", "refund_request"} or any(keyword in msg for keyword in medium_keywords):
            return PriorityResult(level="Medium", reason="The issue may affect money movement or account access.")
        return PriorityResult(level="Low", reason="No immediate financial or security risk was detected.")
