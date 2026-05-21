from app.core.schemas import DraftResult, IntentResult, PolicyResult, ValidationResult


class ValidationNode:
    def process(self, draft: DraftResult, intent: IntentResult, policy: PolicyResult) -> ValidationResult:
        lower_reply = draft.reply.lower()
        checks = {
            "minimum_length": len(draft.reply.strip()) >= 40,
            "policy_grounding": draft.source == "local_template_fallback"
            or self._has_policy_overlap(lower_reply, policy.snippet.lower()),
            "no_connection_error": "connection" not in lower_reply and "lỗi kết nối" not in lower_reply,
            "intent_specific_keyword": self._has_intent_keyword(lower_reply, intent.label),
        }
        issues = [name for name, passed in checks.items() if not passed]
        return ValidationResult(passed=not issues, checks=checks, issues=issues)

    def _has_policy_overlap(self, reply: str, policy: str) -> bool:
        policy_terms = [term for term in policy.replace(".", " ").replace(",", " ").split() if len(term) > 5]
        if not policy_terms:
            return True
        matches = sum(1 for term in policy_terms[:12] if term.lower() in reply)
        return matches >= 1

    def _has_intent_keyword(self, reply: str, intent: str) -> bool:
        required = {
            "fraud_alert": ["otp", "pin", "khóa", "gian lận"],
            "card_lost": ["khóa", "thẻ"],
            "account_blocked": ["xác minh", "mở khóa", "tài khoản"],
            "transfer_issue": ["mã giao dịch", "đối soát", "chuyển"],
            "refund_request": ["hoàn tiền", "mã giao dịch", "7 ngày"],
            "card_not_received": ["giao thẻ", "phát hành", "địa chỉ"],
        }
        keywords = required.get(intent)
        if not keywords:
            return True
        return any(keyword in reply for keyword in keywords)
