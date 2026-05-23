from app.clients.ollama_client import OllamaClient
from app.core.schemas import DraftResult, IntentResult, PolicyResult, PriorityResult


class DraftNode:
    def __init__(self, llm_client=None):
        self.llm_client = llm_client or OllamaClient()

    def process(
        self,
        message: str,
        intent: IntentResult,
        priority: PriorityResult,
        policy: PolicyResult,
    ) -> DraftResult:
        prompt = self._build_prompt(message, intent, priority, policy)
        try:
            reply = self.llm_client.generate(prompt)
            if not reply:
                raise ValueError("empty LLM response")
            return DraftResult(
                reply=reply,
                missing_information=self._missing_information(message, intent.label),
                next_action=self._next_action(intent.label, priority.level),
                source="ollama",
            )
        except Exception as exc:
            return DraftResult(
                reply=self._generate_simulated_reply(intent.label, policy.snippet),
                missing_information=self._missing_information(message, intent.label),
                next_action=self._next_action(intent.label, priority.level),
                source="local_template_fallback",
                error=f"{type(exc).__name__}: {exc}",
            )

    def _build_prompt(
        self,
        message: str,
        intent: IntentResult,
        priority: PriorityResult,
        policy: PolicyResult,
    ) -> str:
        return f"""
You are a banking customer support agent.
Write a concise, professional reply in Vietnamese.
Use only the policy below as grounding.

Customer message: {message}
Detected intent: {intent.label}
Priority: {priority.level}
Policy: {policy.snippet}

Return the final customer-facing reply only.
""".strip()

    def _missing_information(self, message: str, intent: str):
        msg = message.lower()
        missing = []
        if intent in {"transfer_issue", "refund_request"}:
            if not any(token in msg for token in ["id", "mã", "reference", "transaction"]):
                missing.append("transaction_reference")
            if not any(char.isdigit() for char in msg):
                missing.append("amount_or_time")
        if intent in {"card_lost", "account_blocked", "fraud_alert"}:
            if not any(token in msg for token in ["cccd", "cmnd", "id", "phone", "số điện thoại"]):
                missing.append("identity_verification_detail")
        return missing

    def _next_action(self, intent: str, priority: str) -> str:
        if priority == "High":
            return "escalate_to_human_support"
        if intent in {"transfer_issue", "refund_request"}:
            return "request_transaction_details"
        return "send_policy_based_reply"

    def _generate_simulated_reply(self, intent: str, policy: str) -> str:
        templates = {
            "fraud_alert": (
                "Chúng tôi ghi nhận dấu hiệu rủi ro gian lận. Quý khách vui lòng không cung cấp OTP, PIN, "
                "mật khẩu hoặc thông tin thẻ cho bất kỳ ai, đồng thời khóa kênh giao dịch liên quan trên ứng dụng "
                "và chờ nhân viên hỗ trợ liên hệ xác minh."
            ),
            "card_lost": (
                "Chúng tôi rất tiếc về sự cố mất thẻ. Quý khách vui lòng khóa thẻ ngay trên ứng dụng ngân hàng "
                "hoặc gọi hotline 24/7. Sau khi khóa thẻ, ngân hàng sẽ hỗ trợ phát hành lại thẻ sau khi xác minh giấy tờ tùy thân."
            ),
            "account_blocked": (
                "Tài khoản của quý khách cần được xác minh trước khi mở khóa. Vui lòng chuẩn bị giấy tờ tùy thân "
                "và liên hệ hotline hoặc chi nhánh gần nhất để nhân viên kiểm tra nguyên nhân khóa tài khoản."
            ),
            "transfer_issue": (
                "Chúng tôi đã ghi nhận sự cố chuyển tiền của quý khách. Vui lòng cung cấp mã giao dịch, số tiền, "
                "ngân hàng nhận và thời điểm thực hiện để bộ phận hỗ trợ đối soát trong vòng 24 giờ làm việc."
            ),
            "refund_request": (
                "Chúng tôi đã tiếp nhận yêu cầu hoàn tiền. Vui lòng cung cấp mã giao dịch, tên đơn vị chấp nhận thanh toán, "
                "số tiền và thời điểm giao dịch để ngân hàng kiểm tra; thời gian xử lý có thể lên đến 7 ngày làm việc."
            ),
            "card_not_received": (
                "Chúng tôi sẽ hỗ trợ kiểm tra tình trạng phát hành và giao thẻ. Quý khách vui lòng cung cấp ngày đăng ký "
                "mở thẻ và địa chỉ nhận thẻ để ngân hàng xác minh với bộ phận vận chuyển."
            ),
        }
        return templates.get(
            intent,
            f"Cảm ơn quý khách đã liên hệ. Theo chính sách hiện tại: {policy} Vui lòng cung cấp thêm thông tin để chúng tôi hỗ trợ chính xác hơn.",
        )
