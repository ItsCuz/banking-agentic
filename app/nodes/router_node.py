class RouterNode:
    def process(self, priority: str, validation_ok: bool) -> str:
        # Quyết định dựa trên độ ưu tiên và kết quả validation
        if priority == "High" or not validation_ok:
            return "Escalate to Human"
        return "Send Directly"