class ValidationNode:
    def process(self, draft_reply: str, intent: str) -> bool:
        """Kiểm tra câu trả lời có hợp lệ không"""
        # 1. Kiểm tra độ dài (không được quá ngắn)
        if len(draft_reply) < 30:
            return False
        
        # 2. Kiểm tra từ khóa bắt buộc theo Intent
        if intent == "card_lost" and "khóa" not in draft_reply.lower():
            return False
            
        # 3. Kiểm tra xem có bị lỗi kết nối không
        if "Lỗi kết nối" in draft_reply:
            return False
            
        return True