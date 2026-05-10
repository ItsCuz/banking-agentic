import requests
from app.core.settings import settings

class DraftNode:
    def process(self, message: str, intent: str, priority: str, policy: str) -> str:
        # 1. Định nghĩa nội dung biến prompt (Dòng này rất quan trọng) [cite: 162-167]
        prompt = f"""
        Role: Banking Customer Support Agent.
        Context:
        - Customer Message: {message}
        - Detected Intent: {intent}
        - Priority: {priority}
        - Policy: {policy}
        Task: Draft a helpful and professional reply in Vietnamese.
        """
        
        # 2. Thêm header x-Pinggy-No-Screen để bypass trang cảnh báo bảo mật
        headers = {
            "x-Pinggy-No-Screen": "true" 
        }
        
        try:
            # 3. Đưa biến prompt vào payload để gửi đến Ollama [cite: 141-143]
            payload = {
                "model": settings.MODEL_NAME,
                "prompt": prompt, 
                "stream": False
            }
            
            response = requests.post(
                f"{settings.OLLAMA_URL}/api/generate", 
                json=payload, 
                headers=headers,
                timeout=120
            )
            
            # Kiểm tra nếu request thành công
            response.raise_for_status()
            return response.json().get("response", "Lỗi: Không nhận được phản hồi từ AI.")
            
        except Exception as e:
            return f"Error connecting to LLM: {str(e)}"