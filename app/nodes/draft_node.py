import requests
from app.core.settings import settings

class DraftNode:
    def process(self, message: str, intent: str, priority: str, policy: str) -> str:
        prompt = f"""
        Role: Banking Customer Support Agent.
        Context:
        - Customer Message: {message}
        - Detected Intent: {intent}
        - Priority: {priority}
        - Policy: {policy}
        Task: Draft a helpful and professional reply in Vietnamese.
        """
        try:
            payload = {
                "model": settings.MODEL_NAME,
                "prompt": prompt,
                "stream": False
            }
            response = requests.post(f"{settings.OLLAMA_URL}/api/generate", json=payload)
            return response.json().get("response", "Xin lỗi, tôi gặp sự cố khi tạo câu trả lời.")
        except Exception as e:
            return f"Error connecting to LLM: {str(e)}"