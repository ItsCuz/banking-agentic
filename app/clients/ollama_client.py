import requests
from app.clients.base import BaseLLMClient
from app.core.settings import settings

class OllamaClient(BaseLLMClient):
    def __init__(self):
        self.url = f"{settings.OLLAMA_URL}/api/generate"
        self.headers = {"x-Pinggy-No-Screen": "true"} # Vượt rào Pinggy

    def generate(self, prompt: str) -> str:
        payload = {
            "model": settings.MODEL_NAME,
            "prompt": prompt,
            "stream": False
        }
        try:
            response = requests.post(self.url, json=payload, headers=self.headers, timeout=120)
            response.raise_for_status()
            return response.json().get("response", "")
        except Exception as e:
            return f"Lỗi kết nối Ollama: {str(e)}"