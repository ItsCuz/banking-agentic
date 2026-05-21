from app.clients.base import BaseLLMClient
from app.core.settings import settings


class OllamaClient(BaseLLMClient):
    def __init__(self):
        self.url = f"{settings.OLLAMA_URL}/api/generate"
        self.headers = {"x-Pinggy-No-Screen": "true"}

    def generate(self, prompt: str) -> str:
        try:
            import requests
        except ImportError as exc:
            raise RuntimeError("requests is not installed; using local draft fallback") from exc

        payload = {
            "model": settings.MODEL_NAME,
            "prompt": prompt,
            "stream": False,
        }
        response = requests.post(
            self.url,
            json=payload,
            headers=self.headers,
            timeout=settings.OLLAMA_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()
