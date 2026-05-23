import os
from pathlib import Path


class Settings:
    ROOT_DIR = Path(__file__).resolve().parents[2]

    OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
    MODEL_NAME = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")
    INTENT_SERVICE_HOST = os.getenv("INTENT_SERVICE_HOST", "localhost")
    INTENT_SERVICE_PORT = int(os.getenv("INTENT_SERVICE_PORT", "50051"))
    USE_GRPC_INTENT = os.getenv("USE_GRPC_INTENT", "false").lower() == "true"
    INTENT_MODEL_CONFIG = os.getenv(
        "INTENT_MODEL_CONFIG",
        str(ROOT_DIR / "configs" / "inference.yaml"),
    )
    OLLAMA_TIMEOUT_SECONDS = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "180"))


settings = Settings()
