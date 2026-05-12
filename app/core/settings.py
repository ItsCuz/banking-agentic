import os

class Settings:
    # Thay URL này bằng URL từ Pinggy của bạn
    #OLLAMA_URL = "http://your-pinggy-url.a.free.pinggy.link" 
    OLLAMA_URL = "https://vxula-34-125-182-189.run.pinggy-free.link" 
    MODEL_NAME = "gpt-oss:20b"
    # Đường dẫn đến checkpoint Lab 2 (nếu có)
    INTENT_MODEL_CONFIG = "configs/inference.yaml"

settings = Settings()