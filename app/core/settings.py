import os

class Settings:
    # Thay URL này bằng URL từ Pinggy của bạn
    OLLAMA_URL = "http://your-pinggy-url.a.free.pinggy.link" 
    MODEL_NAME = "gpt-oss:20b"
    # Đường dẫn đến checkpoint Lab 2 (nếu có)
    INTENT_MODEL_PATH = "./models/intent_classifier_lab2"

settings = Settings()