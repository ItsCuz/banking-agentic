from app.core.settings import settings
from app.nodes.inference import IntentClassification

class IntentNode:
    def __init__(self):
        # Khởi tạo mô hình bằng interface Lab 2 đã viết [cite: 255, 259]
        # Điều này nạp Tokenizer và Checkpoint vào GPU
        self.classifier = IntentClassification(settings.INTENT_MODEL_CONFIG)

    def process(self, message: str) -> str:
        # Gọi phương thức __call__ để dự đoán nhãn [cite: 256, 260-262]
        # Nhãn trả về sẽ là các class từ BANKING77 (ví dụ: 'card_lost')
        predicted_intent = self.classifier(message)
        
        return predicted_intent