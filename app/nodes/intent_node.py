from app.core.schemas import IntentResult
from app.core.settings import settings
from app.nodes.inference import IntentClassification


class IntentNode:
    def __init__(self):
        self.classifier = IntentClassification(settings.INTENT_MODEL_CONFIG)

    def process(self, message: str) -> IntentResult:
        label = self.classifier(message)
        source = "fallback_rules" if self.classifier.fallback_mode else "fine_tuned_unsloth_checkpoint"
        confidence = 0.65 if self.classifier.fallback_mode else None
        return IntentResult(label=label, confidence=confidence, source=source)
