from app.core.schemas import IntentResult
from app.core.settings import settings
from app.clients.grpc_intent_client import GrpcIntentClient
from app.nodes.inference import IntentClassification


class IntentNode:
    def __init__(self):
        self.grpc_client = GrpcIntentClient()
        self.classifier = None if settings.USE_GRPC_INTENT else IntentClassification(settings.INTENT_MODEL_CONFIG)

    def process(self, message: str) -> IntentResult:
        if settings.USE_GRPC_INTENT:
            try:
                response = self.grpc_client.predict(message)
                return IntentResult(
                    label=response.intent,
                    confidence=response.confidence,
                    source="grpc_intent_service",
                )
            except Exception:
                pass

        if self.classifier is None:
            self.classifier = IntentClassification(settings.INTENT_MODEL_CONFIG)
        label = self.classifier(message)
        source = "fallback_rules" if self.classifier.fallback_mode else "fine_tuned_unsloth_checkpoint"
        confidence = 0.65 if self.classifier.fallback_mode else None
        return IntentResult(label=label, confidence=confidence, source=source)
