from app.core.schemas import IntentResult
from app.core.settings import settings
from app.clients.grpc_intent_client import GrpcIntentClient
from app.nodes.inference import IntentClassification


class IntentNode:
    def __init__(self):
        self.grpc_client = GrpcIntentClient()
        self.classifier = None if settings.USE_GRPC_INTENT else IntentClassification(settings.INTENT_MODEL_CONFIG)
        self.last_grpc_error = None

    def _get_local_classifier(self):
        if self.classifier is None:
            self.classifier = IntentClassification(settings.INTENT_MODEL_CONFIG)
        return self.classifier

    def process(self, message: str) -> IntentResult:
        self.last_grpc_error = None
        if settings.USE_GRPC_INTENT:
            try:
                response = self.grpc_client.predict(message)
                return IntentResult(
                    label=response.intent,
                    confidence=response.confidence,
                    source="grpc_intent_service",
                )
            except Exception as exc:
                self.last_grpc_error = str(exc)

        classifier = self._get_local_classifier()
        label = classifier(message)
        fallback_mode = getattr(classifier, "fallback_mode", True)
        if self.last_grpc_error:
            source = f"fallback_rules_after_grpc_error: {self.last_grpc_error}"
        else:
            source = "fallback_rules" if fallback_mode else "fine_tuned_unsloth_checkpoint"
        confidence = 0.65 if fallback_mode else None
        return IntentResult(label=label, confidence=confidence, source=source)
