import logging
import os
from concurrent import futures

import grpc

import intent_service_pb2
import intent_service_pb2_grpc
from intent_classifier import IntentClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntentService(intent_service_pb2_grpc.IntentServiceServicer):
    def __init__(self):
        self.classifier = IntentClassifier()

    def IntentRecognizer(self, request, context):
        intent, confidence, reason = self.classifier.predict(request.message)
        return intent_service_pb2.IntentResponse(
            intent=intent,
            confidence=float(confidence),
            reason=reason,
        )


def serve():
    port = int(os.getenv("GRPC_PORT", "50051"))
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    intent_service_pb2_grpc.add_IntentServiceServicer_to_server(IntentService(), server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    logger.info("Intent gRPC service listening on port %s", port)
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
