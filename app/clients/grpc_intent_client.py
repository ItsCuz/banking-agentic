from app.core.settings import settings


class GrpcIntentClient:
    def __init__(self, host=None, port=None):
        self.host = host or settings.INTENT_SERVICE_HOST
        self.port = port or settings.INTENT_SERVICE_PORT
        self.target = f"{self.host}:{self.port}"

    def predict(self, message: str):
        try:
            import grpc
            from app.intent_grpc import intent_service_pb2, intent_service_pb2_grpc
        except ImportError as exc:
            raise RuntimeError("grpcio/protobuf dependencies are not installed") from exc

        with grpc.insecure_channel(self.target) as channel:
            stub = intent_service_pb2_grpc.IntentServiceStub(channel)
            request = intent_service_pb2.IntentRequest(message=message)
            return stub.IntentRecognizer(request, timeout=5)
