import grpc

from app.intent_grpc import intent_service_pb2 as intent__service__pb2


class IntentServiceStub(object):
    def __init__(self, channel):
        self.IntentRecognizer = channel.unary_unary(
            "/intent_classify.v1.IntentService/IntentRecognizer",
            request_serializer=intent__service__pb2.IntentRequest.SerializeToString,
            response_deserializer=intent__service__pb2.IntentResponse.FromString,
        )


class IntentServiceServicer(object):
    def IntentRecognizer(self, request, context):
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented")
        raise NotImplementedError("Method not implemented")


def add_IntentServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
        "IntentRecognizer": grpc.unary_unary_rpc_method_handler(
            servicer.IntentRecognizer,
            request_deserializer=intent__service__pb2.IntentRequest.FromString,
            response_serializer=intent__service__pb2.IntentResponse.SerializeToString,
        ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
        "intent_classify.v1.IntentService", rpc_method_handlers
    )
    server.add_generic_rpc_handlers((generic_handler,))
