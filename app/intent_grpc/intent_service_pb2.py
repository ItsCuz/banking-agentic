from google.protobuf import descriptor_pb2 as _descriptor_pb2
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder

_sym_db = _symbol_database.Default()


def _build_file_descriptor():
    file_proto = _descriptor_pb2.FileDescriptorProto()
    file_proto.name = "intent_service.proto"
    file_proto.package = "intent_classify.v1"
    file_proto.syntax = "proto3"

    request = file_proto.message_type.add()
    request.name = "IntentRequest"
    field = request.field.add()
    field.name = "message"
    field.number = 1
    field.label = _descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    field.type = _descriptor_pb2.FieldDescriptorProto.TYPE_STRING

    response = file_proto.message_type.add()
    response.name = "IntentResponse"
    field = response.field.add()
    field.name = "intent"
    field.number = 1
    field.label = _descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    field.type = _descriptor_pb2.FieldDescriptorProto.TYPE_STRING
    field = response.field.add()
    field.name = "confidence"
    field.number = 2
    field.label = _descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    field.type = _descriptor_pb2.FieldDescriptorProto.TYPE_FLOAT
    field = response.field.add()
    field.name = "reason"
    field.number = 3
    field.label = _descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    field.type = _descriptor_pb2.FieldDescriptorProto.TYPE_STRING

    service = file_proto.service.add()
    service.name = "IntentService"
    method = service.method.add()
    method.name = "IntentRecognizer"
    method.input_type = ".intent_classify.v1.IntentRequest"
    method.output_type = ".intent_classify.v1.IntentResponse"

    return file_proto.SerializeToString()


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(_build_file_descriptor())

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, "intent_service_pb2", _globals)
