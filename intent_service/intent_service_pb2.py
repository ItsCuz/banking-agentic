from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder

_sym_db = _symbol_database.Default()

DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(
    b'\n\x14intent_service.proto\x12\x12intent_classify.v1"\x20\n\rIntentRequest\x12\x0f\n\x07message\x18\x01 \x01(\t"@\n\x0eIntentResponse\x12\x0e\n\x06intent\x18\x01 \x01(\t\x12\x12\n\nconfidence\x18\x02 \x01(\x02\x12\x0e\n\x06reason\x18\x03 \x01(\t2s\n\rIntentService\x12b\n\x10IntentRecognizer\x12!.intent_classify.v1.IntentRequest\x1a".intent_classify.v1.IntentResponse"\x00\x62\x06proto3'
)

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, "intent_service_pb2", _globals)
