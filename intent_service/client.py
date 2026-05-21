import argparse

import grpc

import intent_service_pb2
import intent_service_pb2_grpc


def main():
    parser = argparse.ArgumentParser(description="Test the intent gRPC service.")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", default="50051")
    parser.add_argument("--message", required=True)
    args = parser.parse_args()

    with grpc.insecure_channel(f"{args.host}:{args.port}") as channel:
        stub = intent_service_pb2_grpc.IntentServiceStub(channel)
        response = stub.IntentRecognizer(intent_service_pb2.IntentRequest(message=args.message))
        print(f"intent={response.intent}")
        print(f"confidence={response.confidence:.2f}")
        print(f"reason={response.reason}")


if __name__ == "__main__":
    main()
