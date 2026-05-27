# Banking AI-Agent Deployment

This project implements the Lab 4 deployment architecture for a banking AI-agent system using REST, gRPC, Docker, and Docker Compose.

The system is split into multiple services:

- `api-gateway`: FastAPI service that receives external HTTP requests and orchestrates the workflow.
- `intent-service`: gRPC microservice that predicts the banking intent of a customer message.
- `ollama`: HTTP service for response generation.
- `frontend`: Streamlit interface for interacting with the system.

## Architecture

```text
Frontend
  -> API Gateway /run-agent
      -> Intent Service through gRPC
      -> Ollama through HTTP /api/generate
      -> Priority, policy, validation, and routing nodes
  -> Final response + workflow trace
```

The API Gateway returns both the final output and the intermediate workflow trace, including intent detection, priority detection, policy retrieval, draft generation, validation, and routing.

## Project Structure

```text
banking-agentic
|-- app
|   |-- agent/orchestrator.py
|   |-- clients
|   |   |-- base.py
|   |   |-- grpc_intent_client.py
|   |   `-- ollama_client.py
|   |-- core
|   |   |-- schemas.py
|   |   `-- settings.py
|   |-- data/policies.py
|   |-- intent_grpc
|   |   |-- intent_service.proto
|   |   |-- intent_service_pb2.py
|   |   `-- intent_service_pb2_grpc.py
|   |-- main.py
|   `-- nodes
|       |-- draft_node.py
|       |-- intent_node.py
|       |-- policy_node.py
|       |-- priority_node.py
|       |-- router_node.py
|       `-- validation_node.py
|-- intent_service
|   |-- intent_service.proto
|   |-- intent_service_pb2.py
|   |-- intent_service_pb2_grpc.py
|   |-- intent_classifier.py
|   |-- server.py
|   |-- client.py
|   |-- Dockerfile
|   |-- Makefile
|   `-- requirements.txt
|-- frontend
|   |-- interface.py
|   |-- Dockerfile
|   `-- requirements.txt
|-- checkpoints/final_model
|-- configs/inference.yaml
|-- Dockerfile
|-- docker-compose.yml
|-- requirements-api.txt
|-- .env.example
`-- README.md
```

## API Gateway

The API Gateway is implemented with FastAPI.

Endpoints:

```text
GET  /health
GET  /config
POST /run-agent
```

Example request:

```bash
curl -X POST http://localhost:8000/run-agent ^
  -H "Content-Type: application/json" ^
  -d "{\"message\":\"I made a bank transfer this morning, but the recipient has not received the money yet.\"}"
```

## Intent Service gRPC

The gRPC interface is defined in:

```text
intent_service/intent_service.proto
```

The API Gateway uses a generated client under:

```text
app/intent_grpc/
```

The Intent Service exposes:

```proto
service IntentService {
  rpc IntentRecognizer (IntentRequest) returns (IntentResponse) {}
}
```

Response fields:

```text
intent
confidence
reason
```

The service loads the fine-tuned LoRA checkpoint from:

```text
checkpoints/final_model
```

The default base model is:

```text
unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit
```

## Generate gRPC Code

Generated gRPC files are already included. To regenerate them:

```bash
cd intent_service
pip install -r requirements.txt
make
```

This generates:

```text
intent_service_pb2.py
intent_service_pb2_grpc.py
```

## Docker Compose

Run the full system:

```bash
docker compose up --build
```

Open:

```text
Frontend:    http://localhost:8501
API Gateway: http://localhost:8000
Ollama:      http://localhost:11434
gRPC:        localhost:50051
```

Note: `localhost:50051` is a gRPC port, so it is not meant to be opened in a browser.

## Ollama Configuration

By default, Docker Compose uses the `ollama` container:

```text
OLLAMA_URL=http://ollama:11434
OLLAMA_MODEL=gpt-oss:20b
```

If the model is not available in the Ollama container, pull it:

```bash
docker exec -it banking-ollama ollama pull gpt-oss:20b
```

To use Ollama from Google Colab through Pinggy, create a local `.env` file:

```bash
OLLAMA_URL=https://your-pinggy-link.a.free.pinggy.link
OLLAMA_MODEL=gpt-oss:20b
```

Do not include `/api/generate` in `OLLAMA_URL`.

## Useful Commands

Check running containers:

```bash
docker ps
```

Check API Gateway logs:

```bash
docker logs banking-api-gateway --tail 100
```

Check Intent Service logs:

```bash
docker logs banking-intent-service --tail 100
```

Test the gRPC Intent Service:

```bash
docker exec -it banking-intent-service python client.py --message "I made a transfer but the recipient has not received it."
```

Stop services:

```bash
docker compose down
```

Stop services and remove volumes:

```bash
docker compose down -v
```

## Demo Video

The Lab 4 video demonstration should show:

- A short overview of the microservice architecture.
- Running the system using Docker Compose.
- Calling the API Gateway or using the frontend.
- Showing the final workflow output and trace.

Video demo:

[Google Drive demo video](https://drive.google.com/file/d/1m-rsIFH_I-teaRP-LyGfTstxCTPe3fuW/view?usp=sharing)
