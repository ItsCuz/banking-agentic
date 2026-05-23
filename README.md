# Banking Intent + AI-Agent Workflow

This repository combines the requirements of two labs:

- **Project 2:** fine-tune an intent detection model on a sampled BANKING77 subset with Unsloth.
- **Project 3:** build a Banking AI-Agent workflow with intent detection, priority detection, policy retrieval, response drafting, validation, and routing/escalation.
- **Project 4:** deploy the system as a REST + gRPC microservice application with Docker Compose.

## Project Structure

```text
banking-agentic
|-- app
|   |-- agent/orchestrator.py
|   |-- clients/base.py
|   |-- clients/ollama_client.py
|   |-- core/settings.py
|   |-- core/schemas.py
|   |-- data/policies.py
|   |-- main.py
|   `-- nodes
|       |-- intent_node.py
|       |-- inference.py
|       |-- priority_node.py
|       |-- policy_node.py
|       |-- draft_node.py
|       |-- validation_node.py
|       `-- router_node.py
|-- scripts
|   |-- preprocess_data.py
|   |-- train.py
|   |-- evaluate.py
|   `-- inference.py
|-- configs
|   |-- train.yaml
|   `-- inference.yaml
|-- sample_data
|   |-- train.csv
|   |-- test.csv
|   `-- label_mapping.json
|-- examples/sample_requests.json
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
|-- Dockerfile
|-- docker-compose.yml
|-- requirements-api.txt
|-- train.sh
|-- inference.sh
|-- run.py
|-- requirements.txt
`-- README.md
```

## Lab 2: Intent Detection

The standalone inference interface is implemented in `app/nodes/inference.py`:

```python
class IntentClassification:
    def __init__(self, model_path):
        ...

    def __call__(self, message):
        return predicted_label
```

`model_path` points to `configs/inference.yaml`, which contains the checkpoint path and model loading settings.

### Data Preparation

```bash
pip install -r requirements.txt
python scripts/preprocess_data.py --config configs/train.yaml
```

This loads BANKING77, normalizes text, maps labels, samples selected intents, and writes:

- `sample_data/train.csv`
- `sample_data/test.csv`
- `sample_data/label_mapping.json`

### Training with Unsloth

Run training in a GPU environment such as Google Colab or Kaggle:

```bash
python scripts/train.py --config configs/train.yaml --train-file sample_data/train.csv
```

For the recommended lightweight Colab workflow, use:

```text
scripts/colab_finetune_light_intent.py
```

Copy its sections into Google Colab and run them from top to bottom. It fine-tunes:

```text
unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit
```

This model is much lighter than Llama-3-8B and is better suited for RTX 4050-class GPUs.

Main hyperparameters are documented in `configs/train.yaml`:

- base model: `unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit`
- batch size: `8`
- learning rate: `0.0002`
- optimizer: `adamw_8bit`
- epochs: `3`
- max sequence length: `512`
- LoRA rank/alpha/dropout: `16 / 16 / 0.0`

The checkpoint is saved to `checkpoints/final_model`.

### Standalone Inference

```bash
python scripts/inference.py --model-path configs/inference.yaml --message "My transfer was debited but the receiver did not get the money."
```

If GPU/model dependencies are unavailable, the class falls back to deterministic banking keyword rules so the demo remains runnable.

### Test Accuracy

```bash
python scripts/evaluate.py --model-path configs/inference.yaml --test-file sample_data/test.csv
```

The script prints each prediction and the final accuracy on the independent test file.

## Lab 3: Banking AI-Agent

The workflow is controlled by `app/agent/orchestrator.py` and runs these nodes in order:

1. Intent Detection Node
2. Priority or Risk Detection Node
3. Policy Retrieval Node
4. Response Drafting Node
5. Validation Node
6. Router/Escalation Node

The API returns both the final response and a full trace of every node output.

## Run the Agent

Install dependencies:

```bash
pip install -r requirements.txt
```

Optional Ollama configuration:

```bash
set OLLAMA_URL=http://localhost:11434
set OLLAMA_MODEL=gpt-oss:20b
```

Start FastAPI:

```bash
python run.py
```

Open:

```text
http://localhost:8000
```

API example:

```bash
curl -X POST http://localhost:8000/run-agent ^
  -H "Content-Type: application/json" ^
  -d "{\"message\":\"Someone asked me for my OTP and I suspect fraud.\"}"
```

## Lab 4: REST, gRPC, Docker, Docker Compose

The Lab 4 deployment separates the prototype into services:

- `api-gateway`: FastAPI REST gateway. It exposes `/health`, `/config`, and `/run-agent`.
- `intent-service`: independent gRPC service for intent recognition.
- `ollama`: HTTP LLM server used by the response drafting node.
- `frontend`: Streamlit interface for user interaction.

Runtime communication:

```text
Frontend -> API Gateway /run-agent
API Gateway -> Intent Service via gRPC
API Gateway -> Ollama /api/generate via HTTP
API Gateway -> priority, policy, validation, routing nodes locally
```

### Generate gRPC Code

The generated files are already committed, but they can be regenerated from the proto file:

```bash
cd intent_service
pip install -r requirements.txt
make
```

The API Gateway has a copy of the generated bindings under `app/intent_grpc/`.

### Run Intent Service Locally

```bash
cd intent_service
pip install -r requirements.txt
python server.py
```

Test it:

```bash
python client.py --message "Someone asked me for my OTP."
```

### Run API Gateway with gRPC Intent Service

In another terminal:

```bash
set USE_GRPC_INTENT=true
set INTENT_SERVICE_HOST=localhost
set INTENT_SERVICE_PORT=50051
set OLLAMA_URL=http://localhost:11434
python run.py
```

### Docker Build

Build each service:

```bash
docker build -t banking-api-gateway .
docker build -t banking-intent-service ./intent_service
docker build -t banking-frontend ./frontend
```

### Docker Compose

Run the full system:

```bash
docker compose up --build
```

Open:

```text
API Gateway: http://localhost:8000
Frontend: http://localhost:8501
Intent gRPC: localhost:50051
Ollama HTTP: http://localhost:11434
```

If the Ollama container does not already have `gpt-oss:20b`, pull the model:

```bash
docker exec -it banking-ollama ollama pull gpt-oss:20b
```

Call the gateway:

```bash
curl -X POST http://localhost:8000/run-agent ^
  -H "Content-Type: application/json" ^
  -d "{\"message\":\"My account is blocked after entering the wrong password.\"}"
```

The intent service mounts `./checkpoints` into the container so it can load the Lab 2 LoRA checkpoint from `checkpoints/final_model`. If ML dependencies or GPU are unavailable, it uses fallback rules for demo continuity and returns a lower confidence score.

The default intent base model for Lab 4 is:

```text
unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit
```

After fine-tuning in Colab, download `qwen_light_intent_artifacts.zip`, then copy the generated `checkpoints/`, `configs/`, and `sample_data/` folders into this project before rebuilding Docker Compose.

## Demo Video

Add the public Google Drive demo URL here before submission:

```text
Video demo: TODO
```
