# Banking Intent + AI-Agent Workflow

This repository combines the requirements of two labs:

- **Project 2:** fine-tune an intent detection model on a sampled BANKING77 subset with Unsloth.
- **Project 3:** build a Banking AI-Agent workflow with intent detection, priority detection, policy retrieval, response drafting, validation, and routing/escalation.

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

Main hyperparameters are documented in `configs/train.yaml`:

- batch size: `4`
- learning rate: `0.0002`
- optimizer: `adamw_8bit`
- epochs: `3`
- max sequence length: `2048`
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
curl -X POST http://localhost:8000/chat ^
  -H "Content-Type: application/json" ^
  -d "{\"message\":\"Someone asked me for my OTP and I suspect fraud.\"}"
```

## Demo Video

Add the public Google Drive demo URL here before submission:

```text
Video demo: TODO
```
