# Colab notebook script for Lab 2.
# Copy each section into Google Colab cells and run from top to bottom.

# %% [1] Install dependencies
import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "unsloth"])
subprocess.check_call(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "datasets",
        "trl",
        "peft",
        "accelerate",
        "bitsandbytes",
        "scikit-learn",
        "pandas",
        "pyyaml",
        "tqdm",
    ]
)


# %% [2] Create folders and configs
import os
import json
import random
import shutil
from pathlib import Path

import pandas as pd
import torch
import yaml
from datasets import Dataset, load_dataset
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

PROJECT_LABELS = [
    "transfer_issue",
    "card_not_received",
    "card_lost",
    "account_blocked",
    "refund_request",
    "fraud_alert",
    "general_inquiry",
]

BASE_MODEL = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"
OUTPUT_DIR = "checkpoints/final_model"

os.makedirs("configs", exist_ok=True)
os.makedirs("sample_data", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

train_config = {
    "dataset_name": "legacy-datasets/banking77",
    "base_model_name": BASE_MODEL,
    "output_dir": OUTPUT_DIR,
    "max_seq_length": 512,
    "batch_size": 8,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4,
    "epochs": 3,
    "optimizer": "adamw_8bit",
    "lora_r": 16,
    "lora_alpha": 16,
    "lora_dropout": 0.0,
    "random_state": 42,
    "sample_per_source_intent": 150,
}

with open("configs/train.yaml", "w", encoding="utf-8") as f:
    yaml.safe_dump(train_config, f, sort_keys=False)

with open("configs/inference.yaml", "w", encoding="utf-8") as f:
    yaml.safe_dump(
        {
            "checkpoint_path": OUTPUT_DIR,
            "base_model_name": BASE_MODEL,
            "load_in_4bit": True,
            "require_cuda": False,
            "max_new_tokens": 16,
        },
        f,
        sort_keys=False,
    )

print("Config ready:", train_config)


# %% [3] Load BANKING77 and build project-label subset
dataset = load_dataset("legacy-datasets/banking77")
train_raw = pd.DataFrame(dataset["train"])
test_raw = pd.DataFrame(dataset["test"])
label_names = dataset["train"].features["label"].names

train_raw["source_intent"] = train_raw["label"].apply(lambda x: label_names[x])
test_raw["source_intent"] = test_raw["label"].apply(lambda x: label_names[x])

# BANKING77 labels are mapped to the project labels used by the agent.
# This keeps Lab 2 training compatible with Lab 3/4 policy and routing nodes.
LABEL_ALIASES = {
    "transfer_not_received": "transfer_issue",
    "beneficiary_not_verified": "transfer_issue",
    "bank_card": "card_not_received",
    "card_arrival": "card_not_received",
    "cash_withdrawal_card": "card_lost",
    "cash_withdrawal": "refund_request",
    "cash_withdrawal_pending": "transfer_issue",
    "cash_withdrawal_reverted": "refund_request",
    "cash_withdrawal_fee_charged": "refund_request",
    "cash_withdrawal_wrong_exchange_rate": "refund_request",
    "card_not_received": "card_not_received",
}

selected_source_intents = sorted(LABEL_ALIASES.keys())
train_raw = train_raw[train_raw["source_intent"].isin(selected_source_intents)].copy()
test_raw = test_raw[test_raw["source_intent"].isin(selected_source_intents)].copy()

sample_per_intent = train_config["sample_per_source_intent"]
train_df = (
    train_raw.groupby("source_intent", group_keys=False)
    .apply(lambda g: g.sample(min(len(g), sample_per_intent), random_state=42))
    .reset_index(drop=True)
)
test_df = test_raw.reset_index(drop=True)

train_df["message"] = train_df["text"]
test_df["message"] = test_df["text"]
train_df["intent"] = train_df["source_intent"].map(LABEL_ALIASES)
test_df["intent"] = test_df["source_intent"].map(LABEL_ALIASES)

# Add a small project-specific support set for labels that are not directly
# represented in BANKING77 but are required by the agent workflow.
synthetic_examples = [
    ("Someone called me asking for my OTP and said they are from the bank.", "fraud_alert"),
    ("I shared my OTP with a suspicious caller and I think my account is at risk.", "fraud_alert"),
    ("There is an unknown transaction and I suspect fraud on my card.", "fraud_alert"),
    ("My account is blocked after I entered the wrong password three times.", "account_blocked"),
    ("I cannot login because my account has been locked.", "account_blocked"),
    ("The banking app says my account is temporarily blocked.", "account_blocked"),
    ("What are your working hours?", "general_inquiry"),
    ("How can I contact customer support?", "general_inquiry"),
    ("I want to know the bank hotline number.", "general_inquiry"),
]
synthetic_df = pd.DataFrame(synthetic_examples, columns=["message", "intent"])
synthetic_df["source_intent"] = "project_support_example"

train_df = pd.concat([train_df[["message", "intent", "source_intent"]], synthetic_df], ignore_index=True)
test_df = test_df[["message", "intent", "source_intent"]]

label_to_id = {label: i for i, label in enumerate(PROJECT_LABELS)}
with open("sample_data/label_mapping.json", "w", encoding="utf-8") as f:
    json.dump(label_to_id, f, indent=2, ensure_ascii=False)

train_df[["message", "intent", "source_intent"]].to_csv("sample_data/train.csv", index=False)
test_df[["message", "intent", "source_intent"]].to_csv("sample_data/test.csv", index=False)

print("Train size:", len(train_df))
print("Test size:", len(test_df))
print(train_df["intent"].value_counts())
print(train_df.head())


# %% [4] Prompt formatting
INTENT_LIST = PROJECT_LABELS


def normalize_text(text):
    return " ".join(str(text).strip().split())


def build_train_prompt(message, intent):
    labels = ", ".join(INTENT_LIST)
    return f"""You are a banking intent classifier.

Classify the customer message into exactly one intent label.

Allowed intent labels:
{labels}

Customer message:
{normalize_text(message)}

Intent:
{intent}"""


def build_inference_prompt(message):
    labels = ", ".join(INTENT_LIST)
    return f"""You are a banking intent classifier.

Classify the customer message into exactly one intent label.

Allowed intent labels:
{labels}

Customer message:
{normalize_text(message)}

Intent:
"""


train_dataset = Dataset.from_dict(
    {
        "text": [
            build_train_prompt(row["message"], row["intent"])
            for _, row in train_df.iterrows()
        ]
    }
)

print(train_dataset[0]["text"])


# %% [5] Load lightweight model with Unsloth
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=train_config["max_seq_length"],
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=train_config["lora_r"],
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=train_config["lora_alpha"],
    lora_dropout=train_config["lora_dropout"],
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=train_config["random_state"],
)

model.print_trainable_parameters()


# %% [6] Train
from transformers import TrainingArguments
from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    dataset_text_field="text",
    max_seq_length=train_config["max_seq_length"],
    args=TrainingArguments(
        output_dir="checkpoints",
        per_device_train_batch_size=train_config["batch_size"],
        gradient_accumulation_steps=train_config["gradient_accumulation_steps"],
        learning_rate=train_config["learning_rate"],
        num_train_epochs=train_config["epochs"],
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        save_strategy="epoch",
        optim=train_config["optimizer"],
        report_to="none",
    ),
)

trainer.train()


# %% [7] Save checkpoint
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

adapter_path = Path(OUTPUT_DIR) / "adapter_model.safetensors"
adapter_size = adapter_path.stat().st_size
print("Saved:", OUTPUT_DIR)
print("Adapter size:", adapter_size, "bytes")
assert adapter_size > 1_000_000, "Adapter file is too small. Training/save likely failed."


# %% [8] Standalone inference class required by Lab 2
class IntentClassification:
    def __init__(self, model_path):
        with open(model_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        self.checkpoint_path = cfg["checkpoint_path"]
        self.max_new_tokens = int(cfg.get("max_new_tokens", 16))

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.checkpoint_path,
            max_seq_length=512,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(self.model)

    def __call__(self, message):
        prompt = build_inference_prompt(message)
        inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=0.0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        decoded = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        predicted = decoded.strip().splitlines()[0].replace("Intent:", "").strip()

        for label in INTENT_LIST:
            if label.lower() in predicted.lower():
                return label
        return "general_inquiry"


# %% [9] Quick test
classifier = IntentClassification("configs/inference.yaml")

examples = [
    "I made a transfer yesterday but the recipient has not received it.",
    "Someone called me asking for my OTP and said they are from the bank.",
    "My account is blocked after I entered the wrong password three times.",
    "I ordered a card two weeks ago but it has not arrived.",
]

for msg in examples:
    print("Message:", msg)
    print("Predicted:", classifier(msg))
    print()


# %% [10] Evaluate
predictions = []
labels = test_df["intent"].tolist()
for msg in tqdm(test_df["message"].tolist()):
    predictions.append(classifier(msg))

acc = accuracy_score(labels, predictions)
print("Accuracy:", acc)
print(classification_report(labels, predictions, zero_division=0))


# %% [11] Zip artifacts for download
artifact_root = Path("qwen_light_intent_artifacts")
if artifact_root.exists():
    shutil.rmtree(artifact_root)
shutil.copytree("checkpoints", artifact_root / "checkpoints")
shutil.copytree("configs", artifact_root / "configs")
shutil.copytree("sample_data", artifact_root / "sample_data")
shutil.make_archive("qwen_light_intent_artifacts", "zip", artifact_root)

print("Download qwen_light_intent_artifacts.zip and copy checkpoints/, configs/, and sample_data/ into the project.")
