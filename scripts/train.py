import argparse
from pathlib import Path

import pandas as pd
import yaml


def format_prompt(text: str, label: str) -> str:
    return (
        "Classify the banking customer message into one intent label.\n"
        f"Message: {text}\n"
        f"Intent: {label}"
    )


def main():
    parser = argparse.ArgumentParser(description="Fine-tune an Unsloth intent model on BANKING77.")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--train-file", default="sample_data/train.csv")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    try:
        from datasets import Dataset
        from trl import SFTTrainer, SFTConfig
        from unsloth import FastLanguageModel
    except ImportError as exc:
        raise SystemExit(
            "Training requires Unsloth, TRL, datasets, torch, and transformers. "
            "Install the training dependencies in Colab/Kaggle before running this script."
        ) from exc

    train_df = pd.read_csv(args.train_file)
    text_column = "message" if "message" in train_df.columns else "text"
    label_column = "intent" if "intent" in train_df.columns else "label"
    train_df["text"] = train_df.apply(lambda row: format_prompt(row[text_column], row[label_column]), axis=1)
    train_dataset = Dataset.from_pandas(train_df[["text"]])

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["base_model_name"],
        max_seq_length=cfg["max_seq_length"],
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["lora_r"],
        target_modules=cfg["target_modules"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=cfg["random_state"],
    )

    training_args = SFTConfig(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        optim=cfg["optimizer"],
        num_train_epochs=cfg["epochs"],
        max_steps=cfg["max_steps"],
        max_seq_length=cfg["max_seq_length"],
        logging_steps=10,
        save_strategy="steps",
        save_steps=250,
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        args=training_args,
    )
    trainer.train()

    output_dir = Path(cfg["output_dir"])
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved checkpoint to {output_dir}.")


if __name__ == "__main__":
    main()
