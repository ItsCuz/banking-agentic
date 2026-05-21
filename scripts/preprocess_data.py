import argparse
from pathlib import Path

import pandas as pd
import yaml
from datasets import load_dataset
from sklearn.model_selection import train_test_split


def normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def main():
    parser = argparse.ArgumentParser(description="Prepare a sampled BANKING77 subset.")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--output-dir", default="sample_data")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    dataset = load_dataset(cfg["dataset_name"], split="train")
    label_names = dataset.features["label"].names
    selected = set(cfg["selected_intents"])
    label_aliases = cfg.get("label_aliases", {})
    rows = []

    for item in dataset:
        label = label_names[item["label"]]
        if label not in selected:
            continue
        rows.append({"text": normalize_text(item["text"]), "label": label_aliases.get(label, label)})

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("No rows matched selected_intents in the config.")

    sampled = (
        frame.groupby("label", group_keys=False)
        .apply(lambda group: group.sample(min(len(group), cfg["sample_per_intent"]), random_state=cfg["random_state"]))
        .reset_index(drop=True)
    )
    label_to_id = {label: index for index, label in enumerate(sorted(sampled["label"].unique()))}
    sampled["label_id"] = sampled["label"].map(label_to_id)

    train_df, test_df = train_test_split(
        sampled,
        test_size=cfg["test_size"],
        random_state=cfg["random_state"],
        stratify=sampled["label"],
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(output_dir / "train.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)
    pd.Series(label_to_id).to_json(output_dir / "label_mapping.json", force_ascii=False, indent=2)

    print(f"Saved {len(train_df)} train rows and {len(test_df)} test rows to {output_dir}.")
    print(f"Labels: {label_to_id}")


if __name__ == "__main__":
    main()
