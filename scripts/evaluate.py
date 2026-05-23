import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.nodes.inference import IntentClassification


def main():
    parser = argparse.ArgumentParser(description="Evaluate standalone intent inference on a CSV test set.")
    parser.add_argument("--model-path", default="configs/inference.yaml")
    parser.add_argument("--test-file", default="sample_data/test.csv")
    args = parser.parse_args()

    classifier = IntentClassification(args.model_path)
    total = 0
    correct = 0

    with open(args.test_file, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            text = row.get("message") or row.get("text")
            expected = row.get("intent") or row.get("label")
            total += 1
            predicted = classifier(text)
            correct += int(predicted == expected)
            print(f"text={text} | expected={expected} | predicted={predicted}")

    accuracy = correct / total if total else 0.0
    print(f"accuracy={accuracy:.4f} ({correct}/{total})")


if __name__ == "__main__":
    main()
