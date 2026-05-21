import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.nodes.inference import IntentClassification


def main():
    parser = argparse.ArgumentParser(description="Run standalone banking intent inference.")
    parser.add_argument("--model-path", default="configs/inference.yaml")
    parser.add_argument("--message", required=True)
    args = parser.parse_args()

    classifier = IntentClassification(args.model_path)
    predicted_label = classifier(args.message)
    print(predicted_label)


if __name__ == "__main__":
    main()
