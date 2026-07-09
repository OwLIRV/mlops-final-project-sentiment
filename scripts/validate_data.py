import json
import argparse
from collections import Counter
from pathlib import Path


def extract_text_and_label(item):
    text = None
    label = None

    # Варіант 1: простий формат
    if "text" in item:
        text = item.get("text")

    if "label" in item:
        label = item.get("label")

    if "sentiment" in item:
        label = item.get("sentiment")

    # Варіант 2: формат типу review/sentiment
    if "review" in item and text is None:
        text = item.get("review")

    # Варіант 3: Label Studio format
    if "data" in item and isinstance(item["data"], dict):
        data = item["data"]

        if text is None:
            text = data.get("text") or data.get("review") or data.get("comment")

        if label is None:
            label = data.get("label") or data.get("sentiment")

    # Варіант 4: Label Studio annotations
    if label is None and "annotations" in item:
        annotations = item.get("annotations", [])

        for annotation in annotations:
            results = annotation.get("result", [])

            for result in results:
                value = result.get("value", {})

                if "choices" in value and len(value["choices"]) > 0:
                    label = value["choices"][0]
                    break

                if "labels" in value and len(value["labels"]) > 0:
                    label = value["labels"][0]
                    break

            if label is not None:
                break

    if text is not None:
        text = str(text).strip()

    if label is not None:
        label = str(label).strip().lower()

    return text, label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/labeled/reviews_labeled_v3.json")
    args = parser.parse_args()

    path = Path(args.input)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Dataset must be a list of objects")

    labels = []
    empty_texts = 0
    empty_labels = 0

    for i, item in enumerate(data):
        text, label = extract_text_and_label(item)

        if not text:
            empty_texts += 1

        if not label:
            empty_labels += 1
        else:
            labels.append(label)

    print("Dataset validation completed successfully")
    print(f"Total examples: {len(data)}")
    print(f"Empty texts: {empty_texts}")
    print(f"Empty labels: {empty_labels}")

    print("Label distribution:")
    for label, count in Counter(labels).items():
        print(f"  {label}: {count}")

    if empty_texts > 0 or empty_labels > 0:
        print("Warning: some rows have missing text or label")
    else:
        print("All rows contain text and label")


if __name__ == "__main__":
    main()