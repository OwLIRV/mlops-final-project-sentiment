import json
import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def extract_text_and_label(item):
    text = None
    label = None

    # Простий формат
    if "text" in item:
        text = item.get("text")

    if "label" in item:
        label = item.get("label")

    if "sentiment" in item:
        label = item.get("sentiment")

    if "review" in item and text is None:
        text = item.get("review")

    # Формат Label Studio
    if "data" in item and isinstance(item["data"], dict):
        data = item["data"]

        if text is None:
            text = data.get("text") or data.get("review") or data.get("comment")

        if label is None:
            label = data.get("label") or data.get("sentiment")

    # Label Studio annotations
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
    parser.add_argument(
        "--input",
        default="data/labeled/reviews_labeled_v3.json"
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    rows = []

    for item in raw_data:
        text, label = extract_text_and_label(item)

        if text and label:
            rows.append(
                {
                    "text": text,
                    "label": label,
                }
            )

    df = pd.DataFrame(rows)

    train_df, test_df = train_test_split(
        df,
        test_size=0.25,
        random_state=42,
        stratify=df["label"],
    )

    train_df.to_csv(output_dir / "train.csv", index=False, encoding="utf-8")
    test_df.to_csv(output_dir / "test.csv", index=False, encoding="utf-8")

    print("Data split completed successfully")
    print(f"Input dataset: {input_path}")
    print(f"Total examples: {len(df)}")
    print(f"Train size: {len(train_df)}")
    print(f"Test size: {len(test_df)}")

    print("Train label distribution:")
    print(train_df["label"].value_counts())

    print("Test label distribution:")
    print(test_df["label"].value_counts())

    print("Files saved:")
    print(f"  {output_dir / 'train.csv'}")
    print(f"  {output_dir / 'test.csv'}")


if __name__ == "__main__":
    main()