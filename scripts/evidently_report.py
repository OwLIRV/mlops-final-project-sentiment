from pathlib import Path

import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset


REFERENCE_PATH = Path("data/processed/train.csv")
CURRENT_PATH = Path("data/processed/test.csv")
REPORT_PATH = Path("reports/data_drift_report.html")


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(str)

    df["text_length"] = df["text"].str.len()
    df["word_count"] = df["text"].str.split().str.len()
    df["avg_word_length"] = df["text_length"] / df["word_count"].replace(0, 1)

    return df[
        [
            "text_length",
            "word_count",
            "avg_word_length",
            "label",
        ]
    ]


def main():
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    reference_raw = pd.read_csv(REFERENCE_PATH)
    current_raw = pd.read_csv(CURRENT_PATH)

    reference_data = prepare_features(reference_raw)
    current_data = prepare_features(current_raw)

    report = Report(
        metrics=[
            DataDriftPreset(),
        ]
    )

    report.run(
        reference_data=reference_data,
        current_data=current_data,
    )

    report.save_html(str(REPORT_PATH))

    print("Evidently data drift report generated successfully.")
    print(f"Report path: {REPORT_PATH}")
    print(f"Reference dataset: {REFERENCE_PATH}")
    print(f"Current dataset: {CURRENT_PATH}")
    print(f"Reference rows: {len(reference_data)}")
    print(f"Current rows: {len(current_data)}")
    print("Features used for drift analysis:")
    print("  - text_length")
    print("  - word_count")
    print("  - avg_word_length")
    print("  - label")


if __name__ == "__main__":
    main()