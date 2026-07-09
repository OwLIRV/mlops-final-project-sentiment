import argparse
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline


TRAIN_PATH = Path("data/processed/train.csv")
TEST_PATH = Path("data/processed/test.csv")
MODEL_DIR = Path("models")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="sentiment-classifier")
    parser.add_argument("--register", action="store_true")
    args = parser.parse_args()

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("sentiment-classification-final-project")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    X_train = train_df["text"].astype(str)
    y_train = train_df["label"].astype(str)

    X_test = test_df["text"].astype(str)
    y_test = test_df["label"].astype(str)

    model = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )

    params = {
        "model_type": "TF-IDF + LogisticRegression",
        "max_features": 5000,
        "ngram_range": "1,2",
        "class_weight": "balanced",
        "train_size": len(train_df),
        "test_size": len(test_df),
    }

    with mlflow.start_run(run_name="final_project_training") as run:
        mlflow.log_params(params)

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        f1_macro = f1_score(y_test, predictions, average="macro")

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_macro", f1_macro)

        report = classification_report(y_test, predictions)

        report_path = MODEL_DIR / "classification_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)

        model_path = MODEL_DIR / "sentiment_model.joblib"
        joblib.dump(model, model_path)

        mlflow.log_artifact(str(report_path))
        mlflow.log_artifact(str(model_path))

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=X_test.head(3).tolist(),
        )

        print("Training completed successfully")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 macro: {f1_macro:.4f}")
        print(f"Model saved to: {model_path}")
        print(f"Classification report saved to: {report_path}")
        print(f"MLflow run_id: {run.info.run_id}")

        if args.register:
            model_uri = f"runs:/{run.info.run_id}/model"
            result = mlflow.register_model(model_uri, args.model_name)

            client = MlflowClient()

            try:
                client.set_registered_model_alias(
                    name=args.model_name,
                    alias="production",
                    version=result.version,
                )
                print("Alias set: production")
            except Exception as e:
                print(f"Model registered, but alias was not set: {e}")

            print(f"Model registered: {args.model_name}")
            print(f"Model version: {result.version}")


if __name__ == "__main__":
    main()