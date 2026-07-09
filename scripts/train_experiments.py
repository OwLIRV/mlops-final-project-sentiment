import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


TRAIN_PATH = Path("data/processed/train.csv")
TEST_PATH = Path("data/processed/test.csv")
MODEL_DIR = Path("models")


def run_experiment(name, model, params):
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    X_train = train_df["text"].astype(str)
    y_train = train_df["label"].astype(str)

    X_test = test_df["text"].astype(str)
    y_test = test_df["label"].astype(str)

    with mlflow.start_run(run_name=name):
        mlflow.log_params(params)

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        f1_macro = f1_score(y_test, predictions, average="macro")

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_macro", f1_macro)

        report = classification_report(y_test, predictions)

        report_path = MODEL_DIR / f"{name}_classification_report.txt"
        model_path = MODEL_DIR / f"{name}_model.joblib"

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)

        joblib.dump(model, model_path)

        mlflow.log_artifact(str(report_path))
        mlflow.log_artifact(str(model_path))

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=X_test.head(3).tolist(),
        )

        print(f"Experiment: {name}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 macro: {f1_macro:.4f}")
        print("-" * 40)

        return {
            "name": name,
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "model_path": model_path,
        }


def main():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("sentiment-classification-final-project")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    experiments = [
        (
            "logreg_word_tfidf",
            Pipeline(
                steps=[
                    ("tfidf", TfidfVectorizer(max_features=3000, ngram_range=(1, 1))),
                    ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
                ]
            ),
            {
                "model_type": "LogisticRegression",
                "vectorizer": "word_tfidf",
                "max_features": 3000,
                "ngram_range": "1,1",
                "class_weight": "balanced",
            },
        ),
        (
            "logreg_word_bigram_tfidf",
            Pipeline(
                steps=[
                    ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
                    ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
                ]
            ),
            {
                "model_type": "LogisticRegression",
                "vectorizer": "word_tfidf",
                "max_features": 5000,
                "ngram_range": "1,2",
                "class_weight": "balanced",
            },
        ),
        (
            "linear_svm_word_tfidf",
            Pipeline(
                steps=[
                    ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
                    ("classifier", LinearSVC(class_weight="balanced")),
                ]
            ),
            {
                "model_type": "LinearSVC",
                "vectorizer": "word_tfidf",
                "max_features": 5000,
                "ngram_range": "1,2",
                "class_weight": "balanced",
            },
        ),
        (
            "naive_bayes_word_tfidf",
            Pipeline(
                steps=[
                    ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
                    ("classifier", MultinomialNB()),
                ]
            ),
            {
                "model_type": "MultinomialNB",
                "vectorizer": "word_tfidf",
                "max_features": 5000,
                "ngram_range": "1,2",
            },
        ),
        (
            "logreg_char_tfidf",
            Pipeline(
                steps=[
                    ("tfidf", TfidfVectorizer(analyzer="char", max_features=5000, ngram_range=(3, 5))),
                    ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
                ]
            ),
            {
                "model_type": "LogisticRegression",
                "vectorizer": "char_tfidf",
                "max_features": 5000,
                "ngram_range": "3,5",
                "class_weight": "balanced",
            },
        ),
    ]

    results = []

    for name, model, params in experiments:
        result = run_experiment(name, model, params)
        results.append(result)

    best_result = max(results, key=lambda x: x["f1_macro"])

    print("Best experiment:")
    print(f"Name: {best_result['name']}")
    print(f"Accuracy: {best_result['accuracy']:.4f}")
    print(f"F1 macro: {best_result['f1_macro']:.4f}")

    best_model = joblib.load(best_result["model_path"])
    final_model_path = MODEL_DIR / "sentiment_model.joblib"
    joblib.dump(best_model, final_model_path)

    print(f"Best model saved as: {final_model_path}")

    client = MlflowClient()

    latest_versions = client.search_model_versions("name='sentiment-classifier'")

    if latest_versions:
        latest_version = max(latest_versions, key=lambda v: int(v.version))
        client.set_registered_model_alias(
            name="sentiment-classifier",
            alias="production",
            version=latest_version.version,
        )
        print(f"Production alias kept for version: {latest_version.version}")


if __name__ == "__main__":
    main()