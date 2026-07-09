import os
import time

import mlflow
from mlflow.tracking import MlflowClient


EXPERIMENT_NAME = "sentiment-classification-final-project"
MODEL_NAME = "sentiment-classifier"


def main():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient()

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:
        raise ValueError(f"Experiment not found: {EXPERIMENT_NAME}")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.f1_macro DESC"],
        max_results=1,
    )

    if not runs:
        raise ValueError("No runs found")

    best_run = runs[0]
    best_run_id = best_run.info.run_id
    best_f1 = best_run.data.metrics.get("f1_macro")
    best_accuracy = best_run.data.metrics.get("accuracy")
    best_run_name = best_run.data.tags.get("mlflow.runName")

    model_uri = f"runs:/{best_run_id}/model"

    result = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME,
    )

    time.sleep(5)

    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias="production",
        version=result.version,
    )

    print("Best model registered successfully")
    print(f"Best run name: {best_run_name}")
    print(f"Best run id: {best_run_id}")
    print(f"Accuracy: {best_accuracy:.4f}")
    print(f"F1 macro: {best_f1:.4f}")
    print(f"Registered model: {MODEL_NAME}")
    print(f"New model version: {result.version}")
    print("Alias set: production")


if __name__ == "__main__":
    main()