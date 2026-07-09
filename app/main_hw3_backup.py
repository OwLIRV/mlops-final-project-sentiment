import os
import time
import csv
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST


model = None
model_info: Dict[str, Any] = {}


PREDICTIONS_TOTAL = Counter(
    "ml_predictions_total",
    "Total number of ML predictions",
    ["prediction"]
)

PREDICTION_LATENCY = Histogram(
    "ml_prediction_latency_seconds",
    "ML prediction latency in seconds"
)


class PredictRequest(BaseModel):
    texts: List[str]


class PredictionItem(BaseModel):
    text: str
    prediction: str
    probabilities: Optional[Dict[str, float]] = None


class PredictResponse(BaseModel):
    model_uri: str
    tracking_uri: str
    results: List[PredictionItem]


def get_tracking_uri() -> str:
    return os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")


def get_model_uri() -> str:
    return os.getenv(
        "MLFLOW_MODEL_URI",
        "models:/ukrainian-review-sentiment-model/6"
    )


def load_model_from_mlflow():
    tracking_uri = get_tracking_uri()
    model_uri = get_model_uri()

    mlflow.set_tracking_uri(tracking_uri)

    try:
        loaded_model = mlflow.sklearn.load_model(model_uri)
        flavor = "sklearn"
    except Exception:
        loaded_model = mlflow.pyfunc.load_model(model_uri)
        flavor = "pyfunc"

    return loaded_model, {
        "tracking_uri": tracking_uri,
        "model_uri": model_uri,
        "flavor": flavor
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, model_info

    model, model_info = load_model_from_mlflow()

    print("Model loaded from MLflow Model Registry")
    print(f"Model URI: {model_info['model_uri']}")
    print(f"MLflow tracking URI: {model_info['tracking_uri']}")
    print(f"Model flavor: {model_info['flavor']}")

    yield


app = FastAPI(
    title="HW4 Model Inference API with Monitoring",
    description="FastAPI inference service with MLflow Model Registry and Prometheus metrics",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/")
def root():
    return {
        "message": "HW4 inference service with monitoring is running",
        "model_uri": model_info.get("model_uri"),
        "tracking_uri": model_info.get("tracking_uri"),
        "model_flavor": model_info.get("flavor"),
        "available_endpoints": ["/health", "/predict", "/metrics", "/docs"]
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_uri": model_info.get("model_uri"),
        "tracking_uri": model_info.get("tracking_uri"),
        "model_flavor": model_info.get("flavor")
    }


def run_prediction(texts: List[str]):
    try:
        return model.predict(texts)
    except Exception:
        pass

    try:
        return model.predict(pd.DataFrame({"text": texts}))
    except Exception:
        pass

    try:
        return model.predict(pd.DataFrame({"review": texts}))
    except Exception:
        pass

    return model.predict(pd.DataFrame({"reviews": texts}))


def run_probabilities(texts: List[str]):
    if not hasattr(model, "predict_proba"):
        return None, None

    try:
        probabilities = model.predict_proba(texts)
        classes = [str(label) for label in model.classes_]
        return probabilities, classes
    except Exception:
        return None, None


def save_prediction_log(text: str, prediction: str, latency: float):
    os.makedirs("logs", exist_ok=True)

    file_path = "logs/predictions.csv"
    file_exists = os.path.exists(file_path)

    with open(file_path, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "timestamp",
                "text",
                "text_length",
                "prediction",
                "latency"
            ])

        writer.writerow([
            datetime.utcnow().isoformat(),
            text,
            len(text),
            prediction,
            latency
        ])


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    start_time = time.perf_counter()

    if model is None:
        PREDICTIONS_TOTAL.labels(prediction="error").inc()
        raise HTTPException(status_code=500, detail="Model is not loaded")

    if not request.texts:
        PREDICTIONS_TOTAL.labels(prediction="error").inc()
        raise HTTPException(status_code=400, detail="Texts list is empty")

    try:
        predictions = run_prediction(request.texts)
    except Exception as error:
        latency = time.perf_counter() - start_time
        PREDICTION_LATENCY.observe(latency)
        PREDICTIONS_TOTAL.labels(prediction="error").inc()

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}"
        )

    latency = time.perf_counter() - start_time
    PREDICTION_LATENCY.observe(latency)

    probabilities, classes = run_probabilities(request.texts)

    results = []

    for index, text in enumerate(request.texts):
        prediction_label = str(predictions[index])

        PREDICTIONS_TOTAL.labels(prediction=prediction_label).inc()

        save_prediction_log(
            text=text,
            prediction=prediction_label,
            latency=latency
        )

        item: Dict[str, Any] = {
            "text": text,
            "prediction": prediction_label
        }

        if probabilities is not None and classes is not None:
            item["probabilities"] = {
                classes[class_index]: round(
                    float(probabilities[index][class_index]),
                    4
                )
                for class_index in range(len(classes))
            }

        results.append(item)

    return {
        "model_uri": model_info.get("model_uri"),
        "tracking_uri": model_info.get("tracking_uri"),
        "results": results
    }


@app.get("/metrics")
def metrics():
    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )