import os
import time
from typing import List

import mlflow
import mlflow.sklearn
from fastapi import FastAPI
from pydantic import BaseModel
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator


MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MODEL_URI = "models:/sentiment-classifier@production"


app = FastAPI(
    title="Final MLOps Project: Ukrainian Review Sentiment API",
    description="End-to-end MLOps project: MLflow Model Registry, FastAPI serving and Prometheus monitoring",
    version="1.0.0",
)


prediction_counter = Counter(
    "sentiment_predictions_total",
    "Total number of sentiment predictions",
    ["predicted_label"],
)

prediction_latency = Histogram(
    "sentiment_prediction_latency_seconds",
    "Prediction latency in seconds",
)


class PredictionRequest(BaseModel):
    texts: List[str]


class PredictionResponse(BaseModel):
    predictions: List[str]
    latency_seconds: float
    model_uri: str


@app.on_event("startup")
def load_model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    app.state.model = mlflow.sklearn.load_model(MODEL_URI)


@app.get("/")
def root():
    return {
        "service": "Final MLOps Project: Ukrainian Review Sentiment API",
        "status": "running",
        "model_uri": MODEL_URI,
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": hasattr(app.state, "model"),
        "model_uri": MODEL_URI,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    start_time = time.time()

    predictions = app.state.model.predict(request.texts)
    predictions = [str(prediction) for prediction in predictions]

    for prediction in predictions:
        prediction_counter.labels(predicted_label=prediction).inc()

    latency = time.time() - start_time
    prediction_latency.observe(latency)

    return PredictionResponse(
        predictions=predictions,
        latency_seconds=latency,
        model_uri=MODEL_URI,
    )


Instrumentator().instrument(app).expose(app)