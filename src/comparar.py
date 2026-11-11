"""Muestra todos los runs del experimento ordenados por accuracy."""
import mlflow

from config import EXPERIMENTO

runs = mlflow.search_runs(
    experiment_names=[EXPERIMENTO],
    order_by=["metrics.accuracy DESC"],
)
columnas = ["tags.mlflow.runName", "params.n_estimators", "params.max_depth",
            "metrics.accuracy", "metrics.f1_macro"]
print(runs[columnas].to_string(index=False))