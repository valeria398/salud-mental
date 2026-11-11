"""Registra el mejor run del experimento como una versión nueva y la deja como 'challenger'."""
import mlflow
from mlflow import MlflowClient

from config import EXPERIMENTO, MODELO_REGISTRADO

mejores = mlflow.search_runs(
    experiment_names=[EXPERIMENTO],
    order_by=["metrics.accuracy DESC"],
    max_results=1,
)
if mejores.empty:
    raise SystemExit(f"No hay runs en el experimento '{EXPERIMENTO}'. Ejecuta antes src/train.py")

mejor = mejores.iloc[0]
version = mlflow.register_model(mejor["tags.model_uri"], MODELO_REGISTRADO)
MlflowClient().set_registered_model_alias(MODELO_REGISTRADO, "challenger", version.version)
print(f"Registrada la versión {version.version} (accuracy={mejor['metrics.accuracy']:.3f}) como 'challenger'")