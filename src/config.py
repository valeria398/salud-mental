"""Constantes del proyecto. Un solo lugar para nombres, rutas y semilla."""
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent          # carpeta raíz del proyecto
RUTA_DATOS = RAIZ / "data" / "mental_health_dataset.csv"

EXPERIMENTO = "riesgo-salud-mental"        # nombre del experimento en MLflow
MODELO_REGISTRADO = "clasificador-riesgo"  # nombre del modelo en el Model Registry
SEMILLA = 42                               # misma partición de datos en cada ejecución

OBJETIVO = "mental_health_risk"            # columna que se quiere predecir (Low / Medium / High)
COLUMNAS_CATEGORICAS = [
    "gender",
    "employment_status",
    "work_environment",
    "mental_health_history",
    "seeks_treatment",
]