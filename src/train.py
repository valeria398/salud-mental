"""Entrena un modelo y registra parámetros, métricas, artefactos y modelo en MLflow."""
import argparse
import os

import matplotlib
matplotlib.use("Agg")  # genera gráficos sin abrir ventanas
import matplotlib.pyplot as plt
import mlflow
import mlflow.data
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from config import COLUMNAS_CATEGORICAS, EXPERIMENTO, OBJETIVO, RUTA_DATOS, SEMILLA
from data import cargar_datos


def construir_modelo(n_estimators, max_depth):
    """Pipeline: convierte las columnas de texto a números y luego entrena el bosque."""
    preprocesado = ColumnTransformer(
        [("categoricas", OneHotEncoder(handle_unknown="ignore"), COLUMNAS_CATEGORICAS)],
        remainder="passthrough",  # las columnas numéricas pasan sin cambios
    )
    bosque = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, random_state=SEMILLA
    )
    return Pipeline([("preprocesado", preprocesado), ("modelo", bosque)])


def main():
    parser = argparse.ArgumentParser(description="Entrenamiento con seguimiento en MLflow")
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--run-name", default=None, help="nombre legible del run")
    args = parser.parse_args()

    # `mlflow run` ya asigna el experimento; con `python src/train.py` lo asignamos aquí
    if not os.environ.get("MLFLOW_RUN_ID"):
        mlflow.set_experiment(EXPERIMENTO)

    X_train, X_test, y_train, y_test = cargar_datos()

    with mlflow.start_run(run_name=args.run_name):
        # 1) Qué se probó
        mlflow.log_params({
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "seed": SEMILLA,
            "test_size": 0.2,
        })
        mlflow.set_tags({"modelo": "RandomForest", "dataset": "mental_health", "etapa": "desarrollo"})

        # 2) Dejar constancia de qué datos se usaron
        dataset = mlflow.data.from_pandas(
            X_train.assign(**{OBJETIVO: y_train}),
            source=str(RUTA_DATOS),
            name="mental-health-train",
            targets=OBJETIVO,
        )
        mlflow.log_input(dataset, context="training")

        # 3) Entrenar
        modelo = construir_modelo(args.n_estimators, args.max_depth).fit(X_train, y_train)

        # 4) Evaluar con datos que el modelo no vio y guardar las métricas
        pred = modelo.predict(X_test)
        accuracy = accuracy_score(y_test, pred)
        mlflow.log_metrics({
            "accuracy": accuracy,
            "f1_macro": f1_score(y_test, pred, average="macro"),
        })

        # 5) Un gráfico como artefacto
        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=ax)
        mlflow.log_figure(fig, "matriz_confusion.png")
        plt.close(fig)

        # 6) El modelo, con su firma y un ejemplo de entrada
        info = mlflow.sklearn.log_model(
            modelo,
            name="modelo",
            signature=infer_signature(X_train, modelo.predict(X_train)),
            input_example=X_train.head(3),
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )
        mlflow.set_tag("model_uri", info.model_uri)  # para encontrar el modelo después
        print(f"accuracy={accuracy:.3f}  model_uri={info.model_uri}")


if __name__ == "__main__":
    main()