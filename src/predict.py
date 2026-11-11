"""Carga un modelo del registro por alias y predice sobre un archivo CSV."""
import argparse

import mlflow
import pandas as pd

from config import MODELO_REGISTRADO, OBJETIVO

parser = argparse.ArgumentParser(description="Predicción con un modelo del registro")
parser.add_argument("csv", help="archivo con las mismas columnas que se usaron al entrenar")
parser.add_argument("--alias", default="champion", help="alias del modelo (champion, challenger...)")
args = parser.parse_args()

modelo = mlflow.pyfunc.load_model(f"models:/{MODELO_REGISTRADO}@{args.alias}")
datos = pd.read_csv(args.csv)
entrada = datos.drop(columns=[OBJETIVO], errors="ignore")   # por si el CSV trae la respuesta real
datos["prediccion"] = modelo.predict(entrada)
print(datos["prediccion"].value_counts().to_string())
datos.to_csv("predicciones.csv", index=False)
print("Guardado en predicciones.csv")