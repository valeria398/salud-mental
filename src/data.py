"""Carga y partición de datos. Si cambian los datos, se cambia solo este archivo."""
import pandas as pd
from sklearn.model_selection import train_test_split

from config import OBJETIVO, RUTA_DATOS, SEMILLA


def cargar_dataframe():
    """Devuelve el DataFrame completo, tal como está en el CSV."""
    return pd.read_csv(RUTA_DATOS)


def cargar_datos(test_size=0.2):
    """Devuelve X_train, X_test, y_train, y_test con la misma partición siempre."""
    df = cargar_dataframe()
    y = df[OBJETIVO]
    X = df.drop(columns=[OBJETIVO])
    return train_test_split(X, y, test_size=test_size, random_state=SEMILLA, stratify=y)