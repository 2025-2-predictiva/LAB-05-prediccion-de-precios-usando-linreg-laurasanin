import pandas as pd
import numpy as np
import json
import os
import gzip
import pickle

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, median_absolute_error


# =========================================================
# FUNCIONES DE PREPROCESAMIENTO
# =========================================================
def load_and_preprocess(path_train, path_test, current_year=2021):
    """
    Cargar y preprocesar los datos.
    - Crea la columna 'Age' a partir de 'Year'
    - Elimina 'Year' y 'Car_Name'
    """

    # Leer los datasets comprimidos
    train = pd.read_csv(path_train, compression="zip")
    test = pd.read_csv(path_test, compression="zip")

    # Crear la edad del vehículo
    train["Age"] = current_year - train["Year"]
    test["Age"] = current_year - test["Year"]

    # Eliminar columnas innecesarias
    train.drop(columns=["Year", "Car_Name"], inplace=True)
    test.drop(columns=["Year", "Car_Name"], inplace=True)

    return train, test


# =========================================================
# DIVISIÓN DE VARIABLES X E Y
# =========================================================
def split_xy(train, test, target="Present_Price"):
    """
    Separa los conjuntos de entrenamiento y prueba en:
    - X (variables predictoras)
    - y (variable objetivo)
    """

    X_train = train.drop(columns=[target])
    y_train = train[target]

    X_test = test.drop(columns=[target])
    y_test = test[target]

    return X_train, y_train, X_test, y_test


# =========================================================
# CONSTRUCCIÓN DEL PIPELINE
# =========================================================
def build_pipeline(categorical_cols, numeric_cols):
    """
    Crea un pipeline que incluye:
    - OneHotEncoding para variables categóricas
    - MinMaxScaler para variables numéricas
    - Selección de características (K mejores)
    - Modelo de regresión lineal
    """

    # Transformador para procesar categorías y números
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", MinMaxScaler(), numeric_cols)
        ],
        remainder="drop"
    )

    model = LinearRegression()

    # Construcción del pipeline completo
    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("select", SelectKBest(score_func=f_regression)),
        ("model", model)
    ])

    return pipe


# =========================================================
# OPTIMIZACIÓN DEL PIPELINE
# =========================================================
def optimize_pipeline(pipe, X_train, y_train):
    """
    Optimiza los hiperparámetros usando GridSearchCV:
    - k para SelectKBest
    - fit_intercept del modelo lineal
    - positive del modelo lineal
    """

    param_grid = {
        "select__k": range(1, 12),               # Número de variables a seleccionar
        "model__fit_intercept": [True, False],   # Ajustar o no intercepto
        "model__positive": [True, False]         # Forzar coeficientes positivos
    }

    grid = GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=10,                                   # 10 particiones de validación cruzada
        scoring="neg_mean_absolute_error",       # MAE como métrica
        n_jobs=-1,                               # Usar todos los núcleos disponibles
        verbose=1                                # Mostrar progreso
    )

    grid.fit(X_train, y_train)
    return grid


# =========================================================
# GUARDAR MODELO ENTRENADO
# =========================================================
def save_model(grid, output_path="files/models/model.pkl.gz"):
    """Guarda el modelo entrenado comprimido con gzip."""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with gzip.open(output_path, "wb") as f:
        pickle.dump(grid, f)

    print(f"Modelo guardado en: {output_path}")


# =========================================================
# CÁLCULO DE MÉTRICAS
# =========================================================
def compute_metrics(y_true, y_pred, dataset_name):
    """
    Calcula métricas de evaluación:
    - R2
    - MSE
    - MAD (error absoluto mediano)
    """

    return {
        "type": "metrics",
        "dataset": dataset_name,
        "r2": r2_score(y_true, y_pred),
        "mse": mean_squared_error(y_true, y_pred),
        "mad": median_absolute_error(y_true, y_pred),
    }


def save_metrics(metrics, output_path="files/output/metrics.json"):
    """
    Guarda las métricas en formato JSON, una por línea.
    """

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        for m in metrics:
            f.write(json.dumps(m) + "\n")


# =========================================================
# FUNCIÓN PRINCIPAL QUE EJECUTA TODO EL PROCESO
# =========================================================
def main():
    """Ejecuta todo el flujo del proyecto."""

    # --- 1. Cargar y preprocesar los datos ---
    train, test = load_and_preprocess(
        "files/input/train_data.csv.zip",
        "files/input/test_data.csv.zip"
    )

    # --- 2. Separar X e y ---
    X_train, y_train, X_test, y_test = split_xy(train, test)

    # Columnas categóricas y numéricas
    categorical_cols = ["Fuel_Type", "Selling_type", "Transmission"]
    numeric_cols = [col for col in X_train.columns if col not in categorical_cols]

    # --- 3. Crear pipeline ---
    pipe = build_pipeline(categorical_cols, numeric_cols)

    # --- 4. Optimizar el pipeline ---
    grid = optimize_pipeline(pipe, X_train, y_train)
    print("Mejores parámetros encontrados:", grid.best_params_)

    # --- 5. Guardar el modelo ---
    save_model(grid)

    # --- 6. Predecir en train y test ---
    y_train_pred = grid.predict(X_train)
    y_test_pred = grid.predict(X_test)

    # --- 7. Calcular métricas ---
    metrics = [
        compute_metrics(y_train, y_train_pred, "train"),
        compute_metrics(y_test, y_test_pred, "test")
    ]

    # --- 8. Guardar métricas ---
    save_metrics(metrics)
    print("Métricas guardadas correctamente.")


# =========================================================
# EJECUCIÓN DEL SCRIPT
# =========================================================
if __name__ == "__main__":
    main()