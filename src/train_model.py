import pandas as pd
import numpy as np
import joblib  # Usamos joblib, es mejor para scikit-learn
from pathlib import Path
from functools import reduce

# --- Funciones de Ayuda (copiadas de tu script) ---
from sklearn.preprocessing import RobustScaler, PowerTransformer
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor

def mask_outliers_to_nan(df_num: pd.DataFrame, bounds_dict: dict) -> pd.DataFrame:
    df = df_num.copy()
    for c, (low, high) in bounds_dict.items():
        mask_low = df[c] < low
        mask_high = df[c] > high
        df.loc[mask_low | mask_high, c] = np.nan
    return df

def mad_bounds(col_train: pd.Series, Z: float = 3.5,
               q_low: float = 0.001, q_high: float = 0.999):
    med = col_train.median()
    mad = np.median(np.abs(col_train - med))
    if mad > 0:
        low = med - (Z * mad / 0.6745)
        high = med + (Z * mad / 0.6745)
    else:
        low = col_train.quantile(q_low)
        high = col_train.quantile(q_high)
        if low == high:
            low, high = low - 1e-12, high + 1e-12
    return low, high
# --- Fin de Funciones de Ayuda ---


def train_and_save_pipeline(
    output_path: str = "models/pipeline_artifacts.pkl",
    random_state: int = 42
):
    """
    Carga los datos de entrenamiento (CSV), entrena todos los pasos
    de preprocesamiento y el modelo, y guarda todos los artefactos
    en un solo archivo .pkl usando joblib.
    """
    print("Iniciando entrenamiento del pipeline...")

    # --- 1. Cargar Datos de Entrenamiento ---
    try: 
        X_train_csv = pd.read_csv("data/processed/splits/x_train.csv")
        y_train_csv = pd.read_csv("data/processed/splits/y_train.csv")
        X_val_csv = pd.read_csv("data/processed/splits/x_test.csv") # Renombrado en tu script
        y_val_csv = pd.read_csv("data/processed/splits/y_test.csv") # Renombrado en tu script
    except FileNotFoundError as e:
        print(f"Error: No se encontró el archivo CSV: {e.filename}")
        print("Asegúrate de tener x_train.csv, y_train.csv, x_test.csv, y_test.csv")
        return

    X_train = pd.concat([X_train_csv, X_val_csv], ignore_index=True)
    y_train_df = pd.concat([y_train_csv, y_val_csv], ignore_index=True)
    
    # Convertir y_train a Series (1D)
    if isinstance(y_train_df, pd.DataFrame) and y_train_df.shape[1] == 1:
        y_train = y_train_df.iloc[:, 0]
    else:
        y_train = y_train_df.squeeze()

    print(f"Datos de entrenamiento cargados: {X_train.shape[0]} filas.")

    # --- 2. Definir Columnas ---
    # Usamos las 'important_features' como las columnas base
    important_features = [
        'Frio (Kw)', 'Frio_roll_mean_7_lag1', 'Sala Maq (Kw)', 'Envasado (Kw)', 
        'Frio_roll_mean_14_lag1', 'Servicios (Kw)', 'Prod Agua (Kw)', 'KW Gral Planta', 
        'Linea 2 (Kw)', 'CO 2 / Hl', 'EE Caldera / Hl', 'Cocina (Kw)', 
        'ET Linea 5/Hl', 'VAPOR DE LINEA 4 KG', 'Linea 3 (Kw)', 'Resto Serv (Kw)', 
        'Conversion Kg/Mj', 'Restos Planta (Kw)', 'Hl Cerveza L2', 'Frio_roll_mean_3_lag1'
    ]
    
    # Asegurarnos que solo usamos columnas presentes
    common_cols = [c for c in X_train.columns if c in important_features]
    X_train = X_train[common_cols]

    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    other_cols = [c for c in X_train.columns if c not in numeric_cols]

    Xtr_num = X_train[numeric_cols].copy()

    # --- 3. Ajustar (FIT) Preprocesamiento ---
    
    # 3.1. Bounds
    print("Ajustando bounds (MAD)...")
    bounds = {}
    for c in numeric_cols:
        low, high = mad_bounds(Xtr_num[c].dropna(), Z=3.5, q_low=0.001, q_high=0.999)
        bounds[c] = (low, high)
    
    Xtr_num_nan = mask_outliers_to_nan(Xtr_num, bounds)

    # 3.2. Scaler
    print("Ajustando RobustScaler...")
    scaler = RobustScaler()
    Xtr_scaled = pd.DataFrame(scaler.fit_transform(Xtr_num_nan), columns=numeric_cols, index=Xtr_num_nan.index)

    # 3.3. Imputer
    print("Ajustando KNNImputer...")
    imputer = KNNImputer(n_neighbors=5, weights='distance')
    Xtr_imp_scaled = pd.DataFrame(imputer.fit_transform(Xtr_scaled), columns=numeric_cols, index=Xtr_scaled.index)
    
    # Invertir escalado
    Xtr_imp = pd.DataFrame(scaler.inverse_transform(Xtr_imp_scaled), columns=numeric_cols, index=Xtr_imp_scaled.index)

    # 3.4. Reconstruir X_train
    x_train_clean = pd.concat([
        Xtr_imp.reset_index(drop=True),
        X_train[other_cols].reset_index(drop=True)
    ], axis=1)[X_train.columns]

    # 3.5. PowerTransformer
    print("Ajustando PowerTransformer...")
    pt = PowerTransformer(method='yeo-johnson')
    x_train_ready = pd.DataFrame(pt.fit_transform(x_train_clean), columns=x_train_clean.columns)

    # --- 4. Ajustar (FIT) Modelo ---
    print("Ajustando RandomForestRegressor...")
    best_model = RandomForestRegressor(
        n_estimators=1183,
        max_depth=16,
        max_features=1.0,
        bootstrap=True,
        min_samples_split=2,
        min_samples_leaf=2,
        max_samples=0.8654367662002576,
        random_state=random_state,
        n_jobs=-1
    )
    
    best_model.fit(x_train_ready, y_train)

    # --- 5. Guardar Artefactos ---
    print(f"Guardando artefactos en {output_path}...")
    
    artifacts = {
        "common_cols": common_cols,
        "numeric_cols": numeric_cols,
        "other_cols": other_cols,
        "bounds": bounds,
        "scaler": scaler,
        "imputer": imputer,
        "pt": pt,
        "model": best_model
    }
    
    joblib.dump(artifacts, output_path, compress=3)
    print("¡Entrenamiento y guardado completados exitosamente!")


if __name__ == "__main__":
    # Esto te permite ejecutar el script desde la terminal
    # usando: python train_pipeline.py
    train_and_save_pipeline()