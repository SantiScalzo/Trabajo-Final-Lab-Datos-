import pandas as pd
import numpy as np
import joblib
import argparse # <-- Importado para argumentos de consola
from functools import reduce
from typing import Dict, Optional, Tuple, Union
from sklearn.metrics import mean_absolute_error, r2_score # <-- Importado para métricas

# --- Funciones de Ayuda (copiadas de tu script) ---
# Se necesitan para el pre-procesamiento de datos nuevos
def mask_outliers_to_nan(df_num: pd.DataFrame, bounds_dict: dict) -> pd.DataFrame:
    df = df_num.copy()
    for c, (low, high) in bounds_dict.items():
        # Asegurarse de que los tipos son compatibles para la comparación
        c_num = pd.to_numeric(df[c], errors='coerce')
        mask_low = c_num < low
        mask_high = c_num > high
        df.loc[mask_low | mask_high, c] = np.nan
    return df

def preparar_hoja(df: pd.DataFrame, nombre_hoja: str) -> pd.DataFrame:
    df = df.copy()
    if 'FECHA_HORA' in df.columns:
        ts = pd.to_datetime(df['FECHA_HORA'], errors='coerce')
    elif {'DIA', 'HORA'}.issubset(df.columns):
        df['HORA'] = df['HORA'].astype(str).str.extract(r'(\d{1,2}:\d{2}:\d{2})')[0]
        ts = pd.to_datetime(df['DIA'].astype(str) + ' ' + df['HORA'].astype(str), errors='coerce')
    elif 'DIA' in df.columns:
        ts = pd.to_datetime(df['DIA'], errors='coerce')
    else:
        return None
    df['FECHA_HORA'] = ts
    df = df.dropna(subset=['FECHA_HORA']).reset_index(drop=True)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    other_cols = [c for c in df.columns if c not in numeric_cols + ['DIA', 'HORA', 'FECHA_HORA']]
    agg_map = {**{c: 'mean' for c in numeric_cols}, **{c: 'first' for c in other_cols}}
    df = (
        df.drop(columns=['DIA', 'HORA'], errors='ignore')
          .groupby('FECHA_HORA', as_index=False)
          .agg(agg_map)
    )
    return df
# --- Fin de Funciones de Ayuda ---


def run_prediction_from_excel(
    url: str,
    pipeline_path: str = "models/pipeline_artifacts.pkl"
) -> pd.DataFrame: # <-- Cambiado el tipo de retorno
    """
    Carga el pipeline de artefactos guardados y los aplica a un
    nuevo archivo Excel de una URL para generar predicciones.
    Calcula MAE y R2 si los datos de test tienen la columna target.
    """
    print(f"Cargando pipeline desde {pipeline_path}...")
    try:
        artifacts = joblib.load(pipeline_path)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo de pipeline: {pipeline_path}")
        print("Por favor, ejecuta 'train_pipeline.py' primero.")
        return pd.DataFrame(), np.array([])

    # Extraer todos los artefactos
    common_cols = artifacts['common_cols']
    numeric_cols = artifacts['numeric_cols']
    other_cols = artifacts['other_cols']
    bounds = artifacts['bounds']
    scaler = artifacts['scaler']
    imputer = artifacts['imputer']
    pt = artifacts['pt']
    model = artifacts['model']

    print("Cargando y preparando datos de Excel...")
    # --- 1. Cargar y Preparar Datos (Lógica de tu script original) ---
    try:
        # Añadido engine='openpyxl' para evitar errores de motor
        dict_de_hojas = pd.read_excel(url, sheet_name=None, engine='openpyxl')
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo Excel en la ruta: {url}")
        return pd.DataFrame(), np.array([])
    except Exception as e:
        # Captura otros errores, como el formato incorrecto (CSV vs XLSX)
        print(f"Error al leer el archivo Excel: {e}")
        print("Asegúrate de que la ruta es correcta y es un archivo .xlsx, no un .csv")
        return pd.DataFrame(), np.array([])

    dict_de_hojas = {nombre: df for nombre, df in dict_de_hojas.items() if nombre.startswith("Consolidado")}
    hojas_preparadas = [h for nombre, df in dict_de_hojas.items()
                        if (h := preparar_hoja(df, nombre)) is not None and len(h) > 0]
    
    if not hojas_preparadas:
        print("Error: No se encontraron hojas 'Consolidado' válidas en el Excel.")
        return pd.DataFrame(), np.array([])

    df_combinado = reduce(lambda l, r: pd.merge(l, r, on='FECHA_HORA', how='outer'), hojas_preparadas)
    df_combinado = df_combinado.sort_values('FECHA_HORA').reset_index(drop=True)

    indices_ultima_medicion = df_combinado.groupby(
        df_combinado['FECHA_HORA'].dt.date
    )['FECHA_HORA'].idxmax()
    df_combinado = df_combinado.loc[indices_ultima_medicion].reset_index(drop=True)

    # --- 2. Feature Engineering (Lógica de tu script original) ---
    print("Creando features...")
    df_combinado["Frio_diff1_lag1"] = df_combinado["Frio (Kw)"].astype(float).diff().shift(1)
    df_combinado["Frio_diff7_lag1"] = df_combinado["Frio (Kw)"].astype(float).diff().shift(1)
    roll_windows = [3, 7, 14, 28]
    for window in roll_windows:
        df_combinado[f"Frio_roll_mean_{window}_lag1"] = df_combinado["Frio (Kw)"].astype(float).shift(1).rolling(window=window).mean()
        df_combinado[f"Frio_roll_std_{window}_lag1"] = df_combinado["Frio (Kw)"].astype(float).shift(1).rolling(window=window).std()

    # --- CREAR y_true ANTES de modificar X_test ---
    # Replicamos la lógica original: predecir el siguiente valor de Frio (Kw)
    y_true_series = df_combinado["Frio (Kw)"].shift(-1)

    # Este es el set de datos para predecir
    X_test = df_combinado.drop(columns=["FECHA_HORA"], errors='ignore').copy()
    
    # Guardar fechas para el reporte final
    fechas_test = df_combinado["FECHA_HORA"]

    # --- 3. Aplicar Preprocesamiento (SOLO TRANSFORM) ---
    print("Aplicando transformaciones del pipeline...")
    
    # 3.1. Alinear columnas
    # Asegurarse de que X_test tiene todas las columnas de common_cols,
    # rellenando con NaN si falta alguna
    for col in common_cols:
        if col not in X_test.columns:
            X_test[col] = np.nan
    
    X_test = X_test[common_cols] # Reordenar y filtrar

    Xte_num = X_test[numeric_cols].copy()
    
    # 3.2. Aplicar Bounds
    Xte_num_nan = mask_outliers_to_nan(Xte_num, bounds)

    # 3.3. Aplicar Scaler
    # Manejar el caso de que todo sea NaN (p.ej. primera fila)
    Xte_num_nan.fillna(0, inplace=True) # Solución simple, KNN lo manejará
    Xte_scaled = pd.DataFrame(scaler.transform(Xte_num_nan), columns=numeric_cols, index=Xte_num_nan.index)

    # 3.4. Aplicar Imputer
    Xte_imp_scaled = pd.DataFrame(imputer.transform(Xte_scaled), columns=numeric_cols, index=Xte_scaled.index)

    # 3.5. Invertir Scaler
    Xte_imp = pd.DataFrame(scaler.inverse_transform(Xte_imp_scaled), columns=numeric_cols, index=Xte_imp_scaled.index)

    # 3.6. Reconstruir X_test
    x_test_clean = pd.concat([
        Xte_imp.reset_index(drop=True),
        X_test[other_cols].reset_index(drop=True)
    ], axis=1)[common_cols]

    # 3.7. Aplicar PowerTransformer
    x_test_ready = pd.DataFrame(pt.transform(x_test_clean), columns=x_test_clean.columns)

    # --- 4. Generar Predicciones ---
    print("Generando predicciones...")
    y_pred_test = model.predict(x_test_ready)
    
    print("¡Predicción completada!")
    
    # --- 5. Calcular Métricas y Formatear Resultados ---
    
    # Crear un DataFrame de resultados
    df_resultados = pd.DataFrame({
        "FECHA_HORA": fechas_test,
        "prediccion": y_pred_test,
        "y_true": y_true_series # <-- Añadir y_true
    })
    
    # Copia para métricas: eliminar filas donde no se puede comparar
    # (ej. lags iniciales o la última fila de y_true que es NaN)
    df_metrics = df_resultados.dropna(subset=['prediccion', 'y_true'])

    if not df_metrics.empty:
        print("\n=== Métricas de Predicción (Test) ===")
        try:
            mae = mean_absolute_error(df_metrics['y_true'], df_metrics['prediccion'])
            r2 = r2_score(df_metrics['y_true'], df_metrics['prediccion'])
            print(f"MAE : {mae:,.2f}")
            print(f"R²  : {r2:.4f}")
        except Exception as e:
            print(f"No se pudieron calcular las métricas: {e}")
    else:
        print("\nNo hay suficientes datos alineados para calcular métricas.")

    # Eliminar filas donde no se pudo predecir (p.ej. por lags iniciales)
    # Dejamos la columna y_true (tendrá NaN al final) para inspección
    df_resultados_final = df_resultados.dropna(subset=['prediccion'])

    return df_resultados_final


if __name__ == "__main__":
    
    # --- Configuración de Argumentos de Consola ---
    parser = argparse.ArgumentParser(
        description="Ejecuta el pipeline de predicción sobre un archivo Excel."
    )
    
    # Argumento posicional obligatorio: la URL o ruta
    parser.add_argument(
        "url",
        type=str,
        help="URL o ruta local al archivo Excel (.xlsx) con los datos a predecir."
    )
    
    # Argumento opcional: la ruta al pipeline
    parser.add_argument(
        "--pipeline",
        type=str,
        default="models/pipeline_artifacts.pkl",
        help="Ruta al archivo .pkl o .joblib del pipeline guardado (default: models/pipeline_artifacts.pkl)"
    )
    
    args = parser.parse_args()
    
    # --- Fin de la Configuración ---

    try:
        # Usamos los argumentos en lugar de valores fijos
        resultados_df = run_prediction_from_excel(
            url=args.url, 
            pipeline_path=args.pipeline
        )
        
        if not resultados_df.empty:
            print("\n=== Resultados de la Predicción ===")
            print(resultados_df)
            
            # Guardar resultados
            resultados_df.to_csv("predicciones_excel.csv", index=False)
            print("\nPredicciones guardadas en 'predicciones_excel.csv'")
            
    except Exception as e:
        print(f"\nOcurrió un error durante la predicción: {e}")
        print("Asegúrate de que la URL es correcta y los datos tienen el formato esperado.")
        print("Recuerda ejecutar 'python train_pipeline.py' primero.")