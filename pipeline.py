from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from pathlib import Path
from functools import reduce
import numpy as np

def mask_outliers_to_nan(df_num: pd.DataFrame, bounds_dict: dict) -> pd.DataFrame:
    df = df_num.copy()
    for c, (low, high) in bounds_dict.items():
        mask_low  = df[c] < low
        mask_high = df[c] > high
        df.loc[mask_low | mask_high, c] = np.nan
    return df
# =========================
# 3) Límites por feature con MAD (fallback a percentiles si MAD=0)
# =========================
def mad_bounds(col_train: pd.Series, Z: float = 3.5,
            q_low: float = 0.001, q_high: float = 0.999):
    """
    Devuelve (low, high) por MAD; si MAD==0, usa percentiles (q_low, q_high).
    """
    med = col_train.median()
    mad = np.median(np.abs(col_train - med))
    if mad > 0:
        # robust z-score bounds
        # 0.6745 hace que MAD sea comparable a sigma para Normal
        scale = 0.6745 * (col_train - med).abs().median() / mad  # opcional; muchos usan 0.6745 directamente
        # usamos la forma clásica: robust_z = 0.6745*(x-med)/MAD
        # límites: med ± Z * MAD / 0.6745
        low  = med - (Z * mad / 0.6745)
        high = med + (Z * mad / 0.6745)
    else:
        # Sin dispersión: usamos percentiles amplios
        low  = col_train.quantile(q_low)
        high = col_train.quantile(q_high)
        if low == high:
            # todos iguales: expandir un poco para no anular la feature
            low, high = low - 1e-12, high + 1e-12
    return low, high


def preparar_hoja(df: pd.DataFrame, nombre_hoja: str) -> pd.DataFrame:
    df = df.copy()
    
    # Construir FECHA_HORA
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
    #quiero saber si hay nans
    print(f"Hoja {nombre_hoja}: Nulos en FECHA_HORA antes de dropna: {df['FECHA_HORA'].isna().sum()}")
    dias_nulos = df['DIA'][df['FECHA_HORA'].isna()]
    horas_nulas = df['HORA'][df['FECHA_HORA'].isna()]
    if not dias_nulos.empty or not horas_nulas.empty:
        print(f"Días nulos:\n{dias_nulos}")
        print(f"Horas nulas:\n{horas_nulas}") 
    df = df.dropna(subset=['FECHA_HORA']).reset_index(drop=True)
    # 1) Normalizar a una fila por timestamp dentro de la hoja
    #    - numéricas: 'mean' (si tus columnas son intensidades; usa 'sum' si son totales)
    #    - no numéricas: 'first'
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    other_cols = [c for c in df.columns if c not in numeric_cols + ['DIA', 'HORA', 'FECHA_HORA']]
    print(f"Duplicadas por FECHA_HORA en hoja {nombre_hoja} antes de procesmiento: {df.duplicated(subset=['FECHA_HORA']).sum()}")

    agg_map = {**{c: 'mean' for c in numeric_cols}, **{c: 'first' for c in other_cols}}
    df = (
        df.drop(columns=['DIA', 'HORA'], errors='ignore')
          .groupby('FECHA_HORA', as_index=False)
          .agg(agg_map)
    )
    print(f"Duplicadas por FECHA_HORA en hoja {nombre_hoja} despues de procesmiento: {df.duplicated(subset=['FECHA_HORA']).sum()}")
    #quiero mostrar las filas duplicadas
    filas_duplicadas = df[df.duplicated(subset=['FECHA_HORA'], keep=False)]
    if not filas_duplicadas.empty:
        print(f"Filas duplicadas por FECHA_HORA en hoja {nombre_hoja}:\n{filas_duplicadas}")
        print(filas_duplicadas)

    
    return df





def run_pipeline_from_excel_url(
    url: str,
    random_state: int = 42,
) -> Union[float, Tuple[float, Dict[str, object]]]:
    """
    Carga un Excel desde una URL y ejecuta el mismo pipeline del proyecto
    para calcular el MAE sobre un hold-out de test. Deja puntos de extensión
    para reusar tus funciones reales de preprocesamiento y modelo.

    Parámetros
    - url: URL del archivo Excel a procesar (admite HTTP/HTTPS y rutas locales).

    - random_state: semilla para la reproducibilidad del split.

    Retorna
    - mae: Mean Absolute Error en el conjunto de test.
    - opcionalmente un dict con artefactos si return_artifacts=True.

    Notas de integración
    - Reemplaza `preprocessing_fn`, `model_loader` y/o `model_trainer` por las
      funciones reales de tu proyecto para replicar el pipeline fielmente.
    - Si tu pipeline separa train/val/test o usa validación cruzada, adapta la
      lógica de splitting y evaluación según corresponda.
    """


    dict_de_hojas = pd.read_excel(url,sheet_name=None)
    #Solo quiero seleccionar las hojas cuyo nombre empiece con "Consolidado"
    dict_de_hojas = {nombre: df for nombre, df in dict_de_hojas.items() if nombre.startswith("Consolidado")}
    hojas_preparadas = [h for nombre, df in dict_de_hojas.items()
                        if (h := preparar_hoja(df, nombre)) is not None and len(h) > 0]

    # Merge por FECHA_HORA (ya sin duplicados por hoja)
    df_combinado = reduce(lambda l, r: pd.merge(l, r, on='FECHA_HORA', how='outer'), hojas_preparadas)
    df_combinado = df_combinado.sort_values('FECHA_HORA').reset_index(drop=True)

    # Diagnóstico duplicados tras el merge (debería bajar muchísimo)
    print("Duplicadas exactas:", df_combinado.duplicated().sum())
    print("Duplicadas por FECHA_HORA:", df_combinado.duplicated(subset=['FECHA_HORA']).sum())


    # Caso B: quedarme con datos DESDE esa fecha (excluida)
    # df_filtrado = df_combinado[df_combinado['FECHA_HORA'] > fecha_limite].copy()

    print(f"Rango: {df_combinado['FECHA_HORA'].min()} -> {df_combinado['FECHA_HORA'].max()}")
    print(f"Filas: {len(df_combinado)}, Columnas: {df_combinado.shape[1]}")

    # Agrupa por la fecha (la parte de día de 'FECHA_HORA') y encuentra la hora máxima (idxmax) para esa fecha.
    indices_ultima_medicion = df_combinado.groupby(
        df_combinado['FECHA_HORA'].dt.date
    )['FECHA_HORA'].idxmax()

    # --- PASO 3: Filtrar el DataFrame Original ---
    # Selecciona solo las filas correspondientes a los índices de la última medición.
    df_combinado = df_combinado.loc[indices_ultima_medicion]
    
    df_combinado["Frio_diff1_lag1"] = df_combinado["Frio (Kw)"].astype(float).diff().shift(1)
    df_combinado["Frio_diff7_lag1"] = df_combinado["Frio (Kw)"].astype(float).diff().shift(1)
    roll_windows = [3, 7, 14,28]
    for window in roll_windows:
        df_combinado[f"Frio_roll_mean_{window}_lag1"] = df_combinado["Frio (Kw)"].astype(float).shift(1).rolling(window=window).mean()
        df_combinado[f"Frio_roll_std_{window}_lag1"] = df_combinado["Frio (Kw)"].astype(float).shift(1).rolling(window=window).std()

    #Ademas vamos a definir un z score para la columna Frio (Kw)


    y_test = df_combinado["Frio (Kw)"].shift(-1)  # predecir el siguiente valor de Frio (Kw)
    y_test.dropna(inplace=True)
    #pongo los mismos indices en df_combinado
    df_combinado = df_combinado.loc[y_test.index]
    X_test = df_combinado.drop(columns=["FECHA_HORA"]).copy()
    important_features = ['Frio (Kw)', 'Frio_roll_mean_7_lag1', 'Sala Maq (Kw)', 'Envasado (Kw)', 'Frio_roll_mean_14_lag1', 'Servicios (Kw)', 'Prod Agua (Kw)', 'KW Gral Planta', 'Linea 2 (Kw)', 'CO 2 / Hl', 'EE Caldera / Hl', 'Cocina (Kw)', 'ET Linea 5/Hl', 'VAPOR DE LINEA 4 KG', 'Linea 3 (Kw)', 'Resto Serv (Kw)', 'Conversion Kg/Mj', 'Restos Planta (Kw)', 'Hl Cerveza L2', 'Frio_roll_mean_3_lag1']
    X_test = X_test[important_features]
    
    X_train = pd.read_csv("x_train.csv")
    y_train = pd.read_csv("y_train.csv")
    X_val = pd.read_csv("x_test.csv")
    y_val = pd.read_csv("y_test.csv")

    X_train = pd.concat([X_train, X_val], ignore_index=True)
    y_train = pd.concat([y_train, y_val], ignore_index=True)

    # Definir datasets en minúscula usados más abajo
    x_train = X_train.copy()
    x_test = X_test.copy()

    # Alinear columnas: usar solo las comunes para evitar KeyError
    common_cols = [c for c in x_train.columns if c in x_test.columns]
    if len(common_cols) == 0:
        raise ValueError(
            "No hay columnas en común entre train y test. Revisa 'important_features' y los CSVs de train/val."
        )
    x_train = x_train[common_cols]
    x_test = x_test[common_cols]


    numeric_cols = x_train.select_dtypes(include=[np.number]).columns.tolist()
    other_cols   = [c for c in x_train.columns if c not in numeric_cols]

    # Copias aisladas de numéricas
    Xtr_num = x_train[numeric_cols].copy()
    Xte_num = x_test[numeric_cols].copy()

    
    bounds = {}
    for c in numeric_cols:
        low, high = mad_bounds(Xtr_num[c].dropna(), Z=3.5, q_low=0.001, q_high=0.999)
        bounds[c] = (low, high)

    # =========================
    # 4) Reemplazo cellwise: fuera de [low, high] -> np.nan
    #    (usamos límites aprendidos SOLO del train)
    # =========================
    

    Xtr_num_nan = mask_outliers_to_nan(Xtr_num, bounds)
    Xte_num_nan = mask_outliers_to_nan(Xte_num, bounds)

    # Si querés contar cuántas celdas se “nanearon”:
    tr_nan_cells = Xtr_num_nan.isna().sum().sum() - Xtr_num.isna().sum().sum()
    te_nan_cells = Xte_num_nan.isna().sum().sum() - Xte_num.isna().sum().sum()
    print(f"Celdas convertidas a NaN por outliers - train: {tr_nan_cells}, test: {te_nan_cells}")

    # =========================
    # 5) Imputación KNN en ESPACIO ESCALADO y volver a original
    # =========================
    from sklearn.preprocessing import RobustScaler
    from sklearn.impute import KNNImputer

    scaler = RobustScaler()
    Xtr_scaled = pd.DataFrame(scaler.fit_transform(Xtr_num_nan), columns=numeric_cols, index=Xtr_num_nan.index)
    Xte_scaled = pd.DataFrame(scaler.transform(Xte_num_nan),    columns=numeric_cols, index=Xte_num_nan.index)

    imputer = KNNImputer(n_neighbors=5, weights='distance')
    Xtr_imp_scaled = pd.DataFrame(imputer.fit_transform(Xtr_scaled), columns=numeric_cols, index=Xtr_scaled.index)
    Xte_imp_scaled = pd.DataFrame(imputer.transform(Xte_scaled),    columns=numeric_cols, index=Xte_scaled.index)

    # Volvemos a la escala original
    Xtr_imp = pd.DataFrame(scaler.inverse_transform(Xtr_imp_scaled), columns=numeric_cols, index=Xtr_imp_scaled.index)
    Xte_imp = pd.DataFrame(scaler.inverse_transform(Xte_imp_scaled), columns=numeric_cols, index=Xte_imp_scaled.index)

    # =========================
    # 6) Reconstruir datasets finales
    # =========================
    # Alinear índices antes de concatenar para evitar duplicación de filas por union-index
    x_train_clean = pd.concat([
        Xtr_imp.reset_index(drop=True),
        x_train[other_cols].reset_index(drop=True)
    ], axis=1)[x_train.columns]
    # Asegurar que test tenga exactamente las columnas y orden de train
    x_test_clean  = pd.concat([
        Xte_imp.reset_index(drop=True),
        x_test[other_cols].reset_index(drop=True)
    ], axis=1)[x_train.columns]

    # Resultado final para modelar
    x_train = x_train_clean
    x_test  = x_test_clean

    best_model = RandomForestRegressor(
    n_estimators=1183,
    max_depth=16,
    max_features=1.0,
    bootstrap=True,
    min_samples_split=2,
    min_samples_leaf=2,
    max_samples=0.8654367662002576,
    random_state=42,
    n_jobs=-1
    )
    from sklearn.preprocessing import PowerTransformer
    pt = PowerTransformer(method='yeo-johnson')
    x_train_ready = pd.DataFrame(pt.fit_transform(x_train), columns=x_train.columns)
    x_test_ready = pd.DataFrame(pt.transform(x_test), columns=x_test.columns)

    # y_train a 1D por compatibilidad con scikit-learn
    if isinstance(y_train, pd.DataFrame):
        if y_train.shape[1] == 1:
            y_train = y_train.iloc[:, 0]
        else:
            y_train = y_train.squeeze()

    best_model.fit(x_train_ready, y_train)
    y_pred_test = best_model.predict(x_test_ready)
    mse  = mean_squared_error(y_test, y_pred_test)
    rmse = mean_squared_error(y_test, y_pred_test)
    mae  = mean_absolute_error(y_test, y_pred_test)
    r2   = r2_score(y_test, y_pred_test)
    print("\n=== Métricas en TEST del mejor modelo entrenado manualmente ===")
    print(f"MSE : {mse:,.2f}")
    print(f"RMSE: {rmse:,.2f}")
    print(f"MAE : {mae:,.2f}")
    print(f"R²  : {r2:.4f}")

    

    return mae


def preprocess_like_project(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[Dict]]:
    """
    Esqueleto de preprocesamiento para replicar el pipeline del proyecto.
    Implementa aquí los pasos reales (limpieza, imputación, encoding, escalado, etc.).

    Debe devolver un DataFrame listo para modelar y un dict opcional con extras.
    """
    return df, None
