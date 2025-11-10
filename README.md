**Resumen**
- Proyecto de modelado predictivo de consumo/energía a partir de planillas Excel con múltiples hojas "Consolidado". Integra un pipeline reproducible de preparación de datos, ingeniería de características, tratamiento de outliers y nulos, entrenamiento y evaluación de modelos (Random Forest, XGBoost, LightGBM), además de EDA detallado y registro de experimentos.

**Estructura**
- Código principal: `pipeline.py:1`
- Notebooks: `eda.ipynb`, `modelado.ipynb`, `foundational_dataset.ipynb`
- Resultados y logs: `results/`
- Requerimientos: `requirements.txt`

**Datos**
- Fuente: archivo Excel con múltiples hojas; se usan solo hojas cuyo nombre inicia con "Consolidado".
- Llave temporal: se construye `FECHA_HORA` a partir de `DIA`+`HORA` o de la columna `FECHA_HORA` si ya existe.
- Target: `Frio (Kw)`; en inferencia se predice el valor del día siguiente (shift -1) a partir de features del pasado.

**Preparación de Hojas**
- Función `preparar_hoja(...)` en `pipeline.py:20`:
  - Normaliza timestamps a `FECHA_HORA` (conversión robusta desde `DIA`/`HORA`).
  - Elimina nulos en `FECHA_HORA` y agrega duplicados por `FECHA_HORA` (numéricas='mean', otras='first').
  - Devuelve 1 fila por timestamp y reporta duplicados para diagnóstico.

**Construcción del Dataset**
- Función `run_pipeline_from_excel_url(...)` en `pipeline.py:100`:
  - Lee Excel (`pandas.read_excel`) y filtra hojas "Consolidado".
  - Merge outer por `FECHA_HORA` de todas las hojas y orden temporal.
  - Selecciona la última medición por día (idxmax de `FECHA_HORA` por fecha).
  - Ingeniería de variables sobre `Frio (Kw)` con lags y rolling means/std (3, 7, 14, 28).
  - Define `y_test` como `Frio (Kw)` desplazado -1 (predicción del siguiente valor).

**Entrenamiento vs. Inferencia**
- Train/val históricos: se cargan de CSVs en raíz `x_train.csv`, `y_train.csv`, `x_test.csv`, `y_test.csv` para entrenar y consolidar (`pipeline.py:139`).
- Test de producción: se arma a partir del Excel procesado (features alineadas con train).
- Conjunto de features: lista fija `important_features` (ajustar según disponibilidad en train/test).

**Tratamiento de Outliers y Nulos**
- Bounds por MAD por columna (fallback a percentiles amplios si MAD=0) `pipeline.py:56`.
- Reemplazo cell-wise a NaN si está fuera de [low, high].
- Escalado robusto (`RobustScaler`) + imputación KNN en espacio escalado, luego inverse-transform a escala original `pipeline.py:220`.

**Transformaciones y Modelo**
- Transformación `PowerTransformer(method='yeo-johnson')` sobre X antes del modelo.
- Modelo principal cargado: `RandomForestRegressor` con hiperparámetros optimizados `pipeline.py:242`.
- Métricas reportadas: MSE, RMSE, MAE, R² sobre el test de Excel.

**EDA y Experimentos**
- `eda.ipynb`: distribución de variables, outliers (LOF), imputación (KNN), escalado (Robust/Standard), correlaciones (Spearman), clustering de correlación, VIF, winsorización.
- `modelado.ipynb`: pipelines con `PowerTransformer`/`StandardScaler`, `RandomForest`, `Ridge/Lasso`, `XGBoost`, `LightGBM` con early stopping; tuning con Optuna (TPE + pruners); validación temporal (`TimeSeriesSplit`).
- Registros y artefactos en `results/`:
  - Logs: `results/experiment_logs.csv`
  - Resúmenes: `results/experiment_summary_*.md`
  - Figuras y métricas: importancia de features, residuales, predicción vs. verdad para RF y XGBoost (archivos `*.png`, `*.csv`).

**Cómo Ejecutar**
- Requisitos:
  - Python 3.9+ y `pip`.
  - Instalar dependencias: `pip install -r requirements.txt`.
- Notebooks:
  - Abrir `eda.ipynb` y `modelado.ipynb` en Jupyter/VS Code y ejecutar celdas.
- Pipeline desde Python:
  - Preparar `x_train.csv`, `y_train.csv`, `x_test.csv`, `y_test.csv` en la raíz con las columnas esperadas (incluyendo `important_features`).
  - Llamar:
    - `from pipeline import run_pipeline_from_excel_url`
    - `mae = run_pipeline_from_excel_url("ruta/o/url/al.xlsx", random_state=42)`
  - El pipeline imprime métricas y devuelve el MAE del test armado desde Excel.

**Estructura de Resultados**
- Figuras y CSVs de cada experimento se guardan con prefijo de timestamp, por ejemplo:
  - `results/20251110_030137_RandomForest_*`
  - `results/20251110_052302_XGBoost_*`
- Resumen global: `results/experiment_logs.csv` centraliza métricas por corrida.

**Supuestos y Requisitos de Datos**
- Excel con hojas "Consolidado*" que comparten una columna temporal convertible a `FECHA_HORA`.
- Columnas usadas por `important_features` disponibles y consistentes entre train y test.
- `openpyxl` instalado para lectura de Excel.

