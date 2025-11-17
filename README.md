

# Predicción de Consumo de Frío en Planta Cervecera

Este proyecto implementa un sistema completo para predecir el consumo energético de refrigeración (Frío en kW) en una planta cervecera, utilizando datos operativos históricos y un pipeline de Machine Learning reproducible.

---

## Inicio Rápido

### Prerrequisitos

Instalar dependencias:

```bash
pip install -r requirements.txt
```

---

## Uso del Script de Predicción

El script `src/predict.py` permite generar predicciones de consumo de frío (kW) a partir de un archivo Excel en formato adecuado.

### Sintaxis básica

```bash
python src/predict.py <ruta_al_archivo.xlsx>
```

### Ejemplos

1. Predicción con archivo local

   ```bash
   python src/predict.py data/Archivos_xlsx/Planta_2023.xlsx
   ```

2. Predicción con archivo externo

   ```bash
   python src/predict.py C:/datos/nuevos_datos.xlsx
   ```

3. Mostrar ayuda

   ```bash
   python src/predict.py --help
   ```

---

## Formato de Entrada Requerido

El archivo Excel debe incluir:

* Hojas cuyo nombre comience con `"Consolidado"`
  Ejemplos: `Consolidado KPI`, `Consolidado EE`, `Consolidado Agua`, etc.
* Columnas temporales:

  * `DIA` y `HORA`, o
  * `FECHA_HORA`
* Variables operativas: consumo energético, producción, aire, vapor, servicios, etc.

### El script automáticamente:

1. Fusiona todas las hojas *Consolidado*
2. Selecciona la última medición de cada día
3. Genera features de lags y rolling windows
4. Aplica un pipeline completo (outliers, escalado, imputación, etc.)
5. Predice el consumo de frío del día siguiente

---

## Salida del Script

Si el Excel contiene datos históricos, se mostrarán métricas y se generará un archivo CSV:

### Ejemplo de salida

```
=== Métricas de Predicción (Test) ===
MAE : 125.43
R²  : 0.8542
```

Archivo generado: `predicciones_excel.csv`

```csv
FECHA_HORA,prediccion,y_true
2023-03-06 23:59:00,1234.56,1245.32
2023-03-07 23:59:00,1289.45,1298.76
...
```

---

## Notas Importantes

* Primer uso: ejecutar

  ```bash
  python src/train_model.py
  ```

  para generar `models/pipeline_artifacts.pkl`.

* Las primeras filas pueden no tener predicción por falta de historial (lags 1–28 días).

* El modelo requiere las mismas columnas con las que fue entrenado; si faltan, se rellenan con `NaN`.

---

# Contexto General del Proyecto

## Objetivo

Construir un sistema de predicción robusto y reproducible para estimar el consumo de frío del día siguiente en una planta cervecera a partir de series temporales multivariadas.

---

# Estructura del Proyecto

```
proyecto/
│
├── data/
│   ├── Archivos_xlsx/               # Archivos Excel 2020-2023
│   └── processed/
│       ├── foundational_dataset.csv # Dataset unificado diario
│       ├── splits/                  # Train/test
│       └── data_lineage.json        # Documentación del pipeline
│
├── models/
│   └── pipeline_artifacts.pkl       # Pipeline entrenado (pre + modelo)
│
├── notebooks/
│   ├── foundational_dataset.ipynb   # Construcción dataset base
│   ├── eda.ipynb                    # Exploración y limpieza
│   └── modelado.ipynb               # Tuning y evaluación
│
├── src/
│   ├── train_model.py               # Entrenamiento completo
│   └── predict.py                   # Script de inferencia
│
├── results/                         # Logs, gráficos, métricas
├── requirements.txt
└── README.md
```

---

# Pipeline de Datos

## 1. Preparación de Datos (`foundational_dataset.ipynb`)

* Fuente: archivos Excel con hojas *Consolidado*
* Proceso:

  * Unión por timestamp (`FECHA_HORA`)
  * Agregación diaria tomando la última medición del día
  * Limpieza de nombres
  * Features cíclicos (mes, día de semana → seno/coseno)

Salida: `foundational_dataset.csv`
(1190 días diarios entre 2020 y 2023)

---

## 2. Ingeniería de Características

Características generadas para predecir Frio (kW):

* Lags: `Frio_diff1_lag1`, `Frio_diff7_lag1`
* Rolling windows: medias y std de ventanas de 3, 7, 14 y 28 días
* Target: `Frio (Kw)` desplazado -1 día

---

## 3. Preprocesamiento (`eda.ipynb`)

El pipeline incluye:

1. Detección de outliers — MAD (Z=3.5)
2. Escalado — `RobustScaler`
3. Imputación — `KNNImputer (k=5, distance)`
4. Normalización — `PowerTransformer (Yeo-Johnson)`
5. Selección de features — top 20 por varianza y correlación

---

## 4. Modelado (`modelado.ipynb`)

### Modelos evaluados:

* RandomForest
* XGBoost
* LightGBM
* Ridge/Lasso

### Optimización:

* Optuna (TPE + pruners)
* Validación temporal (TimeSeriesSplit)

### Modelo final: RandomForestRegressor

* `n_estimators: 1183`
* `max_depth: 16`
* `max_features: 1.0`

Resultados:

* MAE ≈ 125 kW
* R² ≈ 0.85

---

# Características del Sistema

* Reproducible: pipeline serializado completo
* Robusto: manejo de outliers, NA, duplicados
* Trazable: logs de experimentos en `results/`
* Flexible: acepta nuevos Excel sin modificar código
* Interpretación: importancia de features y análisis residual

---

# Variables Más Importantes

1. Frio_roll_mean_7_lag1
2. Frio (Kw)
3. Frio_roll_mean_14_lag1
4. Sala Maq (Kw)
5. Envasado (Kw)
6. Servicios (Kw)
7. KW Gral Planta
8. Linea 2 (Kw)
9. CO2 / Hl
10. Prod Agua (Kw)

---

# Limitaciones

* 23 días faltantes en datos históricos
* Lags iniciales: primeras 28 filas sin predicción
* Eventos operativos extraordinarios pueden no estar representados

---

# Cómo Entrenar el Modelo

```bash
# (Opcional) Regenerar dataset base
jupyter notebook notebooks/foundational_dataset.ipynb

# Entrenar pipeline completo
python src/train_model.py
```

Generará:
`models/pipeline_artifacts.pkl`

---

# Experimentación

* EDA: `notebooks/eda.ipynb`
* Modelos y tuning: `notebooks/modelado.ipynb`
* Resultados: gráficos, residuales y métricas en `results/`

---
