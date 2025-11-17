# Experiment Summary — 2025-11-17T00:08:30.749807

## Top 10 por MAE de Test

| timestamp                  | model_name     |   test_mae |      test_rmse |   test_r2 |       val_rmse |   val_mae |
|:---------------------------|:---------------|-----------:|---------------:|----------:|---------------:|----------:|
| 2025-11-17T00:07:44.722247 | RandomForest   |    2627.24 | 3424.26        |  0.668345 | 7048.48        |   3817.24 |
| 2025-11-16T22:54:05.250011 | RandomForest   |    2629.26 |    1.17549e+07 |  0.667514 |    4.97499e+07 |   3821.33 |
| 2025-11-16T23:06:56.517899 | RandomForest   |    2629.26 |    1.17549e+07 |  0.667514 |    4.97499e+07 |   3821.33 |
| 2025-11-17T00:07:58.736855 | XGBoost        |    2909.85 | 3772.73        |  0.597408 | 4906.72        |   3694.69 |
| 2025-11-16T23:07:12.214556 | XGBoost        |    2968.29 |    1.49883e+07 |  0.57606  |    2.39484e+07 |   3743.16 |
| 2025-11-16T22:54:21.542343 | XGBoost        |    2968.29 |    1.49883e+07 |  0.57606  |    2.39484e+07 |   3743.16 |
| 2025-11-16T23:07:20.044446 | Ridge (Optuna) |    4843.74 |    4.08069e+07 | -0.154215 |    4.70192e+07 |   5402.61 |
| 2025-11-16T22:54:29.555730 | Ridge (Optuna) |    4850.94 |    4.09242e+07 | -0.157532 |    4.71054e+07 |   5406.85 |
| 2025-11-17T00:08:28.325965 | Ridge (Optuna) |    4949.25 | 6522.8         | -0.203431 | 6950.47        |   5464.94 |
| 2025-11-16T22:54:21.692402 | LightGBM       |    6325.19 |    5.55546e+07 | -0.571351 |    3.37797e+07 |   4724.31 |

## Espacios de búsqueda declarados

- **RandomForest**: `Fixed params (baseline)`
- **RandomForest**: `Fixed params (baseline)`
- **RandomForest**: `Fixed params (baseline)`
- **XGBoost**: `Fixed params; early_stopping_rounds=200`
- **XGBoost**: `Fixed params; early_stopping_rounds=200`
- **XGBoost**: `Fixed params; early_stopping_rounds=200`
- **Ridge (Optuna)**: `alpha ~ loguniform[1e-4, 1e3]; n_trials=50`
- **Ridge (Optuna)**: `alpha ~ loguniform[1e-4, 1e3]; n_trials=50`
- **Ridge (Optuna)**: `alpha ~ loguniform[1e-4, 1e3]; n_trials=50`
- **LightGBM**: `Fixed params; early_stopping_rounds=200`
- **LightGBM**: `Fixed params; early_stopping_rounds=200`
- **LightGBM**: `Fixed params; early_stopping_rounds=200`
- **Lasso (Optuna)**: `alpha ~ loguniform[1e-4, 1e2]; n_trials=50`
- **Lasso (Optuna)**: `alpha ~ loguniform[1e-4, 1e2]; n_trials=50`
- **Lasso (Optuna)**: `alpha ~ loguniform[1e-4, 1e2]; n_trials=50`

## Justificación

- **CRITERIO DE SELECCIÓN: Menor MAE en Test.**
- Se reportan métricas en test del modelo refiteado en todo el train.
- La elección queda respaldada por `results/experiment_logs.csv` y este resumen.