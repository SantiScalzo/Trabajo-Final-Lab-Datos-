# Experiment Summary — 2025-11-17T00:29:58.263058

## Top 10 por MAE de Test

| timestamp                  | model_name     |   test_mae |      test_rmse |   test_r2 |       val_rmse |   val_mae |
|:---------------------------|:---------------|-----------:|---------------:|----------:|---------------:|----------:|
| 2025-11-17T00:07:44.722247 | RandomForest   |    2627.24 | 3424.26        |  0.668345 | 7048.48        |   3817.24 |
| 2025-11-16T22:54:05.250011 | RandomForest   |    2629.26 |    1.17549e+07 |  0.667514 |    4.97499e+07 |   3821.33 |
| 2025-11-16T23:06:56.517899 | RandomForest   |    2629.26 |    1.17549e+07 |  0.667514 |    4.97499e+07 |   3821.33 |
| 2025-11-17T00:29:32.150380 | RandomForest   |    2656.54 | 3461.71        |  0.661052 | 6978.79        |   3754.84 |
| 2025-11-17T00:29:46.468126 | XGBoost        |    2823.35 | 3633.76        |  0.626522 | 4882.65        |   3646.79 |
| 2025-11-17T00:07:58.736855 | XGBoost        |    2909.85 | 3772.73        |  0.597408 | 4906.72        |   3694.69 |
| 2025-11-16T22:54:21.542343 | XGBoost        |    2968.29 |    1.49883e+07 |  0.57606  |    2.39484e+07 |   3743.16 |
| 2025-11-16T23:07:12.214556 | XGBoost        |    2968.29 |    1.49883e+07 |  0.57606  |    2.39484e+07 |   3743.16 |
| 2025-11-17T00:29:55.560226 | Ridge (Optuna) |    4777.9  | 6268.38        | -0.111384 | 7419.41        |   5901.35 |
| 2025-11-16T23:07:20.044446 | Ridge (Optuna) |    4843.74 |    4.08069e+07 | -0.154215 |    4.70192e+07 |   5402.61 |

## Espacios de búsqueda declarados

- **RandomForest**: `Fixed params (baseline)`
- **RandomForest**: `Fixed params (baseline)`
- **RandomForest**: `Fixed params (baseline)`
- **RandomForest**: `Fixed params (baseline)`
- **XGBoost**: `Fixed params; early_stopping_rounds=200`
- **XGBoost**: `Fixed params; early_stopping_rounds=200`
- **XGBoost**: `Fixed params; early_stopping_rounds=200`
- **XGBoost**: `Fixed params; early_stopping_rounds=200`
- **Ridge (Optuna)**: `alpha ~ loguniform[1e-4, 1e3]; n_trials=50`
- **Ridge (Optuna)**: `alpha ~ loguniform[1e-4, 1e3]; n_trials=50`
- **Ridge (Optuna)**: `alpha ~ loguniform[1e-4, 1e3]; n_trials=50`
- **Ridge (Optuna)**: `alpha ~ loguniform[1e-4, 1e3]; n_trials=50`
- **LightGBM**: `Fixed params; early_stopping_rounds=200`
- **LightGBM**: `Fixed params; early_stopping_rounds=200`
- **LightGBM**: `Fixed params; early_stopping_rounds=200`
- **LightGBM**: `Fixed params; early_stopping_rounds=200`
- **Lasso (Optuna)**: `alpha ~ loguniform[1e-4, 1e2]; n_trials=50`
- **Lasso (Optuna)**: `alpha ~ loguniform[1e-4, 1e2]; n_trials=50`
- **Lasso (Optuna)**: `alpha ~ loguniform[1e-4, 1e2]; n_trials=50`
- **Lasso (Optuna)**: `alpha ~ loguniform[1e-4, 1e2]; n_trials=50`

## Justificación

- **CRITERIO DE SELECCIÓN: Menor MAE en Test.**
- Se reportan métricas en test del modelo refiteado en todo el train.
- La elección queda respaldada por `results/experiment_logs.csv` y este resumen.