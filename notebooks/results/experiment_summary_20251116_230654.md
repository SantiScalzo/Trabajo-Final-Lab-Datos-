# Experiment Summary — 2025-11-16T23:07:22.079622

## Top 10 por RMSE de Validación

| timestamp                  | model_name     |    val_rmse |   val_mae |     val_r2 |   test_rmse |   test_mae |   test_r2 |
|:---------------------------|:---------------|------------:|----------:|-----------:|------------:|-----------:|----------:|
| 2025-11-16T22:54:21.542343 | XGBoost        | 2.39484e+07 |   3743.16 |   0.435179 | 1.49883e+07 |    2968.29 |  0.57606  |
| 2025-11-16T23:07:12.214556 | XGBoost        | 2.39484e+07 |   3743.16 |   0.435179 | 1.49883e+07 |    2968.29 |  0.57606  |
| 2025-11-16T23:07:12.334111 | LightGBM       | 3.37797e+07 |   4724.31 |   0.203311 | 5.55546e+07 |    6325.19 | -0.571351 |
| 2025-11-16T22:54:21.692402 | LightGBM       | 3.37797e+07 |   4724.31 |   0.203311 | 5.55546e+07 |    6325.19 | -0.571351 |
| 2025-11-16T23:07:20.044446 | Ridge (Optuna) | 4.70192e+07 |   5402.61 |  -0.108943 | 4.08069e+07 |    4843.74 | -0.154215 |
| 2025-11-16T22:54:29.555730 | Ridge (Optuna) | 4.71054e+07 |   5406.85 |  -0.110975 | 4.09242e+07 |    4850.94 | -0.157532 |
| 2025-11-16T23:06:56.517899 | RandomForest   | 4.97499e+07 |   3821.33 |  -0.173345 | 1.17549e+07 |    2629.26 |  0.667514 |
| 2025-11-16T22:54:05.250011 | RandomForest   | 4.97499e+07 |   3821.33 |  -0.173345 | 1.17549e+07 |    2629.26 |  0.667514 |
| 2025-11-16T22:54:30.582215 | Lasso (Optuna) | 7.83058e+08 |  21729.4  | -17.4683   | 2.88194e+08 |   12623.4  | -7.15151  |
| 2025-11-16T23:07:21.091720 | Lasso (Optuna) | 7.83679e+08 |  21738    | -17.483    | 2.88402e+08 |   12626.8  | -7.15739  |

## Espacios de búsqueda declarados

- **XGBoost**: `Fixed params; early_stopping_rounds=200`
- **XGBoost**: `Fixed params; early_stopping_rounds=200`
- **LightGBM**: `Fixed params; early_stopping_rounds=200`
- **LightGBM**: `Fixed params; early_stopping_rounds=200`
- **Ridge (Optuna)**: `alpha ~ loguniform[1e-4, 1e3]; n_trials=50`
- **Ridge (Optuna)**: `alpha ~ loguniform[1e-4, 1e3]; n_trials=50`
- **RandomForest**: `Fixed params (baseline)`
- **RandomForest**: `Fixed params (baseline)`
- **Lasso (Optuna)**: `alpha ~ loguniform[1e-4, 1e2]; n_trials=50`
- **Lasso (Optuna)**: `alpha ~ loguniform[1e-4, 1e2]; n_trials=50`

## Justificación

- Se elige el mejor modelo por **RMSE de validación** (menor es mejor).
- Se reportan métricas en test del modelo **refiteado en todo el train**.
- La elección queda respaldada por `results/experiment_logs.csv` y este resumen.