# Experiment Summary — 2025-11-10T05:23:34.085435

## Top 10 por RMSE de Validación

| timestamp                  | model_name     |    val_rmse |   val_mae |    val_r2 |   test_rmse |   test_mae |   test_r2 |
|:---------------------------|:---------------|------------:|----------:|----------:|------------:|-----------:|----------:|
| 2025-11-10T03:01:40.476653 | RandomForest   | 1.79712e+07 |   3199.4  |  0.629994 | 1.03115e+07 |    2544.82 |  0.349774 |
| 2025-11-10T03:02:07.690548 | XGBoost        | 2.04578e+07 |   3406.08 |  0.578797 | 1.22134e+07 |    2800.85 |  0.229841 |
| 2025-11-10T05:23:23.991665 | XGBoost        | 2.39484e+07 |   3743.16 |  0.435179 | 1.49883e+07 |    2968.29 |  0.57606  |
| 2025-11-10T05:17:29.386050 | XGBoost        | 2.83509e+07 |   3922.56 |  0.331347 | 1.23607e+07 |    2711.88 |  0.650381 |
| 2025-11-10T05:23:24.157328 | LightGBM       | 3.37797e+07 |   4724.31 |  0.203311 | 5.55546e+07 |    6325.19 | -0.571351 |
| 2025-11-10T03:02:07.850532 | LightGBM       | 4.19847e+07 |   5295.63 |  0.135582 | 7.09844e+07 |    7706.73 | -3.47616  |
| 2025-11-10T05:23:32.126343 | Ridge (Optuna) | 4.7535e+07  |   5427.77 | -0.121108 | 4.15068e+07 |    4886.43 | -0.174012 |
| 2025-11-10T05:16:28.832287 | RandomForest   | 4.88301e+07 |   3746.77 | -0.151651 | 1.18772e+07 |    2663.78 |  0.664054 |
| 2025-11-10T05:23:04.415359 | RandomForest   | 4.97499e+07 |   3821.33 | -0.173345 | 1.17549e+07 |    2629.26 |  0.667514 |
| 2025-11-10T03:02:08.679805 | Ridge (Optuna) | 1.09274e+08 |   8035.5  | -1.24982  | 3.72298e+07 |    4907.05 | -1.34765  |

## Espacios de búsqueda declarados

- **RandomForest**: `Fixed params (baseline)`
- **XGBoost**: `Fixed params; early_stopping_rounds=200`
- **XGBoost**: `Fixed params; early_stopping_rounds=200`
- **XGBoost**: `Fixed params; early_stopping_rounds=200`
- **LightGBM**: `Fixed params; early_stopping_rounds=200`
- **LightGBM**: `Fixed params; early_stopping_rounds=200`
- **Ridge (Optuna)**: `alpha ~ loguniform[1e-4, 1e3]; n_trials=50`
- **RandomForest**: `Fixed params (baseline)`
- **RandomForest**: `Fixed params (baseline)`
- **Ridge (Optuna)**: `alpha ~ loguniform[1e-4, 1e3]; n_trials=50`
- **Lasso (Optuna)**: `alpha ~ loguniform[1e-4, 1e2]; n_trials=50`
- **Lasso (Optuna)**: `alpha ~ loguniform[1e-4, 1e2]; n_trials=50`

## Justificación

- Se elige el mejor modelo por **RMSE de validación** (menor es mejor).
- Se reportan métricas en test del modelo **refiteado en todo el train**.
- La elección queda respaldada por `results/experiment_logs.csv` y este resumen.