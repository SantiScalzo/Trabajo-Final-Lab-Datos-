# Experiment Summary — 2025-11-10T03:02:10.939831

## Top 10 por RMSE de Validación

| timestamp                  | model_name     |    val_rmse |   val_mae |    val_r2 |   test_rmse |   test_mae |   test_r2 |
|:---------------------------|:---------------|------------:|----------:|----------:|------------:|-----------:|----------:|
| 2025-11-10T03:01:40.476653 | RandomForest   | 1.79712e+07 |   3199.4  |  0.629994 | 1.03115e+07 |    2544.82 |  0.349774 |
| 2025-11-10T03:02:07.690548 | XGBoost        | 2.04578e+07 |   3406.08 |  0.578797 | 1.22134e+07 |    2800.85 |  0.229841 |
| 2025-11-10T03:02:07.850532 | LightGBM       | 4.19847e+07 |   5295.63 |  0.135582 | 7.09844e+07 |    7706.73 | -3.47616  |
| 2025-11-10T03:02:08.679805 | Ridge (Optuna) | 1.09274e+08 |   8035.5  | -1.24982  | 3.72298e+07 |    4907.05 | -1.34765  |
| 2025-11-10T03:02:09.777877 | Lasso (Optuna) | 5.22655e+08 |  17277.8  | -9.76088  | 1.6423e+08  |    9491.16 | -9.35609  |

## Espacios de búsqueda declarados

- **RandomForest**: `Fixed params (baseline)`
- **XGBoost**: `Fixed params; early_stopping_rounds=200`
- **LightGBM**: `Fixed params; early_stopping_rounds=200`
- **Ridge (Optuna)**: `alpha ~ loguniform[1e-4, 1e3]; n_trials=50`
- **Lasso (Optuna)**: `alpha ~ loguniform[1e-4, 1e2]; n_trials=50`

## Justificación

- Se elige el mejor modelo por **RMSE de validación** (menor es mejor).
- Se reportan métricas en test del modelo **refiteado en todo el train**.
- La elección queda respaldada por `results/experiment_logs.csv` y este resumen.