# FPL Model

A predictive model for estimating **Fantasy Premier League (FPL)** player points using **XGBoost**, with hyperparameter tuning (`RandomizedSearchCV` + `TimeSeriesSplit`) and experiment tracking via **MLflow**. Includes a FastAPI service and a dashboard for viewing predictions and metrics.

## Overview

The project implements a pipeline that:
1. fetches FPL player data from an external API (per gameweek),
2. tunes `XGBRegressor` hyperparameters using time-order-respecting cross-validation (`TimeSeriesSplit`),
3. trains a production model on data up to a given gameweek,
4. generates point predictions for the upcoming gameweek and evaluates accuracy (MAE),
5. logs parameters, metrics and model artifacts to MLflow,
6. serves predictions through a FastAPI service, displayed on a dashboard.

## Architecture

```
┌─────────────────────┐
                 │   FPL Data API        │
                 │  (gw / gw_max query) │
                 └──────────┬──────────┘
                            │
                            ▼
                     ┌──────────────┐
                     │  FPLModel    │
                     │  - hyperopt  │
                     │  - train     │
                     │  - predict   │
                     └──────┬───────┘
                            │
                 ┌──────────┼──────────┐
                 ▼                     ▼
      ┌────────────────────────┐  ┌────────────────────────┐
      │    MLflow Tracking     │  │  Prediction Snapshots  │
      │    (params/metrics/    │  │ (predictions_history/) │
      │        models)         │  │                        │
      └────────────────────────┘  └────────────────────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │  FastAPI (model_api)  │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │   Dashboard (HTML)    │
                 └─────────────────────┘
```

## Modeling Techniques

### Hyperparameter tuning: `RandomizedSearchCV`

Instead of an exhaustive `GridSearchCV` (864 combinations across `PARAM_GRID`), `RandomizedSearchCV` samples `n_iter=25` combinations, trading a guaranteed global optimum for a much faster tuning cycle.

| Parameter | Values | Meaning |
|---|---|---|
| `n_estimators` | 300, 600, 1000 | number of boosting rounds |
| `max_depth` | 4, 6, 8, 12 | max tree depth |
| `learning_rate` | 0.001–0.1 (8 values) | learning step size |
| `subsample` | 0.4, 0.6, 0.8 | row sampling fraction per tree |
| `colsample_bytree` | 0.4, 0.6, 0.8 | column sampling fraction per tree |

### Cross-validation: `TimeSeriesSplit`

`TimeSeriesSplit` (`N_TS_SPLITS_CV = 4`) is used instead of a shuffled `KFold`, since FPL data has a natural temporal order. It always trains on past gameweeks and validates on future ones, avoiding data leakage.

> `TimeSeriesSplit` splits by row order in the DataFrame, not by the `GW` column — correctness depends on the data being pre-sorted ascending by `GW`.

### Training flow

1. **`hyperopt(gw_number)`** — tunes hyperparameters on data up to `gw_number - 1`.
2. **`train_production(current_gw)`** — trains the final model on all data up to `current_gw`, using the tuned hyperparameters.
3. **`predict(model, next_gw_number)`** — predicts the next, not-yet-played gameweek; MAE is computed once real results are available.

## `FPLModel` Class

### `get_data_in_gw_range(gw_max)` / `get_data_from_gw(gw_number)` *(static)*
Fetch player data from the API — a full range up to `gw_max`, or a single gameweek.

### `_get_feature_target(df)` *(static, private)*
Splits a DataFrame into features (`X`) and target (`y = total_points`), dropping `total_points`, `name`, `GW`, `team`.

### `hyperopt(gw_number)`
Runs `RandomizedSearchCV` (25 iterations, `TimeSeriesSplit`) on data up to `gw_number - 1`. Logs best params and CV MSE to MLflow, stores them in `self.best_params`.

### `train_production(current_gw)`
Trains the production model on all data up to `current_gw`, saves it locally, runs `predict` for the next gameweek, and logs the result to MLflow.

> **Known issue:** the model is built and trained using `self.current_model_params` *before* it gets overwritten with `self.best_params` — the freshly tuned parameters from `hyperopt` are applied one call too late.

### `predict(model, next_gw_number)`
Predicts points for `next_gw_number`, computes MAE against real results if available, and returns the top 30 players by `predicted_points`.

The prediction is also frozen to `predictions_history/gw_{next_gw_number}.csv`, so it can be reconstructed later exactly as it was generated — without recomputing it with a since-retrained model. Powers the dashboard's "Last week: predicted vs real" view.

### `get_historical_prediction(gw_number)`
Reads a frozen prediction from `predictions_history/gw_{gw_number}.csv`. Returns `None` if it doesn't exist.

## API (FastAPI)

| Endpoint | Method | Description |
|---|---|---|
| `/train?gw=X` | `POST` | Runs `hyperopt` + `train_production` in the background |
| `/train/status` | `GET` | Current training status |
| `/predict?gw=X` | `GET` | Predicts points for gameweek `X` with the currently loaded model |
| `/predictions/history?gw=X` | `GET` | Returns the frozen prediction for gameweek `X` |
| `/metrics` | `GET` | Best MAE per gameweek across MLflow runs |

Frontend (`index.html`) built with LLM assistance.
