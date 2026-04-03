import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.xgboost
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
from model.helpers import logger
import requests

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', None)

PARAM_GRID = {
    "n_estimators":     [300, 600, 1000],
    "max_depth":        [4, 6, 8, 12],
    "learning_rate":    [0.001, 0.002, 0.005, 0.008, 0.01, 0.025, 0.05, 0.1],
    "subsample":        [0.4, 0.6, 0.8],
    "colsample_bytree": [0.4, 0.6, 0.8],
}
EXPERIMENT_HYPEROPT   = "FantasyPL_Hyperopt"
EXPERIMENT_PRODUCTION = "FantasyPL_Production"
N_TS_SPLITS_CV = 4

class FPLModel:
    def __init__(self):
        self.current_model_params = None
        self.start_data = None
        self.current_gw_data = None
        self.current_gw = None
        self.current_model = None
        self.mlflow_tracking_uri = "http://127.0.0.1:5000"
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        mlflow.set_experiment("FantasyPL")

    @staticmethod
    def get_data_in_gw_range(gw_max):
        response = requests.get("http://127.0.0.1:8000/data", params={"gw_max": gw_max})
        response.raise_for_status()
        data = pd.DataFrame(response.json())
        return data

    @staticmethod
    def get_data_from_gw(gw_number):
        response = requests.get("http://127.0.0.1:8000/data", params={"gw": gw_number})
        response.raise_for_status()
        return pd.DataFrame(response.json())

    @staticmethod
    def _get_feature_target(df):
        drop_cols = ["total_points", "name", "GW", "team"]
        X = df.drop(columns=[c for c in drop_cols if c in df.columns])
        y = df["total_points"]
        return X, y

    def hyperopt(self, gw_number: int) -> dict:
        mlflow.set_experiment(EXPERIMENT_HYPEROPT)

        data = self.get_data_in_gw_range(gw_number - 1)
        X, y = self._get_feature_target(data)

        tscv = TimeSeriesSplit(n_splits=N_TS_SPLITS_CV)

        model = xgb.XGBRegressor(random_state=42)

        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=PARAM_GRID,
            n_iter=55,
            scoring="neg_mean_squared_error",
            cv=tscv,
            n_jobs=-1,
            random_state=42
        )

        with mlflow.start_run(run_name=f"GW_{gw_number}_randomsearch"):
            random_search.fit(X, y)
            logger.info(f"Random Search running...")
            best_params = random_search.best_params_
            best_score = -random_search.best_score_

            mlflow.log_params(best_params)
            mlflow.log_metric("best_cv_mse", best_score)

        logger.info(f"[GW {gw_number}] Best params: {best_params} | MSE: {best_score:.4f}")

        self.best_params = best_params
        return best_params

    def log_to_mlflow(self, model, metrics: dict, params: dict, gw_number: int):
        with mlflow.start_run(run_name=f"GW_{gw_number}"):
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.xgboost.log_model(model, artifact_path="model")

    def train_production(self, current_gw):
        all_data = self.get_data_in_gw_range(current_gw)
        X_train = all_data.drop(columns=["total_points", "name", "GW", "team"])
        y_train = all_data["total_points"]

        model = xgb.XGBRegressor(**self.current_model_params)
        model.fit(X_train, y_train)

        self.current_model_params = self.best_params
        self.current_model = model
        model.save_model(f"model_production_gw_{current_gw}.json")
        self.log_to_mlflow(model, {}, self.current_model_params, current_gw)
        return self.predict(model, current_gw + 1)

    def predict(self, model, next_gw_number: int) -> pd.DataFrame:
        data = self.get_data_from_gw(next_gw_number)
        X, _ = self._get_feature_target(data)
        data["predicted_points"] = model.predict(X)
        top = (
            data[["name", "team", "predicted_points", "total_points", "position"]]
            .sort_values("predicted_points", ascending=False)
            .head(30)
            .reset_index(drop=True)
        )
        top.index += 1
        return top

