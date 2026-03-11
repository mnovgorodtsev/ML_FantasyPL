import pandas as pd
import numpy as np
import xgboost as xgb
import mlflow
import mlflow.xgboost
from defusedxml.cElementTree import XMLParser
from mlflow.utils.databricks_utils import get_databricks_workspace_info_from_uri
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
from model.helpers import logger
import requests
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', None)

class FPLModel:
    def __init__(self):
        self.current_model_params = None
        self.start_data = None
        self.current_gw_data = None
        self.current_gw = None
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

    def log_to_mlflow(self, model, metrics: dict, params: dict, gw_number: int):
        with mlflow.start_run(run_name=f"GW_{gw_number}"):
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.xgboost.log_model(model, artifact_path="model")

    def train(self, gw_number, prev_model=None):
        params = {
        "n_estimators": 1000,
        "max_depth": 16,
        "learning_rate": 0.005,
        "subsample": 0.3,
        "colsample_bytree": 0.4,
        "random_state": 42,
        "gw_number": gw_number,
        "mode": "initial" if prev_model is None else "incremental"
        }

        logger.info(self.start_data)
        if prev_model is None:
            self.start_data = self.get_data_in_gw_range(gw_number)
            X = self.start_data.drop(columns=["total_points", "name", "GW"])
            y = self.start_data["total_points"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = xgb.XGBRegressor(**{k: v for k, v in params.items() if k not in ["gw_number", "mode"]})
            model.fit(X_train, y_train)
            model.save_model(f"model_gw_")
        else:
            pass

        y_pred = model.predict(X_test)
        metrics = {
            "mse": mean_squared_error(y_test, y_pred),
        }

        logger.info(f"GW {gw_number} metrics: {metrics}")
        model.save_model(f"model_gw_{gw_number}.json")
        self.log_to_mlflow(model, metrics, params, gw_number)
        self.current_model_params = params

    def train_production(self, current_gw):
        all_data = self.get_data_in_gw_range(current_gw)
        X_train = all_data.drop(columns=["total_points", "name", "GW"])
        y_train = all_data["total_points"]

        model = xgb.XGBRegressor(**self.current_model_params)
        model.fit(X_train, y_train)

        self.current_model = model
        model.save_model(f"model_production_gw_{current_gw}.json")
        self.log_to_mlflow(model, {}, self.current_model_params, current_gw)
        return self.predict(model, current_gw + 1)

    def predict(self, model, next_gw_number):
        data = self.get_data_from_gw(next_gw_number)
        X = data.drop(columns=["total_points", "name", "GW"])
        print(data.columns)
        predictions = model.predict(X)
        data["predicted_points"] = predictions
        top10 = (
            data[["name", "predicted_points", "total_points", "position"]].sort_values("predicted_points", ascending=False).head(30)
            .reset_index(drop=True)
        )
        top10.index += 1
        return top10

    def save_model(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save_model(path)
        pass



# FPL_pipeline = FPLModel()
# FPL_pipeline.train(13)
# top10 = FPL_pipeline.train_production(current_gw=13)
# print(top10)
