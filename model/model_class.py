import pandas as pd
import numpy as np
import xgboost as xgb
import mlflow
import mlflow.xgboost
from mlflow.utils.databricks_utils import get_databricks_workspace_info_from_uri
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
from helpers import logger
import requests

class FPLModel:
    def __init__(self):
        self.model = None
        self.start_data = None
        self.current_gw_data = None
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

    def log_to_mlflow(self):
        pass

    def log_to_xgboost(self):
        pass

    def train(self):
        self.start_data = self.get_data_in_gw_range(10)
        logger.info(self.start_data)

    def train_production(self):
        pass

    def predict(self, model):
        pass

    def save_model(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save_model(path)
        pass