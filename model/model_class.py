import pandas as pd
import numpy as np
import xgboost as xgb
import mlflow
import mlflow.xgboost
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

class FPLModel:
    def __init__(self):
        pass

    def get_data(self):
        pass

    def log_to_mlflow(self):
        pass

    def log_to_xgboost(self):
        pass

    def train(self):
        pass

    def train_production(self):
        pass

    def predict(self, model):
        pass