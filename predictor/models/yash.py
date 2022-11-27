import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import time
import pandas as pd
import logging
from datetime import date, timedelta

from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, HuberRegressor
pd.options.mode.chained_assignment = None

import predictor.utils as utils
from predictor.models.predictor_scaffold import Predictor
from predictor.models.unique import HistoricAveragePredictor

class PrevDayPredictor(Predictor):
    def __init__(self):
        pass

    def predict(self, data):
        stations_data = [list(data[station]["wunderground"].iloc[-2][["temp_min","temp_mean","temp_max"]].values) * 5 for station in utils.stations]
        stations_data = np.array(stations_data).flatten()
        return stations_data

def create_regression_data(data, window_size):
    X, y = [], []
    target_data = data[["temp_min","temp_mean","temp_max"]].values
    prediction_window = 5
    for i in range(len(data) - (window_size + prediction_window + 1)):
        X.append(data.values[i:i+window_size,:-1].flatten())
        y.append(target_data[i+window_size+1:i+window_size+1+prediction_window].flatten())
    test_X = data.values[-window_size-1:-1,:-1].flatten().reshape(1, -1) # the final frame used for future prediction
    return np.array(X), np.array(y), test_X

class LRPredictor(Predictor):
    def __init__(self):
        pass

    def predict(self, data):
        stations_data = []
        start = time.time()
        for station in utils.stations:
            window_size = 3
            X, y, test_X = create_regression_data(data[station]["wunderground"], window_size)
            # for i in range(y.shape[1]):
            #     lr = HuberRegressor().fit(X, y[:,i])
            #     stations_data.append(lr.predict(test_X))
            reg = MultiOutputRegressor(GradientBoostingRegressor(n_estimators=20,)).fit(X, y)
            stations_data.append(reg.predict(test_X))
        end = time.time()
        logging.debug(f"Performed prediction in: {end - start} s")
        return np.array(stations_data).flatten()