import numpy as np
import utils
import datetime
import pandas as pd
from datetime import date, timedelta
from sklearn.linear_model import LinearRegression
pd.options.mode.chained_assignment = None

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
        for station in utils.stations:
            window_size = 3
            X, y, test_X = create_regression_data(data[station]["wunderground"], window_size)
            lr = LinearRegression().fit(X, y)
            stations_data.append(lr.predict(test_X))
        return np.array(stations_data).flatten()