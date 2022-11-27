import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import utils
import datetime 
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
from predictor.models.predictor_scaffold import Predictor
from functools import reduce

class BasicOLSPredictor(Predictor):
    def __init__(self):
        pass
    def create_regression_data(self, data, window_size):
        X, y = [], []
        target_data = data[["temp_min","temp_mean","temp_max"]].values
        prediction_window = 5
        for i in range(len(data) - (window_size + prediction_window + 1)):
            X.append(data.values[i:i+window_size,:-1].flatten())
            y.append(target_data[i+window_size+1:i+window_size+1+prediction_window].flatten())
        test_X = data.values[-window_size-1:-1,:-1].flatten().reshape(1, -1) # the final frame used for future prediction
        return np.array(X), np.array(y), test_X
        
    def predict(self, data):
        predictions = []
        for station in utils.stations:
            weather =  self.create_regression_data(data[station]["wunderground"], 3)
            X = weather[0]
            y = weather[1]
            OLSfit = LinearRegression().fit(X,y)
            Xprediction = weather[2]
            pred = OLSfit.predict(Xprediction)
            predictions.append(np.ravel(pred).tolist())
        predictions = reduce(lambda x,y: x+y, predictions)
        return np.array(predictions)

class LassoPredictor(Predictor):
    def __init__(self):
        pass
        
    def create_regression_data(self, data, window_size):
        X, y = [], []
        target_data = data[["temp_min","temp_mean","temp_max"]].values
        prediction_window = 5
        for i in range(len(data) - (window_size + prediction_window + 1)):
            X.append(data.values[i:i+window_size,:-1].flatten())
            y.append(target_data[i+window_size+1:i+window_size+1+prediction_window].flatten())
        test_X = data.values[-window_size-1:-1,:-1].flatten().reshape(1, -1) # the final frame used for future prediction
        return np.array(X), np.array(y), test_X
        
    def predict(self, data):
        predictions = []
        for station in utils.stations:
            weather =  self.create_regression_data(data[station]["wunderground"], 3)
            X = weather[0]
            y = weather[1]
            Lassofit = Lasso(alpha = 1.0).fit(X,y)
            Xprediction = weather[2]
            pred = Lassofit.predict(Xprediction)
            predictions.append(np.ravel(pred).tolist())
        predictions = reduce(lambda x,y: x+y, predictions)
        return np.array(predictions)

class GBTPredictor(Predictor):
    def __init__(self):
        pass

    def create_regression_data(self, data, window_size):
        X, y = [], []
        target_data = data[["temp_min","temp_mean","temp_max"]].values
        prediction_window = 5
        for i in range(len(data) - (window_size + prediction_window + 1)):
            X.append(data.values[i:i+window_size,:-1].flatten())
            y.append(target_data[i+window_size+1:i+window_size+1+prediction_window].flatten())
        test_X = data.values[-window_size-1:-1,:-1].flatten().reshape(1, -1) # the final frame used for future prediction
        return np.array(X), np.array(y), test_X
        
    def predict(self, data):
        predictions = []
        for station in utils.stations:
            weather =  self.create_regression_data(data[station]["wunderground"], 3)
            X = weather[0]
            y = weather[1]
            Xprediction = weather[2]
            GBTfit = MultiOutputRegressor(GradientBoostingRegressor(n_estimators = 20, max_depth = 5)).fit(X, y)
            pred = GBTfit.predict(Xprediction)
            predictions.append(np.ravel(pred).tolist())
        predictions = reduce(lambda x,y: x+y, predictions)
        return np.array(predictions)
