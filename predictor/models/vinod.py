import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import logging
import utils
import datetime
import pandas as pd
import time
from datetime import date, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import Lasso
pd.options.mode.chained_assignment = None

from predictor.models.predictor_scaffold import Predictor
from predictor.models.unique import HistoricAveragePredictor

def create_regression_data(data, window_size, features):
    if features is not None:
        keep_data = data[features]
    else:
        keep_data = data
        
    X, y = [], []
    target_data = data[["temp_min","temp_mean","temp_max"]].values
    prediction_window = 5
    for i in range(len(data) - (window_size + prediction_window + 1)):
        X.append(keep_data.values[i:i+window_size,:-1].flatten())
        y.append(target_data[i+window_size+1:i+window_size+1+prediction_window].flatten())
    test_X = keep_data.values[-window_size-1:-1,:-1].flatten().reshape(1, -1) # the final frame used for future prediction
    return np.array(X), np.array(y), test_X


class PrevDayPredictor(Predictor):
    def __init__(self):
        pass
    def predict(self, data):
        stations_data = [list(data[station]["wunderground"].iloc[-1][["temp_min","temp_mean","temp_max"]].values) * 5 for station in utils.stations]
        stations_data = np.array(stations_data).flatten()
        return stations_data

class PrevDayHistoricalPredictor(Predictor):
    def __init__(self, weights):
        self.prev_day_model = PrevDayPredictor()
        self.historical_day_model = HistoricAveragePredictor()
        self.weights = weights

    def predict(self, data):
        weights = self.weights

        prev_day_pred = self.prev_day_model.predict(data)
        historical_day_pred = self.historical_day_model.predict(data)

        counter = 0

        for index in range(len(prev_day_pred)//3):
            prev_day_val = prev_day_pred[index*3:index*3 + 3]
            historical_day_val= historical_day_pred[index*3:index*3 + 3]

            prev_day_val_weighted = prev_day_val * weights[counter]
            historical_day_val_weighted = historical_day_val * (1-weights[counter])

            prev_day_pred[index*3:index*3 + 3] = prev_day_val_weighted
            historical_day_pred[index*3:index*3 + 3] = historical_day_val_weighted

            counter += 1
            counter %= 5

        predictions = (prev_day_pred + historical_day_pred).round(1)
        return predictions


class MixPrevDayHistoricalPredictor(Predictor):
    def __init__(self):
        self.prev_day_model = PrevDayPredictor()
        self.historical_day_model = HistoricAveragePredictor()

    def create_dataset(self, data, station, day, measure):
        wunderground = data[station]["wunderground"]
        wunderground['date_col'] = pd.to_datetime(wunderground.index).date

        noaa = data[station]["noaa"]
        noaa = noaa.loc[:, [ "TMIN", "TAVG", "TMAX"]].dropna(axis =0)
        noaa = noaa*0.18+32.0

        current_date = wunderground['date_col'].iloc[-1]
        logging.debug(current_date)

        date_range = pd.date_range(current_date - pd.DateOffset(days=365) , current_date - timedelta(days = 1),  freq='d')
        month_day_index= [(date.month, date.day) for date in date_range]

        df = noaa.groupby(by=[noaa.index.month, noaa.index.day]).mean().round(2)
        historical_tmp = df.loc[month_day_index[day:]][measure]

        if measure == "TMIN":
            current_temp = wunderground["temp_min"]
        elif measure == "TMAX":
            current_temp = wunderground["temp_max"]
        else:
            current_temp = wunderground["temp_mean"]

        current_temp_trunc = current_temp.loc[date_range[:-(day)]] #stop 1 days before end

        X =pd.DataFrame(columns=['current', 'historical'])
        X['current'] = current_temp_trunc
        X['historical'] = historical_tmp.values
        y = current_temp.loc[date_range[day:]]

        return X, y

    def predict(self, data):
        measurements = ["TMIN", "TAVG", "TMAX"]

        prev_day_pred = self.prev_day_model.predict(data) #vector of length 300
        historical_day_pred = self.historical_day_model.predict(data) #vector of length 300

        station_reg = []
        for station in utils.stations: #20 stations
            for day in range(1, 6): #5 days
                for meas in measurements: # 3 measurements
                    X, y = self.create_dataset(data, station, day, meas)
                    reg = LinearRegression(fit_intercept = True).fit(X, y)
                    station_reg.append(reg)

        predictions = []
        for index in range(len(prev_day_pred)):
            reg_model = station_reg[index]

            X_test =pd.DataFrame(columns=['current', 'historical'])
            X_test['current'] = [prev_day_pred[index]]
            X_test['historical'] = [historical_day_pred[index]]
            pred = reg_model.predict(X_test)
            predictions.append(pred[0])
            
        return predictions

class MetaPredictor(Predictor):
    def __init__(self, reg, window_size, features= None):
        self.reg = reg
        self.window_size = window_size
        self.features = features

    def predict(self, data):
        stations_data = []
        start = time.time()
        for station in utils.stations:
            window_size = self.window_size
            X, y, test_X = create_regression_data(data[station]["wunderground"], window_size, self.features) 
            self.reg.fit(X, y)
            stations_data.append(self.reg.predict(test_X))
        end = time.time()
        logging.debug(f"Performed prediction in: {end - start} s")
        return np.array(stations_data).flatten()