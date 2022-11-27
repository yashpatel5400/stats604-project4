import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import pandas as pd
import logging
import datetime
from datetime import date
import numpy as np
import sys
from matplotlib import pyplot as plt
import predictor.utils as utils
import time
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from predictor.models.predictor_scaffold import Predictor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor



class HistoricAveragePredictor(Predictor):
    def __init__(self):
        pass

    def predict(self, data):

        stations_data = []
        for station in utils.stations:
            noaa = data[station]["noaa"]
            noaa = noaa.loc[:, [ "TMIN", "TAVG", "TMAX"]].dropna(axis =0)
            noaa = noaa*0.18+32.0
            current_date = noaa.index[-1]
       
            df = noaa.groupby(by=[noaa.index.month, noaa.index.day]).mean().round(2)
            
            for i in range(1, 6):
                date_ = (current_date + pd.DateOffset(days=i))
                stations_data.append(df[df.index == (date_.month, date_.day)].values.flatten())
           
       
             
           
        stations_data = np.array(stations_data).flatten()
        return stations_data



class ArimaPredictor(Predictor):
    def __init__(self):
        pass

    def predict(self, data):

        stations_data = []
        for station in utils.stations:

            df = data[station]["wunderground"]
            temp_max = df['temp_max'].asfreq('D')
            temp_min = df['temp_min'].asfreq('D')
            temp_avg = df['temp_mean'].asfreq('D')
            
            

           
        

            min_model = SARIMAX(temp_min, order=(3,1,1), seasonal_order=(1, 0, 0, 12)).fit(disp=False)
           
    
            avg_model = SARIMAX(temp_avg, order=(3,1,1), seasonal_order=(1, 0, 0, 12)).fit(disp=False)
            max_model = SARIMAX(temp_max, order=(3,1,1), seasonal_order=(1, 0 , 0, 12)).fit(disp=False)
            
            min_fc = min_model.forecast(steps = 5)
            avg_fc = avg_model.forecast(steps = 5)
            max_fc  = max_model.forecast(steps = 5)
            
        
            
            stations_data.append(np.vstack((min_fc.values, avg_fc.values, max_fc.values)).flatten())
            
            
            
             
           
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

class NLinPredict(Predictor):
    def __init__(self):
        pass

    def predict(self, data):
        stations_data = []
        start = time.time()
        for station in utils.stations:
            window_size = 3
            X, y, test_X = create_regression_data(data[station]["wunderground"], window_size) 
            reg = MLPRegressor(random_state=1, hidden_layer_sizes=(20,20, ), max_iter=2000, solver='lbfgs', learning_rate= 'adaptive').fit(X, y)
            stations_data.append(reg.predict(test_X))
        end = time.time()
        logging.debug(f"Performed prediction in: {end - start} s")
        return np.array(stations_data).flatten()   