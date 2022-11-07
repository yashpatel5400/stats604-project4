import numpy as np
import utils
import datetime 
import pandas as pd
from sklearn.linear_model import LinearRegression
from predictor.models.predictor_scaffold import Predictor
from functools import reduce

class BasicOLSPredictor(Predictor):
    def __init__(self):
        pass

    def generate_train(self, location, data):
        df = data[location]["noaa"]
        df = df.loc[:, ["TMIN", "TAVG", "TMAX"]].dropna(axis = 0)
        df['date_col'] = pd.to_datetime(df.index).date
        for i in range (1,6):
            df['ymin'+str(i)] = df['TMIN'].shift(-i)
            df['yavg'+str(i)] = df['TAVG'].shift(-i)
            df['ymax'+str(i)] = df['TMAX'].shift(-i)
        df = df.dropna(axis = 0)
        return df
    def generate_prediction_dat(self,location, data): 
        df = data[location]["wunderground"]
        df['date_col'] = pd.to_datetime(df.index).date
        temp_max = df.groupby(['date_col'], sort=False)['temp'].max()
        temp_min = df.groupby(['date_col'], sort=False)['temp'].min()
        temp_mean = df.groupby(['date_col'], sort=False)['temp'].mean()
        pred = np.asarray([[temp_min[-1], temp_mean[-1], temp_max[-1]]])
        return pred
        
    def predict(self, data):
        predictions = []
        for station in utils.stations:
            weather = self.generate_train(station, data).tail(365)
            X = weather.loc[:, ["TMIN", "TAVG", "TMAX"]]
            y = weather.copy()
            y = y.drop(columns = ["TMIN", "TAVG", "TMAX", 'date_col'])
            X = X*0.18 + 32 # convert to farenheit
            y = y*0.18 + 32 # convert to farenheit
            OLSfit = LinearRegression().fit(X,y)
            Xprediction = self.generate_prediction_dat(station, data)
            pred = OLSfit.predict(Xprediction.reshape(1,-1))
            predictions.append(np.ravel(pred).tolist())
        
        predictions = reduce(lambda x,y: x+y, predictions)
        return np.array(predictions)