import numpy as np
import utils
import datetime 
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from predictor.models.predictor_scaffold import Predictor
from functools import reduce

class BasicOLSPredictor(Predictor):
    def __init__(self):
        pass

    def generate_dat(self, location, data):
        df = data[location]["wunderground"]
        df['date_col'] = pd.to_datetime(df.index).date
        temp_max = df.groupby(['date_col'], sort=False)['temp'].max()
        temp_min = df.groupby(['date_col'], sort=False)['temp'].min()
        temp_mean = df.groupby(['date_col'], sort=False)['temp'].mean()
        pressure = df.groupby(['date_col'], sort=False)['pressure'].mean()
        wndspd = df.groupby(['date_col'], sort=False)['wspd'].mean()
        wnddr = df.groupby(['date_col'], sort=False)['wdir_cardinal'].agg(
            lambda x: pd.Series.mode(x)[0]).astype("category")
        prcp = df.groupby(['date_col'], sort=False)['precip_hrly'].sum()
        dict = {'TMIN': temp_min, 'TAVG': temp_mean, 'TMAX': temp_max, 
        "pressure": pressure, "wndspd": wndspd, "wnddr" :wnddr, 'prcp': prcp }
        X = pd.DataFrame.from_dict(dict)
        for i in range (1,6):
            X['ymin'+str(i)] = X['TMIN'].shift(-i)
            X['yavg'+str(i)] = X['TAVG'].shift(-i)
            X['ymax'+str(i)] = X['TMAX'].shift(-i)
        for i in range(1,3):
            X['TMIN'+str(i)] = X['TMIN'].shift(i)
            X['TAVG'+str(i)] = X['TAVG'].shift(i)
            X['TMAX'+str(i)] = X['TMAX'].shift(i)
        y = X.copy()
        pred = X.copy().tail(1)
        X = X.dropna(axis = 0)
        y = y.dropna(axis = 0)
        X = pd.get_dummies(X, columns = ['wnddr'])
        pred = pd.get_dummies(pred, columns = ['wnddr'])
        y = y.drop(columns = ["TMIN", "TAVG", "TMAX", "pressure", "wndspd", "wnddr", "prcp",
        "TMIN1", "TAVG1", "TMAX1","TMIN2", "TAVG2", "TMAX2",])
        X = X.drop(columns = ['ymin1', 'yavg1', 'ymax1','ymin2', 'yavg2', 'ymax2',
        'ymin3', 'yavg3', 'ymax3', 'ymin4', 'yavg4', 'ymax4', 'ymin5', 'yavg5', 'ymax5' ])
        pred = pred.drop(columns = ['ymin1', 'yavg1', 'ymax1','ymin2', 'yavg2', 'ymax2',
        'ymin3', 'yavg3', 'ymax3', 'ymin4', 'yavg4', 'ymax4', 'ymin5', 'yavg5', 'ymax5' ])
        
        #X = X*0.18 + 32 # convert to farenheit
        #y = y*0.18 + 32 # convert to farenheit
        return X, y, pred
        
    def predict(self, data):
        predictions = []
        for station in utils.stations:
            weather = self.generate_dat(station, data)
            X = weather[0].tail(365)
            y = weather[1].tail(365)
            OLSfit = LinearRegression().fit(X,y)
            Xprediction = weather[2]
            pred = OLSfit.predict(Xprediction)
            predictions.append(np.ravel(pred).tolist())
        predictions = reduce(lambda x,y: x+y, predictions)
        return np.array(predictions)

class RidgePredictor(Predictor):
    def __init__(self):
        pass

    def generate_dat(self, location, data):
        print(location)
        df = data[location]["wunderground"]
        df['date_col'] = pd.to_datetime(df.index).date
        temp_max = df.groupby(['date_col'], sort=False)['temp'].max()
        temp_min = df.groupby(['date_col'], sort=False)['temp'].min()
        temp_mean = df.groupby(['date_col'], sort=False)['temp'].mean()
        pressure = df.groupby(['date_col'], sort=False)['pressure'].mean()
        wndspd = df.groupby(['date_col'], sort=False)['wspd'].mean()
        wnddr = df.groupby(['date_col'], sort=False)['wdir_cardinal'].agg(
            lambda x: pd.Series.mode(x)[0]).astype("category")
        prcp = df.groupby(['date_col'], sort=False)['precip_hrly'].sum()
        dict = {'TMIN': temp_min, 'TAVG': temp_mean, 'TMAX': temp_max, 
        "pressure": pressure, "wndspd": wndspd, "wnddr" :wnddr, 'prcp': prcp }
        X = pd.DataFrame.from_dict(dict)
        for i in range (1,6):
            X['ymin'+str(i)] = X['TMIN'].shift(-i)
            X['yavg'+str(i)] = X['TAVG'].shift(-i)
            X['ymax'+str(i)] = X['TMAX'].shift(-i)
        for i in range(1,3):
            X['TMIN'+str(i)] = X['TMIN'].shift(i)
            X['TAVG'+str(i)] = X['TAVG'].shift(i)
            X['TMAX'+str(i)] = X['TMAX'].shift(i)
        y = X.copy()
        pred = X.copy().tail(1)
        X = X.dropna(axis = 0)
        y = y.dropna(axis = 0)
        X = pd.get_dummies(X, columns = ['wnddr'])
        pred = pd.get_dummies(pred, columns = ['wnddr'])
        y = y.drop(columns = ["TMIN", "TAVG", "TMAX", "pressure", "wndspd", "wnddr", "prcp",
        "TMIN1", "TAVG1", "TMAX1","TMIN2", "TAVG2", "TMAX2",])
        X = X.drop(columns = ['ymin1', 'yavg1', 'ymax1','ymin2', 'yavg2', 'ymax2',
        'ymin3', 'yavg3', 'ymax3', 'ymin4', 'yavg4', 'ymax4', 'ymin5', 'yavg5', 'ymax5' ])
        pred = pred.drop(columns = ['ymin1', 'yavg1', 'ymax1','ymin2', 'yavg2', 'ymax2',
        'ymin3', 'yavg3', 'ymax3', 'ymin4', 'yavg4', 'ymax4', 'ymin5', 'yavg5', 'ymax5' ])
        
        #X = X*0.18 + 32 # convert to farenheit
        #y = y*0.18 + 32 # convert to farenheit
        return X, y, pred
        
    def predict(self, data):
        predictions = []
        for station in utils.stations:
            weather = self.generate_dat(station, data)
            X = weather[0].tail(365)
            y = weather[1].tail(365)
            Huber = Ridge(alpha = 0.1)
            fit = Huber.fit(X,y)
            Xprediction = weather[2]
            pred = fit.predict(Xprediction)
            predictions.append(np.ravel(pred).tolist())
        predictions = reduce(lambda x,y: x+y, predictions)
        return np.array(predictions)