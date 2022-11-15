import pandas as pd
import datetime
from datetime import date
import numpy as np
import sys
from matplotlib import pyplot as plt
import utils
from statsmodels.tsa.statespace.sarimax import SARIMAX
from predictor.models.predictor_scaffold import Predictor




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
       
            df = noaa.groupby(by=[noaa.index.month, noaa.index.day]).median().round(2)
            
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
            noaa = data[station]["noaa"]
            noaa = noaa.loc[:, [ "TMIN", "TAVG", "TMAX"]]
            noaa = noaa.iloc[-500:, :]
            noaa = noaa.asfreq('D')
            #noaa = noaa.dropna(axis =0)
            noaa = noaa*0.18+32.0
            
            

           
        
     
            min_model = SARIMAX(noaa.TMIN, order=(0,1,3)).fit(disp=False)
            min_model.predict().plot(label = "prediction")
            #noaa.TMIN.plot(label = "true")
            #plt.legend()
            #plt.show()
            avg_model = SARIMAX(noaa.TAVG, order=(0,1,3)).fit(disp=False)
            max_model = SARIMAX(noaa.TMAX, order=(0,1,3)).fit(disp=False)
            
            min_fc = min_model.forecast(steps = 5)
            avg_fc = avg_model.forecast(steps = 5)
            max_fc  = max_model.forecast(steps = 5)

        
            
            stations_data.append(np.vstack((min_fc.values, avg_fc.values, max_fc.values)).flatten())
            
            
            
             
           
        stations_data = np.array(stations_data).flatten()
        return stations_data


