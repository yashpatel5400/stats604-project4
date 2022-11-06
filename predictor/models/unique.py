import pandas as pd
import datetime
from datetime import date
import numpy as np
import sys
import utils

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
       
            df = noaa.groupby(by=[noaa.index.month, noaa.index.day]).mean().round(2)
            
            for i in range(1, 6):
                date_ = (current_date + pd.DateOffset(days=i))
                stations_data.append(df[df.index == (date_.month, date_.day)].values.flatten())
           
       
             
           
        stations_data = np.array(stations_data).flatten()
        return stations_data


