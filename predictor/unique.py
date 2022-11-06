#!/usr/bin/env python
"""Main entrypoint for weather predictions"""

__author__ = "Vinod Raman, Unique Subedi, Seamus Somerstep, and Yash Patel"
__license__ = "MIT"
__version__ = "1.0"
__status__ = "Dev"

import pandas as pd
import datetime
from datetime import date
import numpy as np
import sys

sys.path.append('/predictor')
sys.path.append('/data')

import utils

def format_output(date_to_prediction):
    """Takes a dictionary that maps dates to dataframes with the weather stations as 
    rows and min, avg, max as columns and spits back the formatted string desired as output
    """
    start_date = list(date_to_prediction.keys())[0]
    
    return f"{start_date}, {', '.join([', '.join(date_to_prediction[date].values.flatten().astype(str)) for date in date_to_prediction])}"


def noaa_preprocessing(station):
    noaa = pd.read_csv('data/noaa/' + station + '.csv')
    noaa = noaa.rename(columns={"Unnamed: 0" : "date"})
    noaa["date"] = pd.to_datetime(noaa['date']).dt.date
   
   
    noaa = noaa.loc[:, ["date", "TMIN", "TAVG", "TMAX"]]
    noaa = noaa.dropna(axis =0)
    noaa.iloc[:, 1:4] = noaa.iloc[:, 1:4]*0.18 + 32
    
    return noaa


def wunderground_preprocessing(station):
    wunderground = pd.read_csv('data/wunderground/' + station + '.csv')
    wunderground["date"] = pd.to_datetime(wunderground['date']).dt.date
    wunderground = pd.concat([wunderground.groupby(['date'], sort=False)['temp'].max(), wunderground.groupby(['date'], sort=False)['temp'].min(), wunderground.groupby(['date'], sort=False)['temp'].mean()], axis =1)
    wunderground.columns = ["TMIN", "TAVG", "TMAX"]
    wunderground.reset_index(inplace = True)
    
    return wunderground
    
    
        
if __name__ == '__main__':
    current_date = date.today()
    forward_dates = [current_date + datetime.timedelta(days=1 + delta) for delta in range(5)]

    stations_data = {}
    for station in utils.stations:
        
        noaa = noaa_preprocessing(station)
        #wunderground = wunderground_preprocessing(station)
        #df = pd.concat([noaa, wunderground[wunderground.date > noaa.date.iloc[-1]]])
        df = noaa
        df.index = pd.to_datetime(df['date'])
        
        df = df.groupby(by=[df.index.month, df.index.day]).mean()
       
        stations_data[station] = df.round(2)
        


    predictions = {}
    for date_ in forward_dates:
        temp = []
        for (station, data) in stations_data.items():
            temp.append(list(data[data.index == (date_.month, date_.day)].values.flatten()))
     
        predictions[date_] = pd.DataFrame(np.array(temp), columns = ["Min", "Avg", "Max"], index = utils.stations)

    output = format_output(predictions)
    print(output)