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

import utils

def format_output(date_to_prediction):
    """Takes a dictionary that maps dates to dataframes with the weather stations as 
    rows and min, avg, max as columns and spits back the formatted string desired as output
    """
    start_date = list(date_to_prediction.keys())[0]
    return f"{start_date}, {', '.join([', '.join(date_to_prediction[date].values.flatten().astype(str)) for date in date_to_prediction])}"
    
if __name__ == '__main__':
    current_date = date.today()
    forward_dates = [current_date + datetime.timedelta(days=1 + delta) for delta in range(5)]

    stations_data = []
    for station in utils.stations:
        df = pd.read_csv('data/wunderground/' + station + '.csv')
        df['date'] = pd.to_datetime(df['date']).dt.date
        temp_max = df.groupby(['date'], sort=False)['temp'].max()
        print(temp_max)
        temp_min = df.groupby(['date'], sort=False)['temp'].min()
        temp_mean = df.groupby(['date'], sort=False)['temp'].mean()
        stations_data.append([temp_min[-1], round(temp_mean[-1], 1), temp_max[-1]])

    stations_data = np.array(stations_data) 

    predictions = {date: pd.DataFrame(stations_data, columns = ["Min", "Avg", "Max"], index = utils.stations) for date in forward_dates}
    output = format_output(predictions)
    print(output)