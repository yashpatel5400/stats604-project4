import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # ignore FutureWarnings from pd

import datetime
import pickle
import pytz
import pandas as pd
import numpy as np

from raw_data import wunderground_download
import predictor.utils as utils

def process_wunderground(station):
    """Wunderground returns granular (hourly) data points, but we only want daily 
    for prediction: this coarsens the dataset
    """
    wunderground_path = os.path.join(utils.raw_wunderground_cache, f"{station}.csv")
    raw_wunderground_data = pd.read_csv(wunderground_path, index_col=0)
    raw_wunderground_data.index = pd.to_datetime(raw_wunderground_data.index)
    
    local_timezone = pytz.timezone(utils.fetch_timezone(station))
    raw_wunderground_data['date_col'] = pd.to_datetime(raw_wunderground_data.index).tz_convert(local_timezone).date
        
    aggregated_columns = ["temp", "wspd", "pressure", "heat_index", 'dewPt']
    maxes = raw_wunderground_data.groupby(['date_col'], sort=False)[aggregated_columns].max().set_axis([f"{column}_max" for column in aggregated_columns], axis=1, inplace=False).set_index(raw_wunderground_data['date_col'].unique())
    means = raw_wunderground_data.groupby(['date_col'], sort=False)[aggregated_columns].mean().set_axis([f"{column}_mean" for column in aggregated_columns], axis=1, inplace=False).set_index(raw_wunderground_data['date_col'].unique())
    mins  = raw_wunderground_data.groupby(['date_col'], sort=False)[aggregated_columns].min().set_axis([f"{column}_min" for column in aggregated_columns], axis=1, inplace=False).set_index(raw_wunderground_data['date_col'].unique())
    wind_dir = raw_wunderground_data.groupby(['date_col'], sort=False)['wdir_cardinal'].agg(
        lambda x: pd.Series.mode(x)[0]).astype("category").to_frame("wdir_mode").set_index(raw_wunderground_data['date_col'].unique())
    processed_wunderground = pd.concat((mins, means, maxes, wind_dir), axis=1)

    os.makedirs(utils.processed_wunderground_cache, exist_ok=True)
    wunderground_out_path = os.path.join(utils.processed_wunderground_cache, f"{station}.csv")
    processed_wunderground.to_csv(wunderground_out_path)

if __name__ == "__main__":
    for station in utils.stations:
        process_wunderground(station)
        print(f"Processed data for: {station}")