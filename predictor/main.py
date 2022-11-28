#!/usr/bin/env python
"""Main entrypoint for weather predictions"""

__author__ = "Vinod Raman, Unique Subedi, Seamus Somerstep, and Yash Patel"
__license__ = "MIT"
__version__ = "1.0"
__status__ = "Dev"

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import datetime
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor

import utils
from predictor.models.vinod import MetaPredictor
from raw_data.wunderground_download import fetch_wunderground_pd
from data.process_wunderground import process_wunderground_df

import logging
# if you want more verbose debugging information, uncomment the following line
# logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    keep_features = ['temp_min', 'wspd_min', 'pressure_min', 'heat_index_min', 'dewPt_min',
       'temp_mean', 'wspd_mean', 'pressure_mean', 'heat_index_mean',
       'dewPt_mean', 'temp_max', 'wspd_max', 'pressure_max', 'heat_index_max',
       'dewPt_max', 'wdir_mode']
    # reg = MultiOutputRegressor(GradientBoostingRegressor(n_estimators=20,))
    reg = RandomForestRegressor(max_depth=5)
    window_size = 3
    model = MetaPredictor(reg, window_size, keep_features)

    data = utils.load_processed_data() # gets all the historical data from the 1st prediction day
    
    # gets all the more "recent" data that wasn't downloaded initially and updates the "wunderground" entry
    for station in data:
        raw_recent_data = fetch_wunderground_pd(station, predict_date=datetime.date.today(), future_days=0, past_days=30, ignore_cache=True)
        processed_recent_data = process_wunderground_df(raw_recent_data, station)
        data[station]["wunderground"] = data[station]["wunderground"].combine_first(processed_recent_data)
    predictions = model.predict(data)
    
    prediction_date = f"{datetime.date.today():%Y-%m-%d}"
    predictions_rounded = np.around(predictions, 1)
    
    fmt_str_contents = [prediction_date] + list([str(prediction) for prediction in predictions_rounded])
    fmt_str = ", ".join(fmt_str_contents)
    print(fmt_str)
