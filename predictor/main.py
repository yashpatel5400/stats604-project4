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

import utils
from predictor.models.vinod import MetaPredictor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor

if __name__ == "__main__":
    keep_features = ['temp_min', 'wspd_min', 'pressure_min', 'heat_index_min', 'dewPt_min',
       'temp_mean', 'wspd_mean', 'pressure_mean', 'heat_index_mean',
       'dewPt_mean', 'temp_max', 'wspd_max', 'pressure_max', 'heat_index_max',
       'dewPt_max', 'wdir_mode']
    reg = MultiOutputRegressor(GradientBoostingRegressor(n_estimators=20,))
    window_size = 3
    model = MetaPredictor(reg, window_size, keep_features)

    data = utils.load_processed_data()
    predictions = model.predict(data)
    
    prediction_date = f"{datetime.date.today():%Y-%m-%d}"
    predictions_rounded = np.around(predictions, 1)
    
    fmt_str_contents = [prediction_date] + list([str(prediction) for prediction in predictions_rounded])
    fmt_str = ", ".join(fmt_str_contents)
    print(fmt_str)
