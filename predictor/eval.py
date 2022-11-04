import sys
sys.path.append("../")

import os
import utils
import datetime
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from raw_data import wunderground_download
from models.predictor_zeros import ZerosPredictor

def get_station_eval_task(station, prediction_day):
    noaa, _ = utils.load_data()

    columns = ["TMIN", "TAVG", "TMAX"]
    prediction_window = [prediction_day + 1, prediction_day + 5]
    target = noaa[station].iloc[prediction_window[0]:prediction_window[1]+1][columns]

    # *no* data can be used for prediction after 12:00 on the prediction day
    prediction_date = noaa[station].iloc[prediction_day].name
    strict_cutoff = datetime.datetime.strptime(prediction_date, "%Y-%m-%d")
    strict_cutoff = strict_cutoff.replace(hour=12)

    # for everything up to ~3 days ago, we will have NOAA data
    noaa_cutoff_len = 3
    noaa_cutoff = prediction_day - noaa_cutoff_len
    cut_noaa = noaa["PANC"].iloc[:noaa_cutoff+1]

    # for the more recent days, we use Wunderground
    recent_days = wunderground_download.fetch_wunderground(station=station, end_date_str=prediction_date, download_window=noaa_cutoff_len+2) # one day of overlap w/ NOAA
    wunderground_data = pd.DataFrame(recent_days)
    wunderground_data["date"] = wunderground_data["valid_time_gmt"].apply(lambda d: datetime.datetime.fromtimestamp(d))
    wunderground_data = wunderground_data.set_index("date")
    cut_wunderground = wunderground_data[wunderground_data.index < strict_cutoff]

    data = {
        "noaa": cut_noaa,
        "wunderground": cut_wunderground,
    }
    return data, target

def get_full_eval_task(prediction_day):
    cache_dir = "eval"
    os.makedirs(cache_dir, exist_ok=True)
    cache_fn = os.path.join(cache_dir, f"prediction_day.pkl")
    if os.path.exists(cache_fn):
        with open(cache_fn, "rb") as f:
            full_data, full_target = pickle.load(f)
    else:
        full_data = {}
        full_target = []
        for station in utils.stations:
            data, target = get_station_eval_task(station, prediction_day)
            
            full_data[station] = data
            full_target.append(target.values.flatten())
        full_target = np.array(full_target).flatten()
        with open(cache_fn, "wb") as f:
            pickle.dump((full_data, full_target), f)

    return full_data, full_target

def eval(model):
    mses = []
    for prediction_day in range(-375, -300):
        eval_data, eval_target = get_full_eval_task(prediction_day)
        predictions = model.predict(eval_data)
        mse = (np.square(eval_target - predictions)).mean()
        mses.append(mse)
    return mses

if __name__ == "__main__":
    zeros_predictor = ZerosPredictor()
    mses = eval(zeros_predictor)