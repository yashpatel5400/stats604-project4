import sys
sys.path.append("../")

import os
import datetime
import pickle
import pytz
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from raw_data import wunderground_download
import predictor.utils as utils
from predictor.models.predictor_zeros import ZerosPredictor
from predictor.models.vinod import PrevDayPredictor
from predictor.models.unique import HistoricAveragePredictor
from predictor.models.seamus import BasicOLSPredictor
from predictor.models.vinod import PrevDayHistoricalPredictor

def prepare_wunderground_eval_data(station, start_date, eval_len):
    cache_dir = "eval"
    os.makedirs(cache_dir, exist_ok=True)
    cache_fn = os.path.join(cache_dir, f"{station}-{start_date:%Y-%m-%d}-{eval_len}.csv")
    if os.path.exists(cache_fn):
        full_wunderground = pd.read_csv(cache_fn, index_col=0)
        full_wunderground.index = pd.to_datetime(full_wunderground.index)
    else:
        download_window = 5
        window_days = datetime.timedelta(days=download_window)
        num_requests = eval_len // download_window

        full_wunderground = []
        for i in range(num_requests + 3): # need to *include* one before, the current, and one after for padding
            prediction_date = start_date + i * window_days
            wunderground_raw_data = wunderground_download.fetch_wunderground(station=station, end_date_str=f"{prediction_date:%Y-%m-%d}", download_window=download_window)
            wunderground_data = pd.DataFrame(wunderground_raw_data)
            wunderground_data["date"] = wunderground_data["valid_time_gmt"].apply(lambda d: datetime.datetime.fromtimestamp(d))
            wunderground_data = wunderground_data.set_index("date")
            # ARGHHH, the column is named "GMT" but it's actually the local time zone!!
            wunderground_data.index = wunderground_data.index.tz_localize("EST")
            full_wunderground.append(wunderground_data)
        full_wunderground = pd.concat(full_wunderground)
        full_wunderground.to_csv(cache_fn)
    return full_wunderground

def prepare_full_eval_data(start_eval_date, eval_len):
    noaa, _ = utils.load_data()
    full_eval_data = {}
    for station in utils.stations:
        full_eval_data[station] = {}
        full_eval_data[station]["noaa"] = noaa[station]
        full_eval_data[station]["wunderground"] = prepare_wunderground_eval_data(station, start_eval_date, eval_len)
    return full_eval_data

def get_station_eval_task(full_eval_data, prediction_date, station):
    full_noaa = full_eval_data[station]["noaa"]
    full_wunderground = full_eval_data[station]["wunderground"]

    est = pytz.timezone('US/Eastern')
    strict_cutoff = est.localize(prediction_date.replace(hour=12)) # all the predictions are going to be made noon EST

    noaa_cutoff_len = 3
    noaa_cutoff = prediction_date - datetime.timedelta(days=noaa_cutoff_len)
    cut_noaa = full_noaa.iloc[full_noaa.index < noaa_cutoff]

    padded_noaa_cutoff = est.localize(noaa_cutoff - datetime.timedelta(days=1)) # give one day of overlap for Wunderground data
    cut_wunderground = full_wunderground.iloc[np.logical_and(padded_noaa_cutoff <= full_wunderground.index, full_wunderground.index < strict_cutoff)]

    local_timezone = pytz.timezone(utils.fetch_timezone(station))
    forecast_horizon = 5
    target = []
    for forecast_day in range(1, forecast_horizon + 1):
        forecast_date_start = local_timezone.localize(prediction_date + datetime.timedelta(days=forecast_day))
        forecast_date_end = local_timezone.localize(prediction_date + datetime.timedelta(days=forecast_day) + datetime.timedelta(days=1))
        wunderground_forecast = full_wunderground.iloc[np.logical_and(forecast_date_start <= full_wunderground.index, full_wunderground.index < forecast_date_end)]
        temps = wunderground_forecast["temp"]
        target += [temps.max(), temps.min(), temps.mean()]
    
    target = np.array(target)
    data = {
        "noaa": cut_noaa,
        "wunderground": cut_wunderground,
    }
    return data, target

def get_eval_task(full_eval_data, prediction_date):
    full_data = {}
    full_target = []
    for station in utils.stations:
        data, target = get_station_eval_task(full_eval_data, prediction_date, station)
        full_data[station] = data
        full_target.append(target.flatten())
    full_target = np.array(full_target).flatten()
    return full_data, full_target

def eval(start_eval_date, eval_len, model):
    full_eval_data = prepare_full_eval_data(start_eval_date, eval_len)
    
    mses = []
    for day_offset in range(eval_len):
        prediction_date = start_eval_date + datetime.timedelta(days=day_offset)
        eval_data, eval_target = get_eval_task(full_eval_data, prediction_date)
        predictions = model.predict(eval_data)
        mse = (np.square(eval_target - predictions)).mean()
        mses.append(mse)
    return mses

if __name__ == "__main__":
    start_eval_str = "2021-10-01" # when eval period starts (must follow %Y-%m-%d format)
    start_eval_date = datetime.datetime.strptime(start_eval_str, "%Y-%m-%d") 
    eval_len = 10 # how many days we running evaluation for

    zeros_predictor = PrevDayHistoricalPredictor()
    mses = eval(start_eval_date, eval_len, zeros_predictor)
    print(mses)