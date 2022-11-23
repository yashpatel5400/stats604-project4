import sys
sys.path.append("../")

import os
import datetime
import pickle
import pytz
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

import multiprocessing
from multiprocessing.pool import Pool

from raw_data import wunderground_download
import predictor.utils as utils
from predictor.models.predictor_zeros import ZerosPredictor
from predictor.models.vinod import PrevDayPredictor
from predictor.models.unique import ArimaPredictor
from predictor.models.unique import HistoricAveragePredictor
from predictor.models.seamus import BasicOLSPredictor
from predictor.models.seamus import RidgePredictor
from predictor.models.vinod import PrevDayHistoricalPredictor
from predictor.models.vinod import MixPrevDayHistoricalPredictor

def populate_wunderground_data(i, start_date, window_days, station, download_window):
    prediction_date = start_date + i * window_days
    end_date_str = f"{prediction_date:%Y-%m-%d}"
    print(f"Requesting date: {end_date_str}")
    
    wunderground_raw_data = wunderground_download.fetch_wunderground(station=station, end_date_str=f"{prediction_date:%Y-%m-%d}", download_window=download_window)
    wunderground_data = pd.DataFrame(wunderground_raw_data)
    wunderground_data["date"] = wunderground_data["valid_time_gmt"].apply(lambda d: datetime.datetime.fromtimestamp(d))
    wunderground_data = wunderground_data.set_index("date")
    # ARGHHH, the column is named "GMT" but it's actually the local time zone!!
    wunderground_data.index = wunderground_data.index.tz_localize("EST")
    
    return wunderground_data

def populate_wunderground_data_wrapper(args):
  return populate_wunderground_data(*args)

def prepare_wunderground_eval_data(station, start_date, eval_len, wunderground_lookback):
    cache_dir = "eval"
    os.makedirs(cache_dir, exist_ok=True)
    cache_fn = os.path.join(cache_dir, f"{station}-{start_date:%Y-%m-%d}-{eval_len}-{wunderground_lookback}.csv")

    start = time.time()
    if os.path.exists(cache_fn):
        full_wunderground = pd.read_csv(cache_fn, index_col=0)
        full_wunderground.index = pd.to_datetime(full_wunderground.index)
    else:
        download_window = 30
        window_days = datetime.timedelta(days=download_window)
        num_future_requests = eval_len // download_window
        num_past_requests = -(wunderground_lookback // download_window)

        p = Pool(multiprocessing.cpu_count())
        populate_data_args = [(i, start_date, window_days, station, download_window) for i in range(num_past_requests, num_future_requests + 3)]
        full_wunderground = p.map(populate_wunderground_data_wrapper, populate_data_args)
        p.close()
        p.join()
        
        full_wunderground = list(full_wunderground)
        full_wunderground = pd.concat(full_wunderground)
        full_wunderground.to_csv(cache_fn)
    end = time.time()
    print(f"Scraped data in: {end - start} s")
    return full_wunderground

def prepare_full_eval_data(start_eval_date, eval_len, wunderground_lookback):
    noaa, _ = utils.load_data()
    full_eval_data = {}
    for station in utils.stations:
        full_eval_data[station] = {}
        full_eval_data[station]["noaa"] = noaa[station]
        full_eval_data[station]["wunderground"] = prepare_wunderground_eval_data(station, start_eval_date, eval_len, wunderground_lookback)
    return full_eval_data

def get_station_eval_task(full_eval_data, prediction_date, station):
    full_noaa = full_eval_data[station]["noaa"]
    full_wunderground = full_eval_data[station]["wunderground"]

    est = pytz.timezone('US/Eastern')
    strict_cutoff = est.localize(prediction_date.replace(hour=12)) # all the predictions are going to be made noon EST

    noaa_cutoff_len = 3
    noaa_cutoff = prediction_date - datetime.timedelta(days=noaa_cutoff_len)
    cut_noaa = full_noaa.iloc[full_noaa.index < noaa_cutoff]

    cut_wunderground = full_wunderground.iloc[full_wunderground.index < strict_cutoff]

    local_timezone = pytz.timezone(utils.fetch_timezone(station))
    forecast_horizon = 5
    target = []
    for forecast_day in range(1, forecast_horizon + 1):
        forecast_date_start = local_timezone.localize(prediction_date + datetime.timedelta(days=forecast_day))
        forecast_date_end = local_timezone.localize(prediction_date + datetime.timedelta(days=forecast_day) + datetime.timedelta(days=1))
        wunderground_forecast = full_wunderground.iloc[np.logical_and(forecast_date_start <= full_wunderground.index, full_wunderground.index < forecast_date_end)]
        temps = wunderground_forecast["temp"]
        target += [temps.min(), temps.mean(), temps.max()]
    
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

def eval_single_window(start_eval_date, eval_len, wunderground_lookback, model):
    """Runs an evaluation for a window of [start_eval_date, start_eval_date + eval_len] 
    where eval_len is to be specified as the number of days
    
    args:
        start_eval_date: (datetime.datetime) day of first *evaluation*, i.e. first day where predictions are *made*
            Note: that EACH eval day is evaluated for 5 days forward!
        eval_len: how many eval days to include
        wunderground_lookback: how far (in days) *before the first eval day* to extend the Wunderground data
            Note: data scraping will take time proportional to this number
    
    """
    full_eval_data = prepare_full_eval_data(start_eval_date, eval_len, wunderground_lookback)
    
    mses = []
    for day_offset in range(eval_len):
        prediction_date = start_eval_date + datetime.timedelta(days=day_offset)
        eval_data, eval_target = get_eval_task(full_eval_data, prediction_date)
        
        predictions = model.predict(eval_data)
        mse = (np.square(eval_target - predictions)).mean()
        mses.append(mse)
    return mses

def eval(model):
    """Runs evaluations for a windows from 11/30 - 12/10 for multiple years (default: 10 years) 
    using the specified model as the predictor. Returns MSEs as a 20 x 15 matrix, with each station a row
    across the 10 years with the year as the key of a dict, i.e.:
    
    {
        2012: [MSEs],
        2013: [MSEs],
        ...
    }
    """
    
    start_year = 2015
    num_years = 1
    mses_per_year = {}
    wunderground_lookback = 365 # how many days back to return of wunderground data
    eval_len = 10 # how many days we running evaluation for
    
    for year in range(start_year, start_year + num_years):
        start_eval_str = f"{year}-11-30" # when eval period starts (must follow %Y-%m-%d format)
        start_eval_date = datetime.datetime.strptime(start_eval_str, "%Y-%m-%d") 
        mses_per_year[year] = eval_single_window(start_eval_date, eval_len, wunderground_lookback, model)
    return mses_per_year

if __name__ == "__main__":
    model = RidgePredictor()
    eval_mses = eval(model)
    print(eval_mses)