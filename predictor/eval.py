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
from predictor.models.predictor_zeros import ZerosPredictor
# from predictor.models.vinod import PrevDayPredictor
from predictor.models.unique import ArimaPredictor
from predictor.models.unique import HistoricAveragePredictor
from predictor.models.seamus import BasicOLSPredictor
from predictor.models.seamus import LassoPredictor
from predictor.models.seamus import GBTPredictor
from predictor.models.vinod import PrevDayHistoricalPredictor
from predictor.models.vinod import MetaPredictor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor

def prepare_full_eval_data(start_eval_date, eval_len, wunderground_lookback):
    """Prepares data for evaluation for a window of [start_eval_date, start_eval_date + eval_len] 
    where eval_len is to be specified as the number of days. Note that the data returned from this
    is NOT the data that is to be used for evaluation, i.e. each eval_day must be separated after
    this initial bulk fetch (using get_eval_task)
    
    args:
        start_eval_date: (datetime.datetime) day of first *evaluation*, i.e. first day where predictions are *made*
            Note: that EACH eval day is evaluated for 5 days forward!
        eval_len: how many eval days to include
        wunderground_lookback: how far (in days) *before the first eval day* to extend the Wunderground data
            Note: data scraping will take time proportional to this number
    """
    noaa = utils.load_noaa_data()
    full_eval_data = {}
    for station in utils.stations:
        full_eval_data[station] = {}
        full_eval_data[station]["noaa"] = noaa[station]
        full_eval_data[station]["wunderground"] = wunderground_download.fetch_wunderground_pd(
            station, start_eval_date, eval_len, wunderground_lookback)
    return full_eval_data

def get_station_eval_task(full_eval_data, prediction_date, station):
    full_noaa = full_eval_data[station]["noaa"]
    full_wunderground = full_eval_data[station]["wunderground"]

    est = pytz.timezone('US/Eastern')
    strict_cutoff = est.localize(prediction_date.replace(hour=12)) # all the predictions are going to be made noon EST

    local_timezone = pytz.timezone(utils.fetch_timezone(station))
    full_wunderground['date_col'] = pd.to_datetime(full_wunderground.index).tz_convert(local_timezone).date
    
    # cutoff_side = 0: < "prediction cutoff" -- used to construct our dataset
    # cutoff_side = 1: > "prediction cutoff" -- used to construct the evaluation target
    for cutoff_side in range(2):
        if cutoff_side == 0:
            dataset_view = full_wunderground[full_wunderground.index < strict_cutoff]
        else:
            dataset_view = full_wunderground[full_wunderground.index >= strict_cutoff]

        # Wunderground returns granular (hourly) data points, but we only want daily for prediction: this coarsens the dataset
        aggregated_columns = ["temp", "wspd", "pressure", "heat_index", 'dewPt']
        maxes = dataset_view.groupby(['date_col'], sort=False)[aggregated_columns].max().set_axis([f"{column}_max" for column in aggregated_columns], axis=1, inplace=False).set_index(dataset_view['date_col'].unique())
        means = dataset_view.groupby(['date_col'], sort=False)[aggregated_columns].mean().set_axis([f"{column}_mean" for column in aggregated_columns], axis=1, inplace=False).set_index(dataset_view['date_col'].unique())
        mins  = dataset_view.groupby(['date_col'], sort=False)[aggregated_columns].min().set_axis([f"{column}_min" for column in aggregated_columns], axis=1, inplace=False).set_index(dataset_view['date_col'].unique())
        wind_dir = dataset_view.groupby(['date_col'], sort=False)['wdir_cardinal'].agg(
            lambda x: pd.Series.mode(x)[0]).astype("category").to_frame("wdir_mode").set_index(dataset_view['date_col'].unique())
        aggregated_wunderground = pd.concat((mins, means, maxes, wind_dir), axis=1)

        if cutoff_side == 0:
            cut_wunderground = aggregated_wunderground.drop(aggregated_wunderground.index[0], axis=0) # first row is often partial day based on the time zone
        else:
            evaluation_data = aggregated_wunderground

    noaa_cutoff_len = 3
    noaa_cutoff = prediction_date - datetime.timedelta(days=noaa_cutoff_len)
    cut_noaa = full_noaa.iloc[full_noaa.index < noaa_cutoff]
    
    forecast_horizon = 5
    prediction_window = [prediction_date + datetime.timedelta(days=forecast_day) for forecast_day in range(1, forecast_horizon + 1)]
    prediction_targets_df = evaluation_data.loc[prediction_window]
    target = []
    for i in range(len(prediction_targets_df)):
        target.append(prediction_targets_df["temp_min"][i])
        target.append(prediction_targets_df["temp_mean"][i])
        target.append(prediction_targets_df["temp_max"][i])
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
        print(mse)
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
    
    start_year = 2019
    num_years = 1
    mses_per_year = {}
    wunderground_lookback = 1*365 # how many days back to return of wunderground data
    eval_len = 10 # how many days we running evaluation for
    
    for year in range(start_year, start_year + num_years):
        start_eval_str = f"{year}-11-30" # when eval period starts (must follow %Y-%m-%d format)
        start_eval_date = datetime.datetime.strptime(start_eval_str, "%Y-%m-%d") 
        mses_per_year[year] = eval_single_window(start_eval_date, eval_len, wunderground_lookback, model)
    return mses_per_year

if __name__ == "__main__":
    keep_features = ['temp_min', 'wspd_min', 'pressure_min', 'heat_index_min', 'dewPt_min',
       'temp_mean', 'wspd_mean', 'pressure_mean', 'heat_index_mean',
       'dewPt_mean', 'temp_max', 'wspd_max', 'pressure_max', 'heat_index_max',
       'dewPt_max', 'wdir_mode']
    reg = MultiOutputRegressor(GradientBoostingRegressor(n_estimators=20,))
    window_size = 3
    model = MetaPredictor(reg, window_size, keep_features)

    eval_mses = eval(model)
    print(eval_mses)