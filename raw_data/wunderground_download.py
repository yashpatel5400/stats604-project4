import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from math import ceil
import datetime
import json
import requests
import os
import time
import pandas as pd
import logging

import multiprocessing
from multiprocessing.pool import Pool

import predictor.utils as utils

def fetch_wunderground(station, end_date_str="2022-11-03", download_window=5):
    """Downloads data from Wunderground from end_date-download_window to end_date. Note that
    using too large a download_window (i.e. > 20) will cause this to error out
    """

    headers = {
        'sec-ch-ua': '"Chromium";v="106", "Google Chrome";v="106", "Not;A=Brand";v="99"',
        'Accept': 'application/json, text/plain, */*',
        'Referer': 'https://www.wunderground.com/',
        'sec-ch-ua-mobile': '?0',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.0.0 Safari/537.36',
        'sec-ch-ua-platform': '"macOS"',
    }

    end_date = datetime.datetime.strptime(end_date_str, "%Y-%m-%d").date()
    end_date = min(end_date, datetime.date.today()) # can't retrieve future days!

    start_date = end_date - datetime.timedelta(days=(download_window-1))
    start_date_str = f"{start_date:%Y%m%d}"
    end_date_str = f"{end_date:%Y%m%d}"

    params = {
        'apiKey': 'e1f10a1e78da46f5b10a1e78da96f525',
        'units': 'e',
        'startDate': start_date_str,
        'endDate': end_date_str,
    }
    response = requests.get(f'https://api.weather.com/v1/location/{station}:9:US/observations/historical.json', params=params, headers=headers)
    return json.loads(response.text)["observations"]

def fetch_wunderground_pd_window(window_idx, start_date, station, download_window):
    """Downloads a *single window* from Wunderground. This is required since the API blocks large
    simultaneous download requests, so large windows need to be broken into smaller requests. Note that
    the recommended use is to invoke this indirectly through populate_wunderground_data_wrapper with
    multiprocessing to parallelize HTTP requests by using fetch_wunderground_pd instead. See eval.py for an example

    args:
        window_idx: (int) which window is being downloaded (set to 0 if not parallelizing)
        start_date: (datetime) "current" date: note that the past data can be scraped with window_idx < 0
        station: (str) which station
        download_window: (int) size of the window
    """
    window_days = datetime.timedelta(days=download_window)
    prediction_date = start_date + window_idx * window_days
    end_date_str = f"{prediction_date:%Y-%m-%d}"
    logging.info(f"Requesting date: {end_date_str}")
    
    wunderground_raw_data = fetch_wunderground(station=station, end_date_str=f"{prediction_date:%Y-%m-%d}", download_window=download_window)
    wunderground_data = pd.DataFrame(wunderground_raw_data)
    wunderground_data["date"] = wunderground_data["valid_time_gmt"].apply(lambda d: datetime.datetime.fromtimestamp(d))
    wunderground_data = wunderground_data.set_index("date")
    # ARGHHH, the column is named "GMT" but it's actually the local time zone!!
    wunderground_data.index = wunderground_data.index.tz_localize("EST")
    
    return wunderground_data

def fetch_wunderground_pd_window_wrapper(args):
  return fetch_wunderground_pd_window(*args)

def fetch_wunderground_pd(station, predict_date, future_days, past_days, ignore_cache=False):
    """Downloads Wunderground raw data and constructs a full dataset of the following
    window: [predict_date - past_days, predict_date + future_days] 

    args:
        station: (str) which station
        predict_date: (datetime) "current" date of prediction
        future_days: (int) how many days in the future to scrape (note: this can obviously only be
            used for historical data, i.e. for evaluation tasks)
        past_days: (int) how many days in the past to scrape
    """
    os.makedirs(utils.raw_wunderground_cache, exist_ok=True)
    # a bit messy, but to reuse the caching logic between eval and the "final prediction," we have a special
    # case for simplifying the name in the "final prediction" case to match the NOAA caching convention
    if predict_date == datetime.date.today():
        cache_fn = os.path.join(utils.raw_wunderground_cache, f"{station}.csv")
    else:
        cache_fn = os.path.join(utils.raw_wunderground_cache, f"{station}-{predict_date:%Y-%m-%d}-{future_days}-{past_days}.csv")

    start = time.time()
    if not ignore_cache and os.path.exists(cache_fn):
        full_wunderground = pd.read_csv(cache_fn, index_col=0)
        full_wunderground.index = pd.to_datetime(full_wunderground.index)
    else:
        download_window = 30
        num_future_requests = ceil(future_days / download_window)
        num_past_requests = -ceil(past_days / download_window)

        p = Pool(multiprocessing.cpu_count())
        populate_data_args = [(i, predict_date, station, download_window) for i in range(num_past_requests, num_future_requests + 1)]
        full_wunderground = p.map(fetch_wunderground_pd_window_wrapper, populate_data_args)
        p.close()
        p.join()
        
        full_wunderground = list(full_wunderground)
        full_wunderground = pd.concat(full_wunderground)
        if not ignore_cache:
            full_wunderground.to_csv(cache_fn)
    end = time.time()
    logging.info(f"Scraped data in: {end - start} s")
    return full_wunderground

if __name__ == "__main__":
    for station in utils.stations:
        wunderground_days = 365 # get one year of past data
        fetch_wunderground_pd(station, predict_date=datetime.date.today(), future_days=0, past_days=wunderground_days)