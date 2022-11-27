import sys
sys.path.append("../")

import datetime
import json
import requests
import os
import time
import pandas as pd

import multiprocessing
from multiprocessing.pool import Pool

from predictor.utils import stations

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

    end_date = datetime.datetime.strptime(end_date_str, "%Y-%m-%d") # can also use datetime.today() for today's date
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
    print(f"Requesting date: {end_date_str}")
    
    wunderground_raw_data = fetch_wunderground(station=station, end_date_str=f"{prediction_date:%Y-%m-%d}", download_window=download_window)
    wunderground_data = pd.DataFrame(wunderground_raw_data)
    wunderground_data["date"] = wunderground_data["valid_time_gmt"].apply(lambda d: datetime.datetime.fromtimestamp(d))
    wunderground_data = wunderground_data.set_index("date")
    # ARGHHH, the column is named "GMT" but it's actually the local time zone!!
    wunderground_data.index = wunderground_data.index.tz_localize("EST")
    
    return wunderground_data

def fetch_wunderground_pd_window_wrapper(args):
  return fetch_wunderground_pd_window(*args)

def fetch_wunderground_pd(station, predict_date, future_days, past_days):
    """Downloads Wunderground raw data and constructs a full dataset of the following
    window: [predict_date - past_days, predict_date + future_days] 

    args:
        station: (str) which station
        predict_date: (datetime) "current" date of prediction
        future_days: (int) how many days in the future to scrape (note: this can obviously only be
            used for historical data, i.e. for evaluation tasks)
        past_days: (int) how many days in the past to scrape
    """
    cache_dir = "wunderground_cache"
    os.makedirs(cache_dir, exist_ok=True)
    cache_fn = os.path.join(cache_dir, f"{station}-{predict_date:%Y-%m-%d}-{future_days}-{past_days}.csv")

    start = time.time()
    if os.path.exists(cache_fn):
        full_wunderground = pd.read_csv(cache_fn, index_col=0)
        full_wunderground.index = pd.to_datetime(full_wunderground.index)
    else:
        download_window = 30
        num_future_requests = future_days // download_window
        num_past_requests = -(past_days // download_window)

        p = Pool(multiprocessing.cpu_count())
        populate_data_args = [(i, predict_date, station, download_window) for i in range(num_past_requests, num_future_requests + 3)]
        full_wunderground = p.map(fetch_wunderground_pd_window_wrapper, populate_data_args)
        p.close()
        p.join()
        
        full_wunderground = list(full_wunderground)
        full_wunderground = pd.concat(full_wunderground)
        full_wunderground.to_csv(cache_fn)
    end = time.time()
    print(f"Scraped data in: {end - start} s")
    return full_wunderground

if __name__ == "__main__":
    for station in stations:
        data = fetch_wunderground(station=station, end_date_str= str(datetime.date.today()), download_window=20)
        os.makedirs("wunderground", exist_ok=True)
        with open(os.path.join("wunderground", f"{station}.json"), "w") as f:
            json.dump(data, f)