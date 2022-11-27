import os
import pandas as pd
import requests
import json
import logging

# globally referenced paths
raw_wunderground_cache = "data/raw_wunderground"
processed_wunderground_cache = "data/processed_wunderground"
raw_noaa_cache = "data/raw_noaa"
processed_noaa_cache = "data/processed_noaa"

stations = [
    "PANC",
    "KBOI",
    "KORD",
    "KDEN",
    "KDTW",
    "PHNL",
    "KIAH",
    "KMIA",
    "KMIC",
    "KOKC",
    "KBNA",
    "KJFK",
    "KPHX",
    "KPWM",
    "KPDX",
    "KSLC",
    "KSAN",
    "KSFO",
    "KSEA",
    "KDCA",
]
station_latlon_cache = {}
station_timezone_cache = {}

def load_processed_data_src(data_src):
    """Loads data from NOAA or Wunderground (resp. pass in data_src = "noaa" or "wunderground").
    Returns: NOAA_data, Wunderground_data as dictionaries with the station as the key and dataframe as value, i.e.

    {
        "PANC": pd.Dataframe,
        "PHNL": pd.Dataframe,
        ...
    }
    """
    if data_src not in ["noaa", "wunderground"]:
        logging.warning(f"Invalid data source requested: {data_src} -- must be noaa or wunderground")
        return None

    station_to_processed_data = {}
    for station in stations:
        if data_src == "noaa":
            data_src_path = os.path.join(processed_noaa_cache, f"{station}.csv")
        else:
            data_src_path = os.path.join(processed_wunderground_cache, f"{station}.csv")
        station_to_processed_data[station] = pd.read_csv(data_src_path, index_col=0)
        station_to_processed_data[station].index = pd.to_datetime(station_to_processed_data[station].index)
    return station_to_processed_data

def load_processed_data():
    """Loads data from both NOAA or Wunderground. Assumes data has been scraped and preprocessed
    Returns: NOAA_data, Wunderground_data as a single dict with the data src as the inner keys to a dict
    with the station as the key and dataframe as value, i.e.

    {
        "PANC": {
            "noaa": pd.Dataframe,
            "wunderground": pd.Dataframe,
        },
        "PHNL": {
            "noaa": pd.Dataframe,
            "wunderground": pd.Dataframe,
        },
        ...
    }
    """
    processed_data = {}
    for data_src in ["noaa", "wunderground"]:
        station_to_data = load_processed_data_src(data_src)
        for station in station_to_data:
            if station not in processed_data:
                processed_data[station] = {}
            processed_data[station][data_src] = station_to_data[station]
    return processed_data

def fetch_latlon(station):
    global station_latlon_cache
    if station not in station_latlon_cache:
        dummy_url = f"https://www.wunderground.com/history/daily/{station}/date/2021-11-05"
        ans = requests.get(dummy_url)
        script_vals = ans.text.split("&q;:")
        extract_lat_or_lon = lambda lat_or_lon : float([script_vals[i+1] for i, v in enumerate(script_vals) if lat_or_lon in v][0].split(",")[0])
        station_latlon_cache[station] = (extract_lat_or_lon("latitude"), extract_lat_or_lon("longitude"))
    return station_latlon_cache[station]

def fetch_timezone(station):
    global station_latlon_cache
    if station not in station_latlon_cache:
        lat_lon = fetch_latlon(station)

        headers = {
            'sec-ch-ua': '"Google Chrome";v="107", "Chromium";v="107", "Not=A?Brand";v="24"',
            'Accept': 'application/json, text/plain, */*',
            'Referer': 'https://www.wunderground.com/',
            'sec-ch-ua-mobile': '?0',
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36',
            'sec-ch-ua-platform': '"macOS"',
        }

        params = {
            'apiKey': 'e1f10a1e78da46f5b10a1e78da96f525',
            'geocode': f'{round(lat_lon[0], 2)},{round(lat_lon[1], 2)}',
            'format': 'json',
        }

        response = requests.get('https://api.weather.com/v3/dateTime', params=params, headers=headers)
        station_timezone_cache[station] = json.loads(response.text)["ianaTimeZone"]
    return station_timezone_cache[station]