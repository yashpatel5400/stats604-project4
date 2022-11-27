import os
import pandas as pd
import requests
import json

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

def load_noaa_data(station=None):
    """Loads data from NOAA and Wunderground with either the station specified or all if station is None.
    Returns: NOAA_data, Wunderground_data as dictionaries with the station as the key and dataframe as value
    """
    station_to_noaa_data = {}
    for station in stations:
        station_to_noaa_data[station] = pd.read_csv(os.path.join("..", "data", "noaa", f"{station}.csv"), index_col=0)
        station_to_noaa_data[station].index = pd.to_datetime(station_to_noaa_data[station].index)
    return station_to_noaa_data

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