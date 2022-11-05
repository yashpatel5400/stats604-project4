import sys
sys.path.append("../")

import datetime
import json
import requests
import os

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

def fetch_latlon(station):
    dummy_url = f"https://www.wunderground.com/history/daily/{station}/date/2021-11-05"
    ans = requests.get(dummy_url)
    script_vals = ans.text.split("&q;:")
    extract_lat_or_lon = lambda lat_or_lon : float([script_vals[i+1] for i, v in enumerate(script_vals) if lat_or_lon in v][0].split(",")[0])
    return (extract_lat_or_lon("latitude"), extract_lat_or_lon("longitude"))

def fetch_timezone(station):
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
    return json.loads(response.text)

if __name__ == "__main__":
    for station in stations:
        data = fetch_wunderground(station=station, end_date_str= str(datetime.date.today()), download_window=5)
        os.makedirs("wunderground", exist_ok=True)
        with open(os.path.join("wunderground", f"{station}.json"), "w") as f:
            json.dump(data, f)