import sys
sys.path.append("../")

import datetime
import json
import requests
import os

from predictor.utils import stations

def fetch_wunderground(end_date_str="11/03/22", download_window=5):
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

    end_date = datetime.datetime.strptime(end_date_str, "%m/%d/%y") # can also use datetime.today() for today's date
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

if __name__ == "__main__":
    for station in stations:
        data = fetch_wunderground(end_date_str=datetime.today(), download_window=5)
        os.makedirs("wunderground", exist_ok=True)
        with open(os.path.join("wunderground", f"{station}.json"), "w") as f:
            json.dump(data, f)