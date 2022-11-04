import datetime
import json
import requests
import os

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

start_date = datetime.datetime.strptime("11/04/22", "%m/%d/%y")
download_window = 5 # download today and the previous (download_window-1) days, for a total of download_window days

headers = {
    'sec-ch-ua': '"Chromium";v="106", "Google Chrome";v="106", "Not;A=Brand";v="99"',
    'Accept': 'application/json, text/plain, */*',
    'Referer': 'https://www.wunderground.com/',
    'sec-ch-ua-mobile': '?0',
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.0.0 Safari/537.36',
    'sec-ch-ua-platform': '"macOS"',
}

for station in stations[:1]:
    full_data = {}
    for delta in range(download_window):
        date = start_date - datetime.timedelta(days=delta)
        date_str = f"{date:%Y%m%d}"
        params = {
            'apiKey': 'e1f10a1e78da46f5b10a1e78da96f525',
            'units': 'e',
            'startDate': date_str,
            'endDate': date_str,
        }
        response = requests.get(f'https://api.weather.com/v1/location/{station}:9:US/observations/historical.json', params=params, headers=headers)
        data = json.loads(response.text)["observations"]
        full_data[date_str] = data
    
    os.makedirs("wunderground", exist_ok=True)
    with open(os.path.join("wunderground", f"{station}.json"), "w") as f:
        json.dump(full_data, f)