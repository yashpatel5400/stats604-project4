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

headers = {
    'sec-ch-ua': '"Chromium";v="106", "Google Chrome";v="106", "Not;A=Brand";v="99"',
    'Accept': 'application/json, text/plain, */*',
    'Referer': 'https://www.wunderground.com/',
    'sec-ch-ua-mobile': '?0',
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.0.0 Safari/537.36',
    'sec-ch-ua-platform': '"macOS"',
}

for station in stations:
    download_window = 5 # download today and the previous (download_window-1) days, for a total of download_window days
    
    end_date = datetime.datetime.strptime("11/03/22", "%m/%d/%y") # can also use datetime.today() for today's date
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
    data = json.loads(response.text)["observations"]
    
    os.makedirs("wunderground", exist_ok=True)
    with open(os.path.join("wunderground", f"{station}.json"), "w") as f:
        json.dump(data, f)