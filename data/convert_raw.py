import sys
sys.path.append("../")

import os
import json
import datetime
import pandas as pd

from predictor.read_noaa import read_noaa_data_file

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

station = stations[0]

for station in stations:
    noaa_path = f"../raw_data/noaa/{station}.dly"
    noaa_data = read_noaa_data_file(noaa_path)

    os.makedirs("noaa", exist_ok=True)
    noaa_data.to_csv(os.path.join("noaa", f"{station}.csv"))

    wunderground_path = f"../raw_data/wunderground/{station}.json"
    with open(wunderground_path, "r") as f:
        wunderground_raw_data = json.load(f)

    wunderground_data = pd.DataFrame(wunderground_raw_data)
    wunderground_data["date"] = wunderground_data["valid_time_gmt"].apply(lambda d: datetime.datetime.fromtimestamp(d))
    wunderground_data = wunderground_data.set_index("date")

    os.makedirs("wunderground", exist_ok=True)
    wunderground_data.to_csv(os.path.join("wunderground", f"{station}.csv"))