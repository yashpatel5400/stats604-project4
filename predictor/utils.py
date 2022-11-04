import os
import pandas as pd

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

def load_data(station=None):
    """Loads data from NOAA and Wunderground with either the station specified or all if station is None.
    Returns: NOAA_data, Wunderground_data as dictionaries with the station as the key and dataframe as value
    """
    station_to_noaa_data = {}
    station_to_wunderground_data = {}
    for station in stations:
        station_to_noaa_data[station] = pd.read_csv(os.path.join("..", "data", "noaa", f"{station}.csv"), index_col=0)
        station_to_wunderground_data[station] = pd.read_csv(os.path.join("..", "data", "wunderground", f"{station}.csv"), index_col=0)
    return station_to_noaa_data, station_to_wunderground_data