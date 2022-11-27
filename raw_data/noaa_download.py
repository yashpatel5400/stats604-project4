import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import urllib.request
import json
import os
import logging

import predictor.utils as utils

station_code_to_noaa = {
    "PANC": "USW00026451",
    "KBOI": "USW00024131",
    "KORD": "USW00094846",
    "KDEN": "USW00003017",
    "KDTW": "USW00094847",
    "PHNL": "USW00022521",
    "KIAH": "USW00012960",
    "KMIA": "USW00012839",
    "KMIC": "USW00094960",
    "KOKC": "USW00013967",
    "KBNA": "USW00013897",
    "KJFK": "USW00094789",
    "KPHX": "USW00023183",
    "KPWM": "USW00014764",
    "KPDX": "USW00024229",
    "KSLC": "USW00024127",
    "KSAN": "USW00023188",
    "KSFO": "USW00023234",
    "KSEA": "USW00024233",
    "KDCA": "USW00013743",
}

if __name__ == "__main__":
    base_url = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/all/"

    os.makedirs(utils.raw_noaa_cache, exist_ok=True)
    for station_code in station_code_to_noaa:
        url = f"{base_url}/{station_code_to_noaa[station_code]}.dly"
        urllib.request.urlretrieve(url, os.path.join(utils.raw_noaa_cache, f"{station_code}.dly"))
        logging.debug(f"Scraped data for: {station_code}")