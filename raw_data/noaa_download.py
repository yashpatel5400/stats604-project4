import urllib.request
import json
import os

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

    os.makedirs("noaa", exist_ok=True)
    for station_code in station_code_to_noaa:
        url = f"{base_url}/{station_code_to_noaa[station_code]}.dly"
        urllib.request.urlretrieve(url, os.path.join("noaa", f"{station_code}.dly"))