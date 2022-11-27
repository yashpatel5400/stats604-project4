#!/usr/bin/env python
"""Main entrypoint for weather predictions"""

__author__ = "Vinod Raman, Unique Subedi, Seamus Somerstep, and Yash Patel"
__license__ = "MIT"
__version__ = "1.0"
__status__ = "Dev"

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import datetime

import utils

def format_output(date_to_prediction):
    """Takes a dictionary that maps dates to dataframes with the weather stations as 
    rows and min, avg, max as columns and spits back the formatted string desired as output
    """
    start_date = list(date_to_prediction.keys())[0]
    return f"{start_date}, {', '.join([', '.join(date_to_prediction[date].values.flatten().astype(str)) for date in date_to_prediction])}"
    
if __name__ == '__main__':
    start_date = datetime.datetime.strptime("10/10/11", "%m/%d/%y")
    dates = [start_date + datetime.timedelta(days=1 + delta) for delta in range(5)]
    predictions = {date: pd.DataFrame(0, columns = ["Min", "Avg", "Max"], index = utils.stations) for date in dates}
    output = format_output(predictions)
    print(output)