import numpy as np
import utils
import datetime
import pandas as pd

from predictor.models.predictor_scaffold import Predictor

class PrevDayPredictor(Predictor):
    def __init__(self):
        pass

    def predict(self, data):

        stations_data = []
        for station in utils.stations:
            df = data[station]["wunderground"]
            df['date_col'] = pd.to_datetime(df.index).date
            temp_max = df.groupby(['date_col'], sort=False)['temp'].max()
            temp_min = df.groupby(['date_col'], sort=False)['temp'].min()
            temp_mean = df.groupby(['date_col'], sort=False)['temp'].mean()
            stations_data.append([temp_min[-1], round(temp_mean[-1], 1), temp_max[-1]]*5)

        stations_data = np.array(stations_data).flatten()
        return stations_data