import numpy as np
import utils
import datetime
import pandas as pd

from predictor.models.predictor_scaffold import Predictor
from predictor.models.unique import HistoricAveragePredictor

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
            stations_data.append([temp_max[-1], temp_min[-1], round(temp_mean[-1],1)]*5)

        stations_data = np.array(stations_data).flatten()
        return stations_data

class PrevDayHistoricalPredictor(Predictor):
    def __init__(self):
        self.prev_day_model = PrevDayPredictor()
        self.historical_day_model = HistoricAveragePredictor()

    def predict(self, data):
        weights = np.array([1, 0.90, 0.80, 0.70, 0.60])

        prev_day_pred = self.prev_day_model.predict(data)
        historical_day_pred = self.historical_day_model.predict(data)

        counter = 0

        for index in range(len(prev_day_pred)//3):
            prev_day_val = prev_day_pred[index*3:index*3 + 3]
            historical_day_val= historical_day_pred[index*3:index*3 + 3]

            prev_day_val_weighted = prev_day_val * weights[counter]
            historical_day_val_weighted = historical_day_val * (1-weights[counter])

            prev_day_pred[index*3:index*3 + 3] = prev_day_val_weighted
            historical_day_pred[index*3:index*3 + 3] = historical_day_val_weighted

            counter += 1
            counter %= 5

        predictions = (prev_day_pred + historical_day_pred).round(1)
        return predictions