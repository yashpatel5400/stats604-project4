import numpy as np
import utils
import datetime
import pandas as pd
from datetime import date, timedelta
from sklearn.linear_model import LinearRegression
pd.options.mode.chained_assignment = None

from predictor.models.predictor_scaffold import Predictor
from predictor.models.unique import HistoricAveragePredictor

class PrevDayPredictor(Predictor):
    def __init__(self):
        pass

    def predict(self, data):
        stations_data = [list(data[station]["wunderground"].iloc[-2][["temp_min","temp_mean","temp_max"]].values) * 5 for station in utils.stations]
        stations_data = np.array(stations_data).flatten()
        return stations_data