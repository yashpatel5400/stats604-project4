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

    # def predict(self, data):

    #     stations_data = []
    #     for station in utils.stations:
    #         df = data[station]["wunderground"]
    #         df['date_col'] = pd.to_datetime(df.index).date
    #         temp_max = df.groupby(['date_col'], sort=False)['temp'].max()
    #         temp_min = df.groupby(['date_col'], sort=False)['temp'].min()
    #         temp_mean = df.groupby(['date_col'], sort=False)['temp'].mean()
    #         stations_data.append([temp_min[-1], round(temp_mean[-1],2), temp_max[-1]]*5)

    #     stations_data = np.array(stations_data).flatten()
    #     return stations_data
    def predict(self, data):
        stations_data = [list(data[station]["wunderground"].iloc[-1][["temp_min","temp_mean","temp_max"]].values) * 5 for station in utils.stations]
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


class MixPrevDayHistoricalPredictor(Predictor):
    def __init__(self):
        self.prev_day_model = PrevDayPredictor()
        self.historical_day_model = HistoricAveragePredictor()

    def create_dataset(self, data, station, day, measure):
        wunderground = data[station]["wunderground"]
        wunderground['date_col'] = pd.to_datetime(wunderground.index).date

        noaa = data[station]["noaa"]
        noaa = noaa.loc[:, [ "TMIN", "TAVG", "TMAX"]].dropna(axis =0)
        noaa = noaa*0.18+32.0

        current_date = wunderground['date_col'].iloc[-1]
        print(current_date)

        date_range = pd.date_range(current_date - pd.DateOffset(days=365) , current_date - timedelta(days = 1),  freq='d')
        month_day_index= [(date.month, date.day) for date in date_range]

        df = noaa.groupby(by=[noaa.index.month, noaa.index.day]).mean().round(2)
        historical_tmp = df.loc[month_day_index[day:]][measure]

        if measure == "TMIN":
            current_temp = wunderground["temp_min"]
        elif measure == "TMAX":
            current_temp = wunderground["temp_max"]
        else:
            current_temp = wunderground["temp_mean"]

        current_temp_trunc = current_temp.loc[date_range[:-(day)]] #stop 1 days before end

        X =pd.DataFrame(columns=['current', 'historical'])
        X['current'] = current_temp_trunc
        X['historical'] = historical_tmp.values
        y = current_temp.loc[date_range[day:]]

        return X, y

    def predict(self, data):
        measurements = ["TMIN", "TAVG", "TMAX"]

        prev_day_pred = self.prev_day_model.predict(data) #vector of length 300
        historical_day_pred = self.historical_day_model.predict(data) #vector of length 300

        station_reg = []
        for station in utils.stations: #20 stations
            for day in range(1, 6): #5 days
                for meas in measurements: # 3 measurements
                    X, y = self.create_dataset(data, station, day, meas)
                    reg = LinearRegression(fit_intercept = True).fit(X, y)
                    station_reg.append(reg)

        predictions = []
        for index in range(len(prev_day_pred)):
            reg_model = station_reg[index]

            X_test =pd.DataFrame(columns=['current', 'historical'])
            X_test['current'] = [prev_day_pred[index]]
            X_test['historical'] = [historical_day_pred[index]]
            pred = reg_model.predict(X_test)
            predictions.append(pred[0])
            
        return predictions