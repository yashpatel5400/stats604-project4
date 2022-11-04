from abc import ABC, abstractmethod
 
class Predictor(ABC): 
    @abstractmethod
    def predict(self, data):
        """This function will receive a data dictionary in the following format:

        data = {
            "PANC" {
                "noaa": [dataframe of all but the last 3 days],
                "wunderground": [dataframe of the last 3 days up to noon today]
            },
            "KBOI" {
                "noaa": [dataframe of all but the last 3 days],
                "wunderground": [dataframe of the last 3 days up to noon today]
            },
            "KORD" {
                "noaa": [dataframe of all but the last 3 days],
                "wunderground": [dataframe of the last 3 days up to noon today]
            },
            "KDEN": {
                "noaa": [dataframe of all but the last 3 days],
                "wunderground": [dataframe of the last 3 days up to noon today]
            },
            ...
        }

        The output of this function *must* be a *single* vector with 300 entries that whose entries
        follow the final output convention, so:

        output = [
            PANC_min_day1, PANC_avg_day1, PANC_max_day1, PANC_min_day2, PANC_avg_day2, PANC_max_day2, ...,
            KBOI_min_day1, KBOI_avg_day1, KBOI_max_day1, KBOI_min_day2, KBOI_avg_day2, KBOI_max_day2, ...,
            ...
        ]
        """
        pass