import numpy as np

from predictor_scaffold import Predictor

class ZerosPredictor(Predictor):
    def predict(self, data):
        return np.zeros(300)