import numpy as np

from predictor.models.predictor_scaffold import Predictor

class ZerosPredictor(Predictor):
    def __init__(self):
        pass

    def predict(self, data):
        return np.zeros(300)