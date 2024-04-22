import numpy as np
class Normalizer():
    """
    Normalization on the stock prices to standardize the range of 
    these values before feeding the data to the LSTM model.

    Functions:
        fit_transform;
        inverse_transform

    """
    def __init__(self) -> None:
        self.mu = None
        self.sd = None
    
    def fit_transform(self, x):
        self.mu = np.mean(x, axis=(0), keepdims=True)
        self.sd = np.std(x, axis=(0), keepdims=True)
        normalized_x = (x - self.mu) / self.sd
        return normalized_x

    def inverse_transform(self, x):
        return (x*self.sd) + self.mu