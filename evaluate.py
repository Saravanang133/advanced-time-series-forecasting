"""evaluate.py
RMSE and MASE metrics for time series.
"""
import numpy as np
from sklearn.metrics import mean_squared_error

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mase(y_true, y_pred, y_train, seasonality=1):
    # mean absolute scaled error
    n = len(y_true)
    d = np.abs(y_train[seasonality:] - y_train[:-seasonality]).mean()
    return np.mean(np.abs(y_true - y_pred)) / (d + 1e-8)
