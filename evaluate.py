"""evaluate.py

Evaluation metrics for time series forecasting.
Includes RMSE and MASE (Mean Absolute Scaled Error).
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

def rmse(true, pred):
    return np.sqrt(mean_squared_error(true, pred))

def mase(y_true, y_pred, y_train, seasonality=1):
    # y_train used to compute scale (in-sample naive forecast)
    n = len(y_true)
    denom = np.mean(np.abs(y_train[seasonality:] - y_train[:-seasonality]))
    return np.mean(np.abs(y_true - y_pred)) / (denom + 1e-8)

if __name__ == '__main__':
    import numpy as np
    y_true = np.array([1,2,3,4,5])
    y_pred = np.array([1.1,1.9,3.1,3.8,5.2])
    y_train = np.array([0,1,2,3,4,5,6])
    print('RMSE', rmse(y_true,y_pred))
    print('MASE', mase(y_true,y_pred,y_train,seasonality=1))
