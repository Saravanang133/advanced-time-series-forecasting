"""data_generator.py

Generate a multivariate time series dataset (daily frequency) with:
- multiple seasonalities (weekly, yearly)
- trend components
- exogenous variables
- configurable noise and length
"""
import numpy as np
import pandas as pd

def generate_multivariate_series(n_series=3, start='2015-01-01', periods=365*4, freq='D', seed=42):
    np.random.seed(seed)
    idx = pd.date_range(start=start, periods=periods, freq=freq)
    df = pd.DataFrame(index=idx)
    t = np.arange(periods)

    # global trend
    trend = 0.001 * t

    # yearly seasonality (sinusoids)
    yearly = np.array([np.sin(2*np.pi*t/365.25), np.cos(2*np.pi*t/365.25)])
    weekly = np.array([np.sin(2*np.pi*t/7), np.cos(2*np.pi*t/7)])

    for i in range(n_series):
        seasonal = 2.0 * yearly[0] * (0.5 + 0.5*np.random.rand()) + 0.5 * weekly[0] * (0.5 + 0.5*np.random.rand())
        exog = 0.5 * np.random.randn(periods)  # exogenous driver
        noise = 0.3 * np.random.randn(periods)

        series = 10 + (i+1)*trend + seasonal + 0.5 * yearly[1] + exog + noise
        df[f'target_{i+1}'] = series
        df[f'exog_{i+1}'] = exog

    # example: add a calendar effect (holidays) as binary regressor
    df['is_holiday'] = ((df.index.month==1) & (df.index.day==1)).astype(int)
    return df

if __name__ == '__main__':
    df = generate_multivariate_series()
    df.to_csv('generated_data.csv')
    print('Saved generated_data.csv with shape', df.shape)
