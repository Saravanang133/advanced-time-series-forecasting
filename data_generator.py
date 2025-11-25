"""data_generator.py
Generate multivariate daily series with multiple seasonalities and exogenous regressors.
"""
import numpy as np
import pandas as pd

def generate_multivariate_series(n_series=2, start='2018-01-01', periods=365*4, freq='D', seed=42):
    np.random.seed(seed)
    idx = pd.date_range(start=start, periods=periods, freq=freq)
    t = np.arange(periods)
    df = pd.DataFrame(index=idx)
    # Trend component
    trend = 0.001 * t
    for i in range(n_series):
        yearly = 5.0 * np.sin(2 * np.pi * t / 365.25 * (1 + 0.01*i))
        weekly = 1.0 * np.sin(2 * np.pi * t / 7.0)
        exog = 0.5 * np.random.randn(periods)
        noise = 0.4 * np.random.randn(periods)
        series = 50 + (i+1)*trend + yearly + 0.5*weekly + exog + noise
        df[f'target_{i+1}'] = series
        df[f'exog_{i+1}'] = exog
    # simple binary holiday regressor (New Year)
    df['is_newyear'] = ((df.index.month==1) & (df.index.day==1)).astype(int)
    return df

if __name__ == '__main__':
    df = generate_multivariate_series()
    df.to_csv('generated_data.csv')
    print('Saved generated_data.csv', df.shape)
