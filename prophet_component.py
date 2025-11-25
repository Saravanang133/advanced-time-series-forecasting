"""prophet_component.py

Helpers to fit Prophet models on a target series and produce forecasts.
"""
from prophet import Prophet
import pandas as pd

def fit_prophet(y, periods, freq='D', regressors=None, yearly_seasonality=True, weekly_seasonality=True):
    # y: pandas Series indexed by date
    df = y.reset_index().rename(columns={y.name: 'y', y.index.name or 'index': 'ds'})
    if 'ds' not in df.columns:
        df = df.rename(columns={y.index.name: 'ds'})
    df = pd.DataFrame({'ds': y.index, 'y': y.values})
    m = Prophet(yearly_seasonality=yearly_seasonality, weekly_seasonality=weekly_seasonality, daily_seasonality=False)
    if regressors is not None:
        for reg in regressors:
            m.add_regressor(reg)
        # prepare df with regressors
        pass
    m.fit(df)
    future = m.make_future_dataframe(periods=periods, freq=freq)
    forecast = m.predict(future)
    return m, forecast
