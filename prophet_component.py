"""prophet_component.py
Helpers to fit Prophet models with optional external regressors.
"""
import pandas as pd
from prophet import Prophet

def prepare_prophet_df(y_series, regressors_df=None):
    # y_series: pandas Series indexed by date
    df = pd.DataFrame({'ds': y_series.index, 'y': y_series.values})
    if regressors_df is not None:
        # ensure same index and add regressors columns
        reg = regressors_df.reindex(y_series.index).reset_index(drop=True)
        df = pd.concat([df, reg], axis=1)
    return df

def fit_prophet_with_regressors(y_series, regressors_df=None, periods=0, freq='D',
                                yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False):
    df = prepare_prophet_df(y_series, regressors_df)
    m = Prophet(yearly_seasonality=yearly_seasonality, weekly_seasonality=weekly_seasonality, daily_seasonality=daily_seasonality)
    if regressors_df is not None:
        for col in regressors_df.columns:
            m.add_regressor(col)
    m.fit(df)
    future = m.make_future_dataframe(periods=periods, freq=freq)
    # attach regressors into future if provided (will forward-fill / zero)
    if regressors_df is not None and periods>0:
        reg_future = regressors_df.reindex(pd.DatetimeIndex(future['ds'])).fillna(0).reset_index(drop=True)
        future = pd.concat([future.reset_index(drop=True), reg_future], axis=1)
    forecast = m.predict(future)
    return m, forecast
