"""train.py
End-to-end pipeline demonstrating hybrid forecasting with Prophet + LSTM.
This script is intentionally simple but functional. It performs:
- data generation
- expanding-window CV
- Prophet fit on train (with regressors)
- compute residuals on train
- train LSTM on residuals
- iterative forecasting for test horizon where LSTM predicts residuals step-by-step
"""
import os
import numpy as np
import pandas as pd
from data_generator import generate_multivariate_series
from prophet_component import fit_prophet_with_regressors
from evaluate import rmse, mase
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import joblib

# Hyperparameters
TARGET = 'target_1'
EXOG_COLS = ['exog_1', 'is_newyear']
SEQ_LEN = 30
BATCH_SIZE = 64
EPOCHS = 8
LR = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODELS_DIR = 'models'
os.makedirs(MODELS_DIR, exist_ok=True)

class ResidualDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.astype('float32')
        self.y = y.astype('float32')
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

def create_sequences(residuals, exog, seq_len=30):
    X, y = [], []
    for i in range(len(residuals)-seq_len):
        seq = residuals[i:i+seq_len]
        ex = exog[i:i+seq_len]
        X.append(np.column_stack([seq, ex]))
        y.append(residuals[i+seq_len])
    return np.array(X), np.array(y)

def expanding_window_splits(df, initial_train=365*2, test_size=90, n_splits=3):
    splits = []
    start = initial_train
    for i in range(n_splits):
        train_end = start + i*test_size
        test_end = train_end + test_size
        if test_end > len(df):
            break
        train = df.iloc[:train_end]
        test = df.iloc[train_end:test_end]
        splits.append((train, test))
    return splits

def train_lstm(X_train, y_train, X_val, y_val, input_size):
    scaler = StandardScaler()
    X_train_flat = X_train.reshape(-1, X_train.shape[-1])
    X_train_s = scaler.fit_transform(X_train_flat).reshape(X_train.shape)
    X_val_flat = X_val.reshape(-1, X_val.shape[-1])
    X_val_s = scaler.transform(X_val_flat).reshape(X_val.shape)
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.joblib'))

    train_ds = ResidualDataset(X_train_s, y_train)
    val_ds = ResidualDataset(X_val_s, y_val)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = LSTMRegressor(input_size).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE).unsqueeze(-1)
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        # simple val
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE).unsqueeze(-1)
                val_loss += loss_fn(model(xb), yb).item()
        print(f'Epoch {epoch+1}/{EPOCHS} train_loss={total_loss/len(train_loader):.4f} val_loss={val_loss/len(val_loader):.4f}')
    torch.save(model.state_dict(), os.path.join(MODELS_DIR, 'lstm_residuals.pt'))
    return model

def iterative_hybrid_forecast(prophet_model, prophet_history, model_lstm, scaler, last_residuals, exog_future, seq_len=30):
    # prophet_model: fitted Prophet
    # prophet_history: history dataframe used by prophet (ds,y and regressors)
    # model_lstm: trained LSTM model
    # scaler: scaler for LSTM inputs
    # last_residuals: numpy array of residuals up to last train point (length >= seq_len)
    # exog_future: numpy array shape (horizon, n_exog)
    model_lstm.eval()
    preds = []
    resid_history = list(last_residuals[-seq_len:].copy())
    horizon = len(exog_future)
    input_size = 1 + exog_future.shape[1]
    for h in range(horizon):
        # build sequence
        seq_res = np.array(resid_history[-seq_len:])
        ex_slice = exog_future[max(0, h-seq_len+1):h+1]
        # pad ex_slice if short
        if ex_slice.shape[0] < seq_len:
            pad = np.zeros((seq_len - ex_slice.shape[0], ex_slice.shape[1]))
            ex_full = np.vstack([pad, ex_slice])
        else:
            ex_full = ex_slice[-seq_len:]
        X_in = np.column_stack([seq_res, ex_full]).astype('float32')
        X_in_s = scaler.transform(X_in).reshape(1, seq_len, input_size)
        X_t = torch.from_numpy(X_in_s).to(DEVICE)
        with torch.no_grad():
            r_pred = model_lstm(X_t).cpu().numpy().ravel()[0]
        # get prophet point forecast for this horizon (use prophet.predict on future frame)
        # We'll just use prophet_model.predict on the appropriate future ds row created externally by caller.
        preds.append(r_pred)
        resid_history.append(r_pred)  # append predicted residual to history for next step
    return np.array(preds)

def main():
    df = generate_multivariate_series(periods=365*4)
    df.to_csv('generated_data.csv')
    splits = expanding_window_splits(df)
    for si, (train, test) in enumerate(splits):
        print('Split', si, 'train', train.shape, 'test', test.shape)
        # Fit Prophet on train
        m, forecast_full = fit_prophet_with_regressors(train[TARGET], regressors_df=train[EXOG_COLS], periods=len(test))
        # prophet forecast aligned
        prophet_pred = forecast_full['yhat'].values[-len(test):]
        # Compute residuals on train (use prophet.predict on history)
        hist_pred = m.predict(m.history)
        train_resid = train[TARGET].values - hist_pred['yhat'].values
        # Prepare sequences & exog for LSTM
        exog_train = train[EXOG_COLS].values
        exog_test = test[EXOG_COLS].values
        X, y = create_sequences(train_resid, exog_train, seq_len=SEQ_LEN)
        if len(X) < 500:
            # For small example, use 80/20 split
            split_idx = int(len(X)*0.8)
        else:
            split_idx = -200
        X_train, y_train = X[:split_idx], y[:split_idx]
        X_val, y_val = X[split_idx:], y[split_idx:]
        model = train_lstm(X_train, y_train, X_val, y_val, input_size=X.shape[-1])
        # load scaler
        scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.joblib'))
        # iterative hybrid forecast for residuals
        last_resid = train_resid
        model_lstm = model
        residual_preds = iterative_hybrid_forecast(m, m.history, model_lstm, scaler, last_resid, exog_test, seq_len=SEQ_LEN)
        hybrid = prophet_pred[:len(residual_preds)] + residual_preds
        # Evaluate
        truth = test[TARGET].values[:len(hybrid)]
        print('Prophet RMSE:', rmse(truth, prophet_pred[:len(truth)]))
        print('Hybrid RMSE :', rmse(truth, hybrid))
        print('Hybrid MASE :', mase(truth, hybrid, train[TARGET].values, seasonality=7))

if __name__ == '__main__':
    main()
