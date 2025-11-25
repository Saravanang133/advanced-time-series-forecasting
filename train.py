"""train.py

Main pipeline to:
- generate or load data
- run expanding-window cross-validation
- fit LSTM on residuals and Prophet for trend/seasonality
- combine forecasts and evaluate

This script is intentionally simple and modular. Adjust hyperparameters at the top.
"""
import os
import numpy as np
import pandas as pd
from data_generator import generate_multivariate_series
from prophet_component import fit_prophet
from evaluate import rmse, mase
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import joblib

# Hyperparameters
TARGET = 'target_1'
EXOG_COLS = ['exog_1', 'is_holiday']
SEQ_LEN = 30
BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODELS_DIR = 'models'
os.makedirs(MODELS_DIR, exist_ok=True)

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.astype('float32')
        self.y = y.astype('float32')
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTMModel(nn.Module):
    def __init__(self, n_features, hidden=64, n_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden, num_layers=n_layers, batch_first=True)
        self.fc = nn.Linear(hidden, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

def create_sequences(series, exog, seq_len=30):
    X, y = [], []
    for i in range(len(series)-seq_len):
        seq_x = np.column_stack([series[i:i+seq_len], exog[i:i+seq_len]])
        X.append(seq_x)
        y.append(series[i+seq_len])
    return np.array(X), np.array(y)

def train_lstm(X_train, y_train, X_val, y_val, n_features):
    scaler_x = StandardScaler()
    shp = X_train.shape
    X_train_flat = X_train.reshape(-1, X_train.shape[-1])
    X_train_s = scaler_x.fit_transform(X_train_flat).reshape(shp)
    X_val_flat = X_val.reshape(-1, X_val.shape[-1])
    X_val_s = scaler_x.transform(X_val_flat).reshape(X_val.shape)
    joblib.dump(scaler_x, os.path.join(MODELS_DIR, 'scaler_x.joblib'))

    train_ds = TimeSeriesDataset(X_train_s, y_train)
    val_ds = TimeSeriesDataset(X_val_s, y_val)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = LSTMModel(n_features=n_features).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE).unsqueeze(-1)
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()
            train_loss += loss.item()
        # validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE).unsqueeze(-1)
                out = model(xb)
                val_loss += loss_fn(out, yb).item()
        print(f'Epoch {epoch+1}/{EPOCHS} train_loss={train_loss/len(train_loader):.4f} val_loss={val_loss/len(val_loader):.4f}')

    torch.save(model.state_dict(), os.path.join(MODELS_DIR, 'lstm.pt'))
    return model

def expanding_window_cv(df, n_splits=3):
    # Simple expanding window: each split increases train window and keeps a fixed test window
    results = []
    total = len(df)
    test_size = 90
    start = int(total*0.4)
    for split in range(n_splits):
        train_end = start + split*test_size
        test_end = train_end + test_size
        if test_end > total:
            break
        train = df.iloc[:train_end]
        test = df.iloc[train_end:test_end]
        results.append((train, test))
    return results

def main():
    df = generate_multivariate_series(periods=365*4)
    df.to_csv('generated_data.csv', index=True)
    splits = expanding_window_cv(df, n_splits=3)
    # We'll train on target_1 only for the example
    for i, (train, test) in enumerate(splits):
        print('Split', i, 'train', train.shape, 'test', test.shape)
        # Prophet on train target to capture trend/seasonality
        from prophet_component import fit_prophet
        m, forecast = fit_prophet(train[TARGET], periods=len(test))
        # prophet forecast 'yhat' aligned to test index
        prophet_pred = forecast['yhat'].values[-len(test):]

        # residuals on train
        train_resid = train[TARGET].values - m.predict(m.history)['yhat'].values

        # Prepare sequences for LSTM on residuals
        exog_train = train[EXOG_COLS].values
        exog_test = test[EXOG_COLS].values
        seq_len = SEQ_LEN
        X_train, y_train = create_sequences(train_resid, exog_train, seq_len=seq_len)
        # For validation we take the last seq_len portion of train to predict first test point (quick example)
        X_val, y_val = X_train[-200:], y_train[-200:]
        X_train, y_train = X_train[:-200], y_train[:-200]

        n_features = X_train.shape[-1]
        model = train_lstm(X_train, y_train, X_val, y_val, n_features=n_features)

        # To form hybrid forecast: use prophet_pred + lstm_pred_on_residuals (naive demonstration)
        # For a real pipeline you'd iteratively roll forward; here we show a simple vectorized approximation.
        # Prepare last window sequence from train residuals + exog to predict test residuals
        last_window = train_resid[-seq_len:]
        last_exog = exog_test[:len(test)]
        # This is a simplified demonstration: we create fake sequences by sliding last_window with test exog
        X_test_seq = []
        for j in range(len(test)):
            # build sequence where the latter part uses previous predictions (approx)
            seq_series = np.concatenate([last_window[-(seq_len-j):], np.zeros(j)]) if j>0 else last_window
            # to keep shapes consistent, pair with exog slices (pad with zeros if needed)
            ex_slice = np.vstack([np.zeros((max(0, seq_len-len(last_exog)), last_exog.shape[1])), last_exog[:seq_len]])
            X_test_seq.append(np.column_stack([seq_series, ex_slice[:seq_len]]))
        X_test_seq = np.array(X_test_seq).astype('float32')
        # Load scaler and model to predict
        import joblib
        scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler_x.joblib'))
        shp = X_test_seq.shape
        X_test_flat = X_test_seq.reshape(-1, X_test_seq.shape[-1])
        X_test_s = scaler.transform(X_test_flat).reshape(shp)

        model.eval()
        with torch.no_grad():
            X_t = torch.from_numpy(X_test_s).to(torch.float32).to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            preds = model(X_t).cpu().numpy().ravel()

        hybrid_pred = prophet_pred[:len(preds)] + preds

        # Evaluate
        print('RMSE (Prophet only):', rmse(test[TARGET].values[:len(prophet_pred)], prophet_pred))
        print('RMSE (Hybrid):', rmse(test[TARGET].values[:len(hybrid_pred)], hybrid_pred))
        print('MASE (Hybrid):', mase(test[TARGET].values[:len(hybrid_pred)], hybrid_pred, train[TARGET].values, seasonality=7))

if __name__ == '__main__':
    main()
