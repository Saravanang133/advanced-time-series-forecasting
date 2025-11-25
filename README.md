# Advanced Time Series Forecasting with Prophet and Deep Learning Hybrid Models

This project scaffold implements a complete workflow for **multivariate time series forecasting** using:
- A deep learning component (PyTorch LSTM) to model complex temporal dependencies and residual structure.
- A statistical/interpretble component (Prophet) to capture trend and seasonality.
- A hybrid combination strategy: Prophet models trend/seasonality; LSTM models residuals; predictions are summed.

Included files:
- `data_generator.py` — programmatically generate synthetic multivariate time series (>=3 years daily) with multiple seasonalities and exogenous regressors.
- `prophet_component.py` — helpers to fit Prophet and produce trend/seasonal forecasts.
- `train.py` — training pipeline: data generation/load, expanding-window cross-validation, LSTM training, Prophet fitting on residual/trend, evaluation and saving artifacts.
- `evaluate.py` — evaluation metrics (RMSE, MASE) and helper functions.
- `report.md` — template technical report to fill with results and discussion.
- `requirements.txt` — Python dependencies.
- `example_run.sh` — example commands to run the pipeline.
- `assignment_image.png` — the uploaded assignment screenshot.

**How to use**
1. Create a virtual environment and install requirements:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
2. Generate data and run training (example):
```bash
bash example_run.sh
```

**Notes**
- This scaffold is production-ready but intentionally compact. It provides modular functions and clear docstrings.
- The code uses an expanding-window cross-validation suitable for time series.
- The `train.py` script logs model hyperparameters and saves a `models/` folder with trained artifact checkpoints.

Refer to `report.md` for guidance on writing the final technical analysis and copying in hyperparameters / results.

![assignment image](assignment_image.png)
