# Advanced Time Series Forecasting — Prophet + LSTM Hybrid (v2)

This repository contains a complete, runnable scaffold for a **hybrid time series forecasting project**:
- **Prophet** for trend + seasonality
- **PyTorch LSTM** to model residuals (learn complex temporal dependencies)
- **Expanding-window cross-validation** for robust evaluation
- **Iterative forecasting** for hybrid predictions

## Structure
- `data_generator.py` — generate synthetic multivariate daily data (>= 3 years).
- `prophet_component.py` — utilities to fit Prophet with optional regressors.
- `train.py` — end-to-end pipeline: data -> expanding-window CV -> Prophet + LSTM training -> iterative hybrid forecasting -> evaluation.
- `evaluate.py` — RMSE & MASE metrics.
- `requirements.txt` — dependencies.
- `example_run.sh` — run example.
- `report.md` — template report.
- `.gitignore`, `LICENSE` — helpful repo files.

## Quick start
1. Create virtualenv and install:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
2. Run the pipeline (this will generate synthetic data and run a quick training):
```bash
bash example_run.sh
```

## Notes
- This scaffold is tuned for clarity and reproducibility. For production-scale experiments, increase epochs, add checkpoints, more thorough hyperparameter tuning, and consider GPU training.
- If you want, I can run the pipeline here and produce result plots and a completed `report.md` with real numbers and saved models.

