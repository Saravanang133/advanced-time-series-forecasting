#!/bin/bash
python - <<'PY'
from data_generator import generate_multivariate_series
df = generate_multivariate_series()
print('Generated', df.shape)
df.to_csv('generated_data.csv')
PY

python train.py
