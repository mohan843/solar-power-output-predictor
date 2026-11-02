# Solar Power Output Predictor

An explainable renewable-energy machine-learning mini-project that predicts solar power generation from irradiance, temperature, cloud cover, and hour of day.

## Overview

Short-term solar output estimates help operators plan storage, grid supply, and maintenance. This project demonstrates multiple linear regression with standardized weather and solar features, implemented from scratch with a reproducible sample dataset.

## Problem statement

Given environmental readings, estimate photovoltaic output in kilowatts and report training mean absolute error.

## Features

- Multiple linear regression trained with gradient descent.
- Feature standardization for stable optimization.
- Non-negative output handling for physical plausibility.
- CSV sample data, CLI, and unit tests.
- Standard-library-only implementation.

## Technology stack

- Python 3.10+
- Linear regression and gradient descent
- Feature standardization
- CSV, `argparse`, and `math`
- `unittest` for automated testing

## Installation

```bash
git clone https://github.com/mohan843/solar-power-output-predictor.git
cd solar-power-output-predictor
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

No third-party runtime packages are required.

## Execution

```bash
python predictor.py --irradiance 800 --temperature 30 --cloud-cover 5 --hour 12
```

Low-sun example:

```bash
python predictor.py --irradiance 100 --temperature 20 --cloud-cover 85 --hour 8
```

## Testing

```bash
python -m unittest discover -s tests -v
```

## Sample output

```text
Training MAE: 0.48 kW
Predicted solar output: 85.94 kW
```

## Limitations

The sample data is synthetic and small. Real systems need panel orientation, shading, inverter behavior, weather forecasts, outages, uncertainty intervals, and out-of-time validation. Predictions should support operational planning rather than replace safety controls.

## Future improvements

- Add cloud-motion and weather-forecast features.
- Compare with gradient boosting and probabilistic forecasting.
- Add time-based train/test evaluation and prediction intervals.
- Include battery state and grid demand for dispatch planning.
- Build a monitoring dashboard for forecast error and drift.
