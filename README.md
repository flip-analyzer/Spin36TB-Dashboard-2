# Momentum Trading System

A momentum trading system following López de Prado's methodology from "Advances in Financial Machine Learning".

## Features

- Triple barrier labeling
- Fractional differentiation 
- Purged cross-validation
- Momentum-based feature engineering
- ML-driven signal generation

## Installation

```bash
pip install -r requirements.txt
```

## Project Structure

```
momentum_trading/
├── src/
│   ├── data/           # Data handling and preprocessing
│   ├── features/       # Feature engineering
│   ├── labeling/       # Triple barrier labeling
│   ├── validation/     # Purged cross-validation
│   ├── models/         # ML models
│   └── backtesting/    # Backtesting framework
├── notebooks/          # Research and analysis
└── tests/             # Unit tests
```