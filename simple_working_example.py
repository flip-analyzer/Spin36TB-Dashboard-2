#!/usr/bin/env python3
"""
Simple Working Momentum Trading System

This creates a minimal but working example to demonstrate the concepts.
"""

import sys
import os
sys.path.append('momentum_trading/src')

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from data.data_handler import FinancialDataHandler
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def create_simple_momentum_system():
    """Create a simple but working momentum system"""
    print("📈 SIMPLE MOMENTUM TRADING SYSTEM")
    print("=" * 40)
    
    # 1. Load data
    print("\n1. Loading Data...")
    data_handler = FinancialDataHandler(['SPY'])
    data = data_handler.fetch_data('2020-01-01', '2023-12-31')
    spy_data = data['SPY'].dropna()
    
    prices = spy_data['Close']
    volume = spy_data['Volume']
    
    print(f"   ✓ Loaded {len(prices)} price observations")
    
    # 2. Create simple features manually
    print("\n2. Creating Simple Features...")
    
    # Basic returns
    returns_1d = prices.pct_change(1)
    returns_5d = prices.pct_change(5)
    returns_21d = prices.pct_change(21)
    
    # Moving averages
    ma_5 = prices.rolling(5).mean()
    ma_21 = prices.rolling(21).mean()
    ma_63 = prices.rolling(63).mean()
    
    # Price relative to MA
    price_ma5_ratio = prices / ma_5
    price_ma21_ratio = prices / ma_21
    ma5_ma21_ratio = ma_5 / ma_21
    
    # Volatility
    vol_21 = returns_1d.rolling(21).std()
    vol_63 = returns_1d.rolling(63).std()
    vol_ratio = vol_21 / vol_63
    
    # Volume features
    vol_ma = volume.rolling(21).mean()
    volume_ratio = volume / vol_ma
    
    # RSI-like momentum
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rsi = 100 - (100 / (1 + gain / loss))
    
    # Combine features
    features = pd.DataFrame({
        'returns_1d': returns_1d,
        'returns_5d': returns_5d,
        'returns_21d': returns_21d,
        'price_ma5_ratio': price_ma5_ratio,
        'price_ma21_ratio': price_ma21_ratio,
        'ma5_ma21_ratio': ma5_ma21_ratio,
        'vol_21': vol_21,
        'vol_ratio': vol_ratio,
        'volume_ratio': volume_ratio,
        'rsi': rsi
    })
    
    # Clean features
    features = features.dropna()
    
    print(f"   ✓ Created {len(features.columns)} features")
    print(f"   ✓ {len(features)} clean samples")
    
    # 3. Create simple labels
    print("\n3. Creating Labels...")
    
    # Simple future return labeling
    future_returns = returns_1d.shift(-5)  # 5-day future return
    
    # Create labels: 1 for positive, 0 for negative
    labels = (future_returns > 0.005).astype(int)  # 0.5% threshold
    labels[future_returns < -0.005] = -1  # -1 for significant negative
    
    # Align with features
    common_index = features.index.intersection(labels.index)
    X = features.loc[common_index]
    y = labels.loc[common_index]
    
    # Final cleaning
    valid_mask = ~y.isna()
    X = X[valid_mask]
    y = y[valid_mask]
    
    print(f"   ✓ Created {len(y)} labeled samples")
    print(f"   ✓ Label distribution: {y.value_counts().to_dict()}")
    
    # 4. Train/Test split
    print("\n4. Training Model...")
    
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train simple model
    model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    
    print(f"   ✓ Train accuracy: {train_score:.3f}")
    print(f"   ✓ Test accuracy: {test_score:.3f}")
    
    # 5. Simple backtesting
    print("\n5. Simple Backtesting...")
    
    # Generate signals
    signals = model.predict(X_test_scaled)
    test_returns = returns_1d.loc[X_test.index]
    
    # Calculate strategy returns (simplified)
    strategy_returns = signals * test_returns * 0.1  # 10% position size
    
    # Performance metrics
    total_return = (1 + strategy_returns).prod() - 1
    volatility = strategy_returns.std() * np.sqrt(252)
    sharpe = (strategy_returns.mean() * 252) / (volatility) if volatility > 0 else 0
    
    # Benchmark (buy and hold)
    benchmark_returns = test_returns
    benchmark_total = (1 + benchmark_returns).prod() - 1
    
    print(f"   ✓ Strategy Total Return: {total_return:.2%}")
    print(f"   ✓ Strategy Sharpe Ratio: {sharpe:.2f}")
    print(f"   ✓ Benchmark (B&H) Return: {benchmark_total:.2%}")
    
    # 6. Feature importance
    print("\n6. Feature Importance:")
    importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    for feature, imp in importance.head(5).items():
        print(f"   {feature}: {imp:.3f}")
    
    # 7. Validation
    print(f"\n7. System Validation:")
    is_profitable = total_return > 0
    beats_benchmark = total_return > benchmark_total
    reasonable_sharpe = sharpe > 0.5
    
    print(f"   {'✅' if is_profitable else '❌'} Profitable: {total_return:.2%}")
    print(f"   {'✅' if beats_benchmark else '❌'} Beats Benchmark")
    print(f"   {'✅' if reasonable_sharpe else '❌'} Decent Sharpe: {sharpe:.2f}")
    
    if is_profitable and reasonable_sharpe:
        print(f"\n   🎉 SUCCESS: Basic system works!")
        print(f"   📈 Ready for optimization and scaling")
        
        # Monthly estimate
        monthly_return = (1 + total_return) ** (1/12) - 1
        
        # Estimate required capital for $20k/month
        if monthly_return > 0:
            required_capital = 20000 / monthly_return
            print(f"\n   💰 For $20k/month target:")
            print(f"   Monthly return: {monthly_return:.2%}")
            print(f"   Required capital: ${required_capital:,.0f}")
            
            if required_capital < 1000000:  # Under $1M
                print(f"   ✅ Target achievable with reasonable capital")
            else:
                print(f"   ⚠️ Need optimization or more capital")
        
        return {
            'status': 'working',
            'model': model,
            'scaler': scaler,
            'features': X.columns.tolist(),
            'performance': {
                'total_return': total_return,
                'sharpe_ratio': sharpe,
                'monthly_return': monthly_return if 'monthly_return' in locals() else None
            }
        }
    else:
        print(f"\n   ⚠️ System needs improvement")
        return {'status': 'needs_work'}


if __name__ == "__main__":
    result = create_simple_momentum_system()
    
    if result and result['status'] == 'working':
        print(f"\n🚀 NEXT STEPS TO $20K/MONTH:")
        print(f"1. Add more sophisticated features")
        print(f"2. Implement proper triple barrier labeling")
        print(f"3. Add multi-asset diversification")
        print(f"4. Implement better risk management")
        print(f"5. Scale up with more data")
        print(f"6. Create live trading infrastructure")
        
        print(f"\n💡 Foundation is solid - ready to build!")
    else:
        print(f"\n🔧 Focus on fixing basic issues first")