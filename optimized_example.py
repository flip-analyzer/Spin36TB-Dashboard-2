#!/usr/bin/env python3
"""
Optimized Momentum Trading System - Improved Version

This addresses the data alignment issues and numerical problems
from the initial example.
"""

import sys
import os
sys.path.append('momentum_trading/src')

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Import system components
from data.data_handler import FinancialDataHandler
from features.fractional_diff import FractionalDifferentiator
from features.momentum_features import MomentumFeatureEngineer
from labeling.triple_barrier import TripleBarrierLabeler
from validation.purged_cv import PurgedKFold, TimeSeriesValidator
from models.ml_models import MomentumMLModel, EnsembleModel
from backtesting.backtester import MomentumBacktester


def create_optimized_system():
    """Create optimized momentum trading system with proper data handling"""
    print("🚀 OPTIMIZED MOMENTUM TRADING SYSTEM")
    print("=" * 50)
    
    # 1. Load more data with better date range
    print("\n1. Loading Extended Market Data...")
    
    data_handler = FinancialDataHandler(['SPY'])
    
    # Get 5 years of data for better statistics
    start_date = '2019-01-01'
    end_date = '2023-12-31'
    data = data_handler.fetch_data(start_date, end_date)
    spy_data = data['SPY']
    
    print(f"   ✓ Loaded {len(spy_data)} observations")
    print(f"   ✓ Date range: {spy_data.index[0].date()} to {spy_data.index[-1].date()}")
    
    # 2. Conservative Feature Engineering
    print("\n2. Creating Robust Features...")
    
    # Use simpler, more robust feature set
    feature_engineer = MomentumFeatureEngineer(
        lookback_periods=[5, 10, 21, 63],  # Reduced complexity
        volatility_windows=[10, 21, 63],
        use_fractional_diff=False  # Disable for initial optimization
    )
    
    features = feature_engineer.create_momentum_features(
        prices=spy_data['Close'],
        volume=spy_data['Volume']
    )
    
    # Clean features more carefully
    print(f"   → Raw features shape: {features.shape}")
    
    # Replace infinite values with NaN
    features = features.replace([np.inf, -np.inf], np.nan)
    
    # Forward fill then backward fill
    features = features.fillna(method='ffill').fillna(method='bfill')
    
    # Only drop rows where ALL values are NaN
    features = features.dropna(how='all')
    
    # For remaining NaN values, fill with column median
    features = features.fillna(features.median())
    
    print(f"   ✓ Created {features.shape[1]} features")
    print(f"   ✓ Clean feature matrix: {features.shape[0]} samples")
    
    # 3. Improved Triple Barrier Labeling
    print("\n3. Optimizing Label Generation...")
    
    labeler = TripleBarrierLabeler(
        profit_taking_multiple=2.0,  # Higher profit target
        stop_loss_multiple=1.0,      # Tighter stop loss
        min_return=0.002             # Lower minimum return (0.2%)
    )
    
    # Calculate more stable volatility
    returns = spy_data['Close'].pct_change().dropna()
    volatility = returns.rolling(21, min_periods=10).std().fillna(returns.std())
    
    # Generate events on a subset to ensure we have enough training data
    start_idx = max(100, len(features) - 1000)  # Use last 1000 days or start from 100
    t_events = spy_data.index[start_idx:]
    
    events = labeler.get_events(
        close=spy_data['Close'],
        t_events=t_events,
        pt_sl=[2.0, 1.0],
        target=volatility * 1.5,  # 1.5x volatility barriers
        min_ret=0.002
    )
    
    labeled_events = labeler.get_bins(events, spy_data['Close'])
    
    if len(labeled_events) == 0:
        print("   ⚠️ No events generated - adjusting parameters...")
        
        # Fallback with more lenient parameters
        events = labeler.get_events(
            close=spy_data['Close'],
            t_events=t_events[::5],  # Every 5th day
            pt_sl=[1.5, 1.5],        # Symmetric barriers
            target=volatility,        # 1x volatility
            min_ret=0.001            # 0.1% minimum
        )
        labeled_events = labeler.get_bins(events, spy_data['Close'])
    
    print(f"   ✓ Generated {len(labeled_events)} labeled events")
    
    # 4. Align Data Properly
    print("\n4. Aligning Training Data...")
    
    # Find common index with sufficient lookback
    common_index = features.index.intersection(labeled_events.index)
    
    if len(common_index) < 100:
        print(f"   ⚠️ Only {len(common_index)} samples - using all available data")
        
        # Use all overlapping data
        X = features.loc[common_index]
        y = labeled_events.loc[common_index]['bin']
    else:
        # Use the most recent data with sufficient history
        X = features.loc[common_index].tail(500)  # Last 500 samples
        y = labeled_events.loc[common_index].tail(500)['bin']
        common_index = X.index
    
    # Final cleaning
    valid_mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[valid_mask]
    y = y[valid_mask]
    
    print(f"   ✓ Final training set: {X.shape[0]} samples, {X.shape[1]} features")
    
    if len(X) < 50:
        print("   ❌ Insufficient training data - need at least 50 samples")
        return None
    
    # 5. Simple Model Training
    print("\n5. Training Simplified Model...")
    
    # Use single robust model instead of ensemble
    model = MomentumMLModel(
        model_type='random_forest',
        use_feature_selection=True,
        n_features_select=min(10, X.shape[1])  # Limit features
    )
    
    # Simple train-test split instead of CV for debugging
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"   ✓ Train: {len(X_train)} samples, Test: {len(X_test)} samples")
    
    # Train model
    model.fit(X_train, y_train)
    
    # Evaluate
    train_acc = model.evaluate(X_train, y_train)['accuracy']
    test_acc = model.evaluate(X_test, y_test)['accuracy'] if len(X_test) > 0 else 0
    
    print(f"   ✓ Train Accuracy: {train_acc:.3f}")
    print(f"   ✓ Test Accuracy: {test_acc:.3f}")
    
    # 6. Conservative Backtesting
    print("\n6. Running Conservative Backtest...")
    
    # Generate signals on test set
    if len(X_test) > 10:
        signals = pd.Series(model.predict(X_test), index=X_test.index)
        probabilities = pd.DataFrame(model.predict_proba(X_test)).max(axis=1)
        probabilities.index = X_test.index
        
        # Conservative backtester settings
        backtester = MomentumBacktester(
            initial_capital=100000,
            transaction_cost=0.001,
            market_impact=0.0005,
            max_position_size=0.1,  # Only 10% max position
            risk_free_rate=0.02
        )
        
        # Run backtest on test period
        test_prices = spy_data.loc[signals.index]
        backtest_results = backtester.backtest(
            signals=signals,
            prices=test_prices,
            probabilities=probabilities,
            position_sizing_method='fixed'  # Use fixed sizing
        )
        
        print("   ✓ Backtest Results:")
        print(f"     Total Return: {backtest_results['total_return']:.2%}")
        print(f"     Sharpe Ratio: {backtest_results['sharpe_ratio']:.2f}")
        print(f"     Max Drawdown: {backtest_results['max_drawdown']:.2%}")
        print(f"     Win Rate: {backtest_results['win_rate']:.2%}")
        print(f"     Number of Trades: {backtest_results['num_trades']}")
        
        # Basic validation
        is_profitable = backtest_results['total_return'] > 0
        is_reasonable = abs(backtest_results['total_return']) < 10  # Within 1000%
        has_trades = backtest_results['num_trades'] > 0
        
        print(f"\n   📊 System Status:")
        print(f"   {'✅' if is_profitable else '❌'} Profitable")
        print(f"   {'✅' if is_reasonable else '❌'} Reasonable Returns")
        print(f"   {'✅' if has_trades else '❌'} Active Trading")
        
        if is_profitable and is_reasonable and has_trades:
            print(f"\n   🎉 SUCCESS: System shows promise for optimization!")
            return {
                'model': model,
                'backtest_results': backtest_results,
                'feature_importance': model.get_feature_importance(),
                'status': 'optimizable'
            }
        else:
            print(f"\n   ⚠️ System needs more work before scaling")
            return {
                'model': model,
                'backtest_results': backtest_results,
                'issues': {
                    'profitable': is_profitable,
                    'reasonable': is_reasonable,
                    'active': has_trades
                },
                'status': 'needs_work'
            }
    else:
        print("   ❌ Insufficient test data for backtesting")
        return None


if __name__ == "__main__":
    result = create_optimized_system()
    
    if result:
        print(f"\n🎯 NEXT STEPS:")
        
        if result['status'] == 'optimizable':
            print("1. ✅ Scale up data (more symbols, longer timeframe)")
            print("2. ✅ Add fractional differentiation back")
            print("3. ✅ Implement ensemble models")
            print("4. ✅ Add multi-asset diversification")
            print("5. ✅ Create paper trading system")
            
        else:
            print("1. 🔧 Fix data quality and alignment issues")
            print("2. 🔧 Tune labeling parameters")
            print("3. 🔧 Simplify feature engineering")
            print("4. 🔧 Add more robust validation")
            
        print(f"\n💡 Ready to proceed with optimization!")
    else:
        print(f"\n❌ System needs fundamental fixes before proceeding")