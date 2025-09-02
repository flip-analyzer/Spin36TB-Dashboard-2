#!/usr/bin/env python3
"""
Momentum Trading System - Example Usage

This script demonstrates how to use the momentum trading system
following López de Prado's methodology from "Advances in Financial Machine Learning".

Target: Generate $20k/month through systematic momentum trading.
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
from backtesting.backtester import MomentumBacktester, WalkForwardBacktester


def main():
    """Main execution function"""
    print("=" * 60)
    print("MOMENTUM TRADING SYSTEM - EXAMPLE USAGE")
    print("=" * 60)
    
    # 1. DATA ACQUISITION
    print("\n1. Loading market data...")
    
    symbols = ['SPY', 'QQQ', 'IWM', 'XLF', 'XLK']  # Diversified ETF universe
    data_handler = FinancialDataHandler(symbols)
    
    # Fetch 4 years of data
    start_date = '2020-01-01'
    end_date = '2023-12-31'
    data = data_handler.fetch_data(start_date, end_date)
    
    # Focus on SPY for this example
    spy_data = data['SPY']
    print(f"   ✓ Loaded {len(spy_data)} observations for SPY")
    
    # Validate data quality
    quality = data_handler.validate_data_quality('SPY')
    print(f"   ✓ Data quality: {quality['data_completeness']:.1%} complete")
    
    # 2. FEATURE ENGINEERING
    print("\n2. Engineering momentum features...")
    
    feature_engineer = MomentumFeatureEngineer(
        lookback_periods=[5, 10, 21, 63, 126],
        volatility_windows=[10, 21, 63],
        use_fractional_diff=True
    )
    
    features = feature_engineer.create_momentum_features(
        prices=spy_data['Close'],
        volume=spy_data['Volume']
    )
    print(f"   ✓ Created {features.shape[1]} momentum features")
    
    # 3. TRIPLE BARRIER LABELING
    print("\n3. Generating labels with triple barrier method...")
    
    labeler = TripleBarrierLabeler(
        profit_taking_multiple=1.5,
        stop_loss_multiple=1.0,
        min_return=0.005  # 0.5% minimum return threshold
    )
    
    # Calculate dynamic volatility for barriers
    returns = data_handler.get_returns('SPY')
    volatility = data_handler.get_volatility('SPY', window=21)
    
    # Generate events
    t_events = spy_data.index[63:]  # Skip initial warmup period
    events = labeler.get_events(
        close=spy_data['Close'],
        t_events=t_events,
        pt_sl=[1.5, 1.0],  # Asymmetric barriers favor profits
        target=volatility * 2,  # 2x volatility barriers
        min_ret=0.005
    )
    
    labeled_events = labeler.get_bins(events, spy_data['Close'])
    print(f"   ✓ Generated {len(labeled_events)} labeled events")
    
    # Analyze label distribution
    label_dist = labeler.analyze_label_distribution(labeled_events)
    print(f"   ✓ Label distribution: {label_dist['label_percentages']}")
    
    # 4. PREPARE TRAINING DATA
    print("\n4. Preparing training dataset...")
    
    # Align features and labels
    common_index = features.index.intersection(labeled_events.index)
    X = features.loc[common_index].fillna(method='ffill').fillna(0)
    y = labeled_events.loc[common_index]['bin']
    
    # Create sample weights (time decay)
    sample_weights = labeler.get_sample_weights(
        labeled_events.loc[common_index],
        spy_data['Close'],
        method='time_decay'
    )
    
    print(f"   ✓ Training data: X={X.shape}, y={y.shape}")
    
    # 5. MODEL TRAINING WITH PURGED CROSS-VALIDATION
    print("\n5. Training ML models with proper validation...")
    
    # Create ensemble of models
    models = [
        MomentumMLModel(model_type='random_forest', n_features_select=20),
        MomentumMLModel(model_type='gradient_boosting', n_features_select=15),
        MomentumMLModel(model_type='logistic', n_features_select=25)
    ]
    
    # Validate each model
    validator = TimeSeriesValidator()
    cv_results = {}
    
    for i, model in enumerate(models):
        model_name = f"Model_{i+1}_{model.model_type}"
        print(f"   Training {model_name}...")
        
        # Purged cross-validation
        results = validator.validate_model(
            model=model,
            X=X,
            y=y,
            cv_method='purged_kfold',
            n_splits=5,
            sample_weights=sample_weights,
            embargo_pct=0.01,
            purge_pct=0.02
        )
        
        cv_results[model_name] = results
        accuracy = results['accuracy'].mean()
        f1 = results['f1'].mean()
        print(f"     ✓ CV Accuracy: {accuracy:.3f}, F1: {f1:.3f}")
    
    # Train final ensemble
    ensemble = EnsembleModel(models, weights=[0.4, 0.4, 0.2])
    ensemble.fit(X, y, sample_weights=sample_weights)
    print("   ✓ Ensemble model trained")
    
    # 6. BACKTESTING
    print("\n6. Running backtesting with realistic constraints...")
    
    # Generate signals
    signals = pd.Series(ensemble.predict(X), index=X.index)
    probabilities = pd.DataFrame(
        ensemble.predict_proba(X), 
        index=X.index
    ).max(axis=1)  # Use max probability as confidence
    
    # Initialize backtester with realistic parameters
    backtester = MomentumBacktester(
        initial_capital=250000,  # $250k starting capital
        transaction_cost=0.0015,  # 0.15% transaction cost
        market_impact=0.0008,     # 0.08% market impact
        max_position_size=0.25,   # 25% max position
        risk_free_rate=0.025      # 2.5% risk-free rate
    )
    
    # Run backtest
    backtest_results = backtester.backtest(
        signals=signals,
        prices=spy_data.loc[signals.index],
        probabilities=probabilities,
        position_sizing_method='kelly'
    )
    
    print("   ✓ Backtest completed")
    
    # 7. PERFORMANCE ANALYSIS
    print("\n7. Performance Analysis")
    print("-" * 40)
    
    # Key metrics
    total_return = backtest_results['total_return']
    annual_return = backtest_results['annualized_return']
    sharpe_ratio = backtest_results['sharpe_ratio']
    max_drawdown = backtest_results['max_drawdown']
    win_rate = backtest_results['win_rate']
    
    print(f"Total Return:        {total_return:.2%}")
    print(f"Annualized Return:   {annual_return:.2%}")
    print(f"Sharpe Ratio:        {sharpe_ratio:.2f}")
    print(f"Max Drawdown:        {max_drawdown:.2%}")
    print(f"Win Rate:            {win_rate:.2%}")
    print(f"Number of Trades:    {backtest_results['num_trades']}")
    
    # Calculate monthly performance target
    monthly_return = (1 + annual_return) ** (1/12) - 1
    monthly_dollar_return = backtester.initial_capital * monthly_return
    
    print(f"\nMonthly Performance:")
    print(f"Expected Monthly Return: {monthly_return:.2%}")
    print(f"Expected Monthly $ Return: ${monthly_dollar_return:,.2f}")
    
    # Check if target is achievable
    target_monthly = 20000  # $20k target
    required_capital = target_monthly / monthly_return if monthly_return > 0 else float('inf')
    
    print(f"\nTarget Analysis:")
    print(f"Target Monthly Return: ${target_monthly:,.2f}")
    print(f"Required Capital:      ${required_capital:,.2f}")
    
    if required_capital <= 500000:  # Reasonable capital requirement
        print("✓ Target is ACHIEVABLE with reasonable capital")
    else:
        print("⚠ Target requires HIGH capital or strategy optimization")
    
    # 8. RISK ASSESSMENT
    print("\n8. Risk Assessment")
    print("-" * 40)
    
    # Calculate Value at Risk (VaR)
    portfolio_returns = pd.Series(backtest_results.get('returns', []))
    if len(portfolio_returns) > 0:
        var_95 = np.percentile(portfolio_returns, 5)
        var_99 = np.percentile(portfolio_returns, 1)
        print(f"1-day VaR (95%):     {var_95:.2%}")
        print(f"1-day VaR (99%):     {var_99:.2%}")
    
    # Kelly criterion position sizing validation
    print(f"Average Position Size: {np.mean([abs(t.get('target_position', 0)) for t in backtester.trades]):.2%}")
    print(f"Max Position Size:     {backtest_results.get('max_position', 0):.2%}")
    
    # Transaction cost analysis
    cost_drag = backtest_results['cost_drag']
    print(f"Transaction Cost Drag: {cost_drag:.2%}")
    
    # 9. SYSTEM VALIDATION SUMMARY
    print("\n9. System Validation Summary")
    print("=" * 40)
    
    validation_criteria = {
        'Positive Sharpe Ratio': sharpe_ratio > 0,
        'Reasonable Drawdown': max_drawdown > -0.30,  # Less than 30%
        'Sufficient Win Rate': win_rate > 0.45,       # Above 45%
        'Adequate Sample Size': backtest_results['num_trades'] > 50,
        'Profitable Strategy': annual_return > 0.05,  # Above 5%
        'Low Cost Drag': cost_drag < 0.10             # Below 10%
    }
    
    passed_tests = sum(validation_criteria.values())
    total_tests = len(validation_criteria)
    
    print(f"Validation Tests Passed: {passed_tests}/{total_tests}")
    
    for criterion, passed in validation_criteria.items():
        status = "✓" if passed else "✗"
        print(f"  {status} {criterion}")
    
    # Final recommendation
    print(f"\nFINAL RECOMMENDATION:")
    if passed_tests >= 5:
        print("✓ System shows STRONG potential for live trading")
        print("✓ Consider paper trading before live deployment")
        print("✓ Monitor performance and adjust parameters as needed")
    else:
        print("⚠ System needs OPTIMIZATION before live trading")
        print("⚠ Consider additional features or parameter tuning")
        print("⚠ Extend validation period or improve risk management")
    
    # 10. NEXT STEPS
    print(f"\n10. Next Steps for $20k/Month Target")
    print("-" * 40)
    print("1. Scale capital to required level")
    print("2. Implement real-time data feeds") 
    print("3. Set up execution infrastructure")
    print("4. Create monitoring and alerting system")
    print("5. Plan for regime changes and model updates")
    print("6. Consider multiple asset classes for diversification")
    print("7. Implement dynamic position sizing based on volatility")
    print("8. Set up risk management controls and stop-losses")
    
    print(f"\n{'='*60}")
    print("System validation completed successfully!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()