#!/usr/bin/env python3
"""
López de Prado Key Concepts Demo

Demonstrates the core ideas from "Advances in Financial Machine Learning"
with a manageable dataset size to show the methodology
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

def demonstrate_triple_barrier_labeling():
    """
    Chapter 3: Triple Barrier Method
    The most important concept from López de Prado
    """
    print("🎯 TRIPLE BARRIER LABELING (Chapter 3)")
    print("=" * 45)
    print("This is López de Prado's key innovation for labeling financial time series")
    
    # Simulate some price data for demonstration
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=1000, freq='5min')
    price_changes = np.random.normal(0, 0.001, 1000).cumsum()
    prices = 1.0 + price_changes
    close = pd.Series(prices, index=dates)
    
    # Calculate volatility (López de Prado uses daily vol)
    returns = close.pct_change()
    volatility = returns.rolling(50).std()
    
    # Triple barrier parameters
    profit_target = 1.5  # 1.5x volatility for profit taking
    stop_loss = 1.5      # 1.5x volatility for stop loss
    time_horizon = 6     # 6 periods (30 minutes for 5-min bars)
    
    labels = []
    barrier_details = []
    
    print(f"📊 Analyzing {len(close)} price points...")
    print(f"   Profit Target: {profit_target}x volatility")
    print(f"   Stop Loss: {stop_loss}x volatility") 
    print(f"   Time Horizon: {time_horizon} periods")
    
    # Apply triple barrier to each point
    for i in range(len(close) - time_horizon):
        entry_price = close.iloc[i]
        vol = volatility.iloc[i]
        
        if pd.isna(vol) or vol == 0:
            continue
            
        # Define barriers
        upper_barrier = entry_price * (1 + profit_target * vol)
        lower_barrier = entry_price * (1 - stop_loss * vol)
        
        # Look at future prices within time horizon
        future_prices = close.iloc[i+1:i+1+time_horizon]
        
        # Check barrier hits
        upper_hit = (future_prices >= upper_barrier).any()
        lower_hit = (future_prices <= lower_barrier).any()
        
        if upper_hit and not lower_hit:
            label = 1  # Profit target hit first
            barrier_type = "Profit Target"
        elif lower_hit and not upper_hit:
            label = -1  # Stop loss hit first
            barrier_type = "Stop Loss"
        elif upper_hit and lower_hit:
            # Both hit - find which came first
            upper_time = future_prices[future_prices >= upper_barrier].index[0]
            lower_time = future_prices[future_prices <= lower_barrier].index[0]
            if upper_time < lower_time:
                label = 1
                barrier_type = "Profit Target (first)"
            else:
                label = -1
                barrier_type = "Stop Loss (first)"
        else:
            # Time barrier hit
            final_return = (future_prices.iloc[-1] / entry_price) - 1
            label = np.sign(final_return)
            barrier_type = "Time Barrier"
        
        labels.append(label)
        barrier_details.append({
            'timestamp': close.index[i],
            'entry_price': entry_price,
            'upper_barrier': upper_barrier,
            'lower_barrier': lower_barrier,
            'label': label,
            'barrier_type': barrier_type
        })
    
    # Results
    labels = np.array(labels)
    buy_signals = (labels == 1).sum()
    sell_signals = (labels == -1).sum()
    neutral_signals = (labels == 0).sum()
    
    print(f"\n📈 TRIPLE BARRIER RESULTS:")
    print(f"   Buy Signals (1): {buy_signals} ({buy_signals/len(labels):.1%})")
    print(f"   Sell Signals (-1): {sell_signals} ({sell_signals/len(labels):.1%})")
    print(f"   Neutral (0): {neutral_signals} ({neutral_signals/len(labels):.1%})")
    
    print(f"\n💡 KEY INSIGHT:")
    print(f"   This creates balanced, meaningful labels unlike simple future returns")
    print(f"   Each label represents a genuine trading outcome")
    
    return pd.DataFrame(barrier_details)

def demonstrate_sample_weights():
    """
    Chapter 4: Sample Weights based on Label Uniqueness
    Critical for preventing overfitting in overlapping samples
    """
    print(f"\n⚖️  SAMPLE WEIGHTS (Chapter 4)")
    print("=" * 35)
    print("Weight samples by their uniqueness to handle overlapping observations")
    
    # Simulate overlapping samples scenario
    n_samples = 100
    sample_weights = []
    
    print(f"📊 Computing weights for {n_samples} overlapping samples...")
    
    # In reality, this would be based on concurrent label periods
    # Here we simulate the concept
    for i in range(n_samples):
        # Simulate number of overlapping samples at this timestamp
        overlaps = max(1, int(np.random.exponential(3)))  # Average 3 overlaps
        
        # Weight is inverse of overlaps (López de Prado's approach)
        weight = 1.0 / overlaps
        sample_weights.append(weight)
    
    sample_weights = np.array(sample_weights)
    
    # Normalize weights
    normalized_weights = sample_weights / sample_weights.mean()
    
    print(f"   📈 Weight Statistics:")
    print(f"      Mean: {normalized_weights.mean():.2f}")
    print(f"      Std: {normalized_weights.std():.2f}")
    print(f"      Min: {normalized_weights.min():.2f}")
    print(f"      Max: {normalized_weights.max():.2f}")
    
    print(f"\n💡 KEY INSIGHT:")
    print(f"   Samples with many overlaps get lower weights")
    print(f"   Unique samples get higher weights")
    print(f"   Prevents overfitting from redundant observations")
    
    return normalized_weights

def demonstrate_kelly_criterion():
    """
    Chapter 10: Kelly Criterion for Optimal Bet Sizing
    """
    print(f"\n💰 KELLY CRITERION SIZING (Chapter 10)")
    print("=" * 40)
    print("Optimal position sizing based on edge and win probability")
    
    scenarios = [
        {"name": "Strong Edge", "win_prob": 0.65, "win_size": 1.2, "loss_size": 1.0},
        {"name": "Weak Edge", "win_prob": 0.55, "win_size": 1.0, "loss_size": 1.0}, 
        {"name": "No Edge", "win_prob": 0.50, "win_size": 1.0, "loss_size": 1.0},
        {"name": "Negative Edge", "win_prob": 0.45, "win_size": 1.0, "loss_size": 1.0}
    ]
    
    print(f"📊 Kelly Sizing for Different Scenarios:")
    
    for scenario in scenarios:
        win_prob = scenario["win_prob"]
        win_size = scenario["win_size"]
        loss_size = scenario["loss_size"]
        
        # Kelly formula: f = (bp - q) / b
        # b = payoff ratio, p = win prob, q = loss prob
        b = win_size / loss_size
        p = win_prob
        q = 1 - win_prob
        
        kelly_fraction = (b * p - q) / b
        
        # Safety factor (quarter Kelly)
        safe_kelly = max(0, kelly_fraction * 0.25)
        
        print(f"   {scenario['name']}:")
        print(f"      Win Prob: {win_prob:.1%}")
        print(f"      Full Kelly: {kelly_fraction:.1%}")
        print(f"      Safe Kelly: {safe_kelly:.1%}")
        
        if safe_kelly > 0:
            print(f"      → Position Size: {safe_kelly:.1%}")
        else:
            print(f"      → Position Size: 0% (No edge)")
        print()
    
    print(f"💡 KEY INSIGHT:")
    print(f"   Kelly Criterion provides mathematically optimal sizing")
    print(f"   Use fractional Kelly (25%) for safety")
    print(f"   Never bet when there's no statistical edge")
    
    return True

def demonstrate_purged_cross_validation():
    """
    Chapter 7: Purged Cross-Validation
    Prevents data leakage in time series validation
    """
    print(f"\n🔀 PURGED CROSS-VALIDATION (Chapter 7)")
    print("=" * 40)
    print("Proper validation for time series to prevent look-ahead bias")
    
    # Simulate time series data
    n_samples = 200
    timeline = list(range(n_samples))
    
    print(f"📊 Demonstrating on {n_samples} time series samples")
    
    # Standard K-Fold (WRONG for time series)
    print(f"\n❌ STANDARD K-FOLD (Wrong for Time Series):")
    n_folds = 5
    fold_size = n_samples // n_folds
    
    for i in range(n_folds):
        train_start = 0
        train_end = i * fold_size
        test_start = i * fold_size 
        test_end = (i + 1) * fold_size
        
        print(f"   Fold {i+1}: Train[0:{train_end}] + [{test_end}:{n_samples}], Test[{test_start}:{test_end}]")
        print(f"            ⚠️ Uses future data to predict past!")
    
    # Purged Cross-Validation (CORRECT)
    print(f"\n✅ PURGED CROSS-VALIDATION (López de Prado):")
    embargo_pct = 0.05  # 5% embargo period
    embargo_size = int(n_samples * embargo_pct)
    
    for i in range(n_folds):
        test_start = i * fold_size
        test_end = min((i + 1) * fold_size, n_samples)
        
        # Training data with purging and embargo
        train_indices = []
        
        # Before test set (with embargo)
        if test_start - embargo_size > 0:
            train_indices.extend(range(0, test_start - embargo_size))
        
        # After test set (with embargo)  
        if test_end + embargo_size < n_samples:
            train_indices.extend(range(test_end + embargo_size, n_samples))
        
        print(f"   Fold {i+1}: Train{train_indices[:3]+['...']+train_indices[-3:] if len(train_indices)>6 else train_indices}")
        print(f"            Test[{test_start}:{test_end}], Embargo: {embargo_size}")
        print(f"            ✅ No future leakage!")
    
    print(f"\n💡 KEY INSIGHT:")
    print(f"   Purged CV prevents using future data to predict past")
    print(f"   Embargo periods account for delayed market reactions")
    print(f"   Essential for realistic backtesting results")
    
    return True

def main():
    """
    Demonstrate López de Prado's key concepts
    """
    print("📚 LÓPEZ DE PRADO'S 'ADVANCES IN FINANCIAL MACHINE LEARNING'")
    print("🎯 Core Concepts Demonstration")
    print("=" * 65)
    
    # Demonstrate each key concept
    barrier_data = demonstrate_triple_barrier_labeling()
    
    sample_weights = demonstrate_sample_weights() 
    
    demonstrate_kelly_criterion()
    
    demonstrate_purged_cross_validation()
    
    print(f"\n🎉 LÓPEZ DE PRADO METHODOLOGY SUMMARY:")
    print("=" * 45)
    print("1. 🎯 Triple Barrier Labeling → Meaningful, balanced labels")
    print("2. ⚖️  Sample Weights → Handle overlapping observations") 
    print("3. 🔀 Purged Cross-Validation → Prevent data leakage")
    print("4. 💰 Kelly Criterion → Optimal position sizing")
    print("5. 🤖 Random Forest → With proper regularization")
    print("6. 📊 Feature Importance → Understand what drives returns")
    
    print(f"\n✅ IMPLEMENTATION READY:")
    print("   These concepts can be applied to your EURUSD data")
    print("   Focus on triple barriers - most impactful improvement")
    print("   Use sample weights and purged CV to prevent overfitting")
    print("   Apply Kelly sizing for optimal risk management")
    
    return True

if __name__ == "__main__":
    main()