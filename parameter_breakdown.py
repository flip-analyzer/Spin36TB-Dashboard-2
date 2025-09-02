#!/usr/bin/env python3
"""
Parameter Breakdown - Show exactly what we're using
"""

import yfinance as yf
import pandas as pd
import numpy as np

def show_current_parameters():
    """Show all parameters currently being used"""
    
    print("🔍 CURRENT PARAMETER BREAKDOWN")
    print("=" * 50)
    
    print("\n📊 DATASET PARAMETERS:")
    print("   Symbol: SPY (S&P 500 ETF)")
    print("   Time Period: 2 years (period='2y')")
    print("   Interval: Daily (interval='1d')")
    print("   Train/Test Split: 80% train, 20% test")
    
    # Show actual data being used
    spy = yf.Ticker("SPY")
    data = spy.history(period="2y", interval="1d")
    
    print(f"\n📅 ACTUAL DATA LOADED:")
    print(f"   Start Date: {data.index[0].date()}")
    print(f"   End Date: {data.index[-1].date()}")
    print(f"   Total Days: {len(data)}")
    print(f"   Training Days: {int(len(data) * 0.8)}")
    print(f"   Testing Days: {len(data) - int(len(data) * 0.8)}")
    
    print(f"\n🪟 PATTERN WINDOW PARAMETERS:")
    print("   Window Size: 30 candles")
    print("   Pattern Vector Length: 150 elements")
    print("     - 30 normalized close prices")
    print("     - 30 candle body sizes")  
    print("     - 30 upper wick sizes")
    print("     - 30 lower wick sizes")
    print("     - 30 normalized volumes")
    
    print(f"\n🎯 SIMILARITY MATCHING PARAMETERS:")
    print("   Similarity Method: Pearson Correlation")
    print("   Minimum Similarity: 0.7 (70%)")
    print("   Top Patterns Considered: 20")
    print("   Minimum Similar Patterns Required: 5")
    
    print(f"\n📈 PREDICTION PARAMETERS:")
    print("   Future Prediction Window: 5 days ahead")
    print("   Strong Move Threshold: ±1.5%")
    print("   Expected Return Threshold: ±1.0%")
    print("   Confidence Calculation: (strong_moves / total_similar_patterns)")
    
    print(f"\n💰 POSITION SIZING PARAMETERS:")
    print("   Minimum Confidence to Trade: 60%")
    print("   Maximum Position Size: 25%")
    print("   Position Sizing Formula: max_pos × ((confidence - 0.5) / 0.5)")
    print("   Example: 65% confidence = 25% × ((0.65 - 0.5) / 0.5) = 7.5% position")
    
    print(f"\n🔄 BACKTESTING PARAMETERS:")
    print("   Starting Capital: $100,000")
    print("   Hold Period: 5 days")
    print("   Transaction Costs: Not included (assumes perfect execution)")
    print("   Slippage: Not included")
    print("   Win Rate Calculation: Direction prediction accuracy")
    
    # Show sample pattern creation
    print(f"\n🧮 SAMPLE PATTERN CREATION:")
    sample_window = data.tail(30)
    
    print(f"   Sample Window: {sample_window.index[0].date()} to {sample_window.index[-1].date()}")
    print(f"   Base Price (normalization): ${sample_window['Close'].iloc[0]:.2f}")
    print(f"   Price Range in Window: ${sample_window['Close'].min():.2f} - ${sample_window['Close'].max():.2f}")
    print(f"   Volume Range: {sample_window['Volume'].min():,} - {sample_window['Volume'].max():,}")
    
    # Create actual pattern vector
    base_price = sample_window['Close'].iloc[0]
    closes = (sample_window['Close'] / base_price).values
    opens = (sample_window['Open'] / base_price).values
    highs = (sample_window['High'] / base_price).values
    lows = (sample_window['Low'] / base_price).values
    volumes = (sample_window['Volume'] / sample_window['Volume'].mean()).values
    
    body_sizes = np.abs(closes - opens)
    upper_wicks = highs - np.maximum(closes, opens)
    lower_wicks = np.minimum(closes, opens) - lows
    
    pattern_vector = np.concatenate([closes, body_sizes, upper_wicks, lower_wicks, volumes])
    
    print(f"\n   Pattern Vector Stats:")
    print(f"   Vector Length: {len(pattern_vector)}")
    print(f"   Value Range: {pattern_vector.min():.3f} to {pattern_vector.max():.3f}")
    print(f"   Mean: {pattern_vector.mean():.3f}")
    print(f"   Std: {pattern_vector.std():.3f}")
    
    return {
        'data_params': {
            'symbol': 'SPY',
            'period': '2y',
            'interval': '1d',
            'train_split': 0.8
        },
        'pattern_params': {
            'window_size': 30,
            'vector_length': 150,
            'components': ['closes', 'body_sizes', 'upper_wicks', 'lower_wicks', 'volumes']
        },
        'similarity_params': {
            'method': 'pearson_correlation',
            'min_similarity': 0.7,
            'top_k': 20,
            'min_required': 5
        },
        'prediction_params': {
            'future_days': 5,
            'strong_move_threshold': 0.015,
            'expected_return_threshold': 0.01
        },
        'position_params': {
            'min_confidence': 0.6,
            'max_position': 0.25
        },
        'backtest_params': {
            'starting_capital': 100000,
            'hold_period': 5,
            'transaction_costs': 0,
            'slippage': 0
        }
    }


def suggest_parameter_experiments():
    """Suggest different parameters to try"""
    
    print(f"\n🧪 PARAMETER EXPERIMENTS TO TRY:")
    print("=" * 40)
    
    experiments = [
        {
            'name': 'Shorter Windows',
            'changes': {'window_size': 15, 'description': 'Try 15-day patterns instead of 30'},
            'rationale': 'Shorter patterns might be more predictive in fast markets'
        },
        {
            'name': 'Longer Windows', 
            'changes': {'window_size': 45, 'description': 'Try 45-day patterns instead of 30'},
            'rationale': 'Longer patterns might capture bigger trends'
        },
        {
            'name': 'Looser Similarity',
            'changes': {'min_similarity': 0.6, 'description': 'Accept 60% similarity instead of 70%'},
            'rationale': 'More similar patterns = more trading opportunities'
        },
        {
            'name': 'Tighter Similarity',
            'changes': {'min_similarity': 0.8, 'description': 'Require 80% similarity instead of 70%'},
            'rationale': 'Higher quality matches = better predictions'
        },
        {
            'name': 'Faster Predictions',
            'changes': {'future_days': 3, 'description': 'Predict 3 days ahead instead of 5'},
            'rationale': 'Shorter timeframe might be more predictable'
        },
        {
            'name': 'Slower Predictions',
            'changes': {'future_days': 10, 'description': 'Predict 10 days ahead instead of 5'},
            'rationale': 'Longer timeframe might smooth out noise'
        },
        {
            'name': 'More Aggressive',
            'changes': {'min_confidence': 0.55, 'max_position': 0.4, 'description': 'Lower confidence threshold, bigger positions'},
            'rationale': 'Take more trades with bigger size'
        },
        {
            'name': 'More Conservative',
            'changes': {'min_confidence': 0.7, 'max_position': 0.15, 'description': 'Higher confidence threshold, smaller positions'},
            'rationale': 'Only take highest confidence trades'
        }
    ]
    
    for i, exp in enumerate(experiments, 1):
        print(f"{i}. {exp['name']}:")
        print(f"   Change: {exp['changes']['description']}")
        print(f"   Why: {exp['rationale']}")
        print()
    
    return experiments


def show_data_quality():
    """Show the quality of data we're using"""
    
    print(f"\n📋 DATA QUALITY ANALYSIS:")
    print("=" * 30)
    
    spy = yf.Ticker("SPY")
    data = spy.history(period="2y", interval="1d")
    
    print(f"Missing Data:")
    print(f"   Open: {data['Open'].isnull().sum()} missing")
    print(f"   High: {data['High'].isnull().sum()} missing") 
    print(f"   Low: {data['Low'].isnull().sum()} missing")
    print(f"   Close: {data['Close'].isnull().sum()} missing")
    print(f"   Volume: {data['Volume'].isnull().sum()} missing")
    
    print(f"\nPrice Statistics:")
    print(f"   Min Close: ${data['Close'].min():.2f}")
    print(f"   Max Close: ${data['Close'].max():.2f}")
    print(f"   Average Daily Range: {((data['High'] - data['Low']) / data['Close'] * 100).mean():.2f}%")
    
    print(f"\nVolume Statistics:")
    print(f"   Min Volume: {data['Volume'].min():,}")
    print(f"   Max Volume: {data['Volume'].max():,}")
    print(f"   Average Volume: {data['Volume'].mean():,.0f}")
    
    # Check for gaps or unusual moves
    daily_returns = data['Close'].pct_change()
    big_moves = daily_returns[abs(daily_returns) > 0.05]  # >5% moves
    
    print(f"\nUnusual Market Days:")
    print(f"   Big moves (>5%): {len(big_moves)}")
    if len(big_moves) > 0:
        print(f"   Biggest up day: {daily_returns.max():.2%}")
        print(f"   Biggest down day: {daily_returns.min():.2%}")


if __name__ == "__main__":
    params = show_current_parameters()
    experiments = suggest_parameter_experiments() 
    show_data_quality()
    
    print(f"\n❓ WHICH PARAMETERS WOULD YOU LIKE TO EXPERIMENT WITH?")
    print(f"   We can try any of the experiments above, or")
    print(f"   You can suggest your own parameter changes!")