#!/usr/bin/env python3
"""
Debug Pattern System - See what patterns we're finding
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def debug_pattern_system():
    """Debug what patterns we're finding and why no trades"""
    
    print("🔍 DEBUGGING PATTERN SYSTEM")
    print("=" * 40)
    
    # Load data
    ticker = yf.Ticker("SPY")
    data = ticker.history(period="2y", interval="1d")
    print(f"Loaded {len(data)} days")
    
    # Create simple pattern embeddings
    window_size = 20  # Shorter window for debugging
    patterns = []
    future_returns = []
    dates = []
    
    for i in range(len(data) - window_size - 5):
        window = data.iloc[i:i + window_size]
        
        # Simple pattern: just use closing prices normalized to first close
        closes = window['Close'].values
        normalized_closes = closes / closes[0]  # Everything relative to first price
        
        patterns.append(normalized_closes)
        dates.append(window.index[-1])
        
        # Future return 5 days later
        future_price = data.iloc[i + window_size + 5]['Close']
        current_price = window['Close'].iloc[-1]
        future_ret = (future_price / current_price) - 1
        future_returns.append(future_ret)
    
    patterns = np.array(patterns)
    future_returns = np.array(future_returns)
    
    print(f"Created {len(patterns)} patterns")
    
    # Analyze what we found
    print(f"\nFuture Returns Analysis:")
    print(f"Mean: {future_returns.mean():.3f}")
    print(f"Std: {future_returns.std():.3f}")
    print(f"Positive (>1%): {(future_returns > 0.01).sum()}")
    print(f"Negative (<-1%): {(future_returns < -0.01).sum()}")
    print(f"Big moves (>2%): {(np.abs(future_returns) > 0.02).sum()}")
    
    # Test similarity matching on recent patterns
    print(f"\n🔍 Testing Recent Pattern Matches:")
    
    # Get last pattern as test
    test_pattern = patterns[-50]  # 50 days ago
    test_return = future_returns[-50]
    
    print(f"Test pattern future return: {test_return:.3%}")
    
    # Find similar patterns
    similarities = []
    for i, pattern in enumerate(patterns[:-50]):  # Don't include future data
        similarity = np.corrcoef(test_pattern, pattern)[0, 1]
        similarities.append(similarity)
    
    similarities = np.array(similarities)
    
    # Top 10 most similar
    top_indices = np.argsort(similarities)[-10:][::-1]
    
    print(f"\nTop 10 Similar Patterns:")
    for i, idx in enumerate(top_indices):
        sim = similarities[idx]
        ret = future_returns[idx]
        print(f"  {i+1}. Similarity: {sim:.3f}, Future Return: {ret:.3%}")
    
    # Calculate prediction
    top_returns = future_returns[top_indices]
    avg_return = top_returns.mean()
    positive_count = (top_returns > 0.01).sum()
    
    print(f"\nPrediction for test pattern:")
    print(f"Average similar return: {avg_return:.3%}")
    print(f"Positive patterns: {positive_count}/10")
    print(f"Actual return was: {test_return:.3%}")
    
    # Show why no trades were made
    print(f"\n🚫 Why No Trades Were Made:")
    print(f"System requires:")
    print(f"  - Similarity > 0.80 (highest we found: {similarities.max():.3f})")
    print(f"  - Confidence > 0.60 (positive patterns: {positive_count/10:.2f})")
    print(f"  - Strong direction (avg return: {avg_return:.3%})")
    
    # Suggest better parameters
    print(f"\n💡 SUGGESTED IMPROVEMENTS:")
    print(f"1. Lower similarity threshold to 0.60-0.70")
    print(f"2. Lower confidence threshold to 0.55") 
    print(f"3. Require smaller moves (>0.5% instead of >1%)")
    print(f"4. Use shorter windows (10-15 candles)")
    print(f"5. Add volume and volatility to patterns")
    
    return patterns, future_returns, similarities

if __name__ == "__main__":
    patterns, returns, similarities = debug_pattern_system()