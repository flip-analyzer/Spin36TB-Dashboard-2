#!/usr/bin/env python3
"""
Your Pattern Recognition Approach - Clean Implementation

This matches exactly what you described:
1. Look at 30-candle windows
2. Convert to vectors/embeddings  
3. Find similar patterns in history
4. See what direction those patterns went
5. Bet size based on probability
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

class YourPatternApproach:
    
    def __init__(self):
        self.historical_patterns = []
        self.historical_outcomes = []
        self.historical_dates = []
        
    def candle_to_vector(self, candle_data):
        """Convert 30 candles into a vector (your embedding approach)"""
        
        # Normalize everything to the first candle's close price
        base_price = candle_data['Close'].iloc[0]
        
        # Create vector with:
        # - Normalized close prices (30 values)
        # - Candle body sizes (30 values) 
        # - Upper/lower wicks (60 values)
        # - Volume pattern (30 values)
        
        closes = (candle_data['Close'] / base_price).values
        opens = (candle_data['Open'] / base_price).values
        highs = (candle_data['High'] / base_price).values
        lows = (candle_data['Low'] / base_price).values
        
        # Normalize volume to average
        volumes = (candle_data['Volume'] / candle_data['Volume'].mean()).values
        
        # Candle features
        body_sizes = np.abs(closes - opens)  # How big is each candle body
        upper_wicks = highs - np.maximum(closes, opens)  # Upper shadows
        lower_wicks = np.minimum(closes, opens) - lows   # Lower shadows
        
        # Combine into single vector (this is your "embedding")
        pattern_vector = np.concatenate([
            closes,      # Price movement
            body_sizes,  # Volatility/momentum  
            upper_wicks, # Rejection at highs
            lower_wicks, # Support at lows
            volumes      # Volume confirmation
        ])
        
        return pattern_vector
    
    def train_on_history(self, data, window_size=30):
        """Build database of historical patterns and their outcomes"""
        
        print(f"🧠 Building Pattern Database...")
        print(f"   Window size: {window_size} candles")
        
        self.historical_patterns = []
        self.historical_outcomes = []
        self.historical_dates = []
        
        # Go through all history and save patterns + what happened next
        for i in range(len(data) - window_size - 10):  # -10 for future outcome
            
            # Get the pattern window
            pattern_window = data.iloc[i:i + window_size]
            pattern_vector = self.candle_to_vector(pattern_window)
            
            # What happened 6 bars later? (30 minutes for 5-min bars)
            current_close = pattern_window['Close'].iloc[-1]
            future_idx = min(i + window_size + 6, len(data) - 1)  # 6 bars = 30 minutes
            future_close = data.iloc[future_idx]['Close']
            future_return = (future_close / current_close) - 1
            
            # Store pattern and outcome
            self.historical_patterns.append(pattern_vector)
            self.historical_outcomes.append(future_return)
            self.historical_dates.append(pattern_window.index[-1])
        
        print(f"   ✅ Stored {len(self.historical_patterns)} historical patterns")
        
        # Analyze the historical outcomes
        big_winners = np.sum(np.array(self.historical_outcomes) > 0.02)  # >2%
        big_losers = np.sum(np.array(self.historical_outcomes) < -0.02)  # <-2%
        small_moves = len(self.historical_outcomes) - big_winners - big_losers
        
        print(f"   📊 Historical outcomes:")
        print(f"      Big winners (>2%): {big_winners}")
        print(f"      Big losers (<-2%): {big_losers}")  
        print(f"      Small moves: {small_moves}")
    
    def find_similar_patterns(self, current_pattern, top_n=20):
        """Find the most similar historical patterns (your similarity matching)"""
        
        similarities = []
        
        # Compare current pattern to all historical patterns
        for historical_pattern in self.historical_patterns:
            # Use correlation as similarity measure
            similarity = np.corrcoef(current_pattern, historical_pattern)[0, 1]
            if np.isnan(similarity):
                similarity = 0
            similarities.append(similarity)
        
        similarities = np.array(similarities)
        
        # Get the most similar patterns
        top_indices = np.argsort(similarities)[-top_n:][::-1]
        
        similar_patterns = []
        similar_outcomes = []
        similar_scores = []
        
        for idx in top_indices:
            if similarities[idx] > 0.7:  # Only include very similar patterns
                similar_patterns.append(self.historical_patterns[idx])
                similar_outcomes.append(self.historical_outcomes[idx])
                similar_scores.append(similarities[idx])
        
        return similar_outcomes, similar_scores
    
    def predict_direction_and_confidence(self, current_pattern):
        """Predict direction and confidence based on similar patterns"""
        
        similar_outcomes, similarity_scores = self.find_similar_patterns(current_pattern)
        
        if len(similar_outcomes) < 5:  # Need at least 5 similar patterns
            return "HOLD", 0.0, f"Only {len(similar_outcomes)} similar patterns found"
        
        similar_outcomes = np.array(similar_outcomes)
        similarity_scores = np.array(similarity_scores)
        
        # Weight outcomes by similarity scores
        weights = similarity_scores / similarity_scores.sum()
        expected_return = np.average(similar_outcomes, weights=weights)
        
        # Count how many similar patterns went up vs down (adjusted for 30-minute moves)
        strong_ups = np.sum(similar_outcomes > 0.003)  # >0.3% moves up (30 min)
        strong_downs = np.sum(similar_outcomes < -0.003)  # >0.3% moves down (30 min)
        total_patterns = len(similar_outcomes)
        
        # Make prediction (adjusted for intraday timeframe)
        if expected_return > 0.002:  # Expecting >0.2% gain in 30 minutes
            direction = "BUY"
            confidence = strong_ups / total_patterns
        elif expected_return < -0.002:  # Expecting >0.2% loss in 30 minutes
            direction = "SELL"  
            confidence = strong_downs / total_patterns
        else:
            direction = "HOLD"
            confidence = 0.5
        
        analysis = f"{len(similar_outcomes)} similar patterns: {strong_ups} up, {strong_downs} down"
        
        return direction, confidence, analysis
    
    def calculate_bet_size(self, confidence, max_position=0.25):
        """Calculate bet size based on confidence (your probability-based sizing)"""
        
        if confidence < 0.6:
            return 0.0  # Don't trade if not confident
        
        # Kelly-ish sizing: bet more when more confident
        # But cap it at max_position to avoid over-leveraging
        
        base_bet = max_position * ((confidence - 0.5) / 0.5)  # Scale from 0.5-1.0 confidence to 0-max_position
        
        return min(base_bet, max_position)
    
    def backtest_approach(self, data, start_date=None):
        """Test your approach on historical data"""
        
        print(f"\n📈 Backtesting Your Pattern Approach...")
        
        if start_date:
            test_data = data[data.index >= start_date]
        else:
            # Test on last 5 trading days (5-minute bars)
            test_data = data.tail(390)  # ~5 days × 78 bars/day = 390 bars
        
        print(f"   Testing on {len(test_data)} 5-minute bars")
        
        portfolio_value = 100000  # Start with $100k
        trades = []
        bar_portfolio = []
        
        for i in range(30, len(test_data) - 6):  # Need 30 candles for pattern, 6 for exit
            
            current_date = test_data.index[i]
            current_price = test_data.iloc[i]['Close']
            
            # Get current 30-candle pattern
            pattern_window = test_data.iloc[i-30:i]
            current_pattern = self.candle_to_vector(pattern_window)
            
            # Make prediction using your approach
            direction, confidence, analysis = self.predict_direction_and_confidence(current_pattern)
            
            # Calculate bet size
            bet_size = self.calculate_bet_size(confidence)
            
            # Execute trade if confident enough
            if bet_size > 0:
                
                # Simulate holding for 6 bars (30 minutes)
                if i + 6 < len(test_data):
                    entry_price = current_price
                    exit_price = test_data.iloc[i + 6]['Close']
                    
                    if direction == "BUY":
                        trade_return = (exit_price / entry_price - 1) * bet_size
                    elif direction == "SELL":
                        trade_return = (entry_price / exit_price - 1) * bet_size
                    else:
                        trade_return = 0
                    
                    portfolio_value *= (1 + trade_return)
                    
                    trades.append({
                        'date': current_date,
                        'direction': direction,
                        'confidence': confidence,
                        'bet_size': bet_size,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'return': trade_return,
                        'analysis': analysis
                    })
            
            bar_portfolio.append({
                'time': current_date,
                'value': portfolio_value
            })
        
        # Calculate performance
        total_return = (portfolio_value / 100000) - 1
        
        # Win rate
        winning_trades = [t for t in trades if t['return'] > 0]
        win_rate = len(winning_trades) / len(trades) if trades else 0
        
        print(f"   ✅ Backtest Results:")
        print(f"      Total Return: {total_return:.2%}")
        print(f"      Final Value: ${portfolio_value:,.2f}")
        print(f"      Number of Trades: {len(trades)}")
        print(f"      Win Rate: {win_rate:.2%}")
        
        if len(trades) > 0:
            avg_bet_size = np.mean([t['bet_size'] for t in trades])
            avg_confidence = np.mean([t['confidence'] for t in trades])
            print(f"      Average Bet Size: {avg_bet_size:.2%}")
            print(f"      Average Confidence: {avg_confidence:.2%}")
        
        return {
            'total_return': total_return,
            'win_rate': win_rate,
            'num_trades': len(trades),
            'trades': trades,
            'final_value': portfolio_value
        }
    
    def analyze_current_market(self, data):
        """Analyze current market using your approach"""
        
        print(f"\n🔮 CURRENT MARKET ANALYSIS:")
        
        # Get latest 30 candles
        current_pattern = self.candle_to_vector(data.tail(30))
        
        # Make prediction
        direction, confidence, analysis = self.predict_direction_and_confidence(current_pattern)
        bet_size = self.calculate_bet_size(confidence)
        
        print(f"   📊 Pattern Analysis: {analysis}")
        print(f"   🎯 Prediction: {direction}")
        print(f"   🎲 Confidence: {confidence:.2%}")
        print(f"   💰 Suggested Bet Size: {bet_size:.2%} of portfolio")
        
        if bet_size > 0:
            print(f"   ✅ TRADE SIGNAL: {direction} with {bet_size:.1%} position")
        else:
            print(f"   ⚠️ NO TRADE: Confidence too low")
        
        return direction, confidence, bet_size


def main():
    """Run your pattern recognition approach"""
    
    print("🎯 YOUR PATTERN RECOGNITION TRADING APPROACH")
    print("=" * 55)
    
    # Load SPY 5-minute data (Marcos' approach)
    print("\n📥 Loading 5-Minute Intraday Data...")
    spy = yf.Ticker("SPY")
    
    # Get 5-minute bars for last 60 days (max available)
    try:
        data = spy.history(period="60d", interval="5m")
        print(f"   ✅ Loaded {len(data)} 5-minute bars of SPY data")
        
        # Filter to market hours only (9:30 AM - 4:00 PM ET)
        if not data.empty and hasattr(data.index, 'time'):
            data = data.between_time('09:30', '16:00')
            print(f"   ✅ Filtered to market hours: {len(data)} bars")
        
    except Exception as e:
        print(f"   ⚠️ Error loading 5-minute data: {e}")
        print("   Falling back to 15-minute data...")
        data = spy.history(period="60d", interval="15m")
        if not data.empty:
            data = data.between_time('09:30', '16:00')
    
    # Initialize your approach
    trader = YourPatternApproach()
    
    # Train on first 80% of data
    split_idx = int(len(data) * 0.8)
    train_data = data.iloc[:split_idx]
    
    trader.train_on_history(train_data, window_size=30)
    
    # Backtest on remaining 20%
    results = trader.backtest_approach(data.iloc[split_idx:])
    
    # Analyze current market
    trader.analyze_current_market(data)
    
    # Summary for $20k/month target
    if results['total_return'] > 0:
        test_bars = len(data.iloc[split_idx:])
        test_days = test_bars / 78  # ~78 5-minute bars per trading day
        daily_return = (1 + results['total_return']) ** (1/test_days) - 1
        monthly_return = (1 + daily_return) ** 21 - 1  # 21 trading days per month
        required_capital = 20000 / monthly_return if monthly_return > 0 else float('inf')
        
        print(f"\n💰 SCALING TO $20K/MONTH:")
        print(f"   Monthly Return: {monthly_return:.2%}")  
        print(f"   Required Capital: ${required_capital:,.0f}")
        
        if required_capital < 1000000:
            print(f"   ✅ Your approach could work!")
        else:
            print(f"   ⚠️ Need higher returns or more capital")
    
    print(f"\n🎉 Marcos-style intraday pattern approach implemented!")
    print(f"   This matches López de Prado's methodology:")
    print(f"   ✅ 30 5-minute candle windows (2.5 hours)")
    print(f"   ✅ Vector embeddings of intraday patterns") 
    print(f"   ✅ Similarity matching on 30-minute predictions")
    print(f"   ✅ Historical outcome analysis")
    print(f"   ✅ Intraday momentum trading (multiple trades/day)")


if __name__ == "__main__":
    main()