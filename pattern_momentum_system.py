#!/usr/bin/env python3
"""
Pattern-Based Momentum Trading System

This implements the user's approach:
1. Look at 30-candle windows  
2. Convert to vectors (embeddings)
3. Find similar patterns in history
4. Predict direction based on what those patterns did
5. Size bets based on confidence
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class PatternMomentumTrader:
    def __init__(self, window_size=30, similarity_threshold=0.85, min_patterns=5):
        """
        Pattern-based momentum trader
        
        Parameters:
        -----------
        window_size : int
            Number of candles to look at for patterns
        similarity_threshold : float
            Minimum similarity to consider a pattern match
        min_patterns : int
            Minimum number of similar patterns needed
        """
        self.window_size = window_size
        self.similarity_threshold = similarity_threshold
        self.min_patterns = min_patterns
        self.scaler = MinMaxScaler()
        self.pca = PCA(n_components=10)  # Reduce dimensions for easier comparison
        
    def prepare_candle_data(self, data):
        """Convert OHLCV data into normalized candle patterns"""
        
        # Calculate relative prices (everything relative to first close)
        patterns = []
        dates = []
        
        for i in range(len(data) - self.window_size - 5):  # -5 for future return
            window = data.iloc[i:i + self.window_size].copy()
            
            # Normalize to first close price
            base_price = window['Close'].iloc[0]
            
            candle_pattern = {
                'open': (window['Open'] / base_price).values,
                'high': (window['High'] / base_price).values, 
                'low': (window['Low'] / base_price).values,
                'close': (window['Close'] / base_price).values,
                'volume': (window['Volume'] / window['Volume'].mean()).values  # Volume relative to average
            }
            
            # Flatten into single vector
            pattern_vector = np.concatenate([
                candle_pattern['open'],
                candle_pattern['high'], 
                candle_pattern['low'],
                candle_pattern['close'],
                candle_pattern['volume']
            ])
            
            patterns.append(pattern_vector)
            dates.append(window.index[-1])
            
        return np.array(patterns), dates
    
    def calculate_future_returns(self, data, dates, days_forward=5):
        """Calculate what happened X days after each pattern"""
        
        future_returns = []
        
        for date in dates:
            try:
                # Find current price
                current_price = data.loc[date, 'Close']
                
                # Find price N days later
                future_date_idx = data.index.get_loc(date) + days_forward
                if future_date_idx < len(data):
                    future_price = data.iloc[future_date_idx]['Close']
                    future_return = (future_price / current_price) - 1
                    future_returns.append(future_return)
                else:
                    future_returns.append(np.nan)
            except:
                future_returns.append(np.nan)
                
        return np.array(future_returns)
    
    def fit(self, data):
        """Train the pattern recognition system"""
        
        print(f"📊 Training Pattern Recognition System")
        print(f"   Window size: {self.window_size} candles")
        
        # Prepare candle patterns
        patterns, dates = self.prepare_candle_data(data)
        future_returns = self.calculate_future_returns(data, dates)
        
        # Remove invalid patterns
        valid_mask = ~np.isnan(future_returns)
        self.patterns = patterns[valid_mask]
        self.dates = [d for i, d in enumerate(dates) if valid_mask[i]]
        self.future_returns = future_returns[valid_mask]
        
        print(f"   ✓ Created {len(self.patterns)} valid patterns")
        
        # Scale patterns for better comparison
        self.patterns_scaled = self.scaler.fit_transform(self.patterns)
        
        # Optional: Use PCA to reduce dimensions
        self.patterns_pca = self.pca.fit_transform(self.patterns_scaled)
        
        print(f"   ✓ Normalized patterns to {self.patterns_pca.shape[1]} dimensions")
        
        # Analyze pattern outcomes
        positive_returns = (self.future_returns > 0.02).sum()  # >2% moves
        negative_returns = (self.future_returns < -0.02).sum()  # <-2% moves
        neutral = len(self.future_returns) - positive_returns - negative_returns
        
        print(f"   ✓ Pattern outcomes: {positive_returns} up, {negative_returns} down, {neutral} neutral")
        
        return self
    
    def find_similar_patterns(self, current_pattern, top_k=20):
        """Find the most similar historical patterns"""
        
        # Normalize current pattern
        current_scaled = self.scaler.transform(current_pattern.reshape(1, -1))
        current_pca = self.pca.transform(current_scaled)
        
        # Calculate similarity to all historical patterns
        similarities = cosine_similarity(current_pca, self.patterns_pca)[0]
        
        # Get top K most similar patterns
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        top_similarities = similarities[top_indices]
        
        # Filter by similarity threshold
        valid_matches = top_similarities >= self.similarity_threshold
        
        if valid_matches.sum() >= self.min_patterns:
            similar_indices = top_indices[valid_matches]
            similar_returns = self.future_returns[similar_indices]
            similar_similarities = top_similarities[valid_matches]
            
            return similar_returns, similar_similarities
        else:
            return None, None
    
    def predict(self, current_pattern):
        """Predict direction and confidence for current pattern"""
        
        similar_returns, similarities = self.find_similar_patterns(current_pattern)
        
        if similar_returns is None:
            return 0, 0.0  # No prediction, no confidence
        
        # Calculate weighted prediction based on similarity
        weights = similarities / similarities.sum()
        predicted_return = np.average(similar_returns, weights=weights)
        
        # Calculate confidence based on consistency of similar patterns
        positive_returns = (similar_returns > 0.01).sum()
        negative_returns = (similar_returns < -0.01).sum()
        total_patterns = len(similar_returns)
        
        # Direction: 1 for up, -1 for down, 0 for neutral
        if predicted_return > 0.01:
            direction = 1
            confidence = positive_returns / total_patterns
        elif predicted_return < -0.01:
            direction = -1  
            confidence = negative_returns / total_patterns
        else:
            direction = 0
            confidence = 0.5
            
        return direction, confidence
    
    def backtest(self, data, start_date=None, position_size=0.1):
        """Backtest the pattern recognition system"""
        
        print(f"\n🔄 Backtesting Pattern System")
        
        if start_date:
            backtest_data = data[data.index >= start_date]
        else:
            # Use last 6 months for backtesting
            split_idx = int(len(data) * 0.8)
            backtest_data = data.iloc[split_idx:]
            
        print(f"   Testing on {len(backtest_data)} days")
        
        trades = []
        portfolio_value = 100000  # Start with $100k
        cash = portfolio_value
        position = 0
        
        for i in range(self.window_size, len(backtest_data) - 5):
            
            current_date = backtest_data.index[i]
            current_price = backtest_data.iloc[i]['Close']
            
            # Get current pattern
            pattern_window = backtest_data.iloc[i-self.window_size:i]
            current_patterns, _ = self.prepare_candle_data(
                pd.concat([pattern_window], axis=0)
            )
            
            if len(current_patterns) == 0:
                continue
                
            current_pattern = current_patterns[0]
            
            # Make prediction
            direction, confidence = self.predict(current_pattern)
            
            # Trading logic
            if confidence > 0.6 and abs(direction) == 1:  # Only trade with high confidence
                
                # Calculate position size based on confidence
                target_position = direction * position_size * confidence
                
                # Execute trade
                if position != target_position:
                    
                    # Close existing position
                    if position != 0:
                        cash += position * current_price * portfolio_value
                        
                    # Open new position  
                    if target_position != 0:
                        cash -= abs(target_position) * current_price * portfolio_value
                        
                    # Record trade
                    trades.append({
                        'date': current_date,
                        'price': current_price,
                        'direction': direction,
                        'confidence': confidence,
                        'position': target_position,
                        'old_position': position
                    })
                    
                    position = target_position
            
            # Update portfolio value
            portfolio_value = cash + (position * current_price * 100000 if position != 0 else 0)
        
        # Calculate performance
        total_return = (portfolio_value / 100000) - 1
        
        if len(trades) > 0:
            trades_df = pd.DataFrame(trades)
            win_rate = len(trades_df[trades_df['direction'] * 
                             (trades_df['price'].shift(-1) / trades_df['price'] - 1) > 0]) / len(trades_df)
        else:
            win_rate = 0
            
        print(f"   ✓ Total Return: {total_return:.2%}")
        print(f"   ✓ Number of Trades: {len(trades)}")
        print(f"   ✓ Win Rate: {win_rate:.2%}")
        
        return {
            'total_return': total_return,
            'trades': trades,
            'win_rate': win_rate,
            'final_value': portfolio_value
        }


def main():
    """Run the pattern-based momentum system"""
    
    print("🎯 PATTERN-BASED MOMENTUM TRADING")
    print("=" * 40)
    
    # 1. Load data
    print("\n1. Loading Data...")
    ticker = yf.Ticker("SPY")
    data = ticker.history(period="3y", interval="1d")
    
    print(f"   ✓ Loaded {len(data)} days of SPY data")
    
    # 2. Initialize pattern trader
    trader = PatternMomentumTrader(
        window_size=30,        # Your 30-candle windows
        similarity_threshold=0.80,  # 80% similarity required
        min_patterns=5         # Need at least 5 similar patterns
    )
    
    # 3. Train on first 80% of data
    split_idx = int(len(data) * 0.8)
    train_data = data.iloc[:split_idx]
    
    trader.fit(train_data)
    
    # 4. Backtest on remaining 20%
    results = trader.backtest(data)
    
    # 5. Show results
    print(f"\n📊 PATTERN SYSTEM RESULTS:")
    print(f"   Total Return: {results['total_return']:.2%}")
    print(f"   Win Rate: {results['win_rate']:.2%}")
    print(f"   Number of Trades: {len(results['trades'])}")
    
    # Calculate monthly performance
    if results['total_return'] > 0:
        months = 3 * 0.2  # 20% of 3 years ≈ 7.2 months
        monthly_return = (1 + results['total_return']) ** (1/months) - 1
        required_capital = 20000 / monthly_return if monthly_return > 0 else float('inf')
        
        print(f"\n💰 SCALING ANALYSIS:")
        print(f"   Monthly Return: {monthly_return:.2%}")
        print(f"   For $20k/month: ${required_capital:,.0f} needed")
        
        if required_capital < 1000000:
            print(f"   ✅ Target achievable!")
        else:
            print(f"   ⚠️ Need better performance or more capital")
    
    # 6. Show recent predictions
    print(f"\n🔮 RECENT PATTERN ANALYSIS:")
    recent_pattern = data.tail(30)
    patterns, _ = trader.prepare_candle_data(pd.concat([recent_pattern], axis=0))
    
    if len(patterns) > 0:
        direction, confidence = trader.predict(patterns[0])
        direction_text = ["Down", "Neutral", "Up"][direction + 1]
        
        print(f"   Current Pattern Prediction: {direction_text}")
        print(f"   Confidence: {confidence:.2%}")
        
        if confidence > 0.6:
            print(f"   ✅ High confidence - tradeable signal")
        else:
            print(f"   ⚠️ Low confidence - wait for better setup")


if __name__ == "__main__":
    main()