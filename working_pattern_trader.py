#!/usr/bin/env python3
"""
Working Pattern-Based Momentum Trader

Based on debugging, this uses realistic parameters that actually generate trades
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

class WorkingPatternTrader:
    def __init__(self, window_size=15, similarity_threshold=0.70, confidence_threshold=0.55):
        """
        Working pattern trader with realistic parameters
        
        Based on debugging:
        - Lower similarity threshold (0.70 vs 0.80)
        - Lower confidence threshold (0.55 vs 0.60)  
        - Shorter windows (15 vs 30 candles)
        """
        self.window_size = window_size
        self.similarity_threshold = similarity_threshold
        self.confidence_threshold = confidence_threshold
        
    def create_pattern(self, data):
        """Create pattern from OHLCV data"""
        
        # Normalize prices to first close
        closes = data['Close'].values
        opens = data['Open'].values
        highs = data['High'].values  
        lows = data['Low'].values
        volumes = data['Volume'].values
        
        base_price = closes[0]
        base_volume = volumes.mean()
        
        # Create comprehensive pattern
        pattern = np.concatenate([
            closes / base_price,        # Normalized closes
            (highs - closes) / base_price,  # Upper shadows  
            (closes - lows) / base_price,   # Lower shadows
            volumes / base_volume       # Normalized volume
        ])
        
        return pattern
    
    def train(self, data):
        """Train pattern recognition system"""
        
        print(f"🧠 Training Pattern System")
        print(f"   Window: {self.window_size} candles")
        print(f"   Similarity threshold: {self.similarity_threshold}")
        
        self.patterns = []
        self.returns = []
        self.dates = []
        
        # Create training patterns
        for i in range(len(data) - self.window_size - 5):
            
            window = data.iloc[i:i + self.window_size]
            pattern = self.create_pattern(window)
            
            # Future return (5 days ahead)
            current_price = window['Close'].iloc[-1]
            future_price = data.iloc[i + self.window_size + 3]['Close']  # 3 days forward
            future_return = (future_price / current_price) - 1
            
            self.patterns.append(pattern)
            self.returns.append(future_return)
            self.dates.append(window.index[-1])
        
        self.patterns = np.array(self.patterns)
        self.returns = np.array(self.returns)
        
        print(f"   ✅ Created {len(self.patterns)} training patterns")
        
        # Analyze training data
        big_ups = (self.returns > 0.01).sum()
        big_downs = (self.returns < -0.01).sum() 
        neutral = len(self.returns) - big_ups - big_downs
        
        print(f"   📊 Outcomes: {big_ups} up (>1%), {big_downs} down (<-1%), {neutral} neutral")
        
    def predict(self, current_pattern, top_k=15):
        """Predict direction and confidence for current pattern"""
        
        # Find most similar historical patterns
        similarities = []
        for pattern in self.patterns:
            similarity = np.corrcoef(current_pattern, pattern)[0, 1]
            if np.isnan(similarity):
                similarity = 0
            similarities.append(similarity)
        
        similarities = np.array(similarities)
        
        # Get top K most similar patterns above threshold
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        top_sims = similarities[top_indices]
        
        # Filter by similarity threshold
        valid_mask = top_sims >= self.similarity_threshold
        
        if valid_mask.sum() < 3:  # Need at least 3 similar patterns
            return 0, 0.0, []
            
        valid_indices = top_indices[valid_mask]
        valid_sims = top_sims[valid_mask]
        similar_returns = self.returns[valid_indices]
        
        # Calculate weighted prediction
        weights = valid_sims / valid_sims.sum()
        predicted_return = np.average(similar_returns, weights=weights)
        
        # Calculate confidence based on consistency
        strong_ups = (similar_returns > 0.008).sum()  # >0.8% moves
        strong_downs = (similar_returns < -0.008).sum()
        total = len(similar_returns)
        
        if predicted_return > 0.005:  # Predicted up move >0.5%
            direction = 1
            confidence = strong_ups / total
        elif predicted_return < -0.005:  # Predicted down move
            direction = -1
            confidence = strong_downs / total
        else:
            direction = 0
            confidence = 0.5
        
        return direction, confidence, similar_returns
    
    def backtest(self, data, lookback_days=200):
        """Backtest the pattern system"""
        
        print(f"\n📈 Backtesting Pattern System")
        
        # Use last N days for testing
        test_data = data.tail(lookback_days)
        
        portfolio_value = 100000
        cash = portfolio_value
        position = 0
        trades = []
        
        daily_values = []
        
        for i in range(self.window_size + 5, len(test_data)):
            
            current_date = test_data.index[i]
            current_price = test_data.iloc[i]['Close']
            
            # Get pattern for prediction
            pattern_window = test_data.iloc[i - self.window_size:i]
            current_pattern = self.create_pattern(pattern_window)
            
            # Make prediction
            direction, confidence, similar_returns = self.predict(current_pattern)
            
            # Trading decision
            should_trade = (
                confidence > self.confidence_threshold and 
                abs(direction) == 1
            )
            
            if should_trade:
                # Position sizing based on confidence
                base_size = 0.2  # 20% of portfolio
                target_position = direction * base_size * confidence
                
                # Execute trade if different from current position
                if abs(target_position - position) > 0.05:  # 5% minimum change
                    
                    # Record trade
                    trades.append({
                        'date': current_date,
                        'price': current_price,
                        'direction': direction,
                        'confidence': confidence,
                        'target_position': target_position,
                        'old_position': position,
                        'similar_patterns': len(similar_returns)
                    })
                    
                    # Update cash and position
                    cash += (position - target_position) * current_price * portfolio_value
                    position = target_position
            
            # Calculate daily portfolio value
            portfolio_value = cash + (position * current_price * 100000)
            daily_values.append({
                'date': current_date,
                'value': portfolio_value,
                'position': position
            })
        
        # Calculate performance
        if len(daily_values) > 0:
            final_value = daily_values[-1]['value']
            total_return = (final_value / 100000) - 1
            
            # Convert to dataframe for analysis
            values_df = pd.DataFrame(daily_values).set_index('date')
            daily_returns = values_df['value'].pct_change().dropna()
            
            volatility = daily_returns.std() * np.sqrt(252)
            sharpe = (daily_returns.mean() * 252) / volatility if volatility > 0 else 0
            
            # Max drawdown
            rolling_max = values_df['value'].expanding().max()
            drawdown = (values_df['value'] - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            
        else:
            total_return = volatility = sharpe = max_drawdown = 0
            
        # Win rate analysis
        if len(trades) > 0:
            trades_df = pd.DataFrame(trades)
            
            # Simple win rate (did direction match next day's move?)
            wins = 0
            for _, trade in trades_df.iterrows():
                trade_date = trade['date']
                try:
                    next_day_idx = test_data.index.get_loc(trade_date) + 1
                    if next_day_idx < len(test_data):
                        next_price = test_data.iloc[next_day_idx]['Close']
                        actual_direction = np.sign(next_price / trade['price'] - 1)
                        if actual_direction == trade['direction']:
                            wins += 1
                except:
                    pass
                    
            win_rate = wins / len(trades_df) if len(trades_df) > 0 else 0
        else:
            win_rate = 0
            
        print(f"   ✅ Backtest Complete")
        print(f"   📊 Results:")
        print(f"      Total Return: {total_return:.2%}")
        print(f"      Sharpe Ratio: {sharpe:.2f}")
        print(f"      Max Drawdown: {max_drawdown:.2%}")
        print(f"      Volatility: {volatility:.2%}")
        print(f"      Number of Trades: {len(trades)}")
        print(f"      Win Rate: {win_rate:.2%}")
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'num_trades': len(trades),
            'trades': trades,
            'daily_values': daily_values
        }


def main():
    """Run the working pattern trading system"""
    
    print("🎯 WORKING PATTERN-BASED MOMENTUM TRADER")
    print("=" * 50)
    
    # Load data
    print("\n📥 Loading Market Data...")
    ticker = yf.Ticker("SPY")
    data = ticker.history(period="3y", interval="1d")
    
    print(f"   ✅ Loaded {len(data)} days of SPY data")
    
    # Initialize trader
    trader = WorkingPatternTrader(
        window_size=15,           # 15-candle patterns (3 weeks)
        similarity_threshold=0.70, # 70% similarity required  
        confidence_threshold=0.55  # 55% confidence to trade
    )
    
    # Train on first 70% of data
    split_idx = int(len(data) * 0.7)
    train_data = data.iloc[:split_idx]
    
    trader.train(train_data)
    
    # Backtest on remaining 30%
    test_data = data.iloc[split_idx:]
    results = trader.backtest(data)  # Test on all data but trained patterns exclude future
    
    # Analyze results
    print(f"\n🎯 FINAL RESULTS:")
    print(f"   Total Return: {results['total_return']:.2%}")
    print(f"   Sharpe Ratio: {results['sharpe_ratio']:.2f}")  
    print(f"   Win Rate: {results['win_rate']:.2%}")
    print(f"   Max Drawdown: {results['max_drawdown']:.2%}")
    print(f"   Trades Made: {results['num_trades']}")
    
    # Monthly performance estimate
    if results['total_return'] > 0:
        test_months = len(results['daily_values']) / 21  # ~21 trading days per month
        monthly_return = (1 + results['total_return']) ** (1/test_months) - 1
        required_capital_20k = 20000 / monthly_return if monthly_return > 0 else float('inf')
        
        print(f"\n💰 SCALING TO $20K/MONTH:")
        print(f"   Monthly Return: {monthly_return:.2%}")
        print(f"   Required Capital: ${required_capital_20k:,.0f}")
        
        if required_capital_20k < 500000:
            print(f"   ✅ Target achievable with reasonable capital!")
        elif required_capital_20k < 2000000:  
            print(f"   ⚠️ Target achievable but requires significant capital")
        else:
            print(f"   ❌ Target requires optimization")
            
    # Current market prediction
    print(f"\n🔮 CURRENT MARKET PREDICTION:")
    current_pattern = trader.create_pattern(data.tail(15))
    direction, confidence, similar_returns = trader.predict(current_pattern)
    
    direction_text = ["SELL", "HOLD", "BUY"][direction + 1]
    
    print(f"   Signal: {direction_text}")
    print(f"   Confidence: {confidence:.2%}")
    print(f"   Based on {len(similar_returns)} similar patterns")
    
    if confidence > 0.6:
        position_size = confidence * 0.2  # Max 20% position
        print(f"   💡 Suggested Position: {position_size:.1%} of portfolio")
        print(f"   ✅ TRADEABLE SIGNAL")
    else:
        print(f"   ⚠️ Low confidence - wait for better setup")
        
    return results


if __name__ == "__main__":
    results = main()