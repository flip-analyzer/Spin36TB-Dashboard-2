#!/usr/bin/env python3
"""
Intraday Momentum Pattern Trader

This implements a day trading approach:
- 10-20 trades per day
- 1-5 minute candle patterns 
- Same-day entries and exits
- Quick momentum scalping based on pattern recognition
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class IntradayMomentumTrader:
    def __init__(self, 
                 pattern_window=20,      # 20 minute patterns (20x1min candles)
                 similarity_threshold=0.65,  # Looser for more trades
                 confidence_threshold=0.55,  # Lower for more opportunities
                 hold_minutes=30,        # Hold for 30 minutes max
                 max_position=0.05):     # 5% position size (can do 20 trades)
        
        self.pattern_window = pattern_window
        self.similarity_threshold = similarity_threshold
        self.confidence_threshold = confidence_threshold
        self.hold_minutes = hold_minutes
        self.max_position = max_position
        
        self.patterns_database = []
        self.outcomes_database = []
        self.timestamps_database = []
        
    def create_intraday_pattern(self, candle_data):
        """Create pattern from 1-minute OHLCV candles"""
        
        if len(candle_data) < self.pattern_window:
            return None
            
        # Normalize to first candle
        base_price = candle_data['Close'].iloc[0]
        
        # Price movements (relative to base)
        closes = (candle_data['Close'] / base_price).values
        highs = (candle_data['High'] / base_price).values
        lows = (candle_data['Low'] / base_price).values
        
        # Volume profile
        volumes = (candle_data['Volume'] / candle_data['Volume'].mean()).values
        
        # Momentum features
        price_momentum = np.diff(closes)  # Price changes minute-to-minute
        price_momentum = np.concatenate([[0], price_momentum])  # Pad first value
        
        # Volatility features
        minute_ranges = (highs - lows) / closes  # Range as % of close
        
        # Create pattern vector
        pattern = np.concatenate([
            closes,          # Price levels
            price_momentum,  # Price changes
            minute_ranges,   # Volatility
            volumes         # Volume activity
        ])
        
        return pattern
        
    def train_intraday_patterns(self, data):
        """Build database of intraday patterns and outcomes"""
        
        print(f"🧠 Building Intraday Pattern Database")
        print(f"   Pattern window: {self.pattern_window} minutes")
        print(f"   Hold time: {self.hold_minutes} minutes")
        
        self.patterns_database = []
        self.outcomes_database = []
        self.timestamps_database = []
        
        # Process each day's worth of data
        for i in range(len(data) - self.pattern_window - self.hold_minutes):
            
            # Get pattern window
            pattern_end_idx = i + self.pattern_window
            pattern_data = data.iloc[i:pattern_end_idx]
            
            # Create pattern
            pattern = self.create_intraday_pattern(pattern_data)
            if pattern is None:
                continue
                
            # Calculate outcome (what happened in next hold_minutes)
            entry_price = pattern_data['Close'].iloc[-1]
            exit_idx = min(pattern_end_idx + self.hold_minutes, len(data) - 1)
            exit_price = data.iloc[exit_idx]['Close']
            
            outcome = (exit_price / entry_price) - 1
            
            # Store pattern and outcome
            self.patterns_database.append(pattern)
            self.outcomes_database.append(outcome)
            self.timestamps_database.append(pattern_data.index[-1])
            
        print(f"   ✅ Stored {len(self.patterns_database)} intraday patterns")
        
        # Analyze outcomes
        outcomes = np.array(self.outcomes_database)
        positive = (outcomes > 0.002).sum()  # >0.2% moves
        negative = (outcomes < -0.002).sum()
        neutral = len(outcomes) - positive - negative
        
        print(f"   📊 Pattern outcomes:")
        print(f"      Positive (>0.2%): {positive}")
        print(f"      Negative (<-0.2%): {negative}")
        print(f"      Neutral: {neutral}")
        
    def find_similar_intraday_patterns(self, current_pattern, top_k=30):
        """Find similar intraday patterns"""
        
        if len(self.patterns_database) == 0:
            return None, None
            
        similarities = []
        for historical_pattern in self.patterns_database:
            try:
                similarity = np.corrcoef(current_pattern, historical_pattern)[0, 1]
                if np.isnan(similarity):
                    similarity = 0
            except:
                similarity = 0
            similarities.append(similarity)
            
        similarities = np.array(similarities)
        
        # Get top similar patterns
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        top_similarities = similarities[top_indices]
        
        # Filter by threshold
        valid_mask = top_similarities >= self.similarity_threshold
        
        if valid_mask.sum() < 3:
            return None, None
            
        valid_indices = top_indices[valid_mask]
        similar_outcomes = np.array(self.outcomes_database)[valid_indices]
        similar_similarities = top_similarities[valid_mask]
        
        return similar_outcomes, similar_similarities
        
    def predict_intraday_move(self, current_pattern):
        """Predict next 30-minute move based on current pattern"""
        
        similar_outcomes, similarities = self.find_similar_intraday_patterns(current_pattern)
        
        if similar_outcomes is None:
            return "HOLD", 0.0, 0
            
        # Calculate weighted prediction
        weights = similarities / similarities.sum()
        expected_return = np.average(similar_outcomes, weights=weights)
        
        # Count directional moves
        strong_up = (similar_outcomes > 0.003).sum()  # >0.3%
        strong_down = (similar_outcomes < -0.003).sum()
        total = len(similar_outcomes)
        
        # Make decision
        if expected_return > 0.002:  # Expecting >0.2% gain
            direction = "BUY"
            confidence = strong_up / total
        elif expected_return < -0.002:  # Expecting >0.2% loss
            direction = "SELL"
            confidence = strong_down / total
        else:
            direction = "HOLD"
            confidence = 0.5
            
        return direction, confidence, len(similar_outcomes)
        
    def calculate_position_size(self, confidence):
        """Calculate position size for intraday trade"""
        
        if confidence < self.confidence_threshold:
            return 0.0
            
        # Scale position size by confidence
        # Max 5% per trade allows for 20 trades
        base_size = self.max_position
        scaled_size = base_size * ((confidence - self.confidence_threshold) / (1.0 - self.confidence_threshold))
        
        return min(scaled_size, base_size)
        
    def simulate_intraday_trading(self, test_data):
        """Simulate a day of intraday trading"""
        
        print(f"\n📈 Simulating Intraday Trading")
        print(f"   Test data: {len(test_data)} minutes")
        print(f"   Target: 10-20 trades")
        
        portfolio_value = 100000
        trades = []
        active_position = None
        
        for i in range(self.pattern_window, len(test_data) - self.hold_minutes):
            
            current_time = test_data.index[i]
            current_price = test_data.iloc[i]['Close']
            
            # Check if we need to close existing position
            if active_position:
                hold_time = (current_time - active_position['entry_time']).total_seconds() / 60
                
                if hold_time >= self.hold_minutes:
                    # Close position
                    exit_return = (current_price / active_position['entry_price']) - 1
                    if active_position['direction'] == "SELL":
                        exit_return = -exit_return  # Invert for short
                        
                    trade_return = exit_return * active_position['position_size']
                    portfolio_value *= (1 + trade_return)
                    
                    active_position['exit_time'] = current_time
                    active_position['exit_price'] = current_price  
                    active_position['return'] = trade_return
                    trades.append(active_position)
                    
                    active_position = None
            
            # Look for new trade if no active position
            if active_position is None:
                
                # Get current pattern
                pattern_start = i - self.pattern_window
                pattern_data = test_data.iloc[pattern_start:i]
                current_pattern = self.create_intraday_pattern(pattern_data)
                
                if current_pattern is not None:
                    # Make prediction
                    direction, confidence, num_similar = self.predict_intraday_move(current_pattern)
                    position_size = self.calculate_position_size(confidence)
                    
                    # Enter trade if confident
                    if position_size > 0 and direction in ["BUY", "SELL"]:
                        active_position = {
                            'entry_time': current_time,
                            'entry_price': current_price,
                            'direction': direction,
                            'confidence': confidence,
                            'position_size': position_size,
                            'num_similar_patterns': num_similar
                        }
        
        # Close any remaining position at end of day
        if active_position:
            exit_return = (test_data.iloc[-1]['Close'] / active_position['entry_price']) - 1
            if active_position['direction'] == "SELL":
                exit_return = -exit_return
                
            trade_return = exit_return * active_position['position_size']
            portfolio_value *= (1 + trade_return)
            
            active_position['exit_time'] = test_data.index[-1]
            active_position['exit_price'] = test_data.iloc[-1]['Close']
            active_position['return'] = trade_return
            trades.append(active_position)
        
        # Calculate performance
        total_return = (portfolio_value / 100000) - 1
        winning_trades = len([t for t in trades if t['return'] > 0])
        win_rate = winning_trades / len(trades) if trades else 0
        
        print(f"   ✅ Trading Results:")
        print(f"      Total Trades: {len(trades)}")
        print(f"      Win Rate: {win_rate:.2%}")
        print(f"      Total Return: {total_return:.2%}")
        print(f"      Final Value: ${portfolio_value:,.2f}")
        
        if len(trades) > 0:
            avg_confidence = np.mean([t['confidence'] for t in trades])
            avg_position = np.mean([t['position_size'] for t in trades])
            print(f"      Avg Confidence: {avg_confidence:.2%}")
            print(f"      Avg Position Size: {avg_position:.2%}")
        
        return {
            'trades': trades,
            'total_return': total_return,
            'win_rate': win_rate,
            'num_trades': len(trades),
            'final_value': portfolio_value
        }


def get_intraday_data():
    """Get 1-minute intraday data for SPY"""
    
    print("📥 Getting Intraday Data...")
    
    # Get last 7 days of 1-minute data
    spy = yf.Ticker("SPY")
    
    try:
        # Try to get 1-minute data (limited to last 30 days)
        data = spy.history(period="5d", interval="1m")
        
        if len(data) == 0:
            print("   ⚠️ No 1-minute data available, using 5-minute data")
            data = spy.history(period="5d", interval="5m")
            
        if len(data) == 0:
            print("   ⚠️ No 5-minute data available, using hourly data")  
            data = spy.history(period="5d", interval="1h")
        
    except Exception as e:
        print(f"   ⚠️ Error getting intraday data: {e}")
        print("   Using daily data as fallback")
        data = spy.history(period="60d", interval="1d")
    
    # Filter to market hours only (9:30 AM - 4:00 PM ET)
    if not data.empty:
        data = data.between_time('09:30', '16:00')
    
    print(f"   ✅ Loaded {len(data)} intraday bars")
    return data


def main():
    """Run intraday momentum trading system"""
    
    print("⚡ INTRADAY MOMENTUM PATTERN TRADER")
    print("=" * 45)
    print("Target: 10-20 trades per day")
    print("Hold time: 30 minutes max")
    print("Position size: 5% max per trade")
    
    # Get intraday data
    data = get_intraday_data()
    
    if len(data) < 100:
        print("❌ Insufficient data for intraday trading")
        return
    
    # Initialize trader
    trader = IntradayMomentumTrader(
        pattern_window=20,       # 20-bar patterns
        similarity_threshold=0.65, # 65% similarity
        confidence_threshold=0.55, # 55% confidence to trade
        hold_minutes=30,         # 30-minute holds
        max_position=0.05        # 5% max position
    )
    
    # Split data: train on first 80%, test on last 20%
    split_idx = int(len(data) * 0.8)
    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:]
    
    print(f"\nTraining on {len(train_data)} bars")
    print(f"Testing on {len(test_data)} bars")
    
    # Train pattern recognition
    trader.train_intraday_patterns(train_data)
    
    # Simulate trading
    results = trader.simulate_intraday_trading(test_data)
    
    # Analyze results
    print(f"\n🎯 INTRADAY RESULTS:")
    print(f"   Trades per session: {results['num_trades']}")
    print(f"   Win rate: {results['win_rate']:.2%}")
    print(f"   Session return: {results['total_return']:.2%}")
    
    # Scale to daily target
    if results['num_trades'] >= 10 and results['total_return'] > 0:
        daily_target = 0.005  # 0.5% per day target
        sessions_needed = daily_target / results['total_return'] if results['total_return'] > 0 else float('inf')
        
        print(f"\n💰 SCALING ANALYSIS:")
        print(f"   Target daily return: {daily_target:.2%}")  
        print(f"   Current session return: {results['total_return']:.2%}")
        
        if sessions_needed <= 1.5:
            print(f"   ✅ Target achievable! Need {sessions_needed:.1f} sessions")
            
            # Calculate monthly income potential
            monthly_sessions = 20  # ~20 trading days per month
            monthly_return = (1 + results['total_return']) ** monthly_sessions - 1
            required_capital = 20000 / monthly_return if monthly_return > 0 else float('inf')
            
            print(f"   Monthly return potential: {monthly_return:.2%}")
            print(f"   Capital needed for $20k/month: ${required_capital:,.0f}")
            
        else:
            print(f"   ⚠️ Need better performance ({sessions_needed:.1f}x current return)")
    
    return results


if __name__ == "__main__":
    results = main()