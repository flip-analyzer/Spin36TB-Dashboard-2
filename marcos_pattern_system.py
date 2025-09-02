#!/usr/bin/env python3
"""
Marcos López de Prado Pattern Recognition System

Uses your EURUSD 5-minute dataset with proper 30-candle windows
for intraday momentum trading targeting 10-20 trades per day.
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class MarcosPatternTrader:
    def __init__(self, 
                 window_size=30,           # 30 5-minute candles (2.5 hours)
                 similarity_threshold=0.65, # 65% similarity
                 confidence_threshold=0.55, # 55% confidence to trade
                 prediction_horizon=6,      # Predict 6 bars ahead (30 minutes)
                 max_position_size=0.1):    # 10% max position
        
        self.window_size = window_size
        self.similarity_threshold = similarity_threshold
        self.confidence_threshold = confidence_threshold
        self.prediction_horizon = prediction_horizon
        self.max_position_size = max_position_size
        
        self.historical_patterns = []
        self.historical_outcomes = []
        self.pattern_timestamps = []
        
    def load_eurusd_data(self):
        """Load your EURUSD 5-minute dataset"""
        
        print("📊 Loading Your EURUSD 5-Minute Dataset...")
        
        try:
            # Load your data
            data_path = "/Users/jonspinogatti/Desktop/spin35TB/data/candles/eurusd_m5_500k.csv"
            data = pd.read_csv(data_path)
            
            # Parse datetime
            data['time'] = pd.to_datetime(data['time'])
            data = data.set_index('time')
            
            # Rename columns to standard OHLCV format
            data = data.rename(columns={
                'open': 'Open',
                'high': 'High', 
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })
            
            print(f"   ✅ Loaded {len(data):,} 5-minute EURUSD bars")
            print(f"   📅 Date range: {data.index[0]} to {data.index[-1]}")
            print(f"   💰 Price range: {data['Close'].min():.5f} - {data['Close'].max():.5f}")
            
            return data
            
        except Exception as e:
            print(f"   ❌ Error loading data: {e}")
            return None
    
    def create_pattern_vector(self, candle_window):
        """
        Convert 30 5-minute candles into pattern vector (your embedding approach)
        
        This creates a comprehensive representation of 2.5 hours of price action
        """
        if len(candle_window) != self.window_size:
            return None
            
        # Normalize all prices to first candle's close
        base_price = candle_window['Close'].iloc[0]
        
        # Price series (normalized)
        opens = (candle_window['Open'] / base_price).values
        highs = (candle_window['High'] / base_price).values
        lows = (candle_window['Low'] / base_price).values
        closes = (candle_window['Close'] / base_price).values
        
        # Volume pattern (normalized to average)
        volumes = (candle_window['Volume'] / candle_window['Volume'].mean()).values
        
        # Technical patterns within the window
        ranges = (highs - lows) / closes  # Relative ranges
        bodies = np.abs(closes - opens) / closes  # Relative body sizes
        upper_wicks = (highs - np.maximum(opens, closes)) / closes
        lower_wicks = (np.minimum(opens, closes) - lows) / closes
        
        # Momentum features within window
        price_changes = np.diff(closes)
        price_changes = np.concatenate([[0], price_changes])  # Pad first value
        
        # Combine all features into pattern vector (150 elements total)
        pattern_vector = np.concatenate([
            closes,        # 30 elements: normalized closing prices
            ranges,        # 30 elements: bar ranges
            bodies,        # 30 elements: candle bodies
            upper_wicks,   # 30 elements: upper shadows
            lower_wicks,   # 30 elements: lower shadows
        ])
        
        return pattern_vector
    
    def build_pattern_database(self, data, max_patterns=100000):
        """Build database of patterns and their outcomes"""
        
        print(f"\n🧠 Building Pattern Database (Marcos Style)")
        print(f"   Window: {self.window_size} 5-minute candles (2.5 hours)")
        print(f"   Prediction: {self.prediction_horizon} bars ahead (30 minutes)")
        
        self.historical_patterns = []
        self.historical_outcomes = []
        self.pattern_timestamps = []
        
        total_possible = len(data) - self.window_size - self.prediction_horizon
        print(f"   Maximum patterns available: {total_possible:,}")
        
        # Use subset for performance if dataset is huge
        step_size = max(1, total_possible // max_patterns)
        print(f"   Using every {step_size} pattern(s) for training")
        
        patterns_created = 0
        
        for i in range(0, total_possible, step_size):
            if patterns_created >= max_patterns:
                break
                
            # Get 30-candle pattern window
            pattern_start = i
            pattern_end = i + self.window_size
            pattern_window = data.iloc[pattern_start:pattern_end]
            
            # Create pattern vector
            pattern_vector = self.create_pattern_vector(pattern_window)
            if pattern_vector is None:
                continue
                
            # Calculate outcome (what happened prediction_horizon bars later)
            entry_price = pattern_window['Close'].iloc[-1]
            outcome_idx = pattern_end + self.prediction_horizon - 1
            
            if outcome_idx < len(data):
                exit_price = data.iloc[outcome_idx]['Close']
                outcome_return = (exit_price / entry_price) - 1
                
                # Store pattern and outcome
                self.historical_patterns.append(pattern_vector)
                self.historical_outcomes.append(outcome_return)
                self.pattern_timestamps.append(pattern_window.index[-1])
                patterns_created += 1
        
        self.historical_patterns = np.array(self.historical_patterns)
        self.historical_outcomes = np.array(self.historical_outcomes)
        
        print(f"   ✅ Created {len(self.historical_patterns):,} patterns")
        
        # Analyze outcomes
        positive_moves = (self.historical_outcomes > 0.001).sum()  # >10 pip moves
        negative_moves = (self.historical_outcomes < -0.001).sum()
        neutral_moves = len(self.historical_outcomes) - positive_moves - negative_moves
        
        print(f"   📊 Outcome distribution:")
        print(f"      Positive (>10 pips): {positive_moves:,} ({positive_moves/len(self.historical_outcomes)*100:.1f}%)")
        print(f"      Negative (<-10 pips): {negative_moves:,} ({negative_moves/len(self.historical_outcomes)*100:.1f}%)")
        print(f"      Neutral: {neutral_moves:,} ({neutral_moves/len(self.historical_outcomes)*100:.1f}%)")
    
    def find_similar_patterns(self, current_pattern, top_k=50):
        """Find most similar historical patterns"""
        
        if len(self.historical_patterns) == 0:
            return None, None
            
        # Calculate similarities to all historical patterns
        similarities = []
        for i, hist_pattern in enumerate(self.historical_patterns):
            similarity = np.corrcoef(current_pattern, hist_pattern)[0, 1]
            if np.isnan(similarity):
                similarity = 0
            similarities.append(similarity)
        
        similarities = np.array(similarities)
        
        # Get top K most similar patterns
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        top_similarities = similarities[top_indices]
        
        # Filter by similarity threshold
        valid_mask = top_similarities >= self.similarity_threshold
        
        if valid_mask.sum() < 5:  # Need at least 5 similar patterns
            return None, None
            
        similar_indices = top_indices[valid_mask]
        similar_outcomes = self.historical_outcomes[similar_indices]
        similar_similarities = top_similarities[valid_mask]
        
        return similar_outcomes, similar_similarities
    
    def predict_move(self, current_pattern):
        """Predict 30-minute move based on similar patterns"""
        
        similar_outcomes, similarities = self.find_similar_patterns(current_pattern)
        
        if similar_outcomes is None:
            return "HOLD", 0.0, 0
            
        # Calculate weighted prediction
        weights = similarities / similarities.sum()
        expected_return = np.average(similar_outcomes, weights=weights)
        
        # Count strong directional moves
        strong_up = (similar_outcomes > 0.0008).sum()    # >8 pip moves up
        strong_down = (similar_outcomes < -0.0008).sum()  # >8 pip moves down
        total_patterns = len(similar_outcomes)
        
        # Make decision
        if expected_return > 0.0005:  # Expecting >5 pip gain
            direction = "BUY"
            confidence = strong_up / total_patterns
        elif expected_return < -0.0005:  # Expecting >5 pip loss
            direction = "SELL"
            confidence = strong_down / total_patterns
        else:
            direction = "HOLD"
            confidence = 0.5
            
        return direction, confidence, total_patterns
    
    def calculate_position_size(self, confidence):
        """Calculate position size based on confidence"""
        
        if confidence < self.confidence_threshold:
            return 0.0
            
        # Scale position by confidence above threshold
        excess_confidence = confidence - self.confidence_threshold
        max_excess = 1.0 - self.confidence_threshold
        
        position_size = self.max_position_size * (excess_confidence / max_excess)
        
        return min(position_size, self.max_position_size)
    
    def backtest_system(self, data, test_days=30):
        """Backtest the pattern recognition system"""
        
        print(f"\n📈 Backtesting Pattern System")
        print(f"   Testing on last {test_days} days of data")
        
        # Calculate bars per day (approximately)
        bars_per_day = 24 * 60 // 5  # 288 5-minute bars per day
        test_bars = min(test_days * bars_per_day, len(data) // 5)  # Test on 20% max
        
        test_data = data.tail(test_bars)
        print(f"   Testing on {len(test_data):,} bars")
        
        portfolio_value = 100000
        trades = []
        
        for i in range(self.window_size, len(test_data) - self.prediction_horizon):
            
            current_time = test_data.index[i]
            current_price = test_data.iloc[i]['Close']
            
            # Get current pattern
            pattern_start = i - self.window_size
            pattern_window = test_data.iloc[pattern_start:i]
            current_pattern = self.create_pattern_vector(pattern_window)
            
            if current_pattern is None:
                continue
                
            # Make prediction
            direction, confidence, num_similar = self.predict_move(current_pattern)
            position_size = self.calculate_position_size(confidence)
            
            # Execute trade if confident
            if position_size > 0 and direction in ["BUY", "SELL"]:
                
                # Simulate holding for prediction_horizon bars
                exit_idx = i + self.prediction_horizon
                if exit_idx < len(test_data):
                    
                    entry_price = current_price
                    exit_price = test_data.iloc[exit_idx]['Close']
                    
                    if direction == "BUY":
                        trade_return = (exit_price / entry_price - 1) * position_size
                    else:  # SELL
                        trade_return = (entry_price / exit_price - 1) * position_size
                    
                    portfolio_value *= (1 + trade_return)
                    
                    # Record trade
                    trades.append({
                        'time': current_time,
                        'direction': direction,
                        'confidence': confidence,
                        'position_size': position_size,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'trade_return': trade_return,
                        'similar_patterns': num_similar
                    })
        
        # Calculate performance
        total_return = (portfolio_value / 100000) - 1
        winning_trades = len([t for t in trades if t['trade_return'] > 0])
        win_rate = winning_trades / len(trades) if trades else 0
        
        print(f"\n🎯 BACKTEST RESULTS:")
        print(f"   Total Return: {total_return:.2%}")
        print(f"   Number of Trades: {len(trades)}")
        print(f"   Win Rate: {win_rate:.2%}")
        print(f"   Final Portfolio: ${portfolio_value:,.2f}")
        
        if len(trades) > 0:
            avg_confidence = np.mean([t['confidence'] for t in trades])
            avg_position = np.mean([t['position_size'] for t in trades])
            trades_per_day = len(trades) / test_days
            
            print(f"   Average Confidence: {avg_confidence:.2%}")
            print(f"   Average Position: {avg_position:.2%}")
            print(f"   Trades per Day: {trades_per_day:.1f}")
            
            # Check if we hit Marcos' target of 10-20 trades per day
            if 10 <= trades_per_day <= 20:
                print(f"   ✅ Hit target trade frequency!")
            elif trades_per_day < 10:
                print(f"   ⚠️ Trade frequency too low (need 10-20/day)")
            else:
                print(f"   ⚠️ Trade frequency too high (need 10-20/day)")
        
        return {
            'total_return': total_return,
            'num_trades': len(trades),
            'win_rate': win_rate,
            'trades': trades,
            'final_value': portfolio_value
        }
    
    def analyze_current_pattern(self, data):
        """Analyze the most recent 30-candle pattern"""
        
        print(f"\n🔮 CURRENT PATTERN ANALYSIS:")
        
        if len(data) < self.window_size:
            print("   ❌ Insufficient data for pattern analysis")
            return
            
        # Get latest pattern
        latest_pattern = self.create_pattern_vector(data.tail(self.window_size))
        
        if latest_pattern is None:
            print("   ❌ Could not create pattern vector")
            return
            
        # Make prediction
        direction, confidence, num_similar = self.predict_move(latest_pattern)
        position_size = self.calculate_position_size(confidence)
        
        print(f"   📊 Similar patterns found: {num_similar}")
        print(f"   🎯 Prediction: {direction}")
        print(f"   🎲 Confidence: {confidence:.2%}")
        print(f"   💰 Suggested position: {position_size:.2%}")
        
        if position_size > 0:
            print(f"   ✅ TRADE SIGNAL: {direction} with {position_size:.1%} position")
            print(f"   ⏰ Hold for {self.prediction_horizon} bars (30 minutes)")
        else:
            print(f"   ⚠️ NO TRADE: Confidence below {self.confidence_threshold:.0%}")


def main():
    """Run Marcos-style pattern recognition system"""
    
    print("🎯 MARCOS PATTERN RECOGNITION SYSTEM")
    print("=" * 50)
    print("Dataset: Your EURUSD 5-minute data (500k+ bars)")
    print("Method: 30-candle pattern windows (2.5 hours)")
    print("Target: 10-20 trades per day")
    print("Prediction: 30-minute momentum moves")
    
    # Initialize system
    trader = MarcosPatternTrader(
        window_size=30,           # 30 5-minute candles
        similarity_threshold=0.65, # 65% similarity required
        confidence_threshold=0.55, # 55% confidence to trade
        prediction_horizon=6,      # 30 minutes ahead
        max_position_size=0.1     # 10% max position
    )
    
    # Load your data
    data = trader.load_eurusd_data()
    if data is None:
        print("❌ Could not load data - check file path")
        return
    
    # Split data for training and testing
    split_point = int(len(data) * 0.9)  # Train on 90%, test on 10%
    train_data = data.iloc[:split_point]
    test_data = data.iloc[split_point:]
    
    print(f"\n📊 Data Split:")
    print(f"   Training: {len(train_data):,} bars ({train_data.index[0]} to {train_data.index[-1]})")
    print(f"   Testing: {len(test_data):,} bars ({test_data.index[0]} to {test_data.index[-1]})")
    
    # Build pattern database from training data
    trader.build_pattern_database(train_data, max_patterns=50000)
    
    # Backtest on test data
    test_days = len(test_data) // 288  # Approximate days in test set
    results = trader.backtest_system(test_data, test_days=test_days)
    
    # Analyze current market
    trader.analyze_current_pattern(data)
    
    # Calculate scaling potential
    if results['total_return'] > 0 and results['num_trades'] > 0:
        daily_return = (1 + results['total_return']) ** (1/test_days) - 1
        monthly_return = (1 + daily_return) ** 21 - 1  # 21 trading days
        
        print(f"\n💰 SCALING ANALYSIS:")
        print(f"   Daily return: {daily_return:.2%}")
        print(f"   Monthly return: {monthly_return:.2%}")
        
        if monthly_return > 0:
            required_capital = 20000 / monthly_return
            print(f"   Capital for $20k/month: ${required_capital:,.0f}")
            
            if required_capital < 500000:
                print(f"   ✅ Target achievable with reasonable capital!")
            else:
                print(f"   ⚠️ Needs optimization or higher capital")
    
    return results


if __name__ == "__main__":
    results = main()