#!/usr/bin/env python3
"""
Cluster-Based Trading System

Uses cluster analysis results to:
1. Predict direction based on pattern similarity within clusters
2. Calculate confidence from cluster outcome statistics
3. Size positions based on cluster edge and confidence
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class ClusterBasedTrader:
    def __init__(self, window_size=30, n_clusters=8, confidence_threshold=0.60):
        self.window_size = window_size
        self.n_clusters = n_clusters
        self.confidence_threshold = confidence_threshold
        
        # Model components
        self.scaler = None
        self.pca = None
        self.kmeans = None
        self.cluster_stats = {}
        
        # Historical data
        self.patterns_database = []
        self.outcomes_database = []
        self.cluster_labels = []
        
    def create_pattern_vector(self, candle_window):
        """Create 150-element pattern vector from 30 candles"""
        if len(candle_window) != self.window_size:
            return None
            
        base_price = candle_window['Close'].iloc[0]
        
        # Normalize prices to base
        opens = (candle_window['Open'] / base_price).values
        highs = (candle_window['High'] / base_price).values
        lows = (candle_window['Low'] / base_price).values
        closes = (candle_window['Close'] / base_price).values
        volumes = (candle_window['Volume'] / candle_window['Volume'].mean()).values
        
        # Pattern features
        ranges = (highs - lows) / closes
        bodies = np.abs(closes - opens) / closes
        upper_wicks = (highs - np.maximum(opens, closes)) / closes
        lower_wicks = (np.minimum(opens, closes) - lows) / closes
        
        # 150-element vector: 30 each of closes, ranges, bodies, upper_wicks, lower_wicks
        pattern_vector = np.concatenate([
            closes, ranges, bodies, upper_wicks, lower_wicks
        ])
        
        return pattern_vector
    
    def train_cluster_model(self, data):
        """Train the cluster model on historical patterns"""
        print(f"🧠 Training Cluster Model...")
        print(f"   Window: {self.window_size} candles, Clusters: {self.n_clusters}")
        
        self.patterns_database = []
        self.outcomes_database = []
        
        # Extract patterns and outcomes (sample for speed)
        max_patterns = min(1000, (len(data) - self.window_size - 6) // 10)
        step_size = max(1, (len(data) - self.window_size - 6) // max_patterns)
        
        for i in range(0, len(data) - self.window_size - 6, step_size):
            pattern_window = data.iloc[i:i + self.window_size]
            pattern_vector = self.create_pattern_vector(pattern_window)
            
            if pattern_vector is None:
                continue
                
            # 30-minute outcome (6 bars forward)
            entry_price = pattern_window['Close'].iloc[-1]
            exit_idx = min(i + self.window_size + 6, len(data) - 1)
            exit_price = data.iloc[exit_idx]['Close']
            outcome = (exit_price / entry_price) - 1
            
            self.patterns_database.append(pattern_vector)
            self.outcomes_database.append(outcome)
        
        self.patterns_database = np.array(self.patterns_database)
        self.outcomes_database = np.array(self.outcomes_database)
        
        print(f"   ✅ Extracted {len(self.patterns_database):,} patterns")
        
        # Standardize patterns
        self.scaler = StandardScaler()
        patterns_scaled = self.scaler.fit_transform(self.patterns_database)
        
        # Apply PCA for dimensionality reduction
        self.pca = PCA(n_components=20)  # Keep 20 components for clustering
        patterns_pca = self.pca.fit_transform(patterns_scaled)
        
        # Fit K-means clustering
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        self.cluster_labels = self.kmeans.fit_predict(patterns_pca)
        
        # Calculate cluster statistics
        self._calculate_cluster_stats()
        
        print(f"   📊 Cluster Training Complete")
        return True
    
    def _calculate_cluster_stats(self):
        """Calculate predictive statistics for each cluster"""
        print(f"   📊 Analyzing Cluster Performance...")
        
        self.cluster_stats = {}
        
        for cluster_id in range(self.n_clusters):
            mask = self.cluster_labels == cluster_id
            cluster_outcomes = self.outcomes_database[mask]
            
            if len(cluster_outcomes) == 0:
                continue
                
            # Calculate directional probabilities
            strong_up_prob = (cluster_outcomes > 0.001).mean()    # >10 pips
            weak_up_prob = ((cluster_outcomes > 0.0005) & (cluster_outcomes <= 0.001)).mean()
            neutral_prob = ((cluster_outcomes >= -0.0005) & (cluster_outcomes <= 0.0005)).mean()
            weak_down_prob = ((cluster_outcomes < -0.0005) & (cluster_outcomes >= -0.001)).mean()
            strong_down_prob = (cluster_outcomes < -0.001).mean()  # <-10 pips
            
            # Overall directional bias
            up_prob = strong_up_prob + weak_up_prob
            down_prob = strong_down_prob + weak_down_prob
            
            # Expected return
            expected_return = cluster_outcomes.mean()
            
            # Confidence metrics
            volatility = cluster_outcomes.std()
            sharpe_like = expected_return / volatility if volatility > 0 else 0
            
            # Determine primary direction and confidence
            if up_prob > down_prob and up_prob > 0.55:
                direction = "BUY"
                confidence = up_prob
                edge = expected_return if expected_return > 0 else 0
            elif down_prob > up_prob and down_prob > 0.55:
                direction = "SELL" 
                confidence = down_prob
                edge = abs(expected_return) if expected_return < 0 else 0
            else:
                direction = "HOLD"
                confidence = neutral_prob
                edge = 0
            
            self.cluster_stats[cluster_id] = {
                'size': mask.sum(),
                'direction': direction,
                'confidence': confidence,
                'edge': edge,
                'expected_return': expected_return,
                'volatility': volatility,
                'sharpe_like': sharpe_like,
                'up_prob': up_prob,
                'down_prob': down_prob,
                'neutral_prob': neutral_prob,
                'strong_up_prob': strong_up_prob,
                'strong_down_prob': strong_down_prob
            }
            
            print(f"      Cluster {cluster_id}: {mask.sum():4d} patterns, "
                  f"{direction:4s}, conf={confidence:.2%}, edge={edge:.4f}")
    
    def predict_from_pattern(self, current_pattern):
        """Predict direction and confidence for a current pattern"""
        if self.kmeans is None:
            return "HOLD", 0.0, 0.0, "Model not trained"
        
        # Transform pattern through same pipeline
        pattern_scaled = self.scaler.transform([current_pattern])
        pattern_pca = self.pca.transform(pattern_scaled)
        
        # Predict cluster
        cluster_id = self.kmeans.predict(pattern_pca)[0]
        
        if cluster_id not in self.cluster_stats:
            return "HOLD", 0.0, 0.0, f"Unknown cluster {cluster_id}"
        
        stats = self.cluster_stats[cluster_id]
        
        # Get prediction from cluster stats
        direction = stats['direction']
        confidence = stats['confidence']
        expected_return = stats['expected_return']
        
        analysis = (f"Cluster {cluster_id}: {stats['size']} patterns, "
                   f"up={stats['up_prob']:.1%}, down={stats['down_prob']:.1%}")
        
        return direction, confidence, expected_return, analysis
    
    def calculate_position_size(self, direction, confidence, expected_return, 
                               max_position=0.05, kelly_fraction=0.25):
        """Calculate position size using cluster-based Kelly criterion"""
        
        if direction == "HOLD" or confidence < self.confidence_threshold:
            return 0.0
        
        # Basic confidence-based sizing
        confidence_adjustment = (confidence - self.confidence_threshold) / (1.0 - self.confidence_threshold)
        
        # Kelly-inspired sizing based on expected return and confidence
        # Kelly = (bp - q) / b, where b=payoff ratio, p=win prob, q=lose prob
        
        # Simplified: use confidence as win probability and expected return as edge
        if expected_return > 0 and confidence > 0.5:
            # Estimate win/loss ratio from expected return and confidence
            win_prob = confidence
            lose_prob = 1 - confidence
            
            # Assume 1:1 risk/reward for simplicity (can be improved)
            payoff_ratio = 1.0
            
            kelly_fraction_calc = (payoff_ratio * win_prob - lose_prob) / payoff_ratio
            kelly_fraction_calc = max(0, min(kelly_fraction_calc, 0.25))  # Cap at 25%
            
            # Blend Kelly with confidence-based sizing
            kelly_size = kelly_fraction_calc * kelly_fraction
            confidence_size = max_position * confidence_adjustment
            
            # Use more conservative of the two
            position_size = min(kelly_size, confidence_size)
        else:
            # Fallback to confidence-based sizing
            position_size = max_position * confidence_adjustment * 0.5  # More conservative
        
        return min(position_size, max_position)
    
    def analyze_current_market(self, data, lookback_bars=30):
        """Analyze current market conditions"""
        if len(data) < lookback_bars:
            return None
            
        print(f"\n🔮 CURRENT MARKET ANALYSIS")
        
        # Get latest pattern
        current_window = data.tail(lookback_bars)
        current_pattern = self.create_pattern_vector(current_window)
        
        if current_pattern is None:
            return None
        
        # Make prediction
        direction, confidence, expected_return, analysis = self.predict_from_pattern(current_pattern)
        
        # Calculate position size
        position_size = self.calculate_position_size(direction, confidence, expected_return)
        
        print(f"   📊 {analysis}")
        print(f"   🎯 Prediction: {direction}")
        print(f"   🎲 Confidence: {confidence:.2%}")
        print(f"   💰 Expected Return: {expected_return:+.4f} ({expected_return*10000:+.1f} pips)")
        print(f"   📏 Position Size: {position_size:.2%}")
        
        if position_size > 0:
            print(f"   ✅ TRADE SIGNAL: {direction} {position_size:.1%}")
        else:
            print(f"   ⚠️  NO TRADE: Confidence below threshold ({self.confidence_threshold:.1%})")
            
        return {
            'direction': direction,
            'confidence': confidence,
            'expected_return': expected_return,
            'position_size': position_size,
            'analysis': analysis
        }
    
    def backtest_cluster_strategy(self, test_data):
        """Backtest the cluster-based strategy"""
        print(f"\n📈 Backtesting Cluster Strategy...")
        
        portfolio_value = 100000
        trades = []
        
        for i in range(self.window_size, len(test_data) - 6):
            current_time = test_data.index[i]
            
            # Get pattern window
            pattern_window = test_data.iloc[i-self.window_size:i]
            current_pattern = self.create_pattern_vector(pattern_window)
            
            if current_pattern is None:
                continue
            
            # Make prediction
            direction, confidence, expected_return, analysis = self.predict_from_pattern(current_pattern)
            position_size = self.calculate_position_size(direction, confidence, expected_return)
            
            if position_size > 0 and direction in ["BUY", "SELL"]:
                # Execute trade
                entry_price = test_data.iloc[i]['Close']
                exit_price = test_data.iloc[i + 6]['Close']  # 30-minute hold
                
                if direction == "BUY":
                    trade_return = (exit_price / entry_price - 1) * position_size
                else:  # SELL
                    trade_return = (entry_price / exit_price - 1) * position_size
                
                portfolio_value *= (1 + trade_return)
                
                trades.append({
                    'time': current_time,
                    'direction': direction,
                    'confidence': confidence,
                    'position_size': position_size,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'return': trade_return,
                    'portfolio_value': portfolio_value
                })
        
        # Calculate results
        total_return = (portfolio_value / 100000) - 1
        winning_trades = len([t for t in trades if t['return'] > 0])
        win_rate = winning_trades / len(trades) if trades else 0
        
        print(f"   ✅ Backtest Results:")
        print(f"      Total Return: {total_return:.2%}")
        print(f"      Trades: {len(trades)}")
        print(f"      Win Rate: {win_rate:.2%}")
        print(f"      Final Value: ${portfolio_value:,.2f}")
        
        if len(trades) > 0:
            avg_confidence = np.mean([t['confidence'] for t in trades])
            avg_position = np.mean([t['position_size'] for t in trades])
            print(f"      Avg Confidence: {avg_confidence:.2%}")
            print(f"      Avg Position: {avg_position:.2%}")
        
        return {
            'total_return': total_return,
            'win_rate': win_rate,
            'num_trades': len(trades),
            'trades': trades,
            'final_value': portfolio_value
        }


def main():
    """Run cluster-based trading system"""
    
    print("🎯 CLUSTER-BASED EURUSD TRADER")
    print("=" * 40)
    
    # Load EURUSD data
    print("📊 Loading EURUSD Data...")
    try:
        data_path = "/Users/jonspinogatti/Desktop/spin35TB/data/candles/eurusd_m5_500k.csv"
        data = pd.read_csv(data_path)
        
        data['time'] = pd.to_datetime(data['time'])
        data = data.set_index('time')
        data = data.rename(columns={
            'open': 'Open', 'high': 'High', 'low': 'Low', 
            'close': 'Close', 'volume': 'Volume'
        })
        
        print(f"   ✅ Loaded {len(data):,} bars")
    except Exception as e:
        print(f"   ❌ Error loading data: {e}")
        return
    
    # Initialize trader
    trader = ClusterBasedTrader(
        window_size=30,
        n_clusters=8, 
        confidence_threshold=0.60
    )
    
    # Split data for training/testing
    split_idx = int(len(data) * 0.8)
    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:]
    
    print(f"Training: {len(train_data):,} bars")
    print(f"Testing: {len(test_data):,} bars")
    
    # Train cluster model
    if not trader.train_cluster_model(train_data):
        return
    
    # Analyze current market
    current_analysis = trader.analyze_current_market(data)
    
    # Backtest strategy
    backtest_results = trader.backtest_cluster_strategy(test_data)
    
    # Performance analysis
    print(f"\n💰 CLUSTER STRATEGY PERFORMANCE:")
    print(f"   📈 Test Period Return: {backtest_results['total_return']:.2%}")
    print(f"   🎯 Win Rate: {backtest_results['win_rate']:.2%}")
    print(f"   📊 Total Trades: {backtest_results['num_trades']}")
    
    # Scale to monthly income target
    if backtest_results['total_return'] > 0:
        test_days = len(test_data) / (78 * 5)  # ~78 bars per day, 5 days per week
        daily_return = (1 + backtest_results['total_return']) ** (1/test_days) - 1
        monthly_return = (1 + daily_return) ** 21 - 1  # 21 trading days
        
        required_capital = 20000 / monthly_return if monthly_return > 0 else float('inf')
        
        print(f"\n🎯 SCALING TO $20K/MONTH:")
        print(f"   Daily Return: {daily_return:.3%}")
        print(f"   Monthly Return: {monthly_return:.2%}")
        print(f"   Required Capital: ${required_capital:,.0f}")
        
        if required_capital < 500000:
            print(f"   ✅ Cluster approach looks promising!")
        else:
            print(f"   ⚠️  Need better performance or more capital")


if __name__ == "__main__":
    main()