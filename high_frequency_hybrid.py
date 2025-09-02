#!/usr/bin/env python3
"""
High-Frequency Hybrid Trading System
Combines your successful Spin36TB momentum approach with Carson's cluster-based pattern recognition
Target: 20-30 trades per day through multiple signal sources and faster decision cycles
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from typing import Dict, List, Tuple, Optional
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')

class HighFrequencyHybridSystem:
    """
    High-frequency hybrid system combining momentum + clustering for 20-30 trades/day
    Uses your proven momentum foundation with Carson's rapid pattern recognition
    """
    
    def __init__(self, starting_capital=25000):
        self.starting_capital = starting_capital
        
        # HYBRID APPROACH 1: Dual Time Horizons
        self.time_horizons = {
            'micro_signals': {
                'lookback_minutes': 15,      # 3 candles (5min chart) - Carson's approach
                'decision_frequency_minutes': 5,  # Decide every 5 minutes
                'hold_time_minutes': 15,     # Quick scalps
                'confidence_threshold': 0.50, # Lower threshold for micro signals
            },
            'momentum_signals': {
                'lookback_minutes': 75,      # 15 candles - Your approach  
                'decision_frequency_minutes': 5,  # Also every 5 minutes
                'hold_time_minutes': 45,     # Medium-term holds
                'confidence_threshold': 0.60, # Higher threshold for momentum
            },
            'pattern_signals': {
                'lookback_minutes': 150,     # 30 candles - Carson's clustering
                'decision_frequency_minutes': 10, # Every 10 minutes
                'hold_time_minutes': 90,     # Longer pattern holds
                'confidence_threshold': 0.55, # Medium threshold for patterns
            }
        }
        
        # HYBRID APPROACH 2: Multi-Source Signal Generation
        self.signal_sources = {
            'momentum_micro': {
                'enabled': True,
                'weight': 0.25,              # 25% allocation
                'max_concurrent_trades': 3,  # Allow 3 micro trades
                'position_size_base': 0.003, # 0.3% per trade (small & frequent)
            },
            'momentum_standard': {
                'enabled': True, 
                'weight': 0.35,              # 35% allocation - your proven system
                'max_concurrent_trades': 2,  # 2 standard momentum trades
                'position_size_base': 0.008, # 0.8% per trade 
            },
            'cluster_patterns': {
                'enabled': True,
                'weight': 0.25,              # 25% allocation - Carson's system
                'max_concurrent_trades': 2,  # 2 pattern-based trades
                'position_size_base': 0.006, # 0.6% per trade
            },
            'hybrid_confluence': {
                'enabled': True,
                'weight': 0.15,              # 15% allocation - when all agree
                'max_concurrent_trades': 1,  # 1 high-conviction trade
                'position_size_base': 0.015, # 1.5% per trade (bigger when all agree)
            }
        }
        
        # HYBRID APPROACH 3: Aggressive Execution Settings
        self.execution_config = {
            'max_total_concurrent_trades': 8,    # Up to 8 trades running simultaneously
            'max_trades_per_hour': 6,           # 6 trades/hour = ~20-30/day during active hours
            'min_minutes_between_same_source': 10, # 10 min between trades from same source
            'min_minutes_between_any_trade': 3,    # 3 min between any trades
            'active_trading_hours_utc': [(7, 11), (13, 17), (19, 23)], # 12 hours active
            'enable_micro_scalping': True,       # Enable rapid micro trades
        }
        
        # Clustering setup (Carson's approach)
        self.cluster_config = {
            'n_clusters': 8,
            'pattern_window': 30,
            'n_pca_components': 20,
            'min_cluster_size': 10,
            'similarity_threshold': 0.75,
        }
        
        # Tracking
        self.active_trades = []
        self.recent_trades = []
        self.pattern_database = []
        self.cluster_stats = {}
        self.scaler = StandardScaler()
        self.pca = None
        self.kmeans = None
        
        print("🚀 HIGH-FREQUENCY HYBRID SYSTEM INITIALIZED")
        print("=" * 50)
        print("⚡ Target: 20-30 trades per day")
        print("🔄 Decision cycle: Every 5 minutes")
        print("📊 Multi-source signals: Micro + Momentum + Patterns + Confluence")
        print("🎯 Up to 8 concurrent positions")
        
    def should_make_decision(self, signal_source: str) -> bool:
        """
        Check if it's time to make a decision for a specific signal source
        """
        current_time = datetime.now()
        
        # Check if we're in active trading hours
        current_hour = current_time.hour
        active_hours = self.execution_config['active_trading_hours_utc']
        in_active_hours = any(start <= current_hour < end for start, end in active_hours)
        
        if not in_active_hours:
            return False
        
        # Check frequency requirements for this signal source
        source_config = self.time_horizons.get(f"{signal_source}_signals", self.time_horizons['momentum_signals'])
        frequency_minutes = source_config['decision_frequency_minutes']
        
        # Check last decision time for this source
        last_decision = getattr(self, f'last_{signal_source}_decision', None)
        if last_decision:
            minutes_since = (current_time - last_decision).total_seconds() / 60
            if minutes_since < frequency_minutes:
                return False
        
        # Check trade limits
        if not self.can_add_trade(signal_source):
            return False
            
        return True
    
    def can_add_trade(self, signal_source: str) -> bool:
        """
        Check if we can add another trade from this signal source
        """
        current_time = datetime.now()
        
        # Count current active trades
        total_active = len([t for t in self.active_trades if self.is_trade_active(t, current_time)])
        if total_active >= self.execution_config['max_total_concurrent_trades']:
            return False
        
        # Count active trades from this source
        source_active = len([t for t in self.active_trades if t['source'] == signal_source and self.is_trade_active(t, current_time)])
        source_config = self.signal_sources.get(signal_source, {})
        max_concurrent = source_config.get('max_concurrent_trades', 1)
        
        if source_active >= max_concurrent:
            return False
        
        # Check hourly limit
        hour_ago = current_time - timedelta(hours=1)
        recent_trades_count = len([t for t in self.recent_trades if t['timestamp'] > hour_ago])
        if recent_trades_count >= self.execution_config['max_trades_per_hour']:
            return False
        
        # Check minimum time between trades
        if self.recent_trades:
            last_trade_time = self.recent_trades[-1]['timestamp']
            minutes_since = (current_time - last_trade_time).total_seconds() / 60
            
            # Different minimums for same source vs any trade
            same_source_trades = [t for t in self.recent_trades if t['source'] == signal_source]
            if same_source_trades:
                last_same_source = same_source_trades[-1]['timestamp']
                minutes_since_same = (current_time - last_same_source).total_seconds() / 60
                if minutes_since_same < self.execution_config['min_minutes_between_same_source']:
                    return False
            
            if minutes_since < self.execution_config['min_minutes_between_any_trade']:
                return False
        
        return True
    
    def is_trade_active(self, trade: Dict, current_time: datetime) -> bool:
        """Check if a trade is still active"""
        if trade.get('status') in ['closed', 'expired']:
            return False
        
        entry_time = trade['timestamp']
        hold_time_minutes = trade.get('hold_time_minutes', 45)
        
        return (current_time - entry_time).total_seconds() / 60 < hold_time_minutes
    
    def generate_micro_momentum_signal(self, market_data: pd.DataFrame) -> Dict:
        """
        Generate rapid micro-momentum signals (Carson's speed, your momentum logic)
        """
        if len(market_data) < 5:
            return {'signal': 'HOLD', 'confidence': 0, 'source': 'momentum_micro'}
        
        # Very short lookback for micro signals
        window = market_data.tail(3)  # Just 3 candles (15 minutes)
        
        current_price = window['close'].iloc[-1]
        start_price = window['close'].iloc[0]
        
        # Quick momentum calculation
        price_change_pct = (current_price - start_price) / start_price
        volatility = (window['high'] - window['low']).mean() / current_price
        
        # Very sensitive thresholds for micro trading
        if abs(price_change_pct) >= 0.0003 and volatility >= 0.0001:  # 3 pips, 1 pip vol
            direction = 'UP' if price_change_pct > 0 else 'DOWN'
            confidence = min(0.8, abs(price_change_pct) * 200 + volatility * 100)
            
            return {
                'signal': direction,
                'confidence': confidence,
                'source': 'momentum_micro',
                'hold_time_minutes': 15,  # Quick scalps
                'price_change_pct': price_change_pct,
                'volatility': volatility
            }
        
        return {'signal': 'HOLD', 'confidence': 0, 'source': 'momentum_micro'}
    
    def generate_standard_momentum_signal(self, market_data: pd.DataFrame) -> Dict:
        """
        Your proven momentum system (15-candle lookback)
        """
        if len(market_data) < 15:
            return {'signal': 'HOLD', 'confidence': 0, 'source': 'momentum_standard'}
        
        # Your standard 15-candle analysis
        window = market_data.tail(15)
        current_price = window['close'].iloc[-1]
        start_price = window['close'].iloc[0]
        
        price_change_pct = abs(current_price - start_price) / start_price
        volatility = (window['high'] - window['low']).mean() / current_price
        
        # Your proven thresholds
        if price_change_pct >= 0.0006 and volatility >= 0.0002:  # 6 pips, 2 pip vol
            direction = 'UP' if current_price > start_price else 'DOWN'
            confidence = min(0.9, price_change_pct * 100 + volatility * 50)
            
            return {
                'signal': direction,
                'confidence': confidence,
                'source': 'momentum_standard',
                'hold_time_minutes': 45,
                'price_change_pct': price_change_pct,
                'volatility': volatility
            }
        
        return {'signal': 'HOLD', 'confidence': 0, 'source': 'momentum_standard'}
    
    def generate_cluster_pattern_signal(self, market_data: pd.DataFrame) -> Dict:
        """
        Simplified pattern recognition (will enhance once clustering model is trained)
        For now, uses statistical pattern analysis
        """
        if len(market_data) < 30:
            return {'signal': 'HOLD', 'confidence': 0, 'source': 'cluster_patterns'}
        
        # Simple pattern analysis until clustering model is ready
        pattern_window = market_data.tail(30)
        
        # Calculate pattern features
        closes = pattern_window['close'].values
        highs = pattern_window['high'].values
        lows = pattern_window['low'].values
        
        # Pattern analysis
        price_trend = (closes[-1] - closes[0]) / closes[0]  # Overall trend
        volatility = np.std(closes) / np.mean(closes)  # Volatility measure
        range_compression = np.mean(highs - lows) / np.mean(closes)  # Range analysis
        
        # Recent momentum (last 10 vs previous 10)
        recent_avg = np.mean(closes[-10:])
        previous_avg = np.mean(closes[-20:-10])
        momentum = (recent_avg - previous_avg) / previous_avg
        
        # Simple pattern rules (will be replaced with ML clustering)
        signal_strength = abs(price_trend) + abs(momentum)
        
        if signal_strength > 0.002 and volatility > 0.001:  # 20 pips + volatility threshold
            direction = 'UP' if (price_trend + momentum) > 0 else 'DOWN'
            confidence = min(0.85, signal_strength * 200 + volatility * 100)
            
            # Only trade if confidence is reasonable
            if confidence > 0.55:
                return {
                    'signal': direction,
                    'confidence': confidence,
                    'source': 'cluster_patterns',
                    'hold_time_minutes': 90,
                    'pattern_trend': price_trend,
                    'momentum': momentum,
                    'volatility': volatility
                }
        
        return {'signal': 'HOLD', 'confidence': 0, 'source': 'cluster_patterns'}
    
    def generate_confluence_signal(self, micro_signal: Dict, momentum_signal: Dict, pattern_signal: Dict) -> Dict:
        """
        Generate high-conviction signal when multiple sources agree
        """
        signals = [micro_signal, momentum_signal, pattern_signal]
        active_signals = [s for s in signals if s['signal'] != 'HOLD']
        
        if len(active_signals) < 2:
            return {'signal': 'HOLD', 'confidence': 0, 'source': 'hybrid_confluence'}
        
        # Check for agreement
        directions = [s['signal'] for s in active_signals]
        if len(set(directions)) != 1:  # No consensus
            return {'signal': 'HOLD', 'confidence': 0, 'source': 'hybrid_confluence'}
        
        # All agree on direction
        agreed_direction = directions[0]
        avg_confidence = np.mean([s['confidence'] for s in active_signals])
        boosted_confidence = min(0.95, avg_confidence * 1.3)  # 30% confidence boost for agreement
        
        return {
            'signal': agreed_direction,
            'confidence': boosted_confidence,
            'source': 'hybrid_confluence',
            'hold_time_minutes': 60,
            'num_agreeing_signals': len(active_signals),
            'component_signals': active_signals
        }
    
    def create_pattern_vector(self, candle_window: pd.DataFrame) -> np.array:
        """Carson's pattern vector creation"""
        if len(candle_window) != 30:
            return None
            
        base_price = candle_window['close'].iloc[0]
        
        # Normalize prices
        opens = (candle_window['open'] / base_price).values
        highs = (candle_window['high'] / base_price).values
        lows = (candle_window['low'] / base_price).values
        closes = (candle_window['close'] / base_price).values
        
        # Create pattern features
        pattern_features = np.concatenate([
            opens, highs, lows, closes,           # Price action (120 features)
            closes[1:] - closes[:-1],             # Price changes (29 features)
        ])
        
        return pattern_features[:149]  # Exactly 149 features
    
    def predict_from_pattern(self, pattern_vector: np.array) -> Tuple[str, float, float, str]:
        """Carson's prediction method"""
        if self.kmeans is None:
            return "HOLD", 0.0, 0.0, "Model not trained"
        
        try:
            # Transform through pipeline
            pattern_scaled = self.scaler.transform([pattern_vector])
            pattern_pca = self.pca.transform(pattern_scaled)
            cluster_id = self.kmeans.predict(pattern_pca)[0]
            
            # Get cluster stats
            if cluster_id not in self.cluster_stats:
                return "HOLD", 0.0, 0.0, "Unknown cluster"
                
            stats = self.cluster_stats[cluster_id]
            direction = "UP" if stats['direction'] == "BUY" else ("DOWN" if stats['direction'] == "SELL" else "HOLD")
            
            return direction, stats['confidence'], stats['expected_return'], f"Cluster {cluster_id}: {stats['size']} patterns"
            
        except Exception as e:
            return "HOLD", 0.0, 0.0, f"Prediction error: {str(e)}"
    
    def make_high_frequency_decision(self, market_data: pd.DataFrame) -> List[Dict]:
        """
        Main decision engine - checks all signal sources and generates trades
        """
        current_time = datetime.now()
        potential_trades = []
        
        # Generate signals from all sources
        micro_signal = self.generate_micro_momentum_signal(market_data) if self.should_make_decision('momentum_micro') else {'signal': 'HOLD'}
        momentum_signal = self.generate_standard_momentum_signal(market_data) if self.should_make_decision('momentum_standard') else {'signal': 'HOLD'}
        pattern_signal = self.generate_cluster_pattern_signal(market_data) if self.should_make_decision('cluster_patterns') else {'signal': 'HOLD'}
        confluence_signal = self.generate_confluence_signal(micro_signal, momentum_signal, pattern_signal) if self.should_make_decision('hybrid_confluence') else {'signal': 'HOLD'}
        
        # Process each signal
        all_signals = [micro_signal, momentum_signal, pattern_signal, confluence_signal]
        
        for signal in all_signals:
            if signal['signal'] != 'HOLD' and self.can_add_trade(signal['source']):
                # Calculate position size
                position_size = self.calculate_position_size(signal)
                
                if position_size > 0:
                    trade = {
                        'timestamp': current_time,
                        'source': signal['source'],
                        'signal': signal['signal'],
                        'confidence': signal['confidence'],
                        'position_size': position_size,
                        'hold_time_minutes': signal.get('hold_time_minutes', 45),
                        'entry_price': market_data['close'].iloc[-1],
                        'status': 'active'
                    }
                    
                    potential_trades.append(trade)
                    
                    # Update last decision time
                    setattr(self, f"last_{signal['source'].split('_')[0]}_decision", current_time)
        
        return potential_trades
    
    def calculate_position_size(self, signal: Dict) -> float:
        """Calculate position size based on signal source and confidence"""
        source = signal['source']
        confidence = signal['confidence']
        
        if source not in self.signal_sources:
            return 0
        
        source_config = self.signal_sources[source]
        if not source_config['enabled']:
            return 0
        
        base_size = source_config['position_size_base']
        confidence_multiplier = min(2.0, confidence * 1.5)  # Cap at 2x multiplier
        
        return base_size * confidence_multiplier

if __name__ == "__main__":
    print("🚀 High-Frequency Hybrid System Ready")
    print("=" * 50) 
    print("📈 Combining your proven momentum system with Carson's clustering")
    print("⚡ Target: 20-30 trades per day")
    print("🔄 Multiple signal sources running in parallel")
    print("🎯 Up to 8 concurrent positions")
    print("⏱️  Decision cycles every 3-5 minutes")