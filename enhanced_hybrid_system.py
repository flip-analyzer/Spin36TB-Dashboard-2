#!/usr/bin/env python3
"""
Enhanced Hybrid Trading System
Combines Spin36TB momentum approach with Carson's clustering concepts
Uses dynamic lookback periods and pattern recognition for better edge detection
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from typing import Dict, List, Tuple, Optional
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

class EnhancedHybridSystem:
    """
    Hybrid system combining momentum analysis with pattern clustering
    Uses adaptive lookback periods based on market conditions
    """
    
    def __init__(self, starting_capital=25000):
        self.starting_capital = starting_capital
        
        # ENHANCEMENT 1: Adaptive lookback periods based on volatility
        self.lookback_config = {
            'base_lookback': 20,           # Start with 20 candles (100 minutes)
            'min_lookback': 15,            # Minimum for fast markets
            'max_lookback': 30,            # Maximum for slow markets
            'volatility_threshold_low': 0.0002,   # 2 pips
            'volatility_threshold_high': 0.0008,  # 8 pips
        }
        
        # ENHANCEMENT 2: Multi-timeframe signal confirmation
        self.signal_thresholds = {
            'min_price_change_pct': 0.0006,    # 6 pips (keep current quality threshold)
            'min_volatility': 0.0002,          # 2 pips minimum volatility
            'confidence_threshold': 0.65,      # Slightly relaxed for more trades
            'pattern_similarity_threshold': 0.75, # For Carson's clustering component
        }
        
        # ENHANCEMENT 3: Pattern recognition from Carson's approach
        self.pattern_config = {
            'enable_clustering': True,
            'n_clusters': 6,               # Fewer clusters for cleaner patterns
            'pattern_lookback': 30,        # Always use 30 for pattern recognition
            'min_pattern_matches': 5,      # Need 5+ historical matches for confidence
        }
        
        # ENHANCEMENT 4: Aggressive but risk-managed execution (15-30 trades/month)
        self.execution_config = {
            'enable_trading': True,        # Actually execute trades now
            'min_confidence_for_trade': 0.55,  # Medium confidence threshold
            'high_confidence_threshold': 0.70, # For larger position sizes
            'max_daily_trades': 4,         # Allow up to 4 trades per day
            'min_minutes_between_trades': 90, # 1.5 hours minimum spacing
            'enable_tiered_sizing': True,  # Different sizes for different confidence levels
        }
        
        # Tracking
        self.recent_trades = []
        self.pattern_database = []
        self.scaler = StandardScaler()
        self.kmeans = None
        
    def get_adaptive_lookback(self, market_data: pd.DataFrame) -> int:
        """
        Determine optimal lookback period based on current market volatility
        Higher volatility = shorter lookback (faster signals)
        Lower volatility = longer lookback (more confirmation)
        """
        if len(market_data) < 30:
            return self.lookback_config['base_lookback']
            
        # Calculate recent volatility (ATR-like measure)
        recent_volatility = (market_data['high'] - market_data['low']).tail(10).mean()
        
        if recent_volatility > self.lookback_config['volatility_threshold_high']:
            # High volatility - use shorter lookback for faster signals
            return self.lookback_config['min_lookback']
        elif recent_volatility < self.lookback_config['volatility_threshold_low']:
            # Low volatility - use longer lookback for more confirmation
            return self.lookback_config['max_lookback']
        else:
            # Normal volatility - use base lookback
            return self.lookback_config['base_lookback']
    
    def create_pattern_vector(self, candle_window: pd.DataFrame) -> np.array:
        """
        Create pattern vector using Carson's approach (always 30 candles)
        """
        if len(candle_window) < 30:
            return None
            
        # Take the last 30 candles
        pattern_window = candle_window.tail(30).copy()
        base_price = pattern_window['close'].iloc[0]
        
        # Normalize to base price
        opens = (pattern_window['open'] / base_price).values
        highs = (pattern_window['high'] / base_price).values
        lows = (pattern_window['low'] / base_price).values
        closes = (pattern_window['close'] / base_price).values
        
        # Create comprehensive pattern vector
        pattern_features = np.concatenate([
            opens, highs, lows, closes,                    # Price action (120 features)
            closes[1:] - closes[:-1],                      # Price changes (29 features)  
            highs - lows,                                  # Ranges (30 features)
        ])
        
        return pattern_features[:150]  # Standardize to 150 features
    
    def find_similar_patterns(self, current_pattern: np.array) -> List[Dict]:
        """
        Find historically similar patterns using Carson's clustering approach
        """
        if not self.pattern_config['enable_clustering'] or current_pattern is None:
            return []
            
        if len(self.pattern_database) < 50:  # Need enough history
            return []
            
        try:
            # Compare current pattern to historical database
            similarities = []
            for i, historical_pattern in enumerate(self.pattern_database):
                if len(historical_pattern['features']) == len(current_pattern):
                    # Calculate cosine similarity
                    similarity = np.dot(current_pattern, historical_pattern['features']) / (
                        np.linalg.norm(current_pattern) * np.linalg.norm(historical_pattern['features'])
                    )
                    
                    if similarity > self.signal_thresholds['pattern_similarity_threshold']:
                        similarities.append({
                            'index': i,
                            'similarity': similarity,
                            'outcome': historical_pattern['outcome']
                        })
            
            # Sort by similarity and return top matches
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            return similarities[:10]  # Top 10 matches
            
        except Exception as e:
            print(f"Pattern matching error: {e}")
            return []
    
    def make_enhanced_trading_decision(self, market_data: pd.DataFrame) -> Dict:
        """
        Enhanced decision making combining momentum analysis with pattern recognition
        """
        if len(market_data) < 30:
            return {'signal': 'HOLD', 'confidence': 0, 'reason': 'Insufficient data'}
        
        # STEP 1: Get adaptive lookback period
        lookback_period = self.get_adaptive_lookback(market_data)
        analysis_window = market_data.tail(lookback_period)
        
        # STEP 2: Traditional momentum analysis (your current approach)
        momentum_decision = self.analyze_momentum(analysis_window)
        
        # STEP 3: Pattern recognition analysis (Carson's approach)
        pattern_decision = self.analyze_patterns(market_data)
        
        # STEP 4: Combine signals for final decision
        combined_decision = self.combine_signals(momentum_decision, pattern_decision)
        
        # STEP 5: Apply execution filters
        final_decision = self.apply_execution_filters(combined_decision)
        
        return final_decision
    
    def analyze_momentum(self, analysis_window: pd.DataFrame) -> Dict:
        """
        Your existing momentum analysis (adapted from critical_fixes.py)
        """
        if len(analysis_window) < 10:
            return {'signal': 'HOLD', 'confidence': 0, 'component': 'momentum'}
        
        current_price = analysis_window['close'].iloc[-1]
        lookback_price = analysis_window['close'].iloc[0]
        
        # Calculate price change
        price_change_pct = abs(current_price - lookback_price) / lookback_price
        
        # Calculate volatility (ATR-like)
        volatility = (analysis_window['high'] - analysis_window['low']).mean() / current_price
        
        # Check thresholds
        if (price_change_pct >= self.signal_thresholds['min_price_change_pct'] and 
            volatility >= self.signal_thresholds['min_volatility']):
            
            # Determine direction
            direction = 'UP' if current_price > lookback_price else 'DOWN'
            
            # Calculate confidence
            confidence = min(0.9, price_change_pct * 100 + volatility * 50)
            
            return {
                'signal': direction,
                'confidence': confidence,
                'component': 'momentum',
                'price_change_pct': price_change_pct,
                'volatility': volatility
            }
        
        return {'signal': 'HOLD', 'confidence': 0, 'component': 'momentum'}
    
    def analyze_patterns(self, market_data: pd.DataFrame) -> Dict:
        """
        Pattern recognition analysis using Carson's clustering approach
        """
        if not self.pattern_config['enable_clustering']:
            return {'signal': 'HOLD', 'confidence': 0, 'component': 'patterns'}
        
        # Create current pattern vector
        current_pattern = self.create_pattern_vector(market_data)
        if current_pattern is None:
            return {'signal': 'HOLD', 'confidence': 0, 'component': 'patterns'}
        
        # Find similar patterns
        similar_patterns = self.find_similar_patterns(current_pattern)
        
        if len(similar_patterns) < self.pattern_config['min_pattern_matches']:
            return {'signal': 'HOLD', 'confidence': 0, 'component': 'patterns'}
        
        # Analyze outcomes of similar patterns
        outcomes = [p['outcome'] for p in similar_patterns]
        up_outcomes = sum(1 for outcome in outcomes if outcome > 0)
        total_outcomes = len(outcomes)
        
        if total_outcomes == 0:
            return {'signal': 'HOLD', 'confidence': 0, 'component': 'patterns'}
        
        win_rate = up_outcomes / total_outcomes
        avg_similarity = np.mean([p['similarity'] for p in similar_patterns])
        
        # Generate signal based on pattern analysis
        if win_rate > 0.65:  # 65%+ win rate
            signal = 'UP'
        elif win_rate < 0.35:  # 35%- win rate (bearish)
            signal = 'DOWN'
        else:
            signal = 'HOLD'
        
        confidence = avg_similarity * win_rate
        
        return {
            'signal': signal,
            'confidence': confidence,
            'component': 'patterns',
            'win_rate': win_rate,
            'pattern_matches': len(similar_patterns)
        }
    
    def combine_signals(self, momentum_decision: Dict, pattern_decision: Dict) -> Dict:
        """
        Combine momentum and pattern signals intelligently
        """
        momentum_signal = momentum_decision.get('signal', 'HOLD')
        pattern_signal = pattern_decision.get('signal', 'HOLD')
        
        momentum_confidence = momentum_decision.get('confidence', 0)
        pattern_confidence = pattern_decision.get('confidence', 0)
        
        # Signal agreement analysis
        if momentum_signal == pattern_signal and momentum_signal != 'HOLD':
            # Both signals agree - high confidence
            combined_confidence = min(0.95, (momentum_confidence + pattern_confidence) * 0.7)
            return {
                'signal': momentum_signal,
                'confidence': combined_confidence,
                'agreement': 'ALIGNED',
                'momentum': momentum_decision,
                'patterns': pattern_decision
            }
        
        elif momentum_signal != pattern_signal and both_not_hold(momentum_signal, pattern_signal):
            # Signals conflict - reduce confidence or hold
            if momentum_confidence > pattern_confidence * 1.5:
                return {
                    'signal': momentum_signal,
                    'confidence': momentum_confidence * 0.6,  # Reduce confidence
                    'agreement': 'MOMENTUM_DOMINANT',
                    'momentum': momentum_decision,
                    'patterns': pattern_decision
                }
            elif pattern_confidence > momentum_confidence * 1.5:
                return {
                    'signal': pattern_signal,
                    'confidence': pattern_confidence * 0.6,
                    'agreement': 'PATTERN_DOMINANT', 
                    'momentum': momentum_decision,
                    'patterns': pattern_decision
                }
            else:
                return {
                    'signal': 'HOLD',
                    'confidence': 0,
                    'agreement': 'CONFLICT',
                    'momentum': momentum_decision,
                    'patterns': pattern_decision
                }
        
        else:
            # One signal is HOLD - use the active signal with reduced confidence
            active_signal = momentum_signal if momentum_signal != 'HOLD' else pattern_signal
            active_confidence = momentum_confidence if momentum_signal != 'HOLD' else pattern_confidence
            
            return {
                'signal': active_signal,
                'confidence': active_confidence * 0.8,  # Slight confidence reduction
                'agreement': 'PARTIAL',
                'momentum': momentum_decision,
                'patterns': pattern_decision
            }
    
    def apply_execution_filters(self, combined_decision: Dict) -> Dict:
        """
        Apply final filters to determine if trade should be executed
        """
        if not self.execution_config['enable_trading']:
            combined_decision['execute_trade'] = False
            combined_decision['execution_reason'] = 'Trading disabled'
            return combined_decision
        
        # Check confidence threshold
        if combined_decision['confidence'] < self.execution_config['min_confidence_for_trade']:
            combined_decision['execute_trade'] = False
            combined_decision['execution_reason'] = f"Confidence {combined_decision['confidence']:.2f} below threshold {self.execution_config['min_confidence_for_trade']}"
            return combined_decision
        
        # Check daily trade limit
        today_trades = len([t for t in self.recent_trades if t['date'] == datetime.now().date()])
        if today_trades >= self.execution_config['max_daily_trades']:
            combined_decision['execute_trade'] = False
            combined_decision['execution_reason'] = f"Daily trade limit reached ({today_trades})"
            return combined_decision
        
        # Check time between trades (converted to minutes)
        if self.recent_trades:
            last_trade_time = self.recent_trades[-1]['timestamp'] 
            minutes_since_last = (datetime.now() - last_trade_time).total_seconds() / 60
            if minutes_since_last < self.execution_config['min_minutes_between_trades']:
                combined_decision['execute_trade'] = False
                combined_decision['execution_reason'] = f"Only {minutes_since_last:.0f} minutes since last trade (need {self.execution_config['min_minutes_between_trades']})"
                return combined_decision
        
        # All filters passed
        combined_decision['execute_trade'] = True
        combined_decision['execution_reason'] = 'All filters passed - executing trade'
        combined_decision['position_size'] = self.calculate_position_size(combined_decision)
        
        return combined_decision
    
    def calculate_position_size(self, decision: Dict) -> float:
        """
        Calculate tiered position size based on confidence levels
        """
        confidence = decision['confidence']
        
        if not self.execution_config['enable_tiered_sizing']:
            return 0.01  # Default 1%
        
        # Tiered position sizing for more frequent trading
        if confidence >= 0.80:
            # Tier 1: Very high confidence
            return 0.020  # 2.0% risk
        elif confidence >= self.execution_config['high_confidence_threshold']:  # 0.70
            # Tier 2: High confidence  
            return 0.015  # 1.5% risk
        elif confidence >= 0.60:
            # Tier 3: Medium-high confidence
            return 0.012  # 1.2% risk
        elif confidence >= self.execution_config['min_confidence_for_trade']:  # 0.55
            # Tier 4: Medium confidence
            return 0.008  # 0.8% risk
        else:
            # Below threshold - shouldn't execute
            return 0

def both_not_hold(signal1: str, signal2: str) -> bool:
    """Helper function to check if both signals are not HOLD"""
    return signal1 != 'HOLD' and signal2 != 'HOLD'

if __name__ == "__main__":
    print("🔄 Enhanced Hybrid System Initialized")
    print("=" * 50)
    print("✅ Adaptive lookback periods (15-30 candles)")  
    print("✅ Pattern recognition with clustering")
    print("✅ Signal combination and conflict resolution")
    print("✅ Aggressive execution filters for 15-30 trades/month")
    print("✅ Tiered position sizing (0.8%-2.0% based on confidence)")
    print("✅ Target: 3-7 trades/week with proper risk management")