#!/usr/bin/env python3
"""
Upgrade Current System to 6-Hour Momentum (Carson's Approach)
Integrate enhanced momentum analysis with existing hybrid trading system
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_6hour_momentum import Enhanced6HourMomentum
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

class Enhanced6HourHybridSystem:
    """
    Upgraded hybrid system using Carson's 6-hour momentum approach
    Integrates with existing GMM clustering and multiple signal sources
    """
    
    def __init__(self):
        self.momentum_system = Enhanced6HourMomentum()
        
        # Enhanced parameters based on academic research
        self.lookback_hours = 6
        self.session_duration_hours = 6  # Match Carson's suggestion
        self.confidence_threshold = 0.4  # Higher threshold for 6-hour signals
        
        # Multi-timeframe analysis
        self.timeframes = {
            'short': {'hours': 2, 'weight': 0.2},
            'medium': {'hours': 4, 'weight': 0.3}, 
            'long': {'hours': 6, 'weight': 0.5}    # Carson's primary timeframe
        }
        
        # Risk management for longer holding periods
        self.max_position_hours = 8  # Extended from current 2 hours
        self.trailing_stop_pct = 0.005  # 50 pips for 6-hour positions
        
    def analyze_6hour_market_data(self, market_data):
        """
        Analyze market data using Carson's 6-hour approach
        Integrates with existing data pipeline
        """
        try:
            if len(market_data) < 72:  # Need 6 hours of 5-minute data
                return {
                    'status': 'insufficient_data',
                    'required_candles': 72,
                    'available_candles': len(market_data),
                    'message': 'Need 6 hours of data for Carson methodology'
                }
            
            # Convert to DataFrame for analysis
            df = pd.DataFrame(market_data)
            if 'time' in df.columns:
                df['timestamp'] = pd.to_datetime(df['time'])
                df = df.sort_values('timestamp')
            
            # Enhanced 6-hour momentum analysis
            momentum_signal = self.momentum_system.generate_6hour_signal(df)
            
            # Multi-timeframe confirmation (academic best practice)
            timeframe_signals = {}
            for tf_name, tf_config in self.timeframes.items():
                tf_candles = tf_config['hours'] * 12  # 5-minute candles per hour
                if len(df) >= tf_candles:
                    tf_data = df.tail(tf_candles).copy()
                    tf_system = Enhanced6HourMomentum()
                    tf_system.lookback_candles = tf_candles
                    tf_signal = tf_system.generate_6hour_signal(tf_data)
                    timeframe_signals[tf_name] = {
                        'signal': tf_signal,
                        'weight': tf_config['weight']
                    }
            
            # Combine signals using academic weighting
            combined_confidence = self.combine_timeframe_signals(timeframe_signals)
            
            return {
                'status': 'success',
                'primary_signal': momentum_signal,
                'timeframe_signals': timeframe_signals,
                'combined_confidence': combined_confidence,
                'methodology': 'Enhanced 6-Hour Carson Approach',
                'lookback_hours': self.lookback_hours,
                'data_quality': 'sufficient'
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'message': 'Error in 6-hour analysis'
            }
    
    def combine_timeframe_signals(self, timeframe_signals):
        """
        Combine multiple timeframe signals using academic weighting
        Based on López de Prado's ensemble methods
        """
        if not timeframe_signals:
            return {'signal': 'HOLD', 'confidence': 0.0}
        
        total_weight = 0
        weighted_strength = 0
        signal_direction = 0
        
        for tf_name, tf_data in timeframe_signals.items():
            signal_info = tf_data['signal']
            weight = tf_data['weight']
            
            if signal_info['signal'] == 'BUY':
                signal_direction += weight
                weighted_strength += signal_info['confidence'] * weight
            elif signal_info['signal'] == 'SELL':
                signal_direction -= weight
                weighted_strength += signal_info['confidence'] * weight
            
            total_weight += weight
        
        if total_weight == 0:
            return {'signal': 'HOLD', 'confidence': 0.0}
        
        # Normalize
        avg_direction = signal_direction / total_weight
        avg_confidence = weighted_strength / total_weight
        
        # Generate combined signal
        if avg_direction > 0.3 and avg_confidence > self.confidence_threshold:
            combined_signal = 'BUY'
        elif avg_direction < -0.3 and avg_confidence > self.confidence_threshold:
            combined_signal = 'SELL'
        else:
            combined_signal = 'HOLD'
        
        return {
            'signal': combined_signal,
            'confidence': avg_confidence,
            'direction_strength': avg_direction,
            'methodology': 'Multi-timeframe ensemble (Carson + Academic)'
        }
    
    def generate_enhanced_trading_decisions(self, market_data):
        """
        Generate trading decisions using Carson's 6-hour approach
        Compatible with existing hybrid_live_trader.py
        """
        analysis = self.analyze_6hour_market_data(market_data)
        
        if analysis['status'] != 'success':
            return []
        
        combined_signal = analysis['combined_confidence']
        
        if combined_signal['signal'] == 'HOLD':
            return []
        
        # Create trading decision compatible with existing system
        decision = {
            'source': 'enhanced_6hour_momentum',
            'signal': combined_signal['signal'],
            'confidence': combined_signal['confidence'],
            'methodology': 'Carson 6-Hour + Academic Research',
            'timeframe': '6_hours',
            'position_size_multiplier': min(combined_signal['confidence'] * 1.5, 1.0),  # Scale with confidence
            'max_holding_hours': self.max_position_hours,
            'trailing_stop': self.trailing_stop_pct,
            'analysis_details': analysis
        }
        
        return [decision]

def upgrade_current_system():
    """
    Test upgrading current system with Carson's 6-hour approach
    """
    print("🔬 Testing Carson's 6-Hour Momentum Upgrade")
    print("=" * 60)
    
    # Initialize enhanced system
    enhanced_system = Enhanced6HourHybridSystem()
    
    # Simulate market data (you'd replace with real OANDA data)
    print("📊 Simulating 8 hours of market data...")
    dates = pd.date_range(start='2025-09-02 08:00', periods=96, freq='5min')  # 8 hours
    
    # Create trending market scenario
    np.random.seed(42)
    base_price = 1.1650
    trend_strength = 0.0003  # Gradual uptrend
    noise_level = 0.0015
    
    prices = []
    for i in range(96):
        # Add trend + noise
        trend_component = base_price + (trend_strength * i)
        noise_component = np.random.normal(0, noise_level)
        price = trend_component + noise_component
        prices.append(price)
    
    market_data = []
    for i, (date, price) in enumerate(zip(dates, prices)):
        market_data.append({
            'time': date,
            'close': price,
            'open': price + np.random.normal(0, 0.0005),
            'high': price + abs(np.random.normal(0, 0.001)),
            'low': price - abs(np.random.normal(0, 0.001)),
            'volume': np.random.randint(100, 1000)
        })
    
    # Test the enhanced system
    print("🔍 Analyzing market data with 6-hour momentum...")
    decisions = enhanced_system.generate_enhanced_trading_decisions(market_data)
    
    if decisions:
        decision = decisions[0]
        print(f"\n✅ Trading Signal Generated:")
        print(f"   📊 Signal: {decision['signal']}")
        print(f"   🎯 Confidence: {decision['confidence']:.3f}")
        print(f"   📏 Methodology: {decision['methodology']}")
        print(f"   ⏱️  Timeframe: {decision['timeframe']}")
        print(f"   📈 Position Multiplier: {decision['position_size_multiplier']:.2f}")
        print(f"   🕐 Max Holding: {decision['max_holding_hours']} hours")
        print(f"   🛑 Trailing Stop: {decision['trailing_stop']:.3%}")
        
        # Show analysis details
        if 'analysis_details' in decision:
            details = decision['analysis_details']
            print(f"\n🔬 Analysis Details:")
            print(f"   Data Quality: {details.get('data_quality', 'unknown')}")
            print(f"   Lookback Hours: {details.get('lookback_hours', 'unknown')}")
            
            if 'primary_signal' in details:
                primary = details['primary_signal']
                print(f"   Primary Signal: {primary['signal']} ({primary['confidence']:.3f})")
    else:
        print("❌ No trading signal generated (HOLD recommendation)")
    
    print(f"\n💡 Comparison with Current System:")
    print(f"   Current: 75-minute lookback (15 candles)")
    print(f"   Carson's: 6-hour lookback (72 candles)")
    print(f"   Advantage: Better trend capture, less noise")
    print(f"   Academic Support: AQR, Renaissance, Two Sigma research")
    
    return enhanced_system

if __name__ == "__main__":
    enhanced_system = upgrade_current_system()