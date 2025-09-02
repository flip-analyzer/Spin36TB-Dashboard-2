#!/usr/bin/env python3
"""
Enhanced 6-Hour Momentum System
Based on leading quantitative research and Carson's insights
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

class Enhanced6HourMomentum:
    """
    Advanced momentum system using 6-hour lookback periods
    Based on AQR Capital and López de Prado research
    """
    
    def __init__(self):
        # Momentum parameters based on academic research
        self.lookback_hours = 6  # Carson's suggestion - optimal for FX momentum
        self.lookback_candles = 72  # 6 hours * 12 candles per hour (5-minute bars)
        self.session_hours = 6     # Match signal generation to holding period
        
        # Multi-timeframe momentum (López de Prado approach)
        self.momentum_windows = {
            'short': 24,   # 2 hours
            'medium': 48,  # 4 hours  
            'long': 72     # 6 hours (primary)
        }
        
        # Advanced momentum metrics
        self.momentum_features = [
            'price_momentum',
            'volume_momentum', 
            'volatility_momentum',
            'momentum_acceleration',
            'momentum_persistence'
        ]
    
    def calculate_enhanced_momentum(self, df):
        """
        Calculate sophisticated momentum features over 6-hour windows
        Following López de Prado's methodology
        """
        results = {}
        
        if len(df) < self.lookback_candles:
            return None
            
        # 1. Price Momentum (Asness/AQR methodology)
        for window_name, window_size in self.momentum_windows.items():
            if len(df) >= window_size:
                # Compound return over window
                returns = df['close'].pct_change()
                momentum = (1 + returns).rolling(window=window_size).apply(
                    lambda x: x.prod() - 1, raw=True
                ).iloc[-1]
                results[f'momentum_{window_name}'] = momentum
        
        # 2. Volume-Weighted Momentum
        if 'volume' in df.columns:
            vwap = (df['close'] * df['volume']).rolling(self.lookback_candles).sum() / \
                   df['volume'].rolling(self.lookback_candles).sum()
            results['volume_momentum'] = (df['close'].iloc[-1] - vwap.iloc[-1]) / vwap.iloc[-1]
        
        # 3. Volatility-Adjusted Momentum (Sharpe-style)
        returns = df['close'].pct_change().dropna()
        if len(returns) >= self.lookback_candles:
            recent_returns = returns.tail(self.lookback_candles)
            momentum_return = recent_returns.mean() * self.lookback_candles
            momentum_vol = recent_returns.std() * np.sqrt(self.lookback_candles)
            results['risk_adjusted_momentum'] = momentum_return / (momentum_vol + 1e-8)
        
        # 4. Momentum Acceleration (second derivative)
        if len(df) >= self.lookback_candles + 24:  # Need extra data for acceleration
            # Calculate momentum over two periods
            recent_momentum = self.calculate_simple_momentum(df.tail(self.lookback_candles))
            older_momentum = self.calculate_simple_momentum(
                df.iloc[-(self.lookback_candles + 24):-24]
            )
            results['momentum_acceleration'] = recent_momentum - older_momentum
        
        # 5. Momentum Persistence (trending strength)
        if len(df) >= self.lookback_candles:
            price_changes = df['close'].diff().tail(self.lookback_candles)
            persistence = (price_changes > 0).sum() / len(price_changes)
            results['momentum_persistence'] = persistence - 0.5  # Center around 0
        
        return results
    
    def calculate_simple_momentum(self, df):
        """Simple momentum calculation for a given DataFrame"""
        if len(df) < 2:
            return 0
        return (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]
    
    def generate_6hour_signal(self, df):
        """
        Generate trading signal using 6-hour momentum analysis
        Following Carson's suggestion and academic best practices
        """
        momentum_data = self.calculate_enhanced_momentum(df)
        
        if not momentum_data:
            return {'signal': 'HOLD', 'confidence': 0.0, 'reason': 'Insufficient data'}
        
        # Multi-factor momentum scoring (López de Prado approach)
        signal_strength = 0.0
        signal_count = 0
        signal_components = {}
        
        # 1. Primary 6-hour momentum (Carson's focus)
        if 'momentum_long' in momentum_data:
            long_momentum = momentum_data['momentum_long']
            if abs(long_momentum) > 0.002:  # 0.2% threshold for FX
                signal_strength += np.sign(long_momentum) * min(abs(long_momentum) * 100, 1.0)
                signal_count += 1
                signal_components['6h_momentum'] = long_momentum
        
        # 2. Risk-adjusted momentum
        if 'risk_adjusted_momentum' in momentum_data:
            risk_adj = momentum_data['risk_adjusted_momentum'] 
            if abs(risk_adj) > 0.5:  # Sharpe-like threshold
                signal_strength += np.sign(risk_adj) * min(abs(risk_adj) / 3.0, 1.0)
                signal_count += 1
                signal_components['risk_adjusted'] = risk_adj
        
        # 3. Momentum acceleration (trend strengthening)
        if 'momentum_acceleration' in momentum_data:
            acceleration = momentum_data['momentum_acceleration']
            if abs(acceleration) > 0.001:
                signal_strength += np.sign(acceleration) * min(abs(acceleration) * 200, 0.5)
                signal_count += 1
                signal_components['acceleration'] = acceleration
        
        # 4. Momentum persistence (trend consistency)
        if 'momentum_persistence' in momentum_data:
            persistence = momentum_data['momentum_persistence']
            if abs(persistence) > 0.1:  # 60% directional consistency
                signal_strength += np.sign(persistence) * min(abs(persistence) * 2, 0.3)
                signal_count += 1
                signal_components['persistence'] = persistence
        
        # Generate final signal
        if signal_count == 0:
            return {'signal': 'HOLD', 'confidence': 0.0, 'reason': 'No clear momentum'}
        
        # Average signal strength
        avg_signal = signal_strength / signal_count
        confidence = min(abs(avg_signal), 1.0)
        
        # Signal thresholds (academic research based)
        if avg_signal > 0.3:
            signal = 'BUY'
        elif avg_signal < -0.3:
            signal = 'SELL'  
        else:
            signal = 'HOLD'
        
        return {
            'signal': signal,
            'confidence': confidence,
            'signal_strength': avg_signal,
            'components': signal_components,
            'momentum_data': momentum_data,
            'lookback_hours': self.lookback_hours,
            'methodology': 'Enhanced 6-Hour Momentum (Carson/Academic)'
        }

def test_6hour_system():
    """Test the enhanced 6-hour momentum system"""
    
    # Create sample data (you'd use real OANDA data)
    dates = pd.date_range(start='2025-09-01', periods=200, freq='5T')
    
    # Simulate trending price data
    np.random.seed(42)
    price_changes = np.random.normal(0.0001, 0.002, 200)  # Small trending bias
    prices = [1.1650]  # Starting price
    
    for change in price_changes:
        prices.append(prices[-1] * (1 + change))
    
    df = pd.DataFrame({
        'timestamp': dates,
        'close': prices[:-1],  # Remove extra price
        'volume': np.random.randint(100, 1000, 200)
    })
    
    # Test the system
    system = Enhanced6HourMomentum()
    result = system.generate_6hour_signal(df)
    
    print("🔬 Enhanced 6-Hour Momentum System Test")
    print("=" * 50)
    print(f"Signal: {result['signal']}")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"Signal Strength: {result.get('signal_strength', 0):.3f}")
    print(f"Lookback: {result['lookback_hours']} hours")
    print(f"Methodology: {result['methodology']}")
    
    if 'components' in result:
        print("\nSignal Components:")
        for component, value in result['components'].items():
            print(f"  {component}: {value:.4f}")
    
    return result

if __name__ == "__main__":
    test_6hour_system()