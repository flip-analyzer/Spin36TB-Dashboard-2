#!/usr/bin/env python3
"""
Enhanced Risk Management System for Spin36TB
Professional-grade risk controls with adaptive features
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
import os

class EnhancedRiskManager:
    """
    Professional risk management system with:
    1. Dynamic stop losses based on market volatility
    2. Drawdown-based position scaling
    3. Correlation risk monitoring
    4. Market stress detection
    5. Portfolio heat mapping
    """
    
    def __init__(self, starting_capital: float = 25000):
        self.starting_capital = starting_capital
        self.current_capital = starting_capital
        
        # Risk parameters
        self.risk_params = {
            # Position sizing limits
            'max_single_position': 0.022,    # 2.2% max per trade
            'max_portfolio_risk': 0.06,      # 6% max total exposure
            'max_correlation_exposure': 0.04, # 4% max in correlated positions
            
            # Drawdown controls
            'max_daily_drawdown': 0.02,      # 2% daily limit
            'max_total_drawdown': 0.15,      # 15% total limit (emergency stop)
            'drawdown_scale_threshold': 0.05, # Start scaling at 5% drawdown
            
            # Volatility-based stops
            'base_stop_atr_multiplier': 2.0,  # 2x ATR for stops
            'min_stop_pips': 3.0,            # Minimum 3 pip stop
            'max_stop_pips': 25.0,           # Maximum 25 pip stop
            
            # Market stress thresholds
            'high_volatility_threshold': 0.0008,  # 8 pips ATR
            'extreme_volatility_threshold': 0.0015, # 15 pips ATR
            'correlation_stress_threshold': 0.8,    # High correlation warning
        }
        
        # State tracking
        self.daily_stats = {}
        self.position_history = []
        self.risk_alerts = []
        
        # Load existing state if available
        self._load_risk_state()
    
    def _load_risk_state(self):
        """Load previous risk management state"""
        try:
            state_file = '/Users/jonspinogatti/Desktop/spin36TB/risk_state.json'
            if os.path.exists(state_file):
                with open(state_file, 'r') as f:
                    state = json.load(f)
                    self.current_capital = state.get('current_capital', self.starting_capital)
                    self.daily_stats = state.get('daily_stats', {})
                    self.position_history = state.get('position_history', [])
        except:
            pass
    
    def _save_risk_state(self):
        """Save current risk management state"""
        try:
            state = {
                'current_capital': self.current_capital,
                'daily_stats': self.daily_stats,
                'position_history': self.position_history[-100:],  # Keep last 100 positions
                'last_updated': datetime.now().isoformat()
            }
            with open('/Users/jonspinogatti/Desktop/spin36TB/risk_state.json', 'w') as f:
                json.dump(state, f, indent=2)
        except:
            pass
    
    def calculate_dynamic_position_size(self, 
                                      base_size: float,
                                      market_data: pd.DataFrame,
                                      signal_confidence: float,
                                      strategy_type: str) -> Dict:
        """
        Calculate position size with enhanced risk controls
        """
        # Base position size adjustment
        adjusted_size = base_size
        risk_factors = []
        
        # 1. Drawdown-based scaling
        current_drawdown = self.get_current_drawdown()
        if current_drawdown > self.risk_params['drawdown_scale_threshold']:
            drawdown_multiplier = max(0.3, 1.0 - (current_drawdown * 2))
            adjusted_size *= drawdown_multiplier
            risk_factors.append(f"Drawdown scaling: {drawdown_multiplier:.2f}")
        
        # 2. Volatility-based adjustment
        volatility_factor = self.get_volatility_adjustment(market_data)
        adjusted_size *= volatility_factor
        risk_factors.append(f"Volatility factor: {volatility_factor:.2f}")
        
        # 3. Confidence-based scaling
        confidence_multiplier = max(0.5, min(1.5, signal_confidence * 1.5))
        adjusted_size *= confidence_multiplier
        risk_factors.append(f"Confidence scaling: {confidence_multiplier:.2f}")
        
        # 4. Portfolio heat check
        portfolio_heat = self.get_portfolio_heat()
        if portfolio_heat > self.risk_params['max_portfolio_risk']:
            heat_reduction = max(0.2, (self.risk_params['max_portfolio_risk'] / portfolio_heat))
            adjusted_size *= heat_reduction
            risk_factors.append(f"Portfolio heat reduction: {heat_reduction:.2f}")
        
        # 5. Daily risk limit check
        daily_risk_used = self.get_daily_risk_used()
        if daily_risk_used > self.risk_params['max_daily_drawdown'] * 0.8:
            daily_reduction = max(0.1, (self.risk_params['max_daily_drawdown'] - daily_risk_used) / self.risk_params['max_daily_drawdown'])
            adjusted_size *= daily_reduction
            risk_factors.append(f"Daily limit scaling: {daily_reduction:.2f}")
        
        # Apply hard limits
        adjusted_size = min(adjusted_size, self.risk_params['max_single_position'])
        adjusted_size = max(adjusted_size, 0.002)  # Minimum 0.2%
        
        return {
            'position_size': adjusted_size,
            'original_size': base_size,
            'risk_factors': risk_factors,
            'current_drawdown': current_drawdown,
            'portfolio_heat': portfolio_heat,
            'daily_risk_used': daily_risk_used
        }
    
    def calculate_dynamic_stops(self, 
                               market_data: pd.DataFrame, 
                               entry_price: float,
                               direction: str) -> Dict:
        """
        Calculate dynamic stop loss and take profit levels
        """
        # Calculate ATR for volatility-based stops
        recent_data = market_data.tail(14)
        atr = (recent_data['high'] - recent_data['low']).mean()
        
        # Dynamic stop distance
        stop_distance_pips = atr * self.risk_params['base_stop_atr_multiplier'] * 10000
        stop_distance_pips = np.clip(stop_distance_pips, 
                                   self.risk_params['min_stop_pips'],
                                   self.risk_params['max_stop_pips'])
        
        # Calculate stop and target levels
        if direction.upper() == 'BUY':
            stop_loss = entry_price - (stop_distance_pips * 0.0001)
            take_profit = entry_price + (stop_distance_pips * 2.0 * 0.0001)  # 2:1 R/R
        else:  # SELL
            stop_loss = entry_price + (stop_distance_pips * 0.0001)
            take_profit = entry_price - (stop_distance_pips * 2.0 * 0.0001)
        
        # Market stress adjustment
        volatility_regime = self.assess_volatility_regime(atr)
        if volatility_regime == 'EXTREME':
            # Tighten stops in extreme volatility
            stop_distance_pips *= 0.7
            risk_adjustment = "Tightened for extreme volatility"
        elif volatility_regime == 'HIGH':
            # Slightly wider stops in high volatility
            stop_distance_pips *= 1.2
            risk_adjustment = "Widened for high volatility"
        else:
            risk_adjustment = "Normal volatility regime"
        
        return {
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'stop_distance_pips': stop_distance_pips,
            'risk_reward_ratio': 2.0,
            'volatility_regime': volatility_regime,
            'risk_adjustment': risk_adjustment,
            'atr_pips': atr * 10000
        }
    
    def assess_market_stress(self, market_data: pd.DataFrame) -> Dict:
        """
        Detect market stress conditions that warrant position reduction
        """
        stress_factors = []
        stress_level = 0
        
        # 1. Volatility stress
        recent_data = market_data.tail(20)
        current_atr = (recent_data['high'] - recent_data['low']).mean()
        
        if current_atr > self.risk_params['extreme_volatility_threshold']:
            stress_factors.append("Extreme volatility detected")
            stress_level += 3
        elif current_atr > self.risk_params['high_volatility_threshold']:
            stress_factors.append("High volatility detected")
            stress_level += 1
        
        # 2. Price gap stress (large moves between candles)
        price_gaps = abs(recent_data['open'] - recent_data['close'].shift(1)).dropna()
        large_gaps = (price_gaps > current_atr * 0.5).sum()
        if large_gaps > 3:
            stress_factors.append(f"Multiple large price gaps: {large_gaps}")
            stress_level += 2
        
        # 3. Trend acceleration (rapid directional moves)
        price_changes = recent_data['close'].pct_change().abs()
        accelerating_moves = (price_changes > price_changes.mean() + 2 * price_changes.std()).sum()
        if accelerating_moves > 5:
            stress_factors.append(f"Trend acceleration detected: {accelerating_moves} moves")
            stress_level += 1
        
        # 4. Market hours stress (weekend gaps, news times)
        current_hour = datetime.now().hour
        if current_hour in [21, 22, 23] or current_hour in [0, 1, 2]:  # Major news times
            stress_factors.append("High-impact news window")
            stress_level += 1
        
        # Determine overall stress level
        if stress_level >= 4:
            overall_stress = "EXTREME"
            recommendation = "STOP_NEW_TRADES"
        elif stress_level >= 2:
            overall_stress = "HIGH"
            recommendation = "REDUCE_POSITIONS"
        elif stress_level >= 1:
            overall_stress = "MODERATE"
            recommendation = "MONITOR_CLOSELY"
        else:
            overall_stress = "LOW"
            recommendation = "NORMAL_TRADING"
        
        return {
            'stress_level': stress_level,
            'overall_stress': overall_stress,
            'recommendation': recommendation,
            'stress_factors': stress_factors,
            'current_atr_pips': current_atr * 10000
        }
    
    def check_correlation_risk(self, 
                              momentum_signal: str, 
                              mean_reversion_signal: str) -> Dict:
        """
        Monitor correlation risk between strategies
        """
        correlation_risk = "LOW"
        warning = None
        
        # Check if both strategies are signaling same direction
        if momentum_signal == mean_reversion_signal and momentum_signal != 'HOLD':
            correlation_risk = "HIGH"
            warning = f"Both strategies signaling {momentum_signal} - correlation risk"
        
        # Historical correlation check (if we have position history)
        recent_positions = [p for p in self.position_history if 
                          datetime.fromisoformat(p['timestamp']) > datetime.now() - timedelta(days=7)]
        
        if len(recent_positions) > 10:
            momentum_outcomes = [p['outcome'] for p in recent_positions if p['strategy'] == 'momentum']
            mean_rev_outcomes = [p['outcome'] for p in recent_positions if p['strategy'] == 'mean_reversion']
            
            if len(momentum_outcomes) > 5 and len(mean_rev_outcomes) > 5:
                correlation = np.corrcoef(momentum_outcomes[-5:], mean_rev_outcomes[-5:])[0,1]
                if abs(correlation) > self.risk_params['correlation_stress_threshold']:
                    correlation_risk = "EXTREME"
                    warning = f"Strategies highly correlated: {correlation:.2f}"
        
        return {
            'correlation_risk': correlation_risk,
            'warning': warning,
            'recommendation': "REDUCE_EXPOSURE" if correlation_risk == "HIGH" else "NORMAL"
        }
    
    def get_emergency_stops(self) -> Dict:
        """
        Check for conditions requiring immediate position closure
        """
        emergency_conditions = []
        
        # 1. Maximum drawdown breach
        current_drawdown = self.get_current_drawdown()
        if current_drawdown > self.risk_params['max_total_drawdown']:
            emergency_conditions.append(f"Maximum drawdown exceeded: {current_drawdown:.1%}")
        
        # 2. Daily loss limit breach
        daily_loss = self.get_daily_loss()
        if daily_loss > self.risk_params['max_daily_drawdown']:
            emergency_conditions.append(f"Daily loss limit exceeded: {daily_loss:.1%}")
        
        # 3. Rapid capital depletion
        if self.current_capital < self.starting_capital * 0.7:  # 30% total loss
            emergency_conditions.append("Rapid capital depletion detected")
        
        return {
            'emergency_stop_required': len(emergency_conditions) > 0,
            'conditions': emergency_conditions,
            'current_drawdown': current_drawdown,
            'daily_loss': daily_loss,
            'remaining_capital_pct': self.current_capital / self.starting_capital
        }
    
    # Helper methods
    def get_current_drawdown(self) -> float:
        """Calculate current drawdown from peak"""
        if self.current_capital >= self.starting_capital:
            return 0.0
        return (self.starting_capital - self.current_capital) / self.starting_capital
    
    def get_portfolio_heat(self) -> float:
        """Calculate current portfolio risk exposure"""
        # Simplified - in real implementation, would track open positions
        return 0.02  # Placeholder
    
    def get_daily_risk_used(self) -> float:
        """Calculate today's risk utilization"""
        today = datetime.now().date().isoformat()
        return self.daily_stats.get(today, {}).get('risk_used', 0.0)
    
    def get_daily_loss(self) -> float:
        """Calculate today's losses"""
        today = datetime.now().date().isoformat()
        return self.daily_stats.get(today, {}).get('loss_pct', 0.0)
    
    def get_volatility_adjustment(self, market_data: pd.DataFrame) -> float:
        """Get volatility-based position size adjustment"""
        recent_data = market_data.tail(20)
        current_atr = (recent_data['high'] - recent_data['low']).mean()
        
        # Scale relative to normal volatility (3 pips baseline)
        normal_volatility = 0.0003
        volatility_ratio = current_atr / normal_volatility
        
        # Inverse relationship - higher volatility = smaller positions
        return max(0.5, min(1.5, 1.0 / np.sqrt(volatility_ratio)))
    
    def assess_volatility_regime(self, atr: float) -> str:
        """Classify current volatility regime"""
        if atr > self.risk_params['extreme_volatility_threshold']:
            return 'EXTREME'
        elif atr > self.risk_params['high_volatility_threshold']:
            return 'HIGH'
        else:
            return 'NORMAL'
    
    def update_position_outcome(self, 
                               position_size: float, 
                               outcome_pips: float, 
                               strategy: str):
        """Update position tracking and statistics"""
        outcome_pct = (outcome_pips * 0.0001 * position_size)
        
        position_record = {
            'timestamp': datetime.now().isoformat(),
            'position_size': position_size,
            'outcome_pips': outcome_pips,
            'outcome_pct': outcome_pct,
            'strategy': strategy,
            'outcome': 1 if outcome_pips > 0 else 0
        }
        
        self.position_history.append(position_record)
        self.current_capital += (self.starting_capital * outcome_pct)
        
        # Update daily stats
        today = datetime.now().date().isoformat()
        if today not in self.daily_stats:
            self.daily_stats[today] = {'trades': 0, 'pnl_pct': 0.0, 'risk_used': 0.0}
        
        self.daily_stats[today]['trades'] += 1
        self.daily_stats[today]['pnl_pct'] += outcome_pct
        self.daily_stats[today]['risk_used'] += position_size
        
        self._save_risk_state()
    
    def get_risk_summary(self) -> Dict:
        """Get comprehensive risk summary for monitoring"""
        return {
            'current_capital': self.current_capital,
            'total_return': (self.current_capital - self.starting_capital) / self.starting_capital,
            'current_drawdown': self.get_current_drawdown(),
            'daily_risk_used': self.get_daily_risk_used(),
            'portfolio_heat': self.get_portfolio_heat(),
            'recent_trades': len(self.position_history[-10:]),
            'risk_status': 'NORMAL' if self.get_current_drawdown() < 0.05 else 'ELEVATED'
        }

def test_enhanced_risk_manager():
    """Test the enhanced risk management system"""
    import random
    
    # Create sample market data
    dates = pd.date_range('2025-01-01', periods=100, freq='5T')
    market_data = pd.DataFrame({
        'open': [1.0500 + random.uniform(-0.01, 0.01) for _ in range(100)],
        'high': [1.0500 + random.uniform(0, 0.015) for _ in range(100)],
        'low': [1.0500 + random.uniform(-0.015, 0) for _ in range(100)],
        'close': [1.0500 + random.uniform(-0.01, 0.01) for _ in range(100)],
    }, index=dates)
    
    # Test risk manager
    risk_manager = EnhancedRiskManager()
    
    print("🛡️  ENHANCED RISK MANAGEMENT SYSTEM TEST")
    print("=" * 50)
    
    # Test position sizing
    position_result = risk_manager.calculate_dynamic_position_size(
        base_size=0.015,
        market_data=market_data,
        signal_confidence=0.75,
        strategy_type='momentum'
    )
    
    print(f"📊 Position Sizing Test:")
    print(f"   Original size: {position_result['original_size']:.3%}")
    print(f"   Adjusted size: {position_result['position_size']:.3%}")
    print(f"   Risk factors: {position_result['risk_factors']}")
    
    # Test dynamic stops
    stops_result = risk_manager.calculate_dynamic_stops(
        market_data=market_data,
        entry_price=1.0500,
        direction='BUY'
    )
    
    print(f"\n🎯 Dynamic Stops Test:")
    print(f"   Entry: {stops_result.get('entry_price', 1.0500):.4f}")
    print(f"   Stop Loss: {stops_result['stop_loss']:.4f}")
    print(f"   Take Profit: {stops_result['take_profit']:.4f}")
    print(f"   Stop Distance: {stops_result['stop_distance_pips']:.1f} pips")
    print(f"   Volatility Regime: {stops_result['volatility_regime']}")
    
    # Test market stress
    stress_result = risk_manager.assess_market_stress(market_data)
    
    print(f"\n⚠️  Market Stress Assessment:")
    print(f"   Stress Level: {stress_result['overall_stress']}")
    print(f"   Recommendation: {stress_result['recommendation']}")
    print(f"   Factors: {stress_result['stress_factors']}")
    
    # Test emergency stops
    emergency_result = risk_manager.get_emergency_stops()
    
    print(f"\n🚨 Emergency Stop Check:")
    print(f"   Emergency Required: {emergency_result['emergency_stop_required']}")
    print(f"   Conditions: {emergency_result['conditions']}")
    
    print(f"\n📈 Risk Summary:")
    summary = risk_manager.get_risk_summary()
    for key, value in summary.items():
        print(f"   {key}: {value}")

if __name__ == "__main__":
    test_enhanced_risk_manager()