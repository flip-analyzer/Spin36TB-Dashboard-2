#!/usr/bin/env python3
"""
Adaptive Learning System: Machine Learning Enhancement for Risk Management
Learns from trading activity to improve risk percentages and position sizing over time
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import warnings
from typing import Dict, List, Tuple, Optional
from collections import deque
import sys

sys.path.append('/Users/jonspinogatti/Desktop/spin36TB')

warnings.filterwarnings('ignore')

class AdaptiveLearningSystem:
    """
    Machine learning system that learns from trading results to optimize risk management
    """
    
    def __init__(self):
        # Learning parameters
        self.learning_config = {
            'lookback_trades': 100,              # Learn from last 100 trades
            'adaptation_rate': 0.1,              # How fast to adapt (10% per update)
            'min_trades_for_learning': 20,       # Need 20 trades before learning
            'confidence_threshold': 0.7,         # 70% confidence to make changes
            'max_risk_adjustment': 0.5,          # Max 50% change in risk per update
        }
        
        # Performance tracking for learning
        self.trade_history = deque(maxlen=500)  # Keep last 500 trades
        self.learning_metrics = {
            'momentum_performance': {},
            'mean_reversion_performance': {},
            'regime_performance': {},
            'time_of_day_performance': {},
            'volatility_performance': {},
            'correlation_performance': {}
        }
        
        # Adaptive risk parameters (start with defaults, learn over time)
        self.adaptive_params = {
            'momentum_base_risk': 0.005,         # Start at 0.5% base
            'mean_reversion_base_risk': 0.004,   # Start at 0.4% base
            'high_vol_multiplier': 1.3,          # Adjust based on learning
            'low_vol_multiplier': 0.7,           # Adjust based on learning
            'confidence_scaling': 1.0,           # Learn optimal confidence scaling
            'regime_multipliers': {              # Learn regime-specific adjustments
                'HIGH_VOLATILITY': 1.2,
                'LOW_VOLATILITY': 0.8,
                'TRENDING': 1.1,
                'RANGING': 0.9
            }
        }
        
        # Learning state
        self.learning_enabled = True
        self.last_learning_update = None
        self.learning_iterations = 0
        
        print("🧠 ADAPTIVE LEARNING SYSTEM INITIALIZED")
        print("=" * 45)
        print("🎯 Will learn from trading results to optimize risk management")
        print("📊 Minimum trades for learning: 20")
        print("🔄 Learning rate: 10% adaptation per update")
        print("⚖️  Risk adjustment limit: ±50% per update")
    
    def record_trade_result(self, trade_result: Dict, market_context: Dict):
        """
        Record trade result with market context for learning
        """
        # Enhanced trade record with learning features
        learning_record = {
            'timestamp': datetime.now(),
            'strategy': trade_result.get('strategy'),
            'direction': trade_result.get('direction'),
            'entry_price': trade_result.get('entry_price'),
            'exit_price': trade_result.get('exit_price'),
            'position_size': trade_result.get('position_size'),
            'pips_net': trade_result.get('pips_net', 0),
            'return_pct': trade_result.get('return_pct', 0),
            'win': trade_result.get('win', False),
            'exit_reason': trade_result.get('exit_reason'),
            'confidence': trade_result.get('confidence', 0),
            
            # Market context for learning
            'regime': market_context.get('regime'),
            'volatility': market_context.get('volatility', 0),
            'time_of_day': datetime.now().hour,
            'correlation_signal': market_context.get('correlation_signal', 0),
            'signal_strength': market_context.get('signal_strength', 0),
        }
        
        self.trade_history.append(learning_record)
        
        # Trigger learning if enough data
        if len(self.trade_history) >= self.learning_config['min_trades_for_learning']:
            if (self.learning_iterations % 20 == 0 or  # Learn every 20 trades
                self.last_learning_update is None or
                (datetime.now() - self.last_learning_update).days >= 1):  # Or daily
                self.update_learning_models()
    
    def update_learning_models(self):
        """
        Update learning models based on recent trading performance
        """
        if not self.learning_enabled or len(self.trade_history) < self.learning_config['min_trades_for_learning']:
            return
        
        print(f"\n🧠 UPDATING LEARNING MODELS...")
        print(f"   Analyzing {len(self.trade_history)} trades for patterns")
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(list(self.trade_history))
        
        # Learn strategy-specific performance
        self._learn_strategy_performance(df)
        
        # Learn regime-specific performance
        self._learn_regime_performance(df)
        
        # Learn time-based patterns
        self._learn_time_patterns(df)
        
        # Learn volatility impact
        self._learn_volatility_impact(df)
        
        # Update risk parameters based on learning
        self._update_risk_parameters(df)
        
        self.last_learning_update = datetime.now()
        self.learning_iterations += 1
        
        print(f"✅ Learning update complete (iteration {self.learning_iterations})")
    
    def _learn_strategy_performance(self, df: pd.DataFrame):
        """
        Learn how each strategy performs under different conditions
        """
        momentum_trades = df[df['strategy'] == 'MOMENTUM']
        mr_trades = df[df['strategy'] == 'MEAN_REVERSION']
        
        if len(momentum_trades) >= 10:
            momentum_win_rate = momentum_trades['win'].mean()
            momentum_avg_return = momentum_trades['return_pct'].mean()
            momentum_risk_adj_return = momentum_avg_return / momentum_trades['return_pct'].std() if momentum_trades['return_pct'].std() > 0 else 0
            
            self.learning_metrics['momentum_performance'] = {
                'win_rate': momentum_win_rate,
                'avg_return': momentum_avg_return,
                'risk_adjusted_return': momentum_risk_adj_return,
                'sample_size': len(momentum_trades)
            }
        
        if len(mr_trades) >= 10:
            mr_win_rate = mr_trades['win'].mean()
            mr_avg_return = mr_trades['return_pct'].mean()
            mr_risk_adj_return = mr_avg_return / mr_trades['return_pct'].std() if mr_trades['return_pct'].std() > 0 else 0
            
            self.learning_metrics['mean_reversion_performance'] = {
                'win_rate': mr_win_rate,
                'avg_return': mr_avg_return,
                'risk_adjusted_return': mr_risk_adj_return,
                'sample_size': len(mr_trades)
            }
    
    def _learn_regime_performance(self, df: pd.DataFrame):
        """
        Learn how strategies perform in different market regimes
        """
        regime_performance = {}
        
        for regime in df['regime'].unique():
            if pd.isna(regime):
                continue
            
            regime_trades = df[df['regime'] == regime]
            if len(regime_trades) >= 5:  # Need at least 5 trades
                regime_performance[regime] = {
                    'win_rate': regime_trades['win'].mean(),
                    'avg_return': regime_trades['return_pct'].mean(),
                    'trade_count': len(regime_trades),
                    'avg_pips': regime_trades['pips_net'].mean()
                }
        
        self.learning_metrics['regime_performance'] = regime_performance
    
    def _learn_time_patterns(self, df: pd.DataFrame):
        """
        Learn performance patterns by time of day
        """
        time_performance = {}
        
        # Group by hour
        for hour in range(24):
            hour_trades = df[df['time_of_day'] == hour]
            if len(hour_trades) >= 3:
                time_performance[hour] = {
                    'win_rate': hour_trades['win'].mean(),
                    'avg_return': hour_trades['return_pct'].mean(),
                    'trade_count': len(hour_trades)
                }
        
        self.learning_metrics['time_of_day_performance'] = time_performance
    
    def _learn_volatility_impact(self, df: pd.DataFrame):
        """
        Learn how volatility affects performance
        """
        if 'volatility' in df.columns and df['volatility'].notna().any():
            # Divide into volatility quartiles
            volatility_quartiles = df['volatility'].quantile([0.25, 0.5, 0.75])
            
            vol_performance = {}
            for i, (low, high) in enumerate([(0, volatility_quartiles[0.25]),
                                           (volatility_quartiles[0.25], volatility_quartiles[0.5]),
                                           (volatility_quartiles[0.5], volatility_quartiles[0.75]),
                                           (volatility_quartiles[0.75], float('inf'))]):
                
                vol_trades = df[(df['volatility'] > low) & (df['volatility'] <= high)]
                if len(vol_trades) >= 5:
                    vol_performance[f'quartile_{i+1}'] = {
                        'win_rate': vol_trades['win'].mean(),
                        'avg_return': vol_trades['return_pct'].mean(),
                        'trade_count': len(vol_trades),
                        'vol_range': (low, high)
                    }
            
            self.learning_metrics['volatility_performance'] = vol_performance
    
    def _update_risk_parameters(self, df: pd.DataFrame):
        """
        Update risk parameters based on learning
        """
        adaptation_rate = self.learning_config['adaptation_rate']
        max_adjustment = self.learning_config['max_risk_adjustment']
        
        # Adjust strategy base risks based on performance
        if 'momentum_performance' in self.learning_metrics:
            mom_perf = self.learning_metrics['momentum_performance']
            if mom_perf.get('sample_size', 0) >= 20:
                # If momentum is performing well, slightly increase risk
                performance_ratio = mom_perf['risk_adjusted_return']
                adjustment = np.clip(performance_ratio * adaptation_rate, -max_adjustment, max_adjustment)
                
                old_risk = self.adaptive_params['momentum_base_risk']
                new_risk = old_risk * (1 + adjustment)
                new_risk = np.clip(new_risk, 0.002, 0.010)  # Keep within 0.2-1.0%
                
                self.adaptive_params['momentum_base_risk'] = new_risk
                
                print(f"   📈 Momentum base risk: {old_risk:.3f} → {new_risk:.3f} ({adjustment:+.1%})")
        
        # Adjust mean reversion base risk
        if 'mean_reversion_performance' in self.learning_metrics:
            mr_perf = self.learning_metrics['mean_reversion_performance']
            if mr_perf.get('sample_size', 0) >= 20:
                performance_ratio = mr_perf['risk_adjusted_return']
                adjustment = np.clip(performance_ratio * adaptation_rate, -max_adjustment, max_adjustment)
                
                old_risk = self.adaptive_params['mean_reversion_base_risk']
                new_risk = old_risk * (1 + adjustment)
                new_risk = np.clip(new_risk, 0.002, 0.008)  # Keep within 0.2-0.8%
                
                self.adaptive_params['mean_reversion_base_risk'] = new_risk
                
                print(f"   📉 Mean reversion base risk: {old_risk:.3f} → {new_risk:.3f} ({adjustment:+.1%})")
        
        # Adjust regime multipliers
        if self.learning_metrics['regime_performance']:
            for regime, perf in self.learning_metrics['regime_performance'].items():
                if perf['trade_count'] >= 10:
                    # Adjust multiplier based on win rate and average return
                    performance_score = perf['win_rate'] * perf['avg_return']
                    
                    if regime in self.adaptive_params['regime_multipliers']:
                        old_mult = self.adaptive_params['regime_multipliers'][regime]
                        adjustment = performance_score * adaptation_rate
                        adjustment = np.clip(adjustment, -0.2, 0.2)  # Max ±20% change
                        
                        new_mult = old_mult + adjustment
                        new_mult = np.clip(new_mult, 0.5, 1.5)  # Keep within reasonable bounds
                        
                        self.adaptive_params['regime_multipliers'][regime] = new_mult
                        
                        print(f"   🎭 {regime} multiplier: {old_mult:.2f} → {new_mult:.2f}")
    
    def get_adaptive_position_size(self, strategy: str, base_confidence: float, 
                                 market_context: Dict) -> float:
        """
        Calculate position size using learned parameters
        """
        # Get base risk from learned parameters
        if strategy == 'MOMENTUM':
            base_risk = self.adaptive_params['momentum_base_risk']
        else:  # MEAN_REVERSION
            base_risk = self.adaptive_params['mean_reversion_base_risk']
        
        # Apply confidence scaling (learned)
        confidence_scaling = self.adaptive_params['confidence_scaling']
        position_size = base_risk * (0.5 + base_confidence * confidence_scaling)
        
        # Apply regime multiplier (learned)
        regime = market_context.get('regime')
        if regime in self.adaptive_params['regime_multipliers']:
            regime_mult = self.adaptive_params['regime_multipliers'][regime]
            position_size *= regime_mult
        
        # Apply volatility adjustment (learned)
        volatility = market_context.get('volatility', 0)
        if volatility > 0.0005:  # High volatility
            position_size *= self.adaptive_params['high_vol_multiplier']
        elif volatility < 0.0002:  # Low volatility
            position_size *= self.adaptive_params['low_vol_multiplier']
        
        # Apply time-based adjustment if learned
        current_hour = datetime.now().hour
        if (current_hour in self.learning_metrics.get('time_of_day_performance', {}) and
            len(self.trade_history) >= 50):
            hour_perf = self.learning_metrics['time_of_day_performance'][current_hour]
            if hour_perf['trade_count'] >= 5:
                time_mult = 0.8 + (hour_perf['win_rate'] * 0.4)  # 0.8 to 1.2 range
                position_size *= time_mult
        
        # Final bounds
        max_position = 0.012 if strategy == 'MOMENTUM' else 0.008
        position_size = np.clip(position_size, 0.001, max_position)
        
        return position_size
    
    def get_learning_status(self) -> Dict:
        """
        Get comprehensive learning system status
        """
        return {
            'learning_enabled': self.learning_enabled,
            'trades_recorded': len(self.trade_history),
            'learning_iterations': self.learning_iterations,
            'last_update': self.last_learning_update.isoformat() if self.last_learning_update else None,
            'adaptive_params': self.adaptive_params,
            'learning_metrics': self.learning_metrics,
            'next_update_in_trades': 20 - (self.learning_iterations % 20)
        }
    
    def save_learning_state(self, filepath: str):
        """
        Save learning state to file
        """
        state = {
            'adaptive_params': self.adaptive_params,
            'learning_metrics': self.learning_metrics,
            'learning_iterations': self.learning_iterations,
            'last_update': self.last_learning_update.isoformat() if self.last_learning_update else None
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2, default=str)
            print(f"💾 Learning state saved to {filepath}")
        except Exception as e:
            print(f"❌ Error saving learning state: {e}")
    
    def load_learning_state(self, filepath: str):
        """
        Load learning state from file
        """
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            self.adaptive_params = state.get('adaptive_params', self.adaptive_params)
            self.learning_metrics = state.get('learning_metrics', {})
            self.learning_iterations = state.get('learning_iterations', 0)
            
            if state.get('last_update'):
                self.last_learning_update = datetime.fromisoformat(state['last_update'])
            
            print(f"📁 Learning state loaded from {filepath}")
            print(f"   Iterations: {self.learning_iterations}")
            print(f"   Last update: {self.last_learning_update}")
            
        except Exception as e:
            print(f"❌ Error loading learning state: {e}")

def demonstrate_learning_system():
    """
    Demonstrate the adaptive learning system
    """
    print("\n🧠 ADAPTIVE LEARNING SYSTEM DEMO")
    print("=" * 40)
    
    # Initialize learning system
    learning_system = AdaptiveLearningSystem()
    
    # Simulate some trade results to show learning
    print("\n📊 Simulating 30 trades to demonstrate learning...")
    
    strategies = ['MOMENTUM', 'MEAN_REVERSION']
    regimes = ['HIGH_VOLATILITY', 'LOW_VOLATILITY', 'TRENDING', 'RANGING']
    
    np.random.seed(42)
    
    for i in range(30):
        # Simulate trade result
        strategy = np.random.choice(strategies)
        regime = np.random.choice(regimes)
        
        # Simulate performance (mean reversion better in some regimes)
        if strategy == 'MEAN_REVERSION' and regime in ['HIGH_VOLATILITY', 'RANGING']:
            win_prob = 0.8  # Higher win rate
            pips = np.random.normal(5, 3) if np.random.random() < win_prob else np.random.normal(-8, 2)
        else:
            win_prob = 0.55  # Normal win rate
            pips = np.random.normal(8, 5) if np.random.random() < win_prob else np.random.normal(-6, 2)
        
        trade_result = {
            'strategy': strategy,
            'direction': np.random.choice(['UP', 'DOWN']),
            'pips_net': pips,
            'return_pct': pips * 0.0001 * 0.01,  # Rough conversion
            'win': pips > 0,
            'position_size': 0.005,
            'confidence': np.random.uniform(0.4, 0.9),
            'exit_reason': 'Target Hit' if pips > 0 else 'Stop Loss'
        }
        
        market_context = {
            'regime': regime,
            'volatility': np.random.uniform(0.0001, 0.0008),
            'signal_strength': np.random.uniform(1.0, 3.0)
        }
        
        learning_system.record_trade_result(trade_result, market_context)
    
    # Show learning results
    print("\n🎓 LEARNING RESULTS:")
    status = learning_system.get_learning_status()
    
    print(f"   Trades recorded: {status['trades_recorded']}")
    print(f"   Learning iterations: {status['learning_iterations']}")
    
    if 'momentum_performance' in status['learning_metrics']:
        mom_perf = status['learning_metrics']['momentum_performance']
        print(f"   Momentum learned win rate: {mom_perf['win_rate']:.1%}")
    
    if 'mean_reversion_performance' in status['learning_metrics']:
        mr_perf = status['learning_metrics']['mean_reversion_performance']
        print(f"   Mean reversion learned win rate: {mr_perf.get('win_rate', 0):.1%}")
    
    print(f"\n🔧 ADAPTIVE PARAMETERS:")
    print(f"   Momentum base risk: {status['adaptive_params']['momentum_base_risk']:.3f}")
    print(f"   Mean reversion base risk: {status['adaptive_params']['mean_reversion_base_risk']:.3f}")
    
    # Show regime adjustments
    if status['adaptive_params']['regime_multipliers']:
        print(f"   Regime multipliers:")
        for regime, mult in status['adaptive_params']['regime_multipliers'].items():
            print(f"      {regime}: {mult:.2f}")
    
    print(f"\n✅ Learning system successfully adapting risk parameters!")
    
    return learning_system

if __name__ == "__main__":
    learning_system = demonstrate_learning_system()