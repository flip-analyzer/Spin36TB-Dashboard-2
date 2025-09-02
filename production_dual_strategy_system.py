#!/usr/bin/env python3
"""
Production Dual Strategy System
Integrates momentum + mean reversion for live trading
Updated with comprehensive backtest results
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import warnings
from typing import Dict, List, Tuple, Optional
import sys

sys.path.append('/Users/jonspinogatti/Desktop/spin36TB')
from dual_strategy_portfolio import DualStrategyPortfolioManager

warnings.filterwarnings('ignore')

class ProductionDualStrategySystem:
    """
    Production-ready dual strategy system for live trading
    Combines professional momentum and mean reversion strategies
    """
    
    def __init__(self):
        # Load production configuration
        self.production_state = self._load_production_state()
        
        # Initialize dual strategy portfolio
        self.portfolio_manager = DualStrategyPortfolioManager(
            starting_capital=self.production_state['current_capital'],
            allocation_momentum=self.production_state['system_config']['allocation_momentum']
        )
        
        # Production trading controls
        self.trading_controls = {
            'max_daily_trades': self.production_state['system_config']['max_daily_trades'],
            'max_position_size': self.production_state['system_config']['max_position_size'],
            'daily_var_limit': self.production_state['system_config']['daily_var_limit'],
            'max_drawdown_stop': self.production_state['system_config']['max_drawdown_stop'],
            'correlation_threshold': self.production_state['system_config']['correlation_threshold'],
            'emergency_stop_triggers': {
                'consecutive_losses': 5,
                'daily_loss_limit': 0.02,  # 2% daily loss limit
                'correlation_breakdown': 0.5  # Stop if correlation becomes strongly positive
            }
        }
        
        # Session tracking
        self.session_data = {
            'trades_today': 0,
            'daily_pnl': 0.0,
            'consecutive_losses': 0,
            'session_start': datetime.now(),
            'last_trade_time': None,
            'emergency_stop': False,
            'stop_reason': None
        }
        
        print("🚀 PRODUCTION DUAL STRATEGY SYSTEM ONLINE")
        print("=" * 50)
        print(f"💰 Capital: ${self.production_state['current_capital']:,.2f}")
        print(f"📈 Momentum Allocation: {self.production_state['system_config']['allocation_momentum']:.1%}")
        print(f"📉 Mean Reversion Allocation: {self.production_state['system_config']['allocation_mean_reversion']:.1%}")
        print(f"🎯 Target Correlation: {self.production_state['system_config']['correlation_threshold']:+.2f}")
        print(f"⚡ Max Daily Trades: {self.trading_controls['max_daily_trades']}")
        print(f"🛡️  Max Position Size: {self.trading_controls['max_position_size']:.1%}")
        
        # Display backtest performance
        print(f"\n📊 BACKTEST PERFORMANCE:")
        print(f"   Strategy Correlation: {self.production_state['performance_metrics']['strategy_correlation']:+.2f}")
        print(f"   Momentum Win Rate: {self.production_state['strategy_performance']['momentum']['win_rate']:.1%}")
        print(f"   Mean Reversion Win Rate: {self.production_state['strategy_performance']['mean_reversion']['win_rate']:.1%}")
        print(f"   Sharpe Ratio: {self.production_state['risk_metrics']['sharpe_ratio']:.2f}")
    
    def _load_production_state(self) -> Dict:
        """Load production state from JSON"""
        try:
            with open('/Users/jonspinogatti/Desktop/spin36TB/production_state.json', 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Error loading production state: {e}")
            # Return default state
            return {
                'current_capital': 25000,
                'system_config': {
                    'allocation_momentum': 0.6,
                    'allocation_mean_reversion': 0.4,
                    'max_daily_trades': 20,
                    'max_position_size': 0.012,
                    'daily_var_limit': 0.004,
                    'max_drawdown_stop': 0.02,
                    'correlation_threshold': -0.2
                }
            }
    
    def _save_production_state(self):
        """Save current state to production file"""
        self.production_state['current_capital'] = self.portfolio_manager.current_capital
        self.production_state['timestamp'] = datetime.now().isoformat()
        
        try:
            with open('/Users/jonspinogatti/Desktop/spin36TB/production_state.json', 'w') as f:
                json.dump(self.production_state, f, indent=2)
        except Exception as e:
            print(f"❌ Error saving production state: {e}")
    
    def pre_trade_checks(self, market_data: pd.DataFrame) -> Dict:
        """
        Comprehensive pre-trade safety checks
        """
        checks = {
            'safe_to_trade': True,
            'warnings': [],
            'stops': []
        }
        
        # Check daily trade limit
        if self.session_data['trades_today'] >= self.trading_controls['max_daily_trades']:
            checks['safe_to_trade'] = False
            checks['stops'].append(f"Daily trade limit reached: {self.session_data['trades_today']}")
        
        # Check daily loss limit
        if self.session_data['daily_pnl'] <= -self.trading_controls['emergency_stop_triggers']['daily_loss_limit']:
            checks['safe_to_trade'] = False
            checks['stops'].append(f"Daily loss limit exceeded: {self.session_data['daily_pnl']:.2%}")
        
        # Check consecutive losses
        if self.session_data['consecutive_losses'] >= self.trading_controls['emergency_stop_triggers']['consecutive_losses']:
            checks['safe_to_trade'] = False
            checks['stops'].append(f"Too many consecutive losses: {self.session_data['consecutive_losses']}")
        
        # Check emergency stop
        if self.session_data['emergency_stop']:
            checks['safe_to_trade'] = False
            checks['stops'].append(f"Emergency stop active: {self.session_data['stop_reason']}")
        
        # Market data quality check
        if len(market_data) < 80:
            checks['safe_to_trade'] = False
            checks['stops'].append("Insufficient market data for analysis")
        
        # Volatility spike check (risk management)
        if len(market_data) >= 20:
            recent_volatility = (market_data['high'].tail(20) - market_data['low'].tail(20)).mean()
            if recent_volatility > 0.001:  # 10 pips average range (extreme volatility)
                checks['warnings'].append(f"High volatility detected: {recent_volatility/0.0001:.1f} pips")
        
        return checks
    
    def make_production_trading_decision(self, market_data: pd.DataFrame) -> Dict:
        """
        Make production trading decision with all safety checks
        """
        timestamp = datetime.now()
        
        # Pre-trade safety checks
        safety_check = self.pre_trade_checks(market_data)
        
        if not safety_check['safe_to_trade']:
            return {
                'timestamp': timestamp,
                'decision': 'HOLD',
                'reason': 'Safety checks failed',
                'safety_check': safety_check,
                'portfolio_actions': []
            }
        
        # Make portfolio decision
        portfolio_decision = self.portfolio_manager.make_portfolio_decision(market_data)
        
        # Additional production-level risk checks
        if portfolio_decision['total_portfolio_risk'] > self.trading_controls['daily_var_limit']:
            return {
                'timestamp': timestamp,
                'decision': 'HOLD', 
                'reason': f'Portfolio risk too high: {portfolio_decision["total_portfolio_risk"]:.3f}',
                'safety_check': safety_check,
                'portfolio_actions': []
            }
        
        # Check correlation breakdown
        correlation = portfolio_decision['correlation_analysis']['correlation_score']
        if (correlation > self.trading_controls['emergency_stop_triggers']['correlation_breakdown'] and
            len(portfolio_decision['portfolio_actions']) > 1):
            return {
                'timestamp': timestamp,
                'decision': 'HOLD',
                'reason': f'Correlation breakdown detected: {correlation:+.2f}',
                'safety_check': safety_check,
                'portfolio_actions': []
            }
        
        return {
            'timestamp': timestamp,
            'decision': 'TRADE' if portfolio_decision['portfolio_actions'] else 'HOLD',
            'safety_check': safety_check,
            'portfolio_decision': portfolio_decision,
            'portfolio_actions': portfolio_decision['portfolio_actions'],
            'risk_level': portfolio_decision['total_portfolio_risk']
        }
    
    def execute_production_trades(self, production_decision: Dict, market_data: pd.DataFrame) -> List[Dict]:
        """
        Execute trades in production with full tracking
        """
        if production_decision['decision'] != 'TRADE':
            return []
        
        # Execute portfolio trades
        executed_trades = self.portfolio_manager.execute_portfolio_trades(
            production_decision['portfolio_decision'], 
            market_data
        )
        
        # Update session tracking
        self.session_data['trades_today'] += len(executed_trades)
        self.session_data['last_trade_time'] = datetime.now()
        
        # Calculate session PnL
        session_pnl = sum(
            trade.get('return_pct', 0) * trade.get('allocation', 1.0) 
            for trade in executed_trades
        )
        self.session_data['daily_pnl'] += session_pnl
        
        # Update consecutive losses
        if executed_trades and all(trade.get('pips_net', 0) <= 0 for trade in executed_trades):
            self.session_data['consecutive_losses'] += 1
        elif executed_trades and any(trade.get('pips_net', 0) > 0 for trade in executed_trades):
            self.session_data['consecutive_losses'] = 0
        
        # Check for emergency stops
        self._check_emergency_conditions()
        
        # Save state
        self._save_production_state()
        
        return executed_trades
    
    def _check_emergency_conditions(self):
        """Check if emergency stop should be triggered"""
        
        # Daily loss emergency stop
        if self.session_data['daily_pnl'] <= -self.trading_controls['emergency_stop_triggers']['daily_loss_limit']:
            self.session_data['emergency_stop'] = True
            self.session_data['stop_reason'] = f"Daily loss limit: {self.session_data['daily_pnl']:.2%}"
        
        # Consecutive losses emergency stop  
        if self.session_data['consecutive_losses'] >= self.trading_controls['emergency_stop_triggers']['consecutive_losses']:
            self.session_data['emergency_stop'] = True
            self.session_data['stop_reason'] = f"Consecutive losses: {self.session_data['consecutive_losses']}"
        
        # Portfolio drawdown check
        current_drawdown = (self.production_state['risk_metrics']['portfolio_peak'] - self.portfolio_manager.current_capital) / self.production_state['risk_metrics']['portfolio_peak']
        if current_drawdown > self.trading_controls['max_drawdown_stop']:
            self.session_data['emergency_stop'] = True
            self.session_data['stop_reason'] = f"Max drawdown exceeded: {current_drawdown:.2%}"
    
    def get_production_status(self) -> Dict:
        """Get comprehensive production system status"""
        
        portfolio_status = self.portfolio_manager.get_portfolio_status()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'system_status': 'ONLINE' if not self.session_data['emergency_stop'] else 'EMERGENCY_STOP',
            'emergency_stop_reason': self.session_data.get('stop_reason'),
            'session_data': self.session_data,
            'portfolio_status': portfolio_status,
            'trading_controls': self.trading_controls,
            'production_config': self.production_state['system_config'],
            'backtest_metrics': {
                'strategy_correlation': self.production_state['performance_metrics']['strategy_correlation'],
                'momentum_win_rate': self.production_state['strategy_performance']['momentum']['win_rate'],
                'mean_reversion_win_rate': self.production_state['strategy_performance']['mean_reversion']['win_rate'],
                'sharpe_ratio': self.production_state['risk_metrics']['sharpe_ratio']
            }
        }
    
    def reset_daily_session(self):
        """Reset daily session counters"""
        self.session_data = {
            'trades_today': 0,
            'daily_pnl': 0.0,
            'consecutive_losses': 0,
            'session_start': datetime.now(),
            'last_trade_time': None,
            'emergency_stop': False,
            'stop_reason': None
        }
        print("🔄 Daily session reset complete")
    
    def manual_emergency_stop(self, reason: str):
        """Manually trigger emergency stop"""
        self.session_data['emergency_stop'] = True
        self.session_data['stop_reason'] = f"Manual stop: {reason}"
        print(f"🛑 EMERGENCY STOP ACTIVATED: {reason}")
    
    def clear_emergency_stop(self):
        """Clear emergency stop (manual override)"""
        self.session_data['emergency_stop'] = False
        self.session_data['stop_reason'] = None
        self.session_data['consecutive_losses'] = 0
        print("✅ Emergency stop cleared - system ready for trading")

def run_production_system_demo():
    """
    Demonstrate production system capabilities
    """
    print("\n🚀 PRODUCTION DUAL STRATEGY DEMO")
    print("=" * 40)
    
    # Initialize production system
    prod_system = ProductionDualStrategySystem()
    
    # Generate some sample market data
    dates = pd.date_range(start=datetime.now() - timedelta(hours=4), end=datetime.now(), freq='5min')
    np.random.seed(42)
    
    prices = [1.0850]
    for i in range(len(dates) - 1):
        change = np.random.normal(0, 0.0003)
        prices.append(prices[-1] * (1 + change))
    
    market_data = []
    for i, (timestamp, close_price) in enumerate(zip(dates, prices)):
        if i == 0:
            continue
        
        open_price = prices[i-1]
        high = max(open_price, close_price) + abs(np.random.normal(0, 0.0001))
        low = min(open_price, close_price) - abs(np.random.normal(0, 0.0001))
        volume = np.random.uniform(800, 1200)
        
        market_data.append({
            'timestamp': timestamp,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close_price,
            'volume': volume
        })
    
    df = pd.DataFrame(market_data)
    df.set_index('timestamp', inplace=True)
    
    print(f"\n📊 Market Data: {len(df)} candles, price range {df['close'].min():.4f}-{df['close'].max():.4f}")
    
    # Demo production decisions
    print(f"\n🎯 Production Trading Decisions:")
    
    for i in range(3):
        print(f"\n   Decision {i+1}:")
        
        # Make production decision
        decision = prod_system.make_production_trading_decision(df)
        
        print(f"      Decision: {decision['decision']}")
        print(f"      Safety Checks: {'✅ PASS' if decision['safety_check']['safe_to_trade'] else '❌ FAIL'}")
        
        if decision['decision'] == 'TRADE':
            print(f"      Actions: {len(decision['portfolio_actions'])}")
            print(f"      Risk Level: {decision.get('risk_level', 0):.3f}")
            
            # Execute trades
            trades = prod_system.execute_production_trades(decision, df)
            if trades:
                for trade in trades:
                    print(f"         → {trade['strategy']}: {trade['direction']} {trade['pips_net']:+.1f} pips")
        else:
            print(f"      Reason: {decision.get('reason', 'N/A')}")
        
        # Add some randomness for demo
        np.random.seed(np.random.randint(100))
    
    # Show final status
    print(f"\n📊 Production System Status:")
    status = prod_system.get_production_status()
    
    print(f"   System Status: {status['system_status']}")
    print(f"   Trades Today: {status['session_data']['trades_today']}")
    print(f"   Daily PnL: {status['session_data']['daily_pnl']:+.3%}")
    portfolio_value = status.get('portfolio_status', {}).get('performance', {}).get('portfolio', {}).get('current_capital', prod_system.portfolio_manager.current_capital)
    print(f"   Portfolio Value: ${portfolio_value:,.2f}")
    print(f"   Emergency Stop: {status['session_data']['emergency_stop']}")
    
    print(f"\n✅ PRODUCTION SYSTEM DEMO COMPLETE")
    
    return prod_system

if __name__ == "__main__":
    run_production_system_demo()