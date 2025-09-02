#!/usr/bin/env python3
"""
Dual Strategy Portfolio Manager: Momentum + Mean Reversion
Combines Spin36TB momentum system with EURUSD mean reversion
Designed for negative correlation and risk reduction
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from typing import Dict, List, Tuple, Optional
import json
import sys

# Import both systems and enhanced risk manager
sys.path.append('/Users/jonspinogatti/Desktop/spin36TB')
from critical_fixes import ProfessionalSpin36TBSystem
from mean_reversion_system import EURUSDMeanReversionSystem
from enhanced_risk_manager import EnhancedRiskManager

warnings.filterwarnings('ignore')

class DualStrategyPortfolioManager:
    """
    Professional portfolio manager combining momentum and mean reversion strategies
    Optimized for uncorrelated returns and risk reduction
    """
    
    def __init__(self, starting_capital: float = 25000, allocation_momentum: float = 0.6):
        self.starting_capital = starting_capital
        self.current_capital = starting_capital
        self.allocation_momentum = allocation_momentum  # 60% momentum, 40% mean reversion
        self.allocation_mean_reversion = 1.0 - allocation_momentum
        
        # Initialize both systems with allocated capital
        momentum_capital = starting_capital * allocation_momentum
        mean_reversion_capital = starting_capital * self.allocation_mean_reversion
        
        self.momentum_system = ProfessionalSpin36TBSystem(momentum_capital)
        self.mean_reversion_system = EURUSDMeanReversionSystem(mean_reversion_capital)
        
        # Initialize enhanced risk manager
        self.risk_manager = EnhancedRiskManager(starting_capital)
        
        # Portfolio configuration
        self.portfolio_config = {
            'max_simultaneous_trades': 2,        # One per strategy max
            'conflict_resolution': 'BOTH',       # Allow both strategies to trade
            'correlation_threshold': -0.5,       # Expected negative correlation
            'rebalance_frequency': 50,           # Rebalance every 50 trades
            'max_portfolio_risk': 0.012,         # 1.2% max total portfolio risk
        }
        
        # Performance tracking
        self.portfolio_performance = {
            'total_trades': 0,
            'momentum_trades': 0,
            'mean_reversion_trades': 0,
            'simultaneous_trades': 0,
            'conflicting_signals': 0,
            'correlation_samples': [],
            'daily_returns': [],
            'trade_history': []
        }
        
        # Risk management
        self.risk_metrics = {
            'portfolio_var': 0.0,
            'strategy_correlation': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'current_momentum_exposure': 0.0,
            'current_mean_reversion_exposure': 0.0
        }
        
        print("🎭 DUAL STRATEGY PORTFOLIO MANAGER")
        print("=" * 45)
        print(f"💰 Total Capital: ${starting_capital:,}")
        print(f"📈 Momentum Allocation: {allocation_momentum:.1%} (${momentum_capital:,.0f})")
        print(f"📉 Mean Reversion Allocation: {self.allocation_mean_reversion:.1%} (${mean_reversion_capital:,.0f})")
        print("🎯 Strategy: Momentum + Mean Reversion for uncorrelated returns")
        print("⚖️  Risk: Designed for negative correlation and lower volatility")
        print("🔄 Conflict Resolution: Allow both strategies to trade")
    
    def analyze_signal_correlation(self, momentum_decision: Dict, mean_reversion_decision: Dict) -> Dict:
        """
        Analyze correlation between momentum and mean reversion signals
        """
        # Extract signal directions
        momentum_signal = momentum_decision.get('signal', 'HOLD')
        mean_reversion_signal = mean_reversion_decision.get('action', 'HOLD')
        
        # Convert to numeric for correlation analysis
        momentum_numeric = 1 if momentum_signal == 'UP' else (-1 if momentum_signal == 'DOWN' else 0)
        mean_reversion_numeric = 1 if mean_reversion_decision.get('direction') == 'UP' else (-1 if mean_reversion_decision.get('direction') == 'DOWN' else 0)
        
        # Classify relationship
        if momentum_numeric != 0 and mean_reversion_numeric != 0:
            if momentum_numeric == mean_reversion_numeric:
                relationship = 'REINFORCING'  # Same direction
                correlation_score = 1.0
            else:
                relationship = 'OPPOSING'     # Opposite directions (desired)
                correlation_score = -1.0
        elif momentum_numeric != 0 and mean_reversion_numeric == 0:
            relationship = 'MOMENTUM_ONLY'
            correlation_score = 0.0
        elif momentum_numeric == 0 and mean_reversion_numeric != 0:
            relationship = 'MEAN_REVERSION_ONLY'
            correlation_score = 0.0
        else:
            relationship = 'BOTH_HOLD'
            correlation_score = 0.0
        
        return {
            'relationship': relationship,
            'correlation_score': correlation_score,
            'momentum_signal': momentum_signal,
            'mean_reversion_signal': mean_reversion_signal,
            'momentum_numeric': momentum_numeric,
            'mean_reversion_numeric': mean_reversion_numeric
        }
    
    def make_portfolio_decision(self, market_data: pd.DataFrame) -> Dict:
        """
        Make coordinated portfolio decision using both strategies with enhanced risk management
        """
        timestamp = datetime.now()
        
        # ENHANCED: Check for emergency stop conditions first
        emergency_check = self.risk_manager.get_emergency_stops()
        if emergency_check['emergency_stop_required']:
            return {
                'timestamp': timestamp,
                'portfolio_actions': [],
                'emergency_stop': True,
                'emergency_conditions': emergency_check['conditions'],
                'message': "🚨 EMERGENCY STOP: Trading halted due to risk conditions"
            }
        
        # ENHANCED: Assess market stress before making decisions
        market_stress = self.risk_manager.assess_market_stress(market_data)
        if market_stress['recommendation'] == 'STOP_NEW_TRADES':
            return {
                'timestamp': timestamp,
                'portfolio_actions': [],
                'market_stress': market_stress,
                'message': f"⚠️ MARKET STRESS: {market_stress['overall_stress']} - No new trades"
            }
        
        # Get decisions from both strategies
        momentum_decision = self.momentum_system.make_professional_trading_decision(market_data)
        mean_reversion_decision = self.mean_reversion_system.make_mean_reversion_decision(market_data)
        
        # Analyze signal correlation
        correlation_analysis = self.analyze_signal_correlation(momentum_decision, mean_reversion_decision)
        
        # ENHANCED: Check correlation risk
        correlation_risk = self.risk_manager.check_correlation_risk(
            momentum_decision.get('signal', 'HOLD'),
            mean_reversion_decision.get('signal', 'HOLD')
        )
        
        # Track correlation for portfolio analysis
        self.portfolio_performance['correlation_samples'].append(correlation_analysis['correlation_score'])
        
        # Determine portfolio action based on conflict resolution strategy
        portfolio_actions = []
        
        # ENHANCED: Apply risk management to momentum strategy
        if momentum_decision['signal'] in ['UP', 'DOWN']:
            # Calculate enhanced position size
            risk_adjusted_size = self.risk_manager.calculate_dynamic_position_size(
                base_size=momentum_decision.get('position_size', 0.015),
                market_data=market_data,
                signal_confidence=momentum_decision.get('confidence', 0.5),
                strategy_type='momentum'
            )
            
            # Calculate dynamic stops
            entry_price = market_data['close'].iloc[-1]
            dynamic_stops = self.risk_manager.calculate_dynamic_stops(
                market_data=market_data,
                entry_price=entry_price,
                direction=momentum_decision['signal']
            )
            
            # Apply risk adjustments
            enhanced_decision = momentum_decision.copy()
            enhanced_decision['position_size'] = risk_adjusted_size['position_size']
            enhanced_decision['risk_adjusted'] = True
            enhanced_decision['stop_loss'] = dynamic_stops['stop_loss']
            enhanced_decision['take_profit'] = dynamic_stops['take_profit']
            enhanced_decision['risk_factors'] = risk_adjusted_size['risk_factors']
            
            portfolio_actions.append({
                'strategy': 'MOMENTUM',
                'action': 'TRADE',
                'decision': enhanced_decision,
                'allocation': self.allocation_momentum,
                'risk_info': risk_adjusted_size
            })
        
        # ENHANCED: Apply risk management to mean reversion strategy  
        if mean_reversion_decision['action'] == 'TRADE':
            # Calculate enhanced position size
            risk_adjusted_size = self.risk_manager.calculate_dynamic_position_size(
                base_size=mean_reversion_decision.get('position_size', 0.008),
                market_data=market_data,
                signal_confidence=mean_reversion_decision.get('confidence', 0.6),
                strategy_type='mean_reversion'
            )
            
            # Calculate dynamic stops
            entry_price = market_data['close'].iloc[-1]
            dynamic_stops = self.risk_manager.calculate_dynamic_stops(
                market_data=market_data,
                entry_price=entry_price,
                direction=mean_reversion_decision.get('signal', 'BUY')
            )
            
            # Apply risk adjustments
            enhanced_decision = mean_reversion_decision.copy()
            enhanced_decision['position_size'] = risk_adjusted_size['position_size']
            enhanced_decision['risk_adjusted'] = True
            enhanced_decision['stop_loss'] = dynamic_stops['stop_loss']
            enhanced_decision['take_profit'] = dynamic_stops['take_profit']
            enhanced_decision['risk_factors'] = risk_adjusted_size['risk_factors']
            
            portfolio_actions.append({
                'strategy': 'MEAN_REVERSION',
                'action': 'TRADE',
                'decision': enhanced_decision,
                'allocation': self.allocation_mean_reversion,
                'risk_info': risk_adjusted_size
            })
        
        # Portfolio risk check
        total_risk = self._calculate_portfolio_risk(portfolio_actions)
        
        if total_risk > self.portfolio_config['max_portfolio_risk']:
            # Scale down positions proportionally
            scale_factor = self.portfolio_config['max_portfolio_risk'] / total_risk
            for action in portfolio_actions:
                if action['strategy'] == 'MOMENTUM':
                    action['decision']['position_size'] *= scale_factor
                else:  # MEAN_REVERSION
                    action['decision']['position_size'] *= scale_factor
        
        # Count signal types
        if correlation_analysis['relationship'] == 'OPPOSING':
            self.portfolio_performance['conflicting_signals'] += 1
        
        if len(portfolio_actions) == 2:
            self.portfolio_performance['simultaneous_trades'] += 1
        
        return {
            'timestamp': timestamp,
            'portfolio_actions': portfolio_actions,
            'correlation_analysis': correlation_analysis,
            'total_portfolio_risk': total_risk,
            'momentum_decision': momentum_decision,
            'mean_reversion_decision': mean_reversion_decision,
            'actions_count': len(portfolio_actions)
        }
    
    def _calculate_portfolio_risk(self, portfolio_actions: List[Dict]) -> float:
        """
        Calculate total portfolio risk from all proposed actions
        """
        total_risk = 0.0
        
        for action in portfolio_actions:
            if action['action'] == 'TRADE':
                position_size = action['decision']['position_size']
                allocation = action['allocation']
                
                # Risk per strategy = position_size * allocation * leverage
                strategy_risk = position_size * allocation * 2.0  # 2x leverage
                total_risk += strategy_risk
        
        return total_risk
    
    def execute_portfolio_trades(self, portfolio_decision: Dict, market_data: pd.DataFrame) -> List[Dict]:
        """
        Execute all portfolio trades and track results
        """
        executed_trades = []
        
        for action in portfolio_decision['portfolio_actions']:
            if action['action'] == 'TRADE':
                
                if action['strategy'] == 'MOMENTUM':
                    # Execute momentum trade
                    trade_result = self._execute_momentum_trade(action['decision'], market_data)
                    if trade_result:
                        trade_result['strategy'] = 'MOMENTUM'
                        trade_result['allocation'] = action['allocation']
                        executed_trades.append(trade_result)
                        self.portfolio_performance['momentum_trades'] += 1
                
                elif action['strategy'] == 'MEAN_REVERSION':
                    # Execute mean reversion trade
                    trade_result = self.mean_reversion_system.simulate_mean_reversion_trade(
                        action['decision'], market_data
                    )
                    if trade_result:
                        trade_result['strategy'] = 'MEAN_REVERSION'
                        trade_result['allocation'] = action['allocation']
                        executed_trades.append(trade_result)
                        self.portfolio_performance['mean_reversion_trades'] += 1
        
        # Update portfolio performance
        self.portfolio_performance['total_trades'] += len(executed_trades)
        self.portfolio_performance['trade_history'].extend(executed_trades)
        
        # Update capital (simplified - in production would be more sophisticated)
        total_return = sum(trade.get('return_pct', 0) * trade.get('allocation', 1.0) for trade in executed_trades)
        self.current_capital *= (1 + total_return)
        
        return executed_trades
    
    def _execute_momentum_trade(self, decision: Dict, market_data: pd.DataFrame) -> Optional[Dict]:
        """
        Execute momentum trade (simplified simulation)
        """
        entry_price = market_data['close'].iloc[-1]
        direction = decision['direction']
        position_size = decision['position_size']
        confidence = decision['confidence']
        
        # Simulate outcome (using momentum win rate from testing: 66.7%)
        win_rate = 0.667 * confidence  # Scale by confidence
        is_winner = np.random.random() < win_rate
        
        if is_winner:
            pips_gross = np.random.uniform(8, 18)  # Professional momentum targets
            exit_reason = 'Target Hit'
        else:
            pips_gross = -np.random.uniform(6, 12)  # Professional stops
            exit_reason = 'Stop Loss'
        
        # Apply transaction costs
        transaction_costs = 1.5  # pips
        pips_net = pips_gross - transaction_costs
        
        # Calculate return
        leverage = 2.0
        return_pct = (pips_net * 0.0001) * position_size * leverage
        
        exit_price = entry_price + (pips_gross * 0.0001) * (1 if direction == 'UP' else -1)
        
        return {
            'entry_time': market_data.index[-1],
            'direction': direction,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'position_size': position_size,
            'pips_gross': pips_gross,
            'pips_net': pips_net,
            'return_pct': return_pct,
            'exit_reason': exit_reason,
            'confidence': confidence,
            'win': pips_net > 0,
            'type': 'MOMENTUM',
            'transaction_costs': transaction_costs
        }
    
    def calculate_portfolio_performance(self) -> Dict:
        """
        Calculate comprehensive portfolio performance metrics
        """
        if not self.portfolio_performance['trade_history']:
            return {'no_trades': True}
        
        trades = self.portfolio_performance['trade_history']
        
        # Separate by strategy
        momentum_trades = [t for t in trades if t.get('strategy') == 'MOMENTUM']
        mean_reversion_trades = [t for t in trades if t.get('strategy') == 'MEAN_REVERSION']
        
        # Calculate individual strategy performance
        momentum_performance = self._calculate_strategy_performance(momentum_trades, 'MOMENTUM')
        mean_reversion_performance = self._calculate_strategy_performance(mean_reversion_trades, 'MEAN_REVERSION')
        
        # Portfolio-level metrics
        total_return_pct = (self.current_capital - self.starting_capital) / self.starting_capital
        
        # Calculate correlation
        if len(self.portfolio_performance['correlation_samples']) > 10:
            avg_correlation = np.mean(self.portfolio_performance['correlation_samples'])
        else:
            avg_correlation = 0.0
        
        # Strategy distribution
        total_trades = len(trades)
        momentum_pct = len(momentum_trades) / total_trades if total_trades > 0 else 0
        mean_reversion_pct = len(mean_reversion_trades) / total_trades if total_trades > 0 else 0
        
        return {
            'portfolio': {
                'starting_capital': self.starting_capital,
                'current_capital': self.current_capital,
                'total_return_pct': total_return_pct,
                'total_return_usd': self.current_capital - self.starting_capital,
                'total_trades': total_trades
            },
            'momentum_strategy': momentum_performance,
            'mean_reversion_strategy': mean_reversion_performance,
            'portfolio_metrics': {
                'strategy_correlation': avg_correlation,
                'momentum_trade_pct': momentum_pct,
                'mean_reversion_trade_pct': mean_reversion_pct,
                'simultaneous_trades': self.portfolio_performance['simultaneous_trades'],
                'conflicting_signals': self.portfolio_performance['conflicting_signals'],
                'diversification_benefit': avg_correlation < -0.2  # Good if negative correlation
            }
        }
    
    def _calculate_strategy_performance(self, strategy_trades: List[Dict], strategy_name: str) -> Dict:
        """
        Calculate performance for individual strategy
        """
        if not strategy_trades:
            return {'strategy': strategy_name, 'no_trades': True}
        
        winning_trades = [t for t in strategy_trades if t.get('win', False)]
        total_trades = len(strategy_trades)
        win_rate = len(winning_trades) / total_trades
        
        total_pips = sum(t.get('pips_net', 0) for t in strategy_trades)
        avg_pips = total_pips / total_trades
        
        total_return = sum(t.get('return_pct', 0) * t.get('allocation', 1.0) for t in strategy_trades)
        
        return {
            'strategy': strategy_name,
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'win_rate': win_rate,
            'total_pips': total_pips,
            'avg_pips_per_trade': avg_pips,
            'total_return_contribution': total_return
        }
    
    def get_portfolio_status(self) -> Dict:
        """
        Get comprehensive portfolio status
        """
        performance = self.calculate_portfolio_performance()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'portfolio_manager': 'DUAL_STRATEGY',
            'strategies': ['MOMENTUM', 'MEAN_REVERSION'],
            'allocations': {
                'momentum': self.allocation_momentum,
                'mean_reversion': self.allocation_mean_reversion
            },
            'performance': performance,
            'risk_metrics': self.risk_metrics,
            'config': self.portfolio_config
        }

def test_dual_strategy_portfolio():
    """
    Test the dual strategy portfolio manager
    """
    print("\n🎭 TESTING DUAL STRATEGY PORTFOLIO")
    print("=" * 45)
    
    # Initialize portfolio manager
    portfolio = DualStrategyPortfolioManager(
        starting_capital=25000,
        allocation_momentum=0.6  # 60% momentum, 40% mean reversion
    )
    
    # Generate realistic market data
    print("📊 Generating realistic market data...")
    dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='5min')
    dates = dates[dates.weekday < 5]  # Remove weekends
    
    np.random.seed(42)
    base_price = 1.0850
    
    # Generate mixed momentum + mean-reverting price action
    prices = [base_price]
    trend_direction = 1  # Start with uptrend
    mean_price = base_price
    
    for i in range(len(dates) - 1):
        current_price = prices[-1]
        
        # Occasionally change trend (momentum opportunities)
        if np.random.random() < 0.005:  # 0.5% chance per period
            trend_direction *= -1
        
        # Trend component (momentum)
        trend_force = trend_direction * 0.00015 * np.random.uniform(0.5, 1.5)
        
        # Mean reversion component
        deviation = current_price - mean_price
        reversion_force = -0.08 * deviation / mean_price
        
        # Noise
        noise = np.random.normal(0, 0.0003)
        
        # Combine forces
        total_change = trend_force + reversion_force + noise
        new_price = current_price * (1 + total_change)
        prices.append(new_price)
        
        # Slowly adjust mean
        mean_price = mean_price * (1 + trend_direction * 0.00002)
    
    # Create market data
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
    
    print(f"   Generated {len(df)} candles over {(df.index[-1] - df.index[0]).days} days")
    print(f"   Price range: {df['close'].min():.4f} - {df['close'].max():.4f}")
    
    # Test portfolio decisions
    print("\n🎯 Testing Dual Strategy Portfolio Decisions:")
    
    total_decisions = 0
    momentum_only = 0
    mean_reversion_only = 0
    both_strategies = 0
    opposing_signals = 0
    
    # Test every 30 candles
    for i in range(80, len(df), 30):  # Start at 80 for sufficient lookback
        window = df.iloc[max(0, i-80):i]
        
        # Make portfolio decision
        portfolio_decision = portfolio.make_portfolio_decision(window)
        total_decisions += 1
        
        # Execute trades
        executed_trades = portfolio.execute_portfolio_trades(portfolio_decision, window)
        
        # Analyze decision
        actions = portfolio_decision['portfolio_actions']
        correlation = portfolio_decision['correlation_analysis']
        
        if len(actions) == 1:
            if actions[0]['strategy'] == 'MOMENTUM':
                momentum_only += 1
            else:
                mean_reversion_only += 1
        elif len(actions) == 2:
            both_strategies += 1
        
        if correlation['relationship'] == 'OPPOSING':
            opposing_signals += 1
        
        # Display significant decisions
        if len(actions) > 0:
            print(f"\n   Decision {total_decisions}:")
            print(f"      Strategies Active: {len(actions)}")
            print(f"      Signal Relationship: {correlation['relationship']}")
            
            for action in actions:
                strategy = action['strategy']
                decision = action['decision']
                direction = decision.get('direction', decision.get('signal', 'HOLD'))
                
                if strategy == 'MOMENTUM':
                    confidence = decision.get('confidence', 0)
                    regime = decision.get('regime', 'UNKNOWN')
                    print(f"         {strategy}: {direction} ({confidence:.1%} conf, {regime})")
                else:  # MEAN_REVERSION
                    confidence = decision.get('confidence', 0)
                    z_score = decision.get('indicators', {}).get('z_score', 0)
                    print(f"         {strategy}: {direction} ({confidence:.1%} conf, Z={z_score:+.2f})")
            
            for trade in executed_trades:
                print(f"      → {trade['strategy']}: {trade['pips_net']:+.1f} pips ({trade['exit_reason']})")
    
    # Final portfolio analysis
    print(f"\n📊 DUAL STRATEGY PORTFOLIO RESULTS:")
    status = portfolio.get_portfolio_status()
    
    performance = status['performance']
    
    if not performance.get('no_trades'):
        print(f"   Portfolio Value: ${performance['portfolio']['current_capital']:,.2f}")
        print(f"   Total Return: {performance['portfolio']['total_return_pct']:+.2%}")
        print(f"   Total Trades: {performance['portfolio']['total_trades']}")
        
        print(f"\n   📈 Momentum Strategy:")
        mom_perf = performance['momentum_strategy']
        if not mom_perf.get('no_trades'):
            print(f"      Trades: {mom_perf['total_trades']}")
            print(f"      Win Rate: {mom_perf['win_rate']:.1%}")
            print(f"      Avg Pips: {mom_perf['avg_pips_per_trade']:+.1f}")
        
        print(f"\n   📉 Mean Reversion Strategy:")
        mr_perf = performance['mean_reversion_strategy']
        if not mr_perf.get('no_trades'):
            print(f"      Trades: {mr_perf['total_trades']}")
            print(f"      Win Rate: {mr_perf['win_rate']:.1%}")
            print(f"      Avg Pips: {mr_perf['avg_pips_per_trade']:+.1f}")
        
        print(f"\n   🎭 Portfolio Diversification:")
        port_metrics = performance['portfolio_metrics']
        print(f"      Strategy Correlation: {port_metrics['strategy_correlation']:+.2f}")
        print(f"      Momentum Trades: {port_metrics['momentum_trade_pct']:.1%}")
        print(f"      Mean Reversion Trades: {port_metrics['mean_reversion_trade_pct']:.1%}")
        print(f"      Opposing Signals: {port_metrics['conflicting_signals']}")
        print(f"      Diversification Benefit: {port_metrics['diversification_benefit']}")
    
    print(f"\n   🎯 Decision Analysis:")
    print(f"      Total Decisions: {total_decisions}")
    print(f"      Momentum Only: {momentum_only} ({momentum_only/total_decisions:.1%})")
    print(f"      Mean Reversion Only: {mean_reversion_only} ({mean_reversion_only/total_decisions:.1%})")
    print(f"      Both Strategies: {both_strategies} ({both_strategies/total_decisions:.1%})")
    print(f"      Opposing Signals: {opposing_signals} ({opposing_signals/total_decisions:.1%})")
    
    print(f"\n🏛️ PROFESSIONAL ASSESSMENT:")
    
    if performance.get('portfolio', {}).get('total_trades', 0) > 0:
        total_return = performance['portfolio']['total_return_pct']
        correlation = performance['portfolio_metrics']['strategy_correlation']
        
        if total_return > 0:
            print(f"   ✅ Portfolio profitable: {total_return:+.2%}")
        else:
            print(f"   ⚠️  Portfolio unprofitable: {total_return:+.2%}")
        
        if correlation < -0.1:
            print(f"   ✅ Good diversification: Correlation {correlation:+.2f}")
        elif correlation < 0.3:
            print(f"   ⚠️  Moderate diversification: Correlation {correlation:+.2f}")
        else:
            print(f"   ❌ Poor diversification: Correlation {correlation:+.2f}")
        
        if opposing_signals >= total_decisions * 0.2:
            print(f"   ✅ Strategies provide good balance")
        else:
            print(f"   ⚠️  Strategies may be too correlated")
    
    print(f"\n✅ DUAL STRATEGY PORTFOLIO TEST COMPLETE")
    
    return portfolio, status

if __name__ == "__main__":
    test_dual_strategy_portfolio()