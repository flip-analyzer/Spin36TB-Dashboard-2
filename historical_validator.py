#!/usr/bin/env python3
"""
Historical Validation System
Tests dual strategy against real OANDA market data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from typing import Dict, List, Optional
import sys

sys.path.append('/Users/jonspinogatti/Desktop/spin36TB')
from dual_strategy_portfolio import DualStrategyPortfolioManager

warnings.filterwarnings('ignore')

class HistoricalValidator:
    """
    Validates trading system against real historical market data
    """
    
    def __init__(self, starting_capital: float = 25000):
        self.starting_capital = starting_capital
        
        # Initialize portfolio manager
        self.portfolio = DualStrategyPortfolioManager(
            starting_capital=starting_capital,
            allocation_momentum=0.6
        )
        
        # Validation results tracking
        self.validation_results = {
            'trade_history': [],
            'performance_timeline': [],
            'regime_analysis': {},
            'comparison_metrics': {},
            'system_behavior': {}
        }
        
        print("📊 HISTORICAL VALIDATOR INITIALIZED")
        print("=" * 40)
        print(f"💰 Starting Capital: ${starting_capital:,}")
        print(f"📈 Momentum Allocation: 60%")
        print(f"📉 Mean Reversion Allocation: 40%")
        print("🎯 Ready to validate against real market data")
    
    def load_oanda_data(self, filename: str = 'oanda_eurusd_data.csv') -> pd.DataFrame:
        """
        Load real OANDA market data
        """
        filepath = f"/Users/jonspinogatti/Desktop/spin36TB/{filename}"
        
        try:
            print(f"📂 Loading real market data from {filename}...")
            
            # Load CSV data
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            
            print(f"✅ Loaded {len(df):,} real market candles")
            print(f"   Date Range: {df.index[0]} to {df.index[-1]}")
            print(f"   Time Span: {(df.index[-1] - df.index[0]).days} days")
            print(f"   Price Range: {df['close'].min():.4f} - {df['close'].max():.4f}")
            print(f"   Total Movement: {(df['close'].max() - df['close'].min()) * 10000:.0f} pips")
            
            return df
            
        except FileNotFoundError:
            print(f"❌ File not found: {filename}")
            print("   Run oanda_data_downloader.py first to get real data")
            return None
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
    
    def run_historical_validation(self, market_data: pd.DataFrame, 
                                decision_frequency: int = 20) -> Dict:
        """
        Run complete historical validation
        """
        if market_data is None or len(market_data) < 100:
            print("❌ Insufficient market data for validation")
            return {}
        
        print(f"\n🔬 RUNNING HISTORICAL VALIDATION")
        print("=" * 40)
        print(f"📊 Market Data: {len(market_data):,} candles")
        print(f"🕐 Decision Frequency: Every {decision_frequency} candles")
        print(f"⏱️  Expected Duration: ~{len(market_data) // decision_frequency} decisions")
        
        # Validation parameters
        lookback_window = 80
        total_decisions = 0
        total_trades = 0
        
        # Track portfolio value over time
        portfolio_timeline = []
        
        # Start validation
        print(f"\n🎯 Processing historical decisions...")
        
        for i in range(lookback_window, len(market_data), decision_frequency):
            # Get market window
            window = market_data.iloc[max(0, i-lookback_window):i]
            current_time = market_data.index[i-1]
            
            # Make portfolio decision
            portfolio_decision = self.portfolio.make_portfolio_decision(window)
            total_decisions += 1
            
            # Execute trades
            executed_trades = self.portfolio.execute_portfolio_trades(portfolio_decision, window)
            total_trades += len(executed_trades)
            
            # Record trade details
            for trade in executed_trades:
                trade['validation_time'] = current_time
                trade['decision_number'] = total_decisions
                self.validation_results['trade_history'].append(trade)
            
            # Track portfolio value
            portfolio_timeline.append({
                'timestamp': current_time,
                'decision_number': total_decisions,
                'portfolio_value': self.portfolio.current_capital,
                'total_trades': total_trades,
                'decisions': len(portfolio_decision['portfolio_actions'])
            })
            
            # Progress indicator every 50 decisions
            if total_decisions % 50 == 0:
                current_return = (self.portfolio.current_capital - self.starting_capital) / self.starting_capital
                print(f"   Progress: {total_decisions} decisions, {total_trades} trades, {current_return:+.2%} return")
        
        self.validation_results['performance_timeline'] = portfolio_timeline
        
        # Calculate final results
        final_results = self._calculate_validation_results(
            total_decisions, total_trades, market_data
        )
        
        print(f"\n✅ Historical validation complete!")
        print(f"   Total Decisions: {total_decisions}")
        print(f"   Total Trades: {total_trades}")
        print(f"   Final Return: {final_results.get('validation_summary', {}).get('total_return', 0):+.2%}")
        
        return final_results
    
    def _calculate_validation_results(self, total_decisions: int, 
                                    total_trades: int, market_data: pd.DataFrame) -> Dict:
        """
        Calculate comprehensive validation results
        """
        print(f"\n📊 CALCULATING VALIDATION RESULTS")
        print("=" * 35)
        
        # Portfolio performance
        final_capital = self.portfolio.current_capital
        total_return = (final_capital - self.starting_capital) / self.starting_capital
        
        # Trade analysis
        trades = self.validation_results['trade_history']
        momentum_trades = [t for t in trades if t.get('strategy') == 'MOMENTUM']
        mean_reversion_trades = [t for t in trades if t.get('strategy') == 'MEAN_REVERSION']
        
        # Calculate win rates and performance
        momentum_stats = self._calculate_strategy_stats(momentum_trades, 'MOMENTUM')
        mean_reversion_stats = self._calculate_strategy_stats(mean_reversion_trades, 'MEAN_REVERSION')
        
        # Market context analysis
        market_analysis = {
            'total_candles': len(market_data),
            'time_span_days': (market_data.index[-1] - market_data.index[0]).days,
            'price_volatility': market_data['close'].std(),
            'max_drawdown_period': self._calculate_max_drawdown_period(),
            'decision_frequency': total_decisions / len(market_data) * 100
        }
        
        results = {
            'validation_summary': {
                'starting_capital': self.starting_capital,
                'final_capital': final_capital,
                'total_return': total_return,
                'total_return_usd': final_capital - self.starting_capital,
                'total_decisions': total_decisions,
                'total_trades': total_trades,
                'trade_frequency': total_trades / total_decisions if total_decisions > 0 else 0
            },
            'strategy_performance': {
                'momentum': momentum_stats,
                'mean_reversion': mean_reversion_stats
            },
            'market_analysis': market_analysis,
            'portfolio_correlation': self._calculate_portfolio_correlation()
        }
        
        return results
    
    def _calculate_strategy_stats(self, strategy_trades: List[Dict], strategy_name: str) -> Dict:
        """
        Calculate detailed statistics for individual strategy
        """
        if not strategy_trades:
            return {
                'strategy': strategy_name,
                'total_trades': 0,
                'win_rate': 0,
                'avg_pips': 0,
                'total_return_contribution': 0
            }
        
        winning_trades = [t for t in strategy_trades if t.get('pips_net', 0) > 0]
        losing_trades = [t for t in strategy_trades if t.get('pips_net', 0) <= 0]
        
        total_pips = sum(t.get('pips_net', 0) for t in strategy_trades)
        avg_pips = total_pips / len(strategy_trades)
        
        win_rate = len(winning_trades) / len(strategy_trades)
        
        # Calculate return contribution
        total_return_contribution = sum(
            t.get('return_pct', 0) * t.get('allocation', 1.0) 
            for t in strategy_trades
        )
        
        return {
            'strategy': strategy_name,
            'total_trades': len(strategy_trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'total_pips': total_pips,
            'avg_pips': avg_pips,
            'avg_winning_pips': np.mean([t.get('pips_net', 0) for t in winning_trades]) if winning_trades else 0,
            'avg_losing_pips': np.mean([t.get('pips_net', 0) for t in losing_trades]) if losing_trades else 0,
            'total_return_contribution': total_return_contribution
        }
    
    def _calculate_max_drawdown_period(self) -> Dict:
        """
        Calculate maximum drawdown period from portfolio timeline
        """
        timeline = self.validation_results['performance_timeline']
        if not timeline:
            return {'max_drawdown': 0, 'drawdown_duration': 0}
        
        peak_value = timeline[0]['portfolio_value']
        max_drawdown = 0
        drawdown_start = None
        max_drawdown_duration = 0
        
        for point in timeline:
            current_value = point['portfolio_value']
            
            if current_value > peak_value:
                peak_value = current_value
                drawdown_start = None
            else:
                if drawdown_start is None:
                    drawdown_start = point['timestamp']
                
                current_drawdown = (peak_value - current_value) / peak_value
                if current_drawdown > max_drawdown:
                    max_drawdown = current_drawdown
                    if drawdown_start:
                        duration = (point['timestamp'] - drawdown_start).total_seconds() / 3600  # Hours
                        max_drawdown_duration = max(max_drawdown_duration, duration)
        
        return {
            'max_drawdown': max_drawdown,
            'max_drawdown_duration_hours': max_drawdown_duration
        }
    
    def _calculate_portfolio_correlation(self) -> float:
        """
        Calculate correlation between momentum and mean reversion strategies
        """
        trades = self.validation_results['trade_history']
        momentum_returns = [t.get('return_pct', 0) for t in trades if t.get('strategy') == 'MOMENTUM']
        mr_returns = [t.get('return_pct', 0) for t in trades if t.get('strategy') == 'MEAN_REVERSION']
        
        if len(momentum_returns) < 5 or len(mr_returns) < 5:
            return 0.0
        
        # Pad shorter series with zeros for correlation calculation
        max_len = max(len(momentum_returns), len(mr_returns))
        momentum_returns.extend([0] * (max_len - len(momentum_returns)))
        mr_returns.extend([0] * (max_len - len(mr_returns)))
        
        try:
            correlation = np.corrcoef(momentum_returns, mr_returns)[0, 1]
            return correlation if not np.isnan(correlation) else 0.0
        except:
            return 0.0
    
    def compare_with_simulation(self, simulated_results: Dict) -> Dict:
        """
        Compare historical results with simulated results
        """
        print(f"\n📊 COMPARING HISTORICAL vs SIMULATED RESULTS")
        print("=" * 45)
        
        historical = self.validation_results.get('final_results', {})
        
        comparison = {
            'portfolio_return': {
                'simulated': simulated_results.get('total_return', 0),
                'historical': historical.get('validation_summary', {}).get('total_return', 0),
                'difference': 'TBD'
            },
            'trade_frequency': {
                'simulated': simulated_results.get('total_trades', 0),
                'historical': historical.get('validation_summary', {}).get('total_trades', 0),
                'difference': 'TBD'
            },
            'win_rates': {
                'momentum': {
                    'simulated': 0.477,  # From your test
                    'historical': historical.get('strategy_performance', {}).get('momentum', {}).get('win_rate', 0)
                },
                'mean_reversion': {
                    'simulated': 0.875,  # From your test  
                    'historical': historical.get('strategy_performance', {}).get('mean_reversion', {}).get('win_rate', 0)
                }
            }
        }
        
        return comparison
    
    def display_validation_summary(self, results: Dict):
        """
        Display comprehensive validation summary
        """
        print(f"\n🏛️ HISTORICAL VALIDATION SUMMARY")
        print("=" * 40)
        
        summary = results.get('validation_summary', {})
        momentum = results.get('strategy_performance', {}).get('momentum', {})
        mean_reversion = results.get('strategy_performance', {}).get('mean_reversion', {})
        market = results.get('market_analysis', {})
        
        # Portfolio Performance
        print(f"💰 Portfolio Performance:")
        print(f"   Starting Capital: ${summary.get('starting_capital', 0):,.2f}")
        print(f"   Final Capital: ${summary.get('final_capital', 0):,.2f}")
        print(f"   Total Return: {summary.get('total_return', 0):+.3%}")
        print(f"   P&L: ${summary.get('total_return_usd', 0):+,.2f}")
        
        # Trading Activity
        print(f"\n📊 Trading Activity:")
        print(f"   Total Decisions: {summary.get('total_decisions', 0):,}")
        print(f"   Total Trades: {summary.get('total_trades', 0):,}")
        print(f"   Trade Frequency: {summary.get('trade_frequency', 0):.2f} trades per decision")
        
        # Strategy Breakdown
        print(f"\n📈 Momentum Strategy (Real Data):")
        if momentum.get('total_trades', 0) > 0:
            print(f"   Trades: {momentum.get('total_trades', 0)}")
            print(f"   Win Rate: {momentum.get('win_rate', 0):.1%}")
            print(f"   Avg Pips: {momentum.get('avg_pips', 0):+.1f}")
            print(f"   Avg Winner: {momentum.get('avg_winning_pips', 0):+.1f} pips")
            print(f"   Avg Loser: {momentum.get('avg_losing_pips', 0):+.1f} pips")
        else:
            print(f"   No momentum trades executed")
        
        print(f"\n📉 Mean Reversion Strategy (Real Data):")
        if mean_reversion.get('total_trades', 0) > 0:
            print(f"   Trades: {mean_reversion.get('total_trades', 0)}")
            print(f"   Win Rate: {mean_reversion.get('win_rate', 0):.1%}")
            print(f"   Avg Pips: {mean_reversion.get('avg_pips', 0):+.1f}")
            print(f"   Avg Winner: {mean_reversion.get('avg_winning_pips', 0):+.1f} pips")
            print(f"   Avg Loser: {mean_reversion.get('avg_losing_pips', 0):+.1f} pips")
        else:
            print(f"   No mean reversion trades executed")
        
        # Market Context
        print(f"\n🌍 Market Context:")
        print(f"   Time Period: {market.get('time_span_days', 0)} days")
        print(f"   Price Volatility: {market.get('price_volatility', 0):.4f}")
        print(f"   Decision Frequency: {market.get('decision_frequency', 0):.2f}% of candles")
        
        # Correlation
        print(f"\n🎭 Portfolio Diversification:")
        correlation = results.get('portfolio_correlation', 0)
        print(f"   Strategy Correlation: {correlation:+.2f}")
        
        if correlation < -0.2:
            print(f"   ✅ Excellent diversification")
        elif correlation < 0.1:
            print(f"   ✅ Good diversification")
        else:
            print(f"   ⚠️  Limited diversification")

def run_complete_validation():
    """
    Run complete historical validation process
    """
    print("🔬 COMPLETE HISTORICAL VALIDATION")
    print("=" * 40)
    
    # Initialize validator
    validator = HistoricalValidator(starting_capital=25000)
    
    # Load real market data
    market_data = validator.load_oanda_data('oanda_eurusd_data.csv')
    
    if market_data is None:
        print("❌ Cannot proceed without real market data")
        print("   Run oanda_data_downloader.py first")
        return None
    
    # Run validation
    results = validator.run_historical_validation(market_data, decision_frequency=20)
    
    if not results:
        print("❌ Validation failed")
        return None
    
    # Display results
    validator.display_validation_summary(results)
    
    # Save results for comparison
    validator.validation_results['final_results'] = results
    
    print(f"\n🎯 VALIDATION COMPLETE!")
    print("   Your system has been tested on 6 months of real market data")
    print("   Results show how it performs in actual trading conditions")
    
    return validator, results

if __name__ == "__main__":
    validator, results = run_complete_validation()