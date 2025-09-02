#!/usr/bin/env python3
"""
Comprehensive Backtest for Dual Strategy Portfolio
Tests the combined momentum + mean reversion system performance
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from typing import Dict, List, Tuple, Optional
import sys

sys.path.append('/Users/jonspinogatti/Desktop/spin36TB')
from dual_strategy_portfolio import DualStrategyPortfolioManager

warnings.filterwarnings('ignore')

def generate_comprehensive_market_data(days: int = 60) -> pd.DataFrame:
    """
    Generate comprehensive market data for backtesting
    """
    print(f"📊 Generating {days} days of realistic EURUSD data...")
    
    # Create trading hours only (remove weekends)
    start_date = datetime(2024, 1, 1)
    end_date = start_date + timedelta(days=days)
    
    dates = pd.date_range(start=start_date, end=end_date, freq='5min')
    trading_dates = dates[dates.weekday < 5]  # Remove weekends
    
    # Add realistic market hours filter (5 AM - 5 PM EST)
    trading_hours = []
    for dt in trading_dates:
        hour = dt.hour
        if 5 <= hour <= 17:  # Main trading hours
            trading_hours.append(dt)
    
    dates = pd.Series(trading_hours)
    
    np.random.seed(42)  # Reproducible results
    base_price = 1.0850
    
    # Generate realistic price movements with multiple regimes
    prices = [base_price]
    current_regime = 'RANGING'
    regime_duration = 0
    volatility_multiplier = 1.0
    trend_strength = 0.0
    
    for i in range(len(dates) - 1):
        current_price = prices[-1]
        
        # Regime switching logic
        regime_duration += 1
        
        # Switch regimes periodically
        if regime_duration > np.random.exponential(100):  # Average 100 periods per regime
            regime_duration = 0
            regime_choice = np.random.choice(['TRENDING_UP', 'TRENDING_DOWN', 'HIGH_VOL_RANGING', 'LOW_VOL_RANGING'], 
                                           p=[0.25, 0.25, 0.25, 0.25])
            current_regime = regime_choice
            
            # Set regime parameters
            if current_regime == 'TRENDING_UP':
                trend_strength = np.random.uniform(0.0001, 0.0003)
                volatility_multiplier = np.random.uniform(0.8, 1.2)
            elif current_regime == 'TRENDING_DOWN':
                trend_strength = -np.random.uniform(0.0001, 0.0003)
                volatility_multiplier = np.random.uniform(0.8, 1.2)
            elif current_regime == 'HIGH_VOL_RANGING':
                trend_strength = 0.0
                volatility_multiplier = np.random.uniform(1.5, 2.5)
            else:  # LOW_VOL_RANGING
                trend_strength = 0.0
                volatility_multiplier = np.random.uniform(0.3, 0.7)
        
        # Price movement components
        
        # 1. Trend component
        trend_force = trend_strength
        
        # 2. Mean reversion component (stronger in ranging markets)
        mean_reversion_target = np.mean(prices[-min(50, len(prices)):])  # 50-period mean
        reversion_strength = 0.1 if 'RANGING' in current_regime else 0.02
        reversion_force = reversion_strength * (mean_reversion_target - current_price) / current_price
        
        # 3. Random volatility
        base_volatility = 0.0003  # 3 pips base volatility
        noise = np.random.normal(0, base_volatility * volatility_multiplier)
        
        # 4. Occasional spikes (news events)
        if np.random.random() < 0.005:  # 0.5% chance per period
            spike_magnitude = np.random.uniform(0.002, 0.008)  # 20-80 pips
            spike_direction = 1 if np.random.random() > 0.5 else -1
            noise += spike_magnitude * spike_direction
        
        # Combine all forces
        total_change = trend_force + reversion_force + noise
        
        # Apply change
        new_price = current_price * (1 + total_change)
        
        # Keep price in reasonable bounds
        new_price = max(1.0400, min(1.1400, new_price))  # 1000 pip range
        
        prices.append(new_price)
    
    # Create OHLCV data
    market_data = []
    for i, (timestamp, close_price) in enumerate(zip(dates, prices)):
        if i == 0:
            continue
        
        open_price = prices[i-1]
        
        # Create realistic high/low based on volatility
        volatility_pips = abs(close_price - open_price) / 0.0001
        additional_range = np.random.uniform(0.5, 3.0)  # 0.5-3 extra pips
        
        high = max(open_price, close_price) + (additional_range * 0.0001)
        low = min(open_price, close_price) - (additional_range * 0.0001)
        
        # Realistic volume (higher during news/volatility)
        base_volume = np.random.uniform(800, 1200)
        if volatility_pips > 5:  # High volatility periods
            volume = base_volume * np.random.uniform(1.5, 3.0)
        else:
            volume = base_volume
        
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
    
    print(f"   Generated {len(df):,} candles")
    print(f"   Price range: {df['close'].min():.4f} - {df['close'].max():.4f}")
    print(f"   Total pip movement: {(df['close'].max() - df['close'].min()) / 0.0001:.0f} pips")
    
    return df

def run_comprehensive_backtest():
    """
    Run comprehensive backtest of dual strategy system
    """
    print("\n🔬 COMPREHENSIVE DUAL STRATEGY BACKTEST")
    print("=" * 50)
    
    # Generate market data
    market_data = generate_comprehensive_market_data(days=30)  # 1 month of data
    
    # Initialize portfolio
    starting_capital = 25000
    portfolio = DualStrategyPortfolioManager(
        starting_capital=starting_capital,
        allocation_momentum=0.6  # 60% momentum, 40% mean reversion
    )
    
    print(f"\n🎯 Running backtest on {len(market_data):,} candles...")
    
    # Backtest parameters
    lookback_window = 80  # Candles of history needed
    decision_frequency = 20  # Make decision every 20 candles (more realistic)
    
    # Backtest results tracking
    backtest_results = {
        'decisions': [],
        'trades': [],
        'equity_curve': [],
        'regime_analysis': {},
        'correlation_tracking': []
    }
    
    # Run backtest
    decisions_made = 0
    total_trades = 0
    
    for i in range(lookback_window, len(market_data), decision_frequency):
        # Get market window
        window = market_data.iloc[max(0, i-lookback_window):i]
        current_time = market_data.index[i-1]
        
        # Make portfolio decision
        portfolio_decision = portfolio.make_portfolio_decision(window)
        decisions_made += 1
        
        # Execute trades
        executed_trades = portfolio.execute_portfolio_trades(portfolio_decision, window)
        total_trades += len(executed_trades)
        
        # Track results
        backtest_results['decisions'].append({
            'timestamp': current_time,
            'decision_num': decisions_made,
            'portfolio_decision': portfolio_decision,
            'executed_trades': executed_trades
        })
        
        backtest_results['trades'].extend(executed_trades)
        
        # Track equity
        backtest_results['equity_curve'].append({
            'timestamp': current_time,
            'portfolio_value': portfolio.current_capital,
            'total_return': (portfolio.current_capital - starting_capital) / starting_capital
        })
        
        # Track correlation
        correlation_analysis = portfolio_decision['correlation_analysis']
        backtest_results['correlation_tracking'].append(correlation_analysis['correlation_score'])
        
        # Progress indicator
        if decisions_made % 50 == 0:
            current_return = (portfolio.current_capital - starting_capital) / starting_capital
            print(f"   Progress: {decisions_made} decisions, {total_trades} trades, {current_return:+.2%} return")
    
    # Analyze backtest results
    print(f"\n📊 BACKTEST RESULTS ANALYSIS")
    print("=" * 35)
    
    # Portfolio performance
    final_portfolio_value = portfolio.current_capital
    total_return = (final_portfolio_value - starting_capital) / starting_capital
    total_trades_executed = len(backtest_results['trades'])
    
    print(f"🏛️ Portfolio Performance:")
    print(f"   Starting Capital: ${starting_capital:,}")
    print(f"   Final Portfolio Value: ${final_portfolio_value:,.2f}")
    print(f"   Total Return: {total_return:+.2%}")
    print(f"   Total Trades Executed: {total_trades_executed}")
    print(f"   Decisions Made: {decisions_made}")
    
    # Strategy breakdown
    momentum_trades = [t for t in backtest_results['trades'] if t.get('strategy') == 'MOMENTUM']
    mean_reversion_trades = [t for t in backtest_results['trades'] if t.get('strategy') == 'MEAN_REVERSION']
    
    print(f"\n📈 Momentum Strategy:")
    if momentum_trades:
        momentum_winners = [t for t in momentum_trades if t.get('win', False)]
        momentum_win_rate = len(momentum_winners) / len(momentum_trades)
        momentum_avg_pips = np.mean([t.get('pips_net', 0) for t in momentum_trades])
        
        print(f"   Trades: {len(momentum_trades)}")
        print(f"   Win Rate: {momentum_win_rate:.1%}")
        print(f"   Avg Pips per Trade: {momentum_avg_pips:+.1f}")
    else:
        print(f"   No momentum trades executed")
    
    print(f"\n📉 Mean Reversion Strategy:")
    if mean_reversion_trades:
        mr_winners = [t for t in mean_reversion_trades if t.get('win', False)]
        mr_win_rate = len(mr_winners) / len(mean_reversion_trades)
        mr_avg_pips = np.mean([t.get('pips_net', 0) for t in mean_reversion_trades])
        
        print(f"   Trades: {len(mean_reversion_trades)}")
        print(f"   Win Rate: {mr_win_rate:.1%}")
        print(f"   Avg Pips per Trade: {mr_avg_pips:+.1f}")
    else:
        print(f"   No mean reversion trades executed")
    
    # Correlation analysis
    if backtest_results['correlation_tracking']:
        avg_correlation = np.mean(backtest_results['correlation_tracking'])
        opposing_signals = sum(1 for c in backtest_results['correlation_tracking'] if c < 0)
        opposing_pct = opposing_signals / len(backtest_results['correlation_tracking'])
        
        print(f"\n🎭 Diversification Analysis:")
        print(f"   Average Strategy Correlation: {avg_correlation:+.2f}")
        print(f"   Opposing Signals: {opposing_signals} / {len(backtest_results['correlation_tracking'])} ({opposing_pct:.1%})")
        
        if avg_correlation < -0.2:
            print(f"   ✅ Excellent diversification (negative correlation)")
        elif avg_correlation < 0.1:
            print(f"   ✅ Good diversification (low correlation)")
        else:
            print(f"   ⚠️  Poor diversification (high correlation)")
    
    # Risk analysis
    if len(backtest_results['equity_curve']) > 1:
        equity_values = [eq['portfolio_value'] for eq in backtest_results['equity_curve']]
        returns = np.diff(equity_values) / equity_values[:-1]
        
        if len(returns) > 0:
            daily_volatility = np.std(returns) * np.sqrt(252 * 12 * 24)  # Annualized (5min bars)
            max_drawdown = calculate_max_drawdown(equity_values)
            
            if np.std(returns) > 0:
                sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252 * 12 * 24)
            else:
                sharpe_ratio = 0
            
            print(f"\n📊 Risk Metrics:")
            print(f"   Annualized Volatility: {daily_volatility:.1%}")
            print(f"   Maximum Drawdown: {max_drawdown:.2%}")
            print(f"   Sharpe Ratio: {sharpe_ratio:.2f}")
    
    # Monthly target analysis
    monthly_return_needed = 0.80  # 80% per month to reach $20K/month target
    current_monthly_rate = total_return * 12 / 1  # Annualized monthly rate
    
    print(f"\n🎯 Target Analysis ($20K/month goal):")
    print(f"   Monthly Return Needed: {monthly_return_needed:.1%}")
    print(f"   Current Monthly Rate: {current_monthly_rate:.1%}")
    
    if current_monthly_rate >= monthly_return_needed * 0.5:  # 50% of target
        print(f"   ✅ On track for target")
    elif current_monthly_rate > 0:
        print(f"   ⚠️  Below target but profitable")
    else:
        print(f"   ❌ Below target and unprofitable")
    
    print(f"\n🏛️ PROFESSIONAL ASSESSMENT:")
    
    # Overall assessment
    assessment_score = 0
    
    if total_return > 0:
        print(f"   ✅ Strategy is profitable: {total_return:+.2%}")
        assessment_score += 1
    else:
        print(f"   ❌ Strategy is unprofitable: {total_return:+.2%}")
    
    if total_trades_executed >= decisions_made * 0.3:  # 30%+ trade rate
        print(f"   ✅ Good signal generation: {total_trades_executed/decisions_made:.1%} trade rate")
        assessment_score += 1
    else:
        print(f"   ⚠️  Low signal generation: {total_trades_executed/decisions_made:.1%} trade rate")
    
    if avg_correlation < 0:
        print(f"   ✅ Strategies are uncorrelated: {avg_correlation:+.2f}")
        assessment_score += 1
    else:
        print(f"   ⚠️  Strategies may be correlated: {avg_correlation:+.2f}")
    
    if len(momentum_trades) > 0 and len(mean_reversion_trades) > 0:
        print(f"   ✅ Both strategies active")
        assessment_score += 1
    else:
        print(f"   ⚠️  Only one strategy active")
    
    print(f"\n   Overall Assessment: {assessment_score}/4 ({'✅ EXCELLENT' if assessment_score >= 3 else '⚠️  NEEDS WORK' if assessment_score >= 2 else '❌ POOR'})")
    
    print(f"\n✅ COMPREHENSIVE BACKTEST COMPLETE")
    
    return {
        'portfolio': portfolio,
        'backtest_results': backtest_results,
        'final_performance': {
            'total_return': total_return,
            'total_trades': total_trades_executed,
            'avg_correlation': avg_correlation if backtest_results['correlation_tracking'] else 0,
            'assessment_score': assessment_score
        }
    }

def calculate_max_drawdown(equity_curve: List[float]) -> float:
    """
    Calculate maximum drawdown from equity curve
    """
    peak = equity_curve[0]
    max_dd = 0.0
    
    for value in equity_curve:
        if value > peak:
            peak = value
        
        drawdown = (peak - value) / peak
        max_dd = max(max_dd, drawdown)
    
    return max_dd

if __name__ == "__main__":
    result = run_comprehensive_backtest()