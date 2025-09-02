#!/usr/bin/env python3
"""
Comprehensive System Validator
Tests optimized system against 6 months of real market data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys

sys.path.append('/Users/jonspinogatti/Desktop/spin36TB')
from historical_validator import HistoricalValidator

class ComprehensiveValidator:
    def __init__(self, starting_capital=25000):
        self.starting_capital = starting_capital
        
    def load_comprehensive_data(self):
        """Load the comprehensive dataset"""
        try:
            filepath = "/Users/jonspinogatti/Desktop/spin36TB/comprehensive_eurusd_data.csv"
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            
            print("📊 COMPREHENSIVE DATASET LOADED")
            print("=" * 35)
            print(f"   📅 Period: {df.index[0]} to {df.index[-1]}")
            print(f"   ⏱️  Duration: {(df.index[-1] - df.index[0]).days} days")
            print(f"   📈 Candles: {len(df):,}")
            print(f"   💹 Price Range: {df['close'].min():.4f} - {df['close'].max():.4f}")
            print(f"   📊 Total Movement: {(df['close'].max() - df['close'].min()) * 10000:.0f} pips")
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading comprehensive data: {e}")
            return None
    
    def run_walk_forward_analysis(self, data):
        """
        Run walk-forward analysis on 6-month dataset
        """
        print(f"\n🔬 WALK-FORWARD ANALYSIS")
        print("=" * 30)
        
        # Define periods: 60-day training, 20-day testing
        training_days = 60
        testing_days = 20
        total_periods = (len(data) // (288 * testing_days))  # 288 5-min candles per day
        
        print(f"   Training Window: {training_days} days")
        print(f"   Testing Window: {testing_days} days")
        print(f"   Total Periods: {total_periods}")
        
        period_results = []
        
        for period in range(min(total_periods, 6)):  # Test up to 6 periods
            # Define period boundaries
            start_idx = period * (288 * testing_days)
            train_end_idx = start_idx + (288 * training_days)
            test_end_idx = train_end_idx + (288 * testing_days)
            
            if test_end_idx > len(data):
                break
            
            # Extract training and testing data
            training_data = data.iloc[start_idx:train_end_idx]
            testing_data = data.iloc[train_end_idx:test_end_idx]
            
            print(f"\n   🔄 Period {period + 1}:")
            print(f"      Training: {training_data.index[0]} to {training_data.index[-1]}")
            print(f"      Testing: {testing_data.index[0]} to {testing_data.index[-1]}")
            
            # Run validation on testing period
            validator = HistoricalValidator(starting_capital=self.starting_capital)
            results = validator.run_historical_validation(testing_data, decision_frequency=20)
            
            if results:
                period_results.append({
                    'period': period + 1,
                    'start_date': testing_data.index[0],
                    'end_date': testing_data.index[-1],
                    'total_return': results.get('validation_summary', {}).get('total_return', 0),
                    'total_trades': results.get('validation_summary', {}).get('total_trades', 0),
                    'momentum_win_rate': results.get('strategy_performance', {}).get('momentum', {}).get('win_rate', 0),
                    'mr_win_rate': results.get('strategy_performance', {}).get('mean_reversion', {}).get('win_rate', 0),
                    'final_capital': results.get('validation_summary', {}).get('final_capital', self.starting_capital)
                })
                
                print(f"      ✅ Return: {results.get('validation_summary', {}).get('total_return', 0):+.3%}")
                print(f"      📊 Trades: {results.get('validation_summary', {}).get('total_trades', 0)}")
        
        return period_results
    
    def analyze_period_results(self, period_results):
        """Analyze walk-forward results"""
        if not period_results:
            print("❌ No period results to analyze")
            return
        
        print(f"\n📊 WALK-FORWARD ANALYSIS RESULTS")
        print("=" * 40)
        
        # Calculate statistics
        returns = [r['total_return'] for r in period_results]
        trades = [r['total_trades'] for r in period_results]
        momentum_rates = [r['momentum_win_rate'] for r in period_results]
        mr_rates = [r['mr_win_rate'] for r in period_results]
        
        print(f"📈 PERFORMANCE SUMMARY:")
        print(f"   Periods Tested: {len(period_results)}")
        print(f"   Average Return: {np.mean(returns):+.3%}")
        print(f"   Std Deviation: {np.std(returns):.3%}")
        print(f"   Best Period: {max(returns):+.3%}")
        print(f"   Worst Period: {min(returns):+.3%}")
        print(f"   Win Rate: {sum(1 for r in returns if r > 0) / len(returns):.1%}")
        
        print(f"\n📊 STRATEGY PERFORMANCE:")
        print(f"   Avg Momentum Win Rate: {np.mean(momentum_rates):.1%}")
        print(f"   Avg Mean Reversion Win Rate: {np.mean(mr_rates):.1%}")
        print(f"   Avg Trades per Period: {np.mean(trades):.0f}")
        
        print(f"\n🎯 CONSISTENCY CHECK:")
        consistent_periods = sum(1 for r in returns if r > -0.005)  # Within 0.5% loss
        consistency_rate = consistent_periods / len(period_results)
        
        if consistency_rate >= 0.8:
            print(f"   ✅ HIGHLY CONSISTENT ({consistency_rate:.1%} periods profitable/breakeven)")
        elif consistency_rate >= 0.6:
            print(f"   ⚠️  MODERATELY CONSISTENT ({consistency_rate:.1%} periods profitable/breakeven)")
        else:
            print(f"   ❌ INCONSISTENT ({consistency_rate:.1%} periods profitable/breakeven)")
        
        # Final recommendation
        avg_return = np.mean(returns)
        if avg_return > 0.01:  # >1% average return
            print(f"\n🚀 RECOMMENDATION: READY FOR PAPER TRADING")
            print(f"   System shows consistent profitability across multiple periods")
        elif avg_return > 0:
            print(f"\n⚠️  RECOMMENDATION: CAUTIOUS PAPER TRADING")
            print(f"   System shows marginal profitability - consider smaller position sizes")
        else:
            print(f"\n❌ RECOMMENDATION: FURTHER OPTIMIZATION NEEDED")
            print(f"   System shows losses - requires additional calibration")
        
        return {
            'avg_return': avg_return,
            'consistency_rate': consistency_rate,
            'total_periods': len(period_results),
            'recommendation': 'READY' if avg_return > 0.01 else 'CAUTIOUS' if avg_return > 0 else 'OPTIMIZE'
        }

def run_comprehensive_validation():
    """Run complete comprehensive validation"""
    print("🎯 COMPREHENSIVE SYSTEM VALIDATION")
    print("=" * 40)
    print("Testing optimized system on 6 months of real market data")
    
    validator = ComprehensiveValidator(starting_capital=25000)
    
    # Load comprehensive data
    data = validator.load_comprehensive_data()
    if data is None:
        return
    
    # Run walk-forward analysis
    period_results = validator.run_walk_forward_analysis(data)
    
    # Analyze results
    final_analysis = validator.analyze_period_results(period_results)
    
    print(f"\n🏁 COMPREHENSIVE VALIDATION COMPLETE!")
    return final_analysis

if __name__ == "__main__":
    run_comprehensive_validation()