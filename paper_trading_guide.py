#!/usr/bin/env python3
"""
Paper Trading Guide: Step-by-Step Setup
Simple, working guide to start paper trading immediately
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import sys

sys.path.append('/Users/jonspinogatti/Desktop/spin36TB')

def create_simple_market_data():
    """
    Create simple market data that works with our system
    """
    print("📊 Creating market data...")
    
    # Create 5-minute intervals for the last 8 hours
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=8)
    
    dates = []
    current_time = start_time
    while current_time <= end_time:
        # Skip weekends
        if current_time.weekday() < 5:
            dates.append(current_time)
        current_time += timedelta(minutes=5)
    
    # Generate realistic EURUSD prices
    np.random.seed(42)
    base_price = 1.0850
    prices = [base_price]
    
    for i in range(len(dates) - 1):
        change = np.random.normal(0, 0.0003)
        new_price = prices[-1] * (1 + change)
        new_price = max(1.0700, min(1.1000, new_price))
        prices.append(new_price)
    
    # Create DataFrame
    data = []
    for i in range(1, len(dates)):
        data.append({
            'open': prices[i-1],
            'high': max(prices[i-1], prices[i]) + abs(np.random.normal(0, 0.0001)),
            'low': min(prices[i-1], prices[i]) - abs(np.random.normal(0, 0.0001)),
            'close': prices[i],
            'volume': np.random.uniform(800, 1200)
        })
    
    df = pd.DataFrame(data, index=dates[1:])
    
    print(f"   ✅ Generated {len(df)} candles")
    print(f"   📊 Price range: {df['close'].min():.4f} - {df['close'].max():.4f}")
    print(f"   💱 Current EURUSD: {df['close'].iloc[-1]:.4f}")
    
    return df

def paper_trading_step_by_step():
    """
    Step-by-step paper trading guide
    """
    print("📄 PAPER TRADING: STEP-BY-STEP GUIDE")
    print("=" * 45)
    
    print("\n🚀 STEP 1: Initialize Systems")
    print("─" * 30)
    
    # Import and initialize
    try:
        from dual_strategy_portfolio import DualStrategyPortfolioManager
        
        # Initialize portfolio with $25,000
        portfolio = DualStrategyPortfolioManager(
            starting_capital=25000,
            allocation_momentum=0.6
        )
        print("✅ Portfolio manager initialized: $25,000 starting capital")
        
    except Exception as e:
        print(f"❌ Error initializing portfolio: {e}")
        return None
    
    print("\n📊 STEP 2: Generate Market Data")
    print("─" * 35)
    
    # Get market data
    market_data = create_simple_market_data()
    
    print("\n🎯 STEP 3: Make Trading Decision")
    print("─" * 35)
    
    try:
        # Make decision
        decision = portfolio.make_portfolio_decision(market_data)
        
        print(f"   Decision Time: {datetime.now()}")
        print(f"   Portfolio Actions: {len(decision['portfolio_actions'])}")
        print(f"   Signal Relationship: {decision['correlation_analysis']['relationship']}")
        
        # Show each strategy's decision
        momentum_decision = decision['momentum_decision']
        mean_reversion_decision = decision['mean_reversion_decision']
        
        print(f"\n   📈 Momentum System:")
        print(f"      Signal: {momentum_decision['signal']}")
        print(f"      Regime: {momentum_decision.get('regime', 'N/A')}")
        
        print(f"\n   📉 Mean Reversion System:")
        print(f"      Action: {mean_reversion_decision['action']}")
        if mean_reversion_decision['action'] == 'TRADE':
            print(f"      Direction: {mean_reversion_decision['direction']}")
            print(f"      Confidence: {mean_reversion_decision['confidence']:.1%}")
        
    except Exception as e:
        print(f"❌ Error making decision: {e}")
        return None
    
    print("\n💼 STEP 4: Execute Paper Trades")
    print("─" * 35)
    
    if len(decision['portfolio_actions']) > 0:
        try:
            # Execute trades
            trades = portfolio.execute_portfolio_trades(decision, market_data)
            
            print(f"   ✅ Executed {len(trades)} trades:")
            
            for trade in trades:
                strategy = trade.get('strategy', 'UNKNOWN')
                direction = trade.get('direction', 'N/A')
                pips = trade.get('pips_net', 0)
                exit_reason = trade.get('exit_reason', 'N/A')
                
                print(f"      {strategy} {direction}: {pips:+.1f} pips ({exit_reason})")
            
        except Exception as e:
            print(f"❌ Error executing trades: {e}")
            trades = []
    else:
        print("   ⏸️  No trades to execute (HOLD signals)")
        trades = []
    
    print("\n📊 STEP 5: Check Performance")
    print("─" * 30)
    
    try:
        # Get performance
        performance = portfolio.calculate_portfolio_performance()
        
        current_capital = portfolio.current_capital
        starting_capital = portfolio.starting_capital
        total_return = (current_capital - starting_capital) / starting_capital
        
        print(f"   💰 Portfolio Value: ${current_capital:,.2f}")
        print(f"   📈 Total Return: {total_return:+.3%}")
        print(f"   🔢 Total Trades: {performance.get('portfolio', {}).get('total_trades', 0)}")
        
        if not performance.get('no_trades'):
            momentum_perf = performance.get('momentum_strategy', {})
            mr_perf = performance.get('mean_reversion_strategy', {})
            
            if not momentum_perf.get('no_trades'):
                print(f"   📈 Momentum: {momentum_perf['win_rate']:.1%} win rate")
            
            if not mr_perf.get('no_trades'):
                print(f"   📉 Mean Reversion: {mr_perf['win_rate']:.1%} win rate")
        
    except Exception as e:
        print(f"❌ Error checking performance: {e}")
    
    print("\n✅ PAPER TRADING CYCLE COMPLETE!")
    
    return {
        'portfolio': portfolio,
        'decision': decision,
        'trades': trades,
        'market_data': market_data
    }

def show_how_to_continue():
    """
    Show user how to continue paper trading
    """
    print("\n🔄 HOW TO CONTINUE PAPER TRADING:")
    print("=" * 40)
    
    print("1️⃣ Manual Mode (Recommended for learning):")
    print("   • Run this script again for each decision cycle")
    print("   • Study each decision before executing")
    print("   • Understand why the system made each choice")
    
    print("\n2️⃣ Automated Mode:")
    print("   • Set up a loop to run every 5-15 minutes")  
    print("   • Let system trade automatically")
    print("   • Monitor performance periodically")
    
    print("\n3️⃣ Live Data Connection:")
    print("   • Connect to MT5, IB, or OANDA API")
    print("   • Replace generate_market_data() with live feed")
    print("   • Add real-time execution logic")
    
    print("\n📋 Next Steps:")
    print("   • Run several manual cycles to understand the system")
    print("   • Track results in a spreadsheet")
    print("   • Once comfortable, consider automation")
    print("   • Eventually connect to live data feed")

def create_trading_log():
    """
    Create a simple trading log template
    """
    print("\n📝 CREATING TRADING LOG TEMPLATE...")
    
    log_data = {
        'session_start': datetime.now().isoformat(),
        'starting_capital': 25000,
        'trades': [],
        'daily_summaries': [],
        'learning_notes': []
    }
    
    filename = f"trading_log_{datetime.now().strftime('%Y%m%d')}.json"
    filepath = f"/Users/jonspinogatti/Desktop/spin36TB/{filename}"
    
    try:
        with open(filepath, 'w') as f:
            json.dump(log_data, f, indent=2)
        print(f"✅ Trading log created: {filename}")
        print(f"   Use this to track your paper trading progress")
    except Exception as e:
        print(f"❌ Error creating log: {e}")

if __name__ == "__main__":
    # Run the complete paper trading cycle
    result = paper_trading_step_by_step()
    
    if result:
        show_how_to_continue()
        create_trading_log()
        
        print(f"\n🎉 PAPER TRADING SETUP COMPLETE!")
        print(f"   Ready to start your journey to $20K/month!")
    else:
        print(f"\n❌ Setup failed. Please check the error messages above.")