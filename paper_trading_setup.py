#!/usr/bin/env python3
"""
Paper Trading Setup: Complete Guide to Start Paper Trading
Shows exactly how to start trading with real market data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import warnings
from typing import Dict, List, Optional
import sys
import time

sys.path.append('/Users/jonspinogatti/Desktop/spin36TB')
from production_dual_strategy_system import ProductionDualStrategySystem
from adaptive_learning_system import AdaptiveLearningSystem

warnings.filterwarnings('ignore')

class PaperTradingSystem:
    """
    Complete paper trading system with real-time simulation
    """
    
    def __init__(self, starting_capital: float = 25000):
        # Initialize core systems
        self.production_system = ProductionDualStrategySystem()
        self.learning_system = AdaptiveLearningSystem()
        
        # Paper trading configuration
        self.paper_config = {
            'starting_capital': starting_capital,
            'update_frequency': 300,  # 5 minutes (300 seconds)
            'market_hours_only': True,  # Only trade during market hours
            'max_daily_trades': 20,
            'data_source': 'simulated',  # 'live' when connected to real data
            'auto_trading': False,  # Start manual, can enable auto later
        }
        
        # Paper trading state
        self.paper_state = {
            'is_running': False,
            'trades_today': 0,
            'session_start': None,
            'last_decision_time': None,
            'performance_log': [],
            'daily_summary': {},
        }
        
        # Performance tracking
        self.performance_tracker = {
            'equity_curve': [],
            'trade_log': [],
            'daily_stats': {},
            'monthly_stats': {},
        }
        
        print("📄 PAPER TRADING SYSTEM INITIALIZED")
        print("=" * 45)
        print(f"💰 Starting Capital: ${starting_capital:,}")
        print(f"🕐 Update Frequency: {self.paper_config['update_frequency']} seconds")
        print(f"📊 Max Daily Trades: {self.paper_config['max_daily_trades']}")
        print(f"🎯 Ready for paper trading!")
    
    def start_paper_trading(self):
        """
        Start paper trading session
        """
        if self.paper_state['is_running']:
            print("❌ Paper trading already running!")
            return
        
        print("\n🚀 STARTING PAPER TRADING SESSION")
        print("=" * 40)
        
        self.paper_state['is_running'] = True
        self.paper_state['session_start'] = datetime.now()
        self.paper_state['trades_today'] = 0
        
        # Reset daily session in production system
        self.production_system.reset_daily_session()
        
        print("✅ Paper trading session started!")
        print(f"   Session start: {self.paper_state['session_start']}")
        print(f"   Status: ACTIVE")
        print(f"   Mode: {'AUTO' if self.paper_config['auto_trading'] else 'MANUAL'}")
        
        # Start main trading loop
        if self.paper_config['auto_trading']:
            self.run_auto_trading_loop()
        else:
            print("\n📋 Manual Mode Instructions:")
            print("   • Use make_trading_decision() to generate signals")
            print("   • Use execute_paper_trade() to execute trades")
            print("   • Use get_performance_summary() to check results")
            print("   • Use stop_paper_trading() to end session")
    
    def stop_paper_trading(self):
        """
        Stop paper trading session
        """
        if not self.paper_state['is_running']:
            print("❌ Paper trading not running!")
            return
        
        print("\n🛑 STOPPING PAPER TRADING SESSION")
        print("=" * 40)
        
        self.paper_state['is_running'] = False
        session_end = datetime.now()
        session_duration = session_end - self.paper_state['session_start']
        
        # Generate session summary
        session_summary = self.generate_session_summary()
        
        print("✅ Paper trading session stopped!")
        print(f"   Session duration: {session_duration}")
        print(f"   Total trades: {self.paper_state['trades_today']}")
        print(f"   Final P&L: {session_summary['session_pnl']:+.2%}")
        
        # Save session data
        self.save_paper_trading_session()
        
        return session_summary
    
    def generate_market_data(self, periods: int = 100) -> pd.DataFrame:
        """
        Generate realistic market data for paper trading
        In production, this would connect to live data feed
        """
        print(f"📊 Generating market data ({periods} periods)...")
        
        # Create realistic EURUSD data
        dates = pd.date_range(
            start=datetime.now() - timedelta(hours=8),
            end=datetime.now(),
            freq='5min'
        )
        
        # Remove weekends
        dates = dates[dates.weekday < 5]
        dates = dates[-periods:] if len(dates) > periods else dates
        
        np.random.seed(int(datetime.now().timestamp()) % 1000)  # Vary with time
        
        # Start with current approximate EURUSD price
        base_price = 1.0850
        prices = [base_price]
        
        # Generate realistic price movements
        for i in range(len(dates) - 1):
            # Realistic EURUSD movements
            change = np.random.normal(0, 0.0003)  # ~3 pip standard deviation
            
            # Add some trending behavior
            if i % 20 == 0:  # Change trend every ~100 minutes
                trend_strength = np.random.uniform(-0.0001, 0.0001)
            else:
                change += trend_strength
            
            new_price = prices[-1] * (1 + change)
            new_price = max(1.0600, min(1.1200, new_price))  # Keep in reasonable range
            prices.append(new_price)
        
        # Create OHLCV data
        market_data = []
        for i in range(1, len(dates)):  # Start from 1 to have previous price
            timestamp = dates[i]
            close_price = prices[i]
            open_price = prices[i-1]
            
            high = max(open_price, close_price) + abs(np.random.normal(0, 0.0001))
            low = min(open_price, close_price) - abs(np.random.normal(0, 0.0001))
            volume = np.random.uniform(800, 1200)
            
            market_data.append({
                'open': open_price,
                'high': high,
                'low': low,
                'close': close_price,
                'volume': volume
            })
        
        # Create DataFrame with dates as index
        df = pd.DataFrame(market_data, index=dates[1:])
        
        print(f"   Generated {len(df)} candles")
        print(f"   Price range: {df['close'].min():.4f} - {df['close'].max():.4f}")
        print(f"   Latest price: {df['close'].iloc[-1]:.4f}")
        
        return df
    
    def make_trading_decision(self, market_data: pd.DataFrame = None) -> Dict:
        """
        Make a trading decision using current market data
        """
        if not self.paper_state['is_running']:
            print("❌ Start paper trading first!")
            return {}
        
        # Generate market data if not provided
        if market_data is None:
            market_data = self.generate_market_data()
        
        print(f"\n🎯 MAKING TRADING DECISION")
        print(f"   Time: {datetime.now()}")
        print(f"   Latest EURUSD: {market_data['close'].iloc[-1]:.4f}")
        
        # Make production decision
        decision = self.production_system.make_production_trading_decision(market_data)
        
        self.paper_state['last_decision_time'] = datetime.now()
        
        # Display decision
        print(f"   Decision: {decision['decision']}")
        
        if decision['decision'] == 'TRADE':
            actions = decision['portfolio_actions']
            print(f"   Portfolio Actions: {len(actions)}")
            
            for action in actions:
                strategy = action['strategy']
                trade_decision = action['decision']
                direction = trade_decision.get('direction', 'N/A')
                position_size = trade_decision.get('position_size', 0)
                confidence = trade_decision.get('confidence', 0)
                
                print(f"      {strategy}: {direction} @ {position_size:.3f} ({confidence:.1%} confidence)")
        
        else:
            print(f"   Reason: {decision.get('reason', 'No signal')}")
        
        return decision
    
    def execute_paper_trade(self, decision: Dict, market_data: pd.DataFrame) -> List[Dict]:
        """
        Execute paper trade based on decision
        """
        if not self.paper_state['is_running']:
            print("❌ Start paper trading first!")
            return []
        
        if decision['decision'] != 'TRADE':
            print("❌ No trade to execute!")
            return []
        
        print(f"\n💼 EXECUTING PAPER TRADES")
        
        # Execute through production system
        trades = self.production_system.execute_production_trades(decision, market_data)
        
        # Record with learning system
        for trade in trades:
            market_context = {
                'regime': trade.get('regime', 'UNKNOWN'),
                'volatility': (market_data['high'].tail(10) - market_data['low'].tail(10)).mean(),
                'signal_strength': decision.get('signal_strength', 1.0),
                'correlation_signal': decision.get('correlation_analysis', {}).get('correlation_score', 0)
            }
            
            self.learning_system.record_trade_result(trade, market_context)
        
        # Update paper trading state
        self.paper_state['trades_today'] += len(trades)
        
        # Log performance
        current_capital = self.production_system.portfolio_manager.current_capital
        equity_point = {
            'timestamp': datetime.now(),
            'capital': current_capital,
            'trades_count': self.paper_state['trades_today']
        }
        self.performance_tracker['equity_curve'].append(equity_point)
        self.performance_tracker['trade_log'].extend(trades)
        
        # Display results
        for trade in trades:
            print(f"   {trade['strategy']} {trade['direction']}: {trade['pips_net']:+.1f} pips")
        
        print(f"   Portfolio Value: ${current_capital:,.2f}")
        print(f"   Trades Today: {self.paper_state['trades_today']}")
        
        return trades
    
    def run_single_decision_cycle(self) -> Dict:
        """
        Run one complete decision and execution cycle
        """
        # Generate current market data
        market_data = self.generate_market_data()
        
        # Make decision
        decision = self.make_trading_decision(market_data)
        
        # Execute if signal present
        trades = []
        if decision['decision'] == 'TRADE':
            trades = self.execute_paper_trade(decision, market_data)
        
        return {
            'decision': decision,
            'trades': trades,
            'market_data': market_data,
            'timestamp': datetime.now()
        }
    
    def run_auto_trading_loop(self, cycles: int = 10):
        """
        Run automated paper trading for specified cycles
        """
        print(f"\n🔄 RUNNING AUTO TRADING ({cycles} cycles)")
        print("=" * 40)
        
        for cycle in range(cycles):
            if not self.paper_state['is_running']:
                break
            
            print(f"\n📊 Cycle {cycle + 1}/{cycles}")
            print("─" * 25)
            
            try:
                result = self.run_single_decision_cycle()
                
                # Brief pause between cycles
                time.sleep(2)
                
            except KeyboardInterrupt:
                print("\n⏸️  Trading interrupted by user")
                break
            except Exception as e:
                print(f"❌ Error in trading cycle: {e}")
                continue
        
        print(f"\n✅ Auto trading complete!")
        return self.get_performance_summary()
    
    def get_performance_summary(self) -> Dict:
        """
        Get comprehensive performance summary
        """
        if not self.paper_state['session_start']:
            return {'error': 'No session started'}
        
        current_capital = self.production_system.portfolio_manager.current_capital
        starting_capital = self.production_system.portfolio_manager.starting_capital
        
        session_return = (current_capital - starting_capital) / starting_capital
        
        trades = self.performance_tracker['trade_log']
        winning_trades = [t for t in trades if t.get('pips_net', 0) > 0]
        
        summary = {
            'session_start': self.paper_state['session_start'],
            'session_duration': datetime.now() - self.paper_state['session_start'],
            'starting_capital': starting_capital,
            'current_capital': current_capital,
            'session_return': session_return,
            'session_pnl': session_return,
            'total_trades': len(trades),
            'winning_trades': len(winning_trades),
            'win_rate': len(winning_trades) / len(trades) if trades else 0,
            'avg_pips': np.mean([t.get('pips_net', 0) for t in trades]) if trades else 0,
            'learning_status': self.learning_system.get_learning_status()
        }
        
        return summary
    
    def generate_session_summary(self) -> Dict:
        """
        Generate detailed session summary
        """
        summary = self.get_performance_summary()
        
        print(f"\n📊 PAPER TRADING SESSION SUMMARY")
        print("=" * 40)
        print(f"Session Duration: {summary['session_duration']}")
        print(f"Starting Capital: ${summary['starting_capital']:,}")
        print(f"Ending Capital: ${summary['current_capital']:,.2f}")
        print(f"Total Return: {summary['session_return']:+.2%}")
        print(f"Total Trades: {summary['total_trades']}")
        print(f"Win Rate: {summary['win_rate']:.1%}")
        print(f"Average Pips: {summary['avg_pips']:+.1f}")
        
        # Learning summary
        learning = summary['learning_status']
        print(f"\n🧠 Learning Progress:")
        print(f"   Trades Recorded: {learning['trades_recorded']}")
        print(f"   Learning Iterations: {learning['learning_iterations']}")
        print(f"   Next Update: {learning.get('next_update_in_trades', 0)} trades")
        
        return summary
    
    def save_paper_trading_session(self):
        """
        Save paper trading session data
        """
        session_data = {
            'session_summary': self.get_performance_summary(),
            'performance_tracker': self.performance_tracker,
            'paper_state': self.paper_state,
            'timestamp': datetime.now().isoformat()
        }
        
        filename = f"paper_trading_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = f"/Users/jonspinogatti/Desktop/spin36TB/{filename}"
        
        try:
            with open(filepath, 'w') as f:
                json.dump(session_data, f, indent=2, default=str)
            print(f"💾 Session saved to {filename}")
        except Exception as e:
            print(f"❌ Error saving session: {e}")

def demo_paper_trading():
    """
    Demonstrate paper trading setup
    """
    print("\n📄 PAPER TRADING DEMO")
    print("=" * 30)
    
    # Initialize paper trading
    paper_trader = PaperTradingSystem(starting_capital=25000)
    
    # Start session
    paper_trader.start_paper_trading()
    
    # Run a few decision cycles
    print(f"\n🎯 Running 3 decision cycles...")
    
    for i in range(3):
        print(f"\n── Cycle {i+1} ──")
        result = paper_trader.run_single_decision_cycle()
        time.sleep(1)  # Brief pause
    
    # Show results
    summary = paper_trader.generate_session_summary()
    
    # Stop session
    paper_trader.stop_paper_trading()
    
    return paper_trader, summary

if __name__ == "__main__":
    paper_trader, summary = demo_paper_trading()