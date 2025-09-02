#!/usr/bin/env python3
"""
Live Paper Trading System
Trades the optimized Spin36TB system on live OANDA data
"""

import pandas as pd
import numpy as np
import requests
import time
import json
from datetime import datetime, timedelta
import logging
import sys

sys.path.append('/Users/jonspinogatti/Desktop/spin36TB')
from dual_strategy_portfolio import DualStrategyPortfolioManager
from forex_market_scheduler import ForexMarketScheduler
from adaptive_learning_system import AdaptiveLearningSystem

class LivePaperTrader:
    def __init__(self, starting_capital=25000):
        # OANDA Configuration
        self.api_token = "fdf72f4b5166602e39e001d1c1cb38b0-524c6662baa3187260b34c006eeab1fd"
        self.account_id = "101-001-31365224-001"
        self.api_url = 'https://api-fxpractice.oanda.com'
        self.headers = {
            'Authorization': f'Bearer {self.api_token}',
            'Content-Type': 'application/json'
        }
        
        # Trading System
        self.portfolio = DualStrategyPortfolioManager(
            starting_capital=starting_capital,
            allocation_momentum=0.6
        )
        self.starting_capital = starting_capital
        self.current_capital = starting_capital
        
        # Enhanced Systems
        self.market_scheduler = ForexMarketScheduler()
        self.learning_system = AdaptiveLearningSystem()
        
        # Paper Trading State
        self.active_trades = []
        self.trade_history = []
        self.live_data_buffer = []
        self.last_decision_time = None
        self.decision_interval = 5  # 5 minutes between decisions
        
        # Performance tracking
        self.session_stats = {
            'start_time': datetime.now(),
            'total_decisions': 0,
            'total_trades': 0,
            'total_return': 0.0,
            'momentum_trades': 0,
            'mr_trades': 0,
            'momentum_wins': 0,
            'mr_wins': 0
        }
        
        # Logging setup
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('/Users/jonspinogatti/Desktop/spin36TB/live_trading.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        print("🚀 LIVE PAPER TRADER INITIALIZED")
        print("=" * 40)
        print(f"💰 Starting Capital: ${starting_capital:,}")
        print(f"📊 OANDA Practice Account: {self.account_id}")
        print(f"🎯 Ready for live paper trading!")
    
    def get_live_market_data(self, candles_needed=100):
        """
        Get live market data from OANDA
        """
        try:
            url = f"{self.api_url}/v3/instruments/EUR_USD/candles"
            params = {
                'granularity': 'M5',  # 5-minute candles
                'count': candles_needed,
                'price': 'MBA'
            }
            
            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                candles = data['candles']
                
                market_data = []
                for candle in candles:
                    if candle['complete']:
                        market_data.append({
                            'timestamp': pd.to_datetime(candle['time']),
                            'open': float(candle['mid']['o']),
                            'high': float(candle['mid']['h']),
                            'low': float(candle['mid']['l']),
                            'close': float(candle['mid']['c']),
                            'volume': int(candle['volume'])
                        })
                
                df = pd.DataFrame(market_data)
                df.set_index('timestamp', inplace=True)
                return df
                
            else:
                self.logger.error(f"Failed to get market data: {response.status_code}")
                return None
                
        except Exception as e:
            self.logger.error(f"Market data error: {e}")
            return None
    
    def make_trading_decision(self, market_data):
        """
        Make trading decision using the optimized portfolio system
        """
        try:
            # Make portfolio decision
            decision = self.portfolio.make_portfolio_decision(market_data)
            self.session_stats['total_decisions'] += 1
            
            if not decision or not decision.get('portfolio_actions'):
                self.logger.info("No trading signals generated")
                return None
            
            # Log decision
            self.logger.info(f"🎯 Portfolio Decision: {len(decision['portfolio_actions'])} actions")
            
            return decision
            
        except Exception as e:
            self.logger.error(f"Decision error: {e}")
            return None
    
    def execute_paper_trades(self, decision, current_price):
        """
        Execute paper trades (virtual trades, no real money)
        """
        if not decision or not decision.get('portfolio_actions'):
            return []
        
        executed_trades = []
        
        for action in decision['portfolio_actions']:
            try:
                # Create paper trade
                trade = {
                    'trade_id': f"PAPER_{int(time.time())}_{len(self.active_trades)}",
                    'strategy': action.get('strategy', 'UNKNOWN'),
                    'direction': action.get('direction', 'HOLD'),
                    'entry_price': current_price,
                    'position_size': action.get('position_size', 0),
                    'confidence': action.get('confidence', 0),
                    'entry_time': datetime.now(),
                    'status': 'ACTIVE',
                    'allocation': action.get('allocation', 0),
                    'expected_pips': action.get('expected_pips', 0)
                }
                
                if trade['direction'] != 'HOLD' and trade['position_size'] > 0:
                    self.active_trades.append(trade)
                    executed_trades.append(trade)
                    
                    self.session_stats['total_trades'] += 1
                    if trade['strategy'] == 'MOMENTUM':
                        self.session_stats['momentum_trades'] += 1
                    elif trade['strategy'] == 'MEAN_REVERSION':
                        self.session_stats['mr_trades'] += 1
                    
                    self.logger.info(f"📈 PAPER TRADE: {trade['strategy']} {trade['direction']} "
                                   f"{trade['position_size']:.3f}% @ {current_price:.4f}")
                
            except Exception as e:
                self.logger.error(f"Trade execution error: {e}")
        
        return executed_trades
    
    def update_active_trades(self, current_price):
        """
        Update active paper trades and close completed ones
        """
        trades_to_close = []
        
        for trade in self.active_trades[:]:  # Copy list for safe iteration
            try:
                # Calculate current P&L
                if trade['direction'] == 'UP':
                    pips = (current_price - trade['entry_price']) * 10000
                else:
                    pips = (trade['entry_price'] - current_price) * 10000
                
                # Check exit conditions
                should_close = False
                exit_reason = ""
                
                # Time-based exit (2 hours max)
                time_elapsed = (datetime.now() - trade['entry_time']).total_seconds() / 3600
                if time_elapsed > 2:
                    should_close = True
                    exit_reason = "Time limit (2 hours)"
                
                # Profit target (based on strategy)
                elif trade['strategy'] == 'MOMENTUM' and pips > 8:
                    should_close = True
                    exit_reason = f"Profit target (+{pips:.1f} pips)"
                
                elif trade['strategy'] == 'MEAN_REVERSION' and pips > 5:
                    should_close = True
                    exit_reason = f"Profit target (+{pips:.1f} pips)"
                
                # Stop loss
                elif pips < -10:
                    should_close = True
                    exit_reason = f"Stop loss ({pips:.1f} pips)"
                
                if should_close:
                    # Close trade
                    trade['exit_price'] = current_price
                    trade['exit_time'] = datetime.now()
                    trade['pips'] = pips
                    trade['exit_reason'] = exit_reason
                    trade['status'] = 'CLOSED'
                    
                    # Calculate return
                    leverage = 2.0  # Conservative 2x leverage
                    trade_return = (pips * 0.0001) * trade['position_size'] * leverage * trade['allocation']
                    trade['return_pct'] = trade_return
                    
                    # Update capital
                    dollar_return = self.current_capital * trade_return
                    self.current_capital += dollar_return
                    self.session_stats['total_return'] += trade_return
                    
                    # Track wins
                    if pips > 0:
                        if trade['strategy'] == 'MOMENTUM':
                            self.session_stats['momentum_wins'] += 1
                        elif trade['strategy'] == 'MEAN_REVERSION':
                            self.session_stats['mr_wins'] += 1
                    
                    # Move to history
                    self.trade_history.append(trade)
                    trades_to_close.append(trade)
                    
                    self.logger.info(f"🔒 TRADE CLOSED: {trade['strategy']} {exit_reason} "
                                   f"({pips:+.1f} pips, {trade_return:+.4%} return)")
            
            except Exception as e:
                self.logger.error(f"Trade update error: {e}")
        
        # Remove closed trades from active list
        for trade in trades_to_close:
            if trade in self.active_trades:
                self.active_trades.remove(trade)
    
    def display_live_status(self, current_price):
        """
        Display current trading status
        """
        print(f"\n📊 LIVE TRADING STATUS - {datetime.now().strftime('%H:%M:%S')}")
        print("=" * 50)
        print(f"💰 Current Capital: ${self.current_capital:,.2f} ({self.session_stats['total_return']:+.3%})")
        print(f"📈 Current EURUSD: {current_price:.4f}")
        print(f"🎯 Active Trades: {len(self.active_trades)}")
        print(f"📊 Decisions Made: {self.session_stats['total_decisions']}")
        print(f"💼 Total Trades: {self.session_stats['total_trades']}")
        
        # Strategy performance
        momentum_wr = (self.session_stats['momentum_wins'] / max(1, self.session_stats['momentum_trades'])) * 100
        mr_wr = (self.session_stats['mr_wins'] / max(1, self.session_stats['mr_trades'])) * 100
        
        print(f"\n🔥 MOMENTUM: {self.session_stats['momentum_trades']} trades, {momentum_wr:.1f}% win rate")
        print(f"⚡ MEAN REV: {self.session_stats['mr_trades']} trades, {mr_wr:.1f}% win rate")
        
        # Show active trades
        if self.active_trades:
            print(f"\n🎪 ACTIVE TRADES:")
            for trade in self.active_trades:
                if trade['direction'] == 'UP':
                    pips = (current_price - trade['entry_price']) * 10000
                else:
                    pips = (trade['entry_price'] - current_price) * 10000
                
                duration = (datetime.now() - trade['entry_time']).total_seconds() / 60
                print(f"   {trade['strategy'][:3]} {trade['direction']} @ {trade['entry_price']:.4f} "
                      f"({pips:+.1f} pips, {duration:.0f}m)")
    
    def run_live_trading_session(self, duration_minutes=60):
        """
        Run live paper trading session
        """
        print(f"\n🚀 STARTING LIVE PAPER TRADING SESSION")
        print(f"⏱️  Duration: {duration_minutes} minutes")
        print(f"📊 Decision Interval: {self.decision_interval} minutes")
        print("=" * 50)
        
        session_start = datetime.now()
        session_end = session_start + timedelta(minutes=duration_minutes)
        
        self.logger.info(f"Starting live trading session: {duration_minutes} minutes")
        
        try:
            while datetime.now() < session_end:
                try:
                    # ENHANCED: Check if markets are open before trading
                    should_run, market_reason, market_details = self.market_scheduler.should_run_trading_system()
                    
                    if not should_run:
                        sleep_duration = self.market_scheduler.get_sleep_duration()
                        self.logger.info(f"💤 {market_reason}")
                        self.logger.info(f"⏰ Sleeping for {sleep_duration/3600:.1f} hours until markets open")
                        print(f"💤 Markets closed - sleeping until {market_details['market_status']}")
                        time.sleep(min(sleep_duration, 3600))  # Max 1 hour sleep chunks
                        continue
                    
                    # Get live market data
                    market_data = self.get_live_market_data(100)
                    
                    if market_data is not None and len(market_data) > 50:
                        current_price = market_data['close'].iloc[-1]
                        
                        # Update active trades
                        self.update_active_trades(current_price)
                        
                        # Check if time for new decision
                        now = datetime.now()
                        if (self.last_decision_time is None or 
                            (now - self.last_decision_time).total_seconds() >= self.decision_interval * 60):
                            
                            # Make trading decision
                            decision = self.make_trading_decision(market_data)
                            
                            if decision:
                                # Execute paper trades
                                new_trades = self.execute_paper_trades(decision, current_price)
                                
                            self.last_decision_time = now
                        
                        # Display status every 5 updates
                        if self.session_stats['total_decisions'] % 5 == 0:
                            self.display_live_status(current_price)
                    
                    else:
                        self.logger.warning("Insufficient market data")
                    
                    # Wait before next update (30 seconds)
                    time.sleep(30)
                
                except KeyboardInterrupt:
                    print("\n⏹️  Session stopped by user")
                    break
                
                except Exception as e:
                    self.logger.error(f"Session error: {e}")
                    time.sleep(5)  # Brief pause on error
        
        finally:
            # Close any remaining active trades
            if self.active_trades:
                print(f"\n🔒 Closing {len(self.active_trades)} remaining trades...")
                current_price = market_data['close'].iloc[-1] if market_data is not None else 1.1600
                for trade in self.active_trades[:]:
                    self.update_active_trades(current_price)
            
            self.display_final_session_results()
    
    def display_final_session_results(self):
        """
        Display final session results
        """
        print(f"\n🏁 LIVE PAPER TRADING SESSION COMPLETE")
        print("=" * 45)
        
        session_duration = (datetime.now() - self.session_stats['start_time']).total_seconds() / 3600
        final_return = (self.current_capital - self.starting_capital) / self.starting_capital
        
        print(f"⏱️  Session Duration: {session_duration:.1f} hours")
        print(f"💰 Starting Capital: ${self.starting_capital:,}")
        print(f"💰 Final Capital: ${self.current_capital:,.2f}")
        print(f"📈 Total Return: {final_return:+.3%}")
        print(f"💵 Dollar P&L: ${self.current_capital - self.starting_capital:+,.2f}")
        
        print(f"\n📊 TRADING ACTIVITY:")
        print(f"   Total Decisions: {self.session_stats['total_decisions']}")
        print(f"   Total Trades: {self.session_stats['total_trades']}")
        print(f"   Closed Trades: {len(self.trade_history)}")
        
        if self.session_stats['momentum_trades'] > 0:
            momentum_wr = (self.session_stats['momentum_wins'] / self.session_stats['momentum_trades']) * 100
            print(f"   Momentum: {self.session_stats['momentum_trades']} trades, {momentum_wr:.1f}% win rate")
        
        if self.session_stats['mr_trades'] > 0:
            mr_wr = (self.session_stats['mr_wins'] / self.session_stats['mr_trades']) * 100
            print(f"   Mean Reversion: {self.session_stats['mr_trades']} trades, {mr_wr:.1f}% win rate")
        
        # Save session results
        self.save_session_results()
        
        print(f"\n✅ Session results saved to live_trading.log")
        print(f"🚀 Ready for next trading session!")
    
    def save_session_results(self):
        """
        Save detailed session results to file
        """
        try:
            results = {
                'session_summary': {
                    'start_time': self.session_stats['start_time'].isoformat(),
                    'end_time': datetime.now().isoformat(),
                    'starting_capital': self.starting_capital,
                    'final_capital': self.current_capital,
                    'total_return': (self.current_capital - self.starting_capital) / self.starting_capital,
                    'total_decisions': self.session_stats['total_decisions'],
                    'total_trades': self.session_stats['total_trades']
                },
                'strategy_performance': {
                    'momentum_trades': self.session_stats['momentum_trades'],
                    'momentum_wins': self.session_stats['momentum_wins'],
                    'mr_trades': self.session_stats['mr_trades'],
                    'mr_wins': self.session_stats['mr_wins']
                },
                'trade_history': [
                    {
                        'strategy': t['strategy'],
                        'direction': t['direction'],
                        'entry_price': t['entry_price'],
                        'exit_price': t.get('exit_price', 0),
                        'pips': t.get('pips', 0),
                        'return_pct': t.get('return_pct', 0),
                        'duration_minutes': (t.get('exit_time', datetime.now()) - t['entry_time']).total_seconds() / 60
                    }
                    for t in self.trade_history
                ]
            }
            
            filename = f"/Users/jonspinogatti/Desktop/spin36TB/session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            self.logger.info(f"Session results saved: {filename}")
            
        except Exception as e:
            self.logger.error(f"Failed to save session results: {e}")

def start_continuous_trading():
    """
    Start continuous trading that automatically handles market hours
    """
    print("🎯 SPIN36TB CONTINUOUS LIVE TRADING")
    print("=" * 40)
    print("🕒 Market-aware: Automatically sleeps when markets closed")
    print("🔄 Continuous operation: Runs indefinitely until stopped")
    print("💡 Press Ctrl+C to stop")
    print("=" * 40)
    
    trader = LivePaperTrader(starting_capital=25000)
    
    try:
        while True:
            # Check market status
            should_run, reason, details = trader.market_scheduler.should_run_trading_system()
            
            if should_run:
                print(f"\n🚀 {reason}")
                print(f"📊 {details['market_status']}")
                print(f"⭐ Quality: {details['quality_score']:.1f} - {details['quality_description']}")
                
                # Run trading session for market hours or until markets close
                # Use shorter sessions to allow frequent market status checks
                session_duration = 240  # 4 hours max, but will check market status regularly
                
                trader.run_live_trading_session(duration_minutes=session_duration)
                
                # Brief pause between sessions
                time.sleep(60)
                
            else:
                sleep_duration = trader.market_scheduler.get_sleep_duration()
                sleep_hours = sleep_duration / 3600
                
                print(f"\n💤 {reason}")
                print(f"⏰ {details['market_status']}")
                print(f"😴 Sleeping for {sleep_hours:.1f} hours until markets open...")
                
                # Sleep in chunks to allow for interruption
                total_sleep = 0
                chunk_size = min(3600, sleep_duration)  # 1 hour max chunks
                
                while total_sleep < sleep_duration:
                    remaining = sleep_duration - total_sleep
                    current_chunk = min(chunk_size, remaining)
                    
                    print(f"💤 Sleeping {current_chunk/60:.0f} more minutes...")
                    time.sleep(current_chunk)
                    total_sleep += current_chunk
                    
                    # Double-check market status in case schedule changed
                    should_run_now, _, _ = trader.market_scheduler.should_run_trading_system()
                    if should_run_now:
                        print("🎯 Markets opened early - breaking from sleep!")
                        break
                
    except KeyboardInterrupt:
        print("\n⏹️ Continuous trading stopped by user")
        print("📊 Final session summary saved")
        trader.save_session_results()
    except Exception as e:
        print(f"\n❌ Error in continuous trading: {e}")
        trader.logger.error(f"Continuous trading error: {e}")

def start_live_paper_trading():
    """
    Start single live paper trading session (legacy method)
    """
    print("🎯 SPIN36TB SINGLE SESSION TRADING")
    print("=" * 40)
    print("⚠️ Note: For continuous operation, use start_continuous_trading()")
    print("=" * 40)
    
    trader = LivePaperTrader(starting_capital=25000)
    
    # Check if markets are open before starting
    should_run, reason, details = trader.market_scheduler.should_run_trading_system()
    
    if not should_run:
        print(f"\n💤 {reason}")
        print(f"⏰ {details['market_status']}")
        print("❌ Cannot start single session - markets are closed")
        return
    
    # Start trading session
    print(f"\n🚀 {reason}")
    print("💡 Press Ctrl+C to stop early")
    
    trader.run_live_trading_session(duration_minutes=120)  # 2 hours

if __name__ == "__main__":
    # Use continuous trading by default for automatic market awareness
    start_continuous_trading()