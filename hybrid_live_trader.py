#!/usr/bin/env python3
"""
Hybrid Live Paper Trader
Integrates your existing successful system with high-frequency hybrid approach
Target: 20-30 trades per day with multiple concurrent positions
"""

import pandas as pd
import numpy as np
import time
import requests
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import sys
import warnings

# Import your existing systems
sys.path.append('/Users/jonspinogatti/Desktop/spin36TB')
from dual_strategy_portfolio import DualStrategyPortfolioManager
from forex_market_scheduler import ForexMarketScheduler
from adaptive_learning_system import AdaptiveLearningSystem
from high_frequency_hybrid import HighFrequencyHybridSystem
from upgrade_to_6hour_momentum import Enhanced6HourHybridSystem

warnings.filterwarnings('ignore')

class HybridLivePaperTrader:
    """
    Enhanced live paper trader combining your proven system with high-frequency hybrid approach
    """
    
    def __init__(self, starting_capital=25000):
        # OANDA Configuration (keep your existing setup)
        self.api_token = "fdf72f4b5166602e39e001d1c1cb38b0-524c6662baa3187260b34c006eeab1fd"
        self.account_id = "101-003-29503497-001"
        self.base_url = "https://api-fxpractice.oanda.com"
        
        # Enhanced Systems
        self.portfolio = DualStrategyPortfolioManager(starting_capital)
        self.market_scheduler = ForexMarketScheduler()
        self.learning_system = AdaptiveLearningSystem()
        self.hybrid_system = HighFrequencyHybridSystem(starting_capital)  # NEW
        self.enhanced_6hour_system = Enhanced6HourHybridSystem()  # Carson's 6-hour approach
        
        # Enhanced Trading State (support multiple concurrent trades)
        self.active_trades = []          # Multiple trades can be active
        self.trade_history = []
        self.live_data_buffer = []
        self.last_decision_time = None   # Track last decision for your original system
        self.last_hybrid_check = None    # Track last hybrid system check
        self.last_6hour_check = None     # Track last 6-hour momentum check
        
        # Enhanced Session Stats
        self.session_stats = {
            'starting_capital': starting_capital,
            'current_capital': starting_capital,
            'total_decisions': 0,
            'hybrid_decisions': 0,        # NEW: Hybrid system decisions
            'traditional_decisions': 0,   # Your original system decisions
            '6hour_decisions': 0,        # NEW: 6-hour momentum decisions
            'momentum_trades': 0,
            'mr_trades': 0,
            'micro_trades': 0,           # NEW: Micro momentum trades
            'pattern_trades': 0,         # NEW: Pattern-based trades
            'confluence_trades': 0,      # NEW: Confluence trades
            '6hour_momentum_trades': 0,  # NEW: Carson's 6-hour momentum trades
            'total_return': 0.0,
            'max_concurrent_trades': 0   # Track max concurrent positions
        }
        
        # Logging setup
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('/Users/jonspinogatti/Desktop/spin36TB/hybrid_trading.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        print("🚀 HYBRID LIVE PAPER TRADER INITIALIZED")
        print("=" * 50)
        print("⚡ High-frequency hybrid system enabled")
        print("🔄 Triple decision engines: Traditional + Hybrid + 6-Hour Momentum") 
        print("📊 Target: 20-30 trades per day")
        print("🎯 Multiple concurrent positions supported")
        print("⏰ Carson's 6-hour momentum approach integrated")
        
    def get_live_data(self, count=100):
        """Enhanced data fetching with better error handling - now supports 6-hour lookback"""
        try:
            # For 6-hour momentum analysis, we need 72 candles (6 hours * 12 candles per hour)
            # But also support shorter lookbacks for other systems
            adjusted_count = max(count, 72)  # Ensure we have enough for 6-hour analysis
            
            headers = {"Authorization": f"Bearer {self.api_token}"}
            params = {
                "count": adjusted_count,
                "granularity": "M5",  # 5-minute candles for high-frequency
                "price": "M"
            }
            
            response = requests.get(
                f"{self.base_url}/v3/instruments/EUR_USD/candles",
                headers=headers,
                params=params,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                df_data = []
                
                for candle in data['candles']:
                    if candle['complete']:
                        df_data.append({
                            'time': pd.to_datetime(candle['time']).tz_localize(None),  # Remove timezone for compatibility
                            'open': float(candle['mid']['o']),
                            'high': float(candle['mid']['h']),
                            'low': float(candle['mid']['l']),
                            'close': float(candle['mid']['c']),
                            'volume': int(candle.get('volume', 1))
                        })
                
                df = pd.DataFrame(df_data)
                if not df.empty:
                    df = df.sort_values('time').reset_index(drop=True)
                    self.live_data_buffer = df.to_dict('records')
                    return df
                    
            else:
                self.logger.error(f"API error: {response.status_code}")
                return self.get_fallback_data()
                
        except Exception as e:
            self.logger.error(f"Data fetch error: {e}")
            return self.get_fallback_data()
    
    def get_fallback_data(self):
        """Fallback to buffered data if live fetch fails"""
        if self.live_data_buffer:
            return pd.DataFrame(self.live_data_buffer)
        return pd.DataFrame()
    
    def make_traditional_portfolio_decision(self, market_data):
        """Your existing portfolio decision system (every 5 minutes)"""
        try:
            # Your original dual-strategy approach
            decision = self.portfolio.make_portfolio_decision(market_data)
            self.session_stats['total_decisions'] += 1
            self.session_stats['traditional_decisions'] += 1
            
            if not decision or not decision.get('portfolio_actions'):
                self.logger.info("Traditional system: No trading signals generated")
                return None
            
            # Log decision with source identifier
            actions_count = len(decision['portfolio_actions'])
            self.logger.info(f"🎯 Traditional Portfolio Decision: {actions_count} actions")
            
            return decision
            
        except Exception as e:
            self.logger.error(f"Traditional decision error: {e}")
            return None
    
    def make_hybrid_decisions(self, market_data):
        """NEW: High-frequency hybrid decision system"""
        try:
            # High-frequency hybrid approach (every 3-5 minutes)
            potential_trades = self.hybrid_system.make_high_frequency_decision(market_data)
            
            if potential_trades:
                self.session_stats['hybrid_decisions'] += len(potential_trades)
                self.session_stats['total_decisions'] += len(potential_trades)
                
                self.logger.info(f"⚡ Hybrid System Decisions: {len(potential_trades)} new signals")
                
                for trade in potential_trades:
                    self.logger.info(f"   📊 {trade['source']}: {trade['signal']} (confidence: {trade['confidence']:.2%})")
                
                return potential_trades
            
            return []
            
        except Exception as e:
            self.logger.error(f"Hybrid decision error: {e}")
            return []
    
    def make_6hour_momentum_decisions(self, market_data):
        """NEW: Carson's 6-hour momentum decision system (academic research-backed)"""
        try:
            # Carson's 6-hour momentum approach with academic backing
            momentum_decisions = self.enhanced_6hour_system.generate_enhanced_trading_decisions(market_data)
            
            if momentum_decisions:
                self.session_stats['6hour_decisions'] += len(momentum_decisions)
                self.session_stats['total_decisions'] += len(momentum_decisions)
                
                self.logger.info(f"⏰ 6-Hour Momentum Decisions: {len(momentum_decisions)} Carson-backed signals")
                
                for decision in momentum_decisions:
                    confidence = decision.get('confidence', 0)
                    signal = decision.get('signal', 'UNKNOWN')
                    methodology = decision.get('methodology', 'Unknown')
                    self.logger.info(f"   📊 6h_momentum: {signal} (confidence: {confidence:.3f}, {methodology})")
                
                return momentum_decisions
            
            return []
            
        except Exception as e:
            self.logger.error(f"6-Hour momentum decision error: {e}")
            return []
    
    def execute_paper_trades(self, decisions, current_price):
        """Enhanced trade execution supporting multiple concurrent positions"""
        if not decisions:
            return []
        
        executed_trades = []
        current_time = datetime.now()
        
        # Handle traditional portfolio decisions
        if isinstance(decisions, dict) and 'portfolio_actions' in decisions:
            for action in decisions['portfolio_actions']:
                trade = self.create_paper_trade(action, current_price, 'traditional', current_time)
                if trade:
                    executed_trades.append(trade)
        
        # Handle hybrid decisions (list of trade dictionaries)
        elif isinstance(decisions, list):
            for decision in decisions:
                trade = self.create_paper_trade_from_hybrid(decision, current_price, current_time)
                if trade:
                    executed_trades.append(trade)
        
        # Add to active trades and update stats
        for trade in executed_trades:
            self.active_trades.append(trade)
            self.update_trade_stats(trade)
            
            self.logger.info(f"📄 PAPER TRADE OPENED: {trade['source']} {trade['direction']} "
                           f"@ {trade['entry_price']:.4f} (size: {trade['position_size']:.3f})")
        
        # Update max concurrent trades stat
        self.session_stats['max_concurrent_trades'] = max(
            self.session_stats['max_concurrent_trades'], 
            len(self.active_trades)
        )
        
        return executed_trades
    
    def create_paper_trade(self, action, current_price, source, current_time):
        """Create paper trade from traditional portfolio action"""
        try:
            trade = {
                'trade_id': f"PAPER_{int(time.time())}_{len(self.active_trades)}",
                'source': source,
                'strategy': action.get('strategy', 'UNKNOWN'),
                'direction': action.get('direction', 'HOLD'),
                'entry_price': current_price,
                'position_size': action.get('position_size', 0),
                'confidence': action.get('confidence', 0),
                'entry_time': current_time,
                'hold_time_minutes': 60,  # Default hold time for traditional system
                'stop_loss': current_price * (0.985 if action.get('direction') == 'UP' else 1.015),
                'take_profit': current_price * (1.022 if action.get('direction') == 'UP' else 0.978),
                'status': 'active'
            }
            
            return trade if trade['direction'] in ['UP', 'DOWN'] else None
            
        except Exception as e:
            self.logger.error(f"Error creating traditional trade: {e}")
            return None
    
    def create_paper_trade_from_hybrid(self, decision, current_price, current_time):
        """Create paper trade from hybrid system decision"""
        try:
            trade = {
                'trade_id': f"HYBRID_{int(time.time())}_{len(self.active_trades)}",
                'source': decision['source'],
                'strategy': decision['source'].upper(),
                'direction': decision['signal'],
                'entry_price': current_price,
                'position_size': decision['position_size'],
                'confidence': decision['confidence'],
                'entry_time': current_time,
                'hold_time_minutes': self._get_hold_time_minutes(decision),
                'stop_loss': self._calculate_stop_loss(decision, current_price),
                'take_profit': self._calculate_take_profit(decision, current_price),
                'trailing_stop': decision.get('trailing_stop', 0.012),  # Default or 6-hour system value
                'status': 'active'
            }
            
            return trade if trade['direction'] in ['UP', 'DOWN'] else None
            
        except Exception as e:
            self.logger.error(f"Error creating hybrid trade: {e}")
            return None
    
    def _get_hold_time_minutes(self, decision):
        """Get appropriate holding time based on decision source"""
        source = decision.get('source', '')
        
        # Carson's 6-hour system gets extended holding period
        if 'enhanced_6hour' in source or 'max_holding_hours' in decision:
            max_hours = decision.get('max_holding_hours', 8)  # 8 hours default for 6-hour system
            return max_hours * 60  # Convert to minutes
        
        # Default short-term holding periods for other systems
        return decision.get('hold_time_minutes', 45)
    
    def _calculate_stop_loss(self, decision, current_price):
        """Calculate stop loss based on decision source and volatility"""
        source = decision.get('source', '')
        
        # Carson's 6-hour system uses wider stops (50 pips as per academic research)
        if 'enhanced_6hour' in source:
            trailing_stop = decision.get('trailing_stop', 0.005)  # 50 pips
            return current_price * (1 - trailing_stop if decision['signal'] == 'UP' else 1 + trailing_stop)
        
        # Tighter stops for short-term systems
        return current_price * (0.988 if decision['signal'] == 'UP' else 1.012)
    
    def _calculate_take_profit(self, decision, current_price):
        """Calculate take profit based on decision source and expected move"""
        source = decision.get('source', '')
        
        # Carson's 6-hour system uses wider profit targets (proportional to longer holding)
        if 'enhanced_6hour' in source:
            confidence = decision.get('confidence', 0.5)
            # Scale profit target with confidence: 1.5-3x stop distance
            profit_multiplier = 1.5 + (confidence * 1.5)  # 1.5x to 3x
            trailing_stop = decision.get('trailing_stop', 0.005)
            profit_target = trailing_stop * profit_multiplier
            return current_price * (1 + profit_target if decision['signal'] == 'UP' else 1 - profit_target)
        
        # Standard profit targets for short-term systems
        return current_price * (1.018 if decision['signal'] == 'UP' else 0.982)
    
    def update_trade_stats(self, trade):
        """Update session statistics based on trade source"""
        source = trade['source']
        
        if 'momentum' in source:
            if 'micro' in source:
                self.session_stats['micro_trades'] += 1
            elif 'enhanced_6hour' in source or '6h' in source:
                self.session_stats['6hour_momentum_trades'] += 1
            else:
                self.session_stats['momentum_trades'] += 1
        elif 'pattern' in source or 'cluster' in source:
            self.session_stats['pattern_trades'] += 1
        elif 'confluence' in source or 'hybrid' in source:
            self.session_stats['confluence_trades'] += 1
        elif trade['strategy'] == 'MEAN_REVERSION':
            self.session_stats['mr_trades'] += 1
        else:
            self.session_stats['momentum_trades'] += 1
    
    def manage_active_trades(self, current_price):
        """Enhanced trade management for multiple concurrent positions"""
        current_time = datetime.now()
        closed_trades = []
        
        for trade in self.active_trades[:]:  # Create copy to iterate safely
            # Check if trade should be closed
            should_close, reason = self.should_close_trade(trade, current_price, current_time)
            
            if should_close:
                # Close the trade
                closed_trade = self.close_paper_trade(trade, current_price, reason, current_time)
                if closed_trade:
                    closed_trades.append(closed_trade)
                    self.active_trades.remove(trade)
                    self.trade_history.append(closed_trade)
                    
                    # Log trade closure
                    pnl = closed_trade['pnl']
                    pips = closed_trade['pips']
                    self.logger.info(f"📄 PAPER TRADE CLOSED: {closed_trade['source']} "
                                   f"({reason}) PnL: ${pnl:+.2f} ({pips:+.1f} pips)")
        
        return closed_trades
    
    def should_close_trade(self, trade, current_price, current_time):
        """Determine if trade should be closed"""
        # Time-based exit
        hold_time = (current_time - trade['entry_time']).total_seconds() / 60
        if hold_time >= trade['hold_time_minutes']:
            return True, "time_exit"
        
        # Stop loss
        if trade['direction'] == 'UP' and current_price <= trade['stop_loss']:
            return True, "stop_loss"
        elif trade['direction'] == 'DOWN' and current_price >= trade['stop_loss']:
            return True, "stop_loss"
        
        # Take profit
        if trade['direction'] == 'UP' and current_price >= trade['take_profit']:
            return True, "take_profit"
        elif trade['direction'] == 'DOWN' and current_price <= trade['take_profit']:
            return True, "take_profit"
        
        return False, "active"
    
    def close_paper_trade(self, trade, current_price, reason, current_time):
        """Close a paper trade and calculate P&L"""
        try:
            exit_price = current_price
            entry_price = trade['entry_price']
            position_size = trade['position_size']
            
            # Calculate P&L
            if trade['direction'] == 'UP':
                pips = (exit_price - entry_price) * 10000
                pnl = (exit_price / entry_price - 1) * position_size * self.session_stats['starting_capital']
            else:  # DOWN
                pips = (entry_price - exit_price) * 10000
                pnl = (entry_price / exit_price - 1) * position_size * self.session_stats['starting_capital']
            
            # Update capital
            self.session_stats['current_capital'] += pnl
            self.session_stats['total_return'] = ((self.session_stats['current_capital'] / 
                                                  self.session_stats['starting_capital']) - 1) * 100
            
            # Create closed trade record
            closed_trade = trade.copy()
            closed_trade.update({
                'exit_price': exit_price,
                'exit_time': current_time,
                'exit_reason': reason,
                'pnl': pnl,
                'pips': pips,
                'status': 'closed',
                'hold_duration_minutes': (current_time - trade['entry_time']).total_seconds() / 60
            })
            
            return closed_trade
            
        except Exception as e:
            self.logger.error(f"Error closing trade: {e}")
            return None
    
    def run_hybrid_trading_session(self, duration_minutes=60):
        """Enhanced trading session with hybrid approach"""
        self.logger.info(f"🚀 Starting HYBRID trading session: {duration_minutes} minutes")
        
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)
        
        traditional_cycle = 0  # For 5-minute traditional decisions
        hybrid_cycle = 0       # For 3-minute hybrid checks
        sixhour_cycle = 0      # For 10-minute 6-hour momentum checks
        
        while datetime.now() < end_time:
            try:
                # Get market status
                should_run, reason, market_details = self.market_scheduler.should_run_trading_system()
                
                if not should_run:
                    self.logger.info(f"💤 {reason}")
                    time.sleep(300)  # 5-minute sleep during market closure
                    continue
                
                # Get live market data
                market_data = self.get_live_data(100)  # Get 100 candles for pattern analysis
                if market_data.empty:
                    self.logger.warning("📊 No market data - retrying in 30 seconds")
                    time.sleep(30)
                    continue
                
                current_price = market_data['close'].iloc[-1]
                current_time = datetime.now()
                
                # Manage existing trades (every cycle)
                closed_trades = self.manage_active_trades(current_price)
                
                # TRADITIONAL SYSTEM: Every 5 minutes (your proven approach)
                traditional_cycle += 1
                if traditional_cycle >= 10:  # 30 seconds * 10 = 5 minutes
                    if (self.last_decision_time is None or 
                        (current_time - self.last_decision_time).total_seconds() >= 300):
                        
                        traditional_decision = self.make_traditional_portfolio_decision(market_data)
                        if traditional_decision:
                            executed = self.execute_paper_trades(traditional_decision, current_price)
                            self.last_decision_time = current_time
                    
                    traditional_cycle = 0
                
                # HYBRID SYSTEM: Every 3 minutes (high-frequency approach)
                hybrid_cycle += 1
                if hybrid_cycle >= 6:  # 30 seconds * 6 = 3 minutes
                    if (self.last_hybrid_check is None or 
                        (current_time - self.last_hybrid_check).total_seconds() >= 180):
                        
                        hybrid_decisions = self.make_hybrid_decisions(market_data)
                        if hybrid_decisions:
                            executed = self.execute_paper_trades(hybrid_decisions, current_price)
                            self.last_hybrid_check = current_time
                    
                    hybrid_cycle = 0
                
                # 6-HOUR MOMENTUM SYSTEM: Every 10 minutes (Carson's academic approach)
                sixhour_cycle += 1
                if sixhour_cycle >= 20:  # 30 seconds * 20 = 10 minutes
                    if (self.last_6hour_check is None or 
                        (current_time - self.last_6hour_check).total_seconds() >= 600):
                        
                        sixhour_decisions = self.make_6hour_momentum_decisions(market_data)
                        if sixhour_decisions:
                            executed = self.execute_paper_trades(sixhour_decisions, current_price)
                            self.last_6hour_check = current_time
                    
                    sixhour_cycle = 0
                
                # Display session status
                self.display_session_status(current_price)
                
                # Sleep for 30 seconds before next cycle
                time.sleep(30)
                
            except KeyboardInterrupt:
                self.logger.info("🛑 Trading session interrupted by user")
                break
            except Exception as e:
                self.logger.error(f"Session error: {e}")
                time.sleep(30)
        
        # Session complete
        self.finalize_session()
    
    def display_session_status(self, current_price):
        """Enhanced status display"""
        print(f"\n{'='*60}")
        print(f"⏰ TIME: {datetime.now().strftime('%H:%M:%S')} | 💰 EURUSD: {current_price:.4f}")
        print(f"💼 CAPITAL: ${self.session_stats['current_capital']:,.2f} "
              f"({self.session_stats['total_return']:+.2f}%)")
        
        # Decision stats
        print(f"🎯 DECISIONS: {self.session_stats['total_decisions']} total "
              f"(Traditional: {self.session_stats['traditional_decisions']}, "
              f"Hybrid: {self.session_stats['hybrid_decisions']}, "
              f"6H-Momentum: {self.session_stats['6hour_decisions']})")
        
        # Trade stats by source
        print(f"📈 TRADES: Momentum: {self.session_stats['momentum_trades']}, "
              f"6H-Mom: {self.session_stats['6hour_momentum_trades']}, "
              f"Micro: {self.session_stats['micro_trades']}, "
              f"Patterns: {self.session_stats['pattern_trades']}, "
              f"Confluence: {self.session_stats['confluence_trades']}, "
              f"Mean Rev: {self.session_stats['mr_trades']}")
        
        # Active trades
        active_count = len(self.active_trades)
        max_concurrent = self.session_stats['max_concurrent_trades']
        print(f"🎪 ACTIVE TRADES: {active_count} current (max concurrent: {max_concurrent})")
        
        if self.active_trades:
            for trade in self.active_trades[-3:]:  # Show last 3 active trades
                duration = (datetime.now() - trade['entry_time']).total_seconds() / 60
                pips = ((current_price - trade['entry_price']) * 10000 if trade['direction'] == 'UP' 
                       else (trade['entry_price'] - current_price) * 10000)
                print(f"   {trade['source'][:12]} {trade['direction']} @ {trade['entry_price']:.4f} "
                      f"({pips:+.1f} pips, {duration:.0f}m)")
    
    def finalize_session(self):
        """Finalize trading session with summary"""
        # Close any remaining active trades
        current_price = self.get_live_data(1)['close'].iloc[-1] if not self.get_live_data(1).empty else 1.0
        
        for trade in self.active_trades[:]:
            closed_trade = self.close_paper_trade(trade, current_price, "session_end", datetime.now())
            if closed_trade:
                self.trade_history.append(closed_trade)
        
        self.active_trades.clear()
        
        # Save session data
        session_data = {
            'session_summary': self.session_stats,
            'trade_history': [dict(trade) for trade in self.trade_history],
            'final_timestamp': datetime.now().isoformat()
        }
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'/Users/jonspinogatti/Desktop/spin36TB/hybrid_session_{timestamp}.json'
        
        with open(filename, 'w') as f:
            json.dump(session_data, f, indent=2, default=str)
        
        self.logger.info(f"💾 Session data saved to {filename}")
        
        # Print final summary
        print(f"\n{'='*60}")
        print(f"🏁 HYBRID TRADING SESSION COMPLETE")
        print(f"{'='*60}")
        print(f"💰 Final Capital: ${self.session_stats['current_capital']:,.2f}")
        print(f"📈 Total Return: {self.session_stats['total_return']:+.2f}%")
        print(f"🎯 Total Decisions: {self.session_stats['total_decisions']}")
        print(f"📊 Total Trades: {len(self.trade_history)}")
        print(f"⚡ Max Concurrent: {self.session_stats['max_concurrent_trades']}")

if __name__ == "__main__":
    print("🚀 Initializing Hybrid Live Paper Trading System...")
    
    # Create hybrid trader instance
    trader = HybridLivePaperTrader(starting_capital=25000)
    
    try:
        # Start hybrid trading session
        trader.run_hybrid_trading_session(duration_minutes=120)  # 2-hour session
        
    except Exception as e:
        print(f"❌ Error in main execution: {e}")
        import traceback
        traceback.print_exc()