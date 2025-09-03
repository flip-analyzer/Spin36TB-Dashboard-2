#!/usr/bin/env python3
"""
Spin36TB Automated PRADO Trading Daemon

Automatically:
- Starts when forex markets open
- Runs continuously during trading hours
- Restarts daily with fresh logging
- Handles all errors and recovery
- Logs everything to files
"""

import sys
import os
import time
import logging
import signal
import json
from datetime import datetime, timedelta
from pathlib import Path
import threading

# Add project directory to path
sys.path.append('/Users/jonspinogatti/Desktop/spin36TB')

from hybrid_live_trader_prado import HybridLivePaperTraderPrado

class AutoPradoDaemon:
    """24/7 Automated PRADO Trading Daemon"""
    
    def __init__(self):
        self.running = False
        self.trader = None
        self.current_session_thread = None
        self.setup_logging()
        self.load_config()
        self.setup_signal_handlers()
        
    def setup_logging(self):
        """Setup comprehensive file logging"""
        # Create logs directory
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Daily log file with timestamp
        today = datetime.now().strftime("%Y%m%d")
        log_file = log_dir / f"auto_prado_{today}.log"
        
        # Setup logging configuration
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()  # Also print to console
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("🤖 Auto PRADO Daemon Logging Initialized")
        
    def load_config(self):
        """Load daemon configuration"""
        self.config = {
            'starting_capital': 25000,
            'max_session_hours': 24,  # Maximum session length
            'restart_hour': 17,  # 5 PM EST - after NY close
            'market_check_interval': 300,  # Check market status every 5 minutes
            'error_retry_delay': 60,  # Wait 1 minute after errors
            'max_consecutive_errors': 5
        }
        
        # Try to load custom config if exists
        config_file = Path("auto_prado_config.json")
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    custom_config = json.load(f)
                    self.config.update(custom_config)
                    self.logger.info(f"📋 Loaded custom configuration: {config_file}")
            except Exception as e:
                self.logger.warning(f"⚠️ Could not load config: {e}")
    
    def setup_signal_handlers(self):
        """Setup graceful shutdown handlers"""
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.logger.info(f"🛑 Received signal {signum} - shutting down gracefully...")
        self.running = False
        
        if self.current_session_thread and self.current_session_thread.is_alive():
            self.logger.info("⏳ Waiting for current trading session to finish...")
            # Give it time to finish current cycle
            time.sleep(30)
            
    def is_market_hours(self):
        """Check if forex markets are open"""
        try:
            if hasattr(self, 'trader') and self.trader:
                should_run, reason, details = self.trader.market_scheduler.should_run_trading_system()
                return should_run, reason
            else:
                # Fallback check - forex is open 24/5 except weekends
                now = datetime.now()
                weekday = now.weekday()  # 0 = Monday, 6 = Sunday
                
                # Forex closes Friday 5 PM EST, reopens Sunday 5 PM EST
                if weekday == 6:  # Sunday
                    return now.hour >= 17, "Sunday evening - markets opening"
                elif weekday == 5:  # Saturday
                    return False, "Saturday - markets closed"
                elif weekday == 4 and now.hour >= 17:  # Friday after 5 PM
                    return False, "Friday evening - markets closed"
                else:
                    return True, "Forex markets open"
                    
        except Exception as e:
            self.logger.error(f"Error checking market hours: {e}")
            return False, f"Error checking market status: {e}"
    
    def should_restart_daily(self):
        """Check if it's time for daily restart"""
        now = datetime.now()
        return now.hour == self.config['restart_hour'] and now.minute < 5
    
    def initialize_trader(self):
        """Initialize the trading system"""
        try:
            self.logger.info("🚀 Initializing PRADO Trading System...")
            self.trader = HybridLivePaperTraderPrado(
                starting_capital=self.config['starting_capital']
            )
            self.logger.info("✅ Trading system initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize trader: {e}")
            return False
    
    def export_dashboard_data(self):
        """Export data for Streamlit dashboard"""
        try:
            if not self.trader:
                return
            
            # Get current trader stats
            stats = self.trader.session_stats
            trade_history = getattr(self.trader, 'trade_history', [])
            active_trades = getattr(self.trader, 'active_trades', [])
            
            # Calculate key metrics
            total_trades = len(trade_history)
            winning_trades = len([t for t in trade_history if t.get('success', False)])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            total_pips = sum([t.get('pips', 0) for t in trade_history])
            current_capital = stats.get('current_capital', stats.get('starting_capital', 25000))
            total_return = stats.get('total_return', 0)
            
            # PRADO specific stats
            prado_perf = stats.get('prado_performance', {})
            prado_enhanced = stats.get('prado_enhanced', 0)
            prado_filtered = stats.get('prado_filtered', 0)
            
            # Create dashboard data
            dashboard_data = {
                'timestamp': datetime.now().isoformat(),
                'system_status': 'ACTIVE' if self.running else 'STOPPED',
                'account_balance': current_capital,
                'total_return_pct': total_return,
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'win_rate': win_rate,
                'total_pips': total_pips,
                'active_trades_count': len(active_trades),
                
                # PRADO stats
                'prado_enhanced': prado_enhanced,
                'prado_filtered': prado_filtered,
                'prado_win_rate': prado_perf.get('winning_trades', 0) / max(prado_perf.get('total_closed_trades', 1), 1),
                'meta_labeling_accuracy': 0.75,  # Placeholder - could be calculated
                
                # Recent trades (last 10)
                'recent_trades': trade_history[-10:] if trade_history else [],
                'active_trades': active_trades,
                
                # Performance by coach
                'momentum_trades': stats.get('momentum_trades', 0),
                'academic_trades': stats.get('academic_trades', 0),
                'traditional_trades': stats.get('traditional_trades', 0),
                'gmm_trades': stats.get('gmm_trades', 0),
                
                # Market info
                'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Write to dashboard file
            dashboard_file = Path('dashboard_data.json')
            with open(dashboard_file, 'w') as f:
                json.dump(dashboard_data, f, indent=2, default=str)
            
            # Also update trade_summary.json for compatibility
            trade_summary = {
                'current_capital': current_capital,
                'total_return': total_return,
                'total_trades': total_trades,
                'win_rate': win_rate,
                'total_pips': total_pips,
                'trade_history': trade_history,
                'session_stats': stats,
                'last_updated': datetime.now().isoformat()
            }
            
            with open('trade_summary.json', 'w') as f:
                json.dump(trade_summary, f, indent=2, default=str)
                
            self.logger.debug(f"📊 Dashboard data exported: {total_trades} trades, ${current_capital:,.2f}")
            
        except Exception as e:
            self.logger.error(f"Error exporting dashboard data: {e}")
    
    def run_trading_session(self):
        """Run a trading session in a separate thread"""
        try:
            self.logger.info("🌟 Starting automated trading session...")
            
            # Run for max session hours or until market close
            max_minutes = self.config['max_session_hours'] * 60
            
            # Use the existing trading method but in a controlled way
            self.trader.run_live_prado_session(duration_minutes=max_minutes)
            
        except Exception as e:
            self.logger.error(f"❌ Trading session error: {e}")
            raise
    
    def run_daemon(self):
        """Main daemon loop"""
        self.running = True
        consecutive_errors = 0
        last_restart_day = datetime.now().day
        
        self.logger.info("🤖 AUTO PRADO DAEMON STARTING")
        self.logger.info("=" * 60)
        self.logger.info(f"📊 Starting Capital: ${self.config['starting_capital']:,}")
        self.logger.info(f"🔄 Daily Restart Time: {self.config['restart_hour']}:00")
        self.logger.info(f"⏰ Market Check Interval: {self.config['market_check_interval']}s")
        
        while self.running:
            try:
                current_day = datetime.now().day
                
                # Check for daily restart
                if (current_day != last_restart_day or 
                    self.should_restart_daily()):
                    
                    self.logger.info("🔄 Daily restart triggered")
                    if self.current_session_thread and self.current_session_thread.is_alive():
                        self.logger.info("⏳ Stopping current session for restart...")
                        # Let current session finish gracefully
                        time.sleep(60)
                    
                    # Reinitialize for fresh start
                    if self.initialize_trader():
                        last_restart_day = current_day
                        self.logger.info("✅ Daily restart completed")
                    else:
                        self.logger.error("❌ Failed to restart - will retry")
                        time.sleep(self.config['error_retry_delay'])
                        continue
                
                # Check market status
                market_open, reason = self.is_market_hours()
                
                if not market_open:
                    self.logger.info(f"💤 Markets closed: {reason}")
                    
                    # Stop current session if running
                    if self.current_session_thread and self.current_session_thread.is_alive():
                        self.logger.info("🛑 Stopping trading session - markets closed")
                        self.running = False  # This will stop the trader
                        time.sleep(60)
                        self.running = True  # Reset for next check
                        
                    time.sleep(self.config['market_check_interval'])
                    continue
                
                # Markets are open - ensure we have a trading session running
                if not self.current_session_thread or not self.current_session_thread.is_alive():
                    
                    # Initialize trader if needed
                    if not self.trader:
                        if not self.initialize_trader():
                            consecutive_errors += 1
                            if consecutive_errors >= self.config['max_consecutive_errors']:
                                self.logger.critical(f"🚨 Too many initialization failures - stopping daemon")
                                break
                            time.sleep(self.config['error_retry_delay'])
                            continue
                    
                    # Start new trading session
                    self.logger.info(f"🚀 Starting new trading session - {reason}")
                    self.current_session_thread = threading.Thread(
                        target=self.run_trading_session,
                        daemon=True
                    )
                    self.current_session_thread.start()
                    consecutive_errors = 0  # Reset error counter on successful start
                
                else:
                    self.logger.debug(f"✅ Trading session running - {reason}")
                
                # Export dashboard data every check
                self.export_dashboard_data()
                
                # Wait before next check
                time.sleep(self.config['market_check_interval'])
                
            except KeyboardInterrupt:
                self.logger.info("🛑 Daemon interrupted by user")
                break
                
            except Exception as e:
                consecutive_errors += 1
                self.logger.error(f"❌ Daemon error #{consecutive_errors}: {e}")
                
                if consecutive_errors >= self.config['max_consecutive_errors']:
                    self.logger.critical(f"🚨 Too many consecutive errors - stopping daemon")
                    break
                
                self.logger.info(f"⏳ Waiting {self.config['error_retry_delay']}s before retry...")
                time.sleep(self.config['error_retry_delay'])
        
        # Cleanup
        self.shutdown()
    
    def shutdown(self):
        """Graceful shutdown"""
        self.logger.info("🔄 Shutting down Auto PRADO Daemon...")
        self.running = False
        
        if self.current_session_thread and self.current_session_thread.is_alive():
            self.logger.info("⏳ Waiting for trading session to complete...")
            self.current_session_thread.join(timeout=60)
        
        # Save final status
        try:
            status = {
                'shutdown_time': datetime.now().isoformat(),
                'final_capital': getattr(self.trader, 'session_stats', {}).get('current_capital', 0),
                'total_trades': len(getattr(self.trader, 'trade_history', [])),
                'uptime_hours': 0  # Could track this if needed
            }
            
            status_file = Path("logs") / f"final_status_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(status_file, 'w') as f:
                json.dump(status, f, indent=2)
                
            self.logger.info(f"💾 Final status saved to: {status_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving final status: {e}")
        
        self.logger.info("✅ Auto PRADO Daemon shutdown complete")

def main():
    """Main entry point"""
    print("🤖 Spin36TB Auto PRADO Daemon")
    print("=" * 50)
    print("🔄 Automated 24/7 PRADO Trading System")
    print("📊 Continuous learning and adaptation")
    print("📁 All activity logged to files")
    print("⚠️  Press Ctrl+C to stop gracefully")
    print()
    
    daemon = AutoPradoDaemon()
    daemon.run_daemon()

if __name__ == "__main__":
    main()