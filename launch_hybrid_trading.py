#!/usr/bin/env python3
"""
Launch Hybrid Trading System
Deploy the high-frequency hybrid system for live paper trading
Target: 20-30 trades per day with multiple signal sources
"""

import sys
import time
from datetime import datetime
import signal

sys.path.append('/Users/jonspinogatti/Desktop/spin36TB')
from hybrid_live_trader import HybridLivePaperTrader

class HybridTradingLauncher:
    """
    Deploy and monitor the hybrid trading system
    """
    
    def __init__(self):
        self.trader = None
        self.running = True
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        print(f"\n🛑 Received signal {signum} - Initiating graceful shutdown...")
        self.running = False
        
        if self.trader and self.trader.active_trades:
            print("🏁 Closing active trades before shutdown...")
            # Close active trades at market price
            try:
                current_price = self.trader.get_live_data(1)['close'].iloc[-1]
                for trade in self.trader.active_trades[:]:
                    closed_trade = self.trader.close_paper_trade(trade, current_price, "system_shutdown", datetime.now())
                    if closed_trade:
                        self.trader.trade_history.append(closed_trade)
                        print(f"   ✅ Closed {trade['source']} trade: ${closed_trade['pnl']:+.2f}")
                
                self.trader.active_trades.clear()
            except Exception as e:
                print(f"❌ Error closing trades: {e}")
        
        print("✅ Graceful shutdown complete")
        sys.exit(0)
    
    def display_startup_info(self):
        """Display system startup information"""
        print("🚀 LAUNCHING HIGH-FREQUENCY HYBRID TRADING SYSTEM")
        print("=" * 60)
        print("📊 SYSTEM CONFIGURATION:")
        print("   • Multiple signal sources: Momentum Micro + Standard + Patterns + Confluence")
        print("   • Decision cycles: Every 3-5 minutes")
        print("   • Target frequency: 20-30 trades per day")
        print("   • Max concurrent positions: 8")
        print("   • Position sizes: 0.3% - 2.0% based on confidence")
        print("   • Trading hours: 12 hours/day (peak sessions)")
        print()
        print("🎯 EXPECTED PERFORMANCE:")
        print("   • Daily trades: 20-30")
        print("   • Trade distribution:")
        print("     - Micro momentum: 8-12 trades/day (quick scalps)")
        print("     - Standard momentum: 4-6 trades/day (proven system)")
        print("     - Pattern analysis: 3-4 trades/day (statistical patterns)")
        print("     - Confluence signals: 1-2 trades/day (high conviction)")
        print("   • Risk management: Multi-layered with time-based exits")
        print()
        print("🔧 CONTROLS:")
        print("   • Ctrl+C: Graceful shutdown (closes all positions)")
        print("   • Monitor logs: hybrid_trading.log")
        print("   • Session data: hybrid_session_*.json")
        print("=" * 60)
    
    def run_continuous_trading(self):
        """Run continuous hybrid trading with automatic restarts"""
        session_count = 0
        
        while self.running:
            try:
                session_count += 1
                print(f"\n🔄 STARTING TRADING SESSION #{session_count}")
                print(f"⏰ Session start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Create fresh trader instance for each session
                self.trader = HybridLivePaperTrader(starting_capital=25000)
                
                # Run 2-hour trading session
                self.trader.run_hybrid_trading_session(duration_minutes=120)
                
                if self.running:  # Only continue if not shutting down
                    print(f"✅ Session #{session_count} completed successfully")
                    print("⏳ 5-minute break before next session...")
                    time.sleep(300)  # 5-minute break between sessions
                
            except KeyboardInterrupt:
                print("\n🛑 Manual shutdown requested")
                break
                
            except Exception as e:
                print(f"❌ Session #{session_count} error: {e}")
                print("🔄 Restarting in 60 seconds...")
                
                if self.running:  # Only sleep if not shutting down
                    time.sleep(60)
    
    def launch(self):
        """Main launch sequence"""
        try:
            self.display_startup_info()
            
            # System readiness check
            print("🔍 SYSTEM READINESS CHECK:")
            test_trader = HybridLivePaperTrader(starting_capital=25000)
            
            # Test data connection
            print("   📊 Testing data connection...")
            market_data = test_trader.get_live_data(10)
            if market_data.empty:
                print("   ❌ Data connection failed")
                return False
            print(f"   ✅ Data connection active (price: {market_data['close'].iloc[-1]:.4f})")
            
            # Test decision engines
            print("   🧠 Testing decision engines...")
            hybrid_decisions = test_trader.make_hybrid_decisions(market_data)
            print(f"   ✅ Decision engines ready ({len(hybrid_decisions)} initial signals)")
            
            print("\n🚀 ALL SYSTEMS GO - LAUNCHING HYBRID TRADING!")
            print("=" * 60)
            
            # Start continuous trading
            self.run_continuous_trading()
            
        except Exception as e:
            print(f"❌ Launch failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        return True

def main():
    """Main entry point"""
    launcher = HybridTradingLauncher()
    
    print("🎯 High-Frequency Hybrid Trading System")
    print(f"📅 Launch time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        success = launcher.launch()
        
        if success:
            print("\n✅ Trading system shutdown complete")
        else:
            print("\n💥 Trading system failed to launch")
            
    except Exception as e:
        print(f"\n❌ Critical error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()