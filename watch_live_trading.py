#!/usr/bin/env python3
"""
Live Trading Monitor
Watch your paper trading session in real-time
"""

import requests
import time
from datetime import datetime

class LiveTradingMonitor:
    def __init__(self):
        # OANDA Configuration
        self.api_token = "fdf72f4b5166602e39e001d1c1cb38b0-524c6662baa3187260b34c006eeab1fd"
        self.account_id = "101-001-31365224-001"
        self.api_url = 'https://api-fxpractice.oanda.com'
        self.headers = {
            'Authorization': f'Bearer {self.api_token}',
            'Content-Type': 'application/json'
        }
    
    def get_current_price(self):
        """Get current EURUSD price"""
        try:
            url = f"{self.api_url}/v3/instruments/EUR_USD/candles"
            params = {'granularity': 'M5', 'count': 1, 'price': 'MBA'}
            response = requests.get(url, headers=self.headers, params=params, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                if data['candles']:
                    candle = data['candles'][-1]
                    return float(candle['mid']['c'])
            return None
        except:
            return None
    
    def get_account_status(self):
        """Get OANDA account status"""
        try:
            url = f"{self.api_url}/v3/accounts/{self.account_id}"
            response = requests.get(url, headers=self.headers, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                account = data['account']
                return {
                    'balance': float(account['balance']),
                    'currency': account['currency'],
                    'unrealized_pl': float(account.get('unrealizedPL', 0))
                }
            return None
        except:
            return None
    
    def read_log_tail(self, lines=10):
        """Read last lines from trading log"""
        try:
            with open('/Users/jonspinogatti/Desktop/spin36TB/live_trading.log', 'r') as f:
                log_lines = f.readlines()
                return log_lines[-lines:] if log_lines else []
        except:
            return []
    
    def display_live_dashboard(self):
        """Display live trading dashboard"""
        while True:
            try:
                # Clear screen (works on Mac/Linux)
                print('\033[2J\033[H')
                
                print("🚀 SPIN36TB LIVE PAPER TRADING MONITOR")
                print("=" * 55)
                print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Current price
                current_price = self.get_current_price()
                if current_price:
                    print(f"📈 EURUSD: {current_price:.4f}")
                else:
                    print("📈 EURUSD: Unable to fetch")
                
                # Account status
                account = self.get_account_status()
                if account:
                    print(f"💰 Account Balance: {account['currency']} {account['balance']:,.2f}")
                    if account['unrealized_pl'] != 0:
                        print(f"💼 Unrealized P&L: {account['unrealized_pl']:+.2f}")
                
                print("\n📊 RECENT TRADING ACTIVITY:")
                print("-" * 55)
                
                # Recent log entries
                log_lines = self.read_log_tail(8)
                for line in log_lines:
                    if line.strip():
                        # Clean up log format for display
                        clean_line = line.strip().split(' - ', 2)
                        if len(clean_line) >= 3:
                            timestamp = clean_line[0]
                            level = clean_line[1]
                            message = clean_line[2]
                            
                            # Color code messages
                            if "PAPER TRADE" in message:
                                print(f"🔵 {timestamp[-8:]}: {message}")
                            elif "TRADE CLOSED" in message:
                                print(f"🟢 {timestamp[-8:]}: {message}")
                            elif "Portfolio Decision" in message:
                                print(f"🎯 {timestamp[-8:]}: {message}")
                            elif "ERROR" in level:
                                print(f"🔴 {timestamp[-8:]}: {message}")
                            else:
                                print(f"ℹ️  {timestamp[-8:]}: {message}")
                
                print(f"\n🔄 Refreshing every 10 seconds... (Ctrl+C to exit)")
                time.sleep(10)
                
            except KeyboardInterrupt:
                print(f"\n👋 Live monitor stopped")
                break
            except Exception as e:
                print(f"Monitor error: {e}")
                time.sleep(5)

def start_live_monitor():
    """Start the live trading monitor"""
    monitor = LiveTradingMonitor()
    monitor.display_live_dashboard()

if __name__ == "__main__":
    start_live_monitor()