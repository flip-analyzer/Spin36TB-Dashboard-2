#!/usr/bin/env python3
"""
Auto PRADO Trading Monitor

Real-time monitoring dashboard for the automated trading daemon
"""

import os
import sys
import time
import json
import psutil
from datetime import datetime, timedelta
from pathlib import Path

def clear_screen():
    """Clear terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def get_daemon_status():
    """Check if daemon is running"""
    pid_file = Path("logs/auto_prado.pid")
    
    if not pid_file.exists():
        return False, "PID file not found"
    
    try:
        with open(pid_file, 'r') as f:
            pid = int(f.read().strip())
        
        if psutil.pid_exists(pid):
            process = psutil.Process(pid)
            return True, {
                'pid': pid,
                'status': process.status(),
                'cpu_percent': process.cpu_percent(),
                'memory_mb': process.memory_info().rss / 1024 / 1024,
                'start_time': datetime.fromtimestamp(process.create_time())
            }
        else:
            return False, "Process not found"
            
    except Exception as e:
        return False, f"Error: {e}"

def get_latest_log_entries(n=10):
    """Get latest log entries"""
    today = datetime.now().strftime("%Y%m%d")
    log_file = Path(f"logs/auto_prado_{today}.log")
    
    if not log_file.exists():
        return ["No log file found for today"]
    
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
            return [line.strip() for line in lines[-n:]]
    except Exception as e:
        return [f"Error reading logs: {e}"]

def get_trading_stats():
    """Get current trading statistics"""
    # Look for the most recent session file
    log_dir = Path("logs")
    session_files = list(log_dir.glob("live_prado_session_*.json"))
    
    if not session_files:
        return None
    
    # Get the most recent session file
    latest_session = max(session_files, key=lambda x: x.stat().st_mtime)
    
    try:
        with open(latest_session, 'r') as f:
            return json.load(f)
    except Exception as e:
        return {"error": f"Could not read session stats: {e}"}

def format_uptime(start_time):
    """Format uptime duration"""
    if isinstance(start_time, datetime):
        uptime = datetime.now() - start_time
        days = uptime.days
        hours, remainder = divmod(uptime.seconds, 3600)
        minutes, _ = divmod(remainder, 60)
        
        if days > 0:
            return f"{days}d {hours}h {minutes}m"
        elif hours > 0:
            return f"{hours}h {minutes}m"
        else:
            return f"{minutes}m"
    return "Unknown"

def main():
    """Main monitoring loop"""
    print("🤖 Auto PRADO Trading Monitor")
    print("Press Ctrl+C to exit")
    print()
    
    try:
        while True:
            clear_screen()
            
            print("🤖 SPIN36TB AUTO PRADO TRADING MONITOR")
            print("=" * 60)
            print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print()
            
            # Daemon Status
            running, status = get_daemon_status()
            
            if running:
                print("✅ DAEMON STATUS: RUNNING")
                print(f"   PID: {status['pid']}")
                print(f"   Status: {status['status']}")
                print(f"   CPU: {status['cpu_percent']:.1f}%")
                print(f"   Memory: {status['memory_mb']:.1f} MB")
                print(f"   Uptime: {format_uptime(status['start_time'])}")
            else:
                print("❌ DAEMON STATUS: NOT RUNNING")
                print(f"   Reason: {status}")
            
            print()
            
            # Trading Statistics
            print("📊 TRADING STATISTICS")
            print("-" * 30)
            
            stats = get_trading_stats()
            if stats:
                if 'error' in stats:
                    print(f"⚠️  {stats['error']}")
                else:
                    print(f"💰 Capital: ${stats.get('current_capital', 0):,.2f}")
                    print(f"📈 Return: {stats.get('total_return', 0):+.2f}%")
                    print(f"📊 Total Trades: {stats.get('total_trades', 0)}")
                    print(f"🎯 Win Rate: {stats.get('win_rate', 0)*100:.1f}%")
                    print(f"📌 Active Trades: {stats.get('active_trades', 0)}")
                    
                    prado_perf = stats.get('prado_performance', {})
                    if prado_perf:
                        print(f"🎯 PRADO Wins: {prado_perf.get('winning_trades', 0)}")
                        print(f"📊 Total Pips: {prado_perf.get('total_pips', 0):+.1f}")
            else:
                print("📊 No trading statistics available")
            
            print()
            
            # Recent Log Entries
            print("📝 RECENT ACTIVITY (Last 8 entries)")
            print("-" * 50)
            
            log_entries = get_latest_log_entries(8)
            for entry in log_entries:
                # Truncate long lines
                if len(entry) > 100:
                    entry = entry[:97] + "..."
                print(f"   {entry}")
            
            print()
            print("🔄 Refreshing in 30 seconds... (Ctrl+C to exit)")
            
            # Wait 30 seconds
            time.sleep(30)
            
    except KeyboardInterrupt:
        print("\n👋 Monitor stopped by user")
    except Exception as e:
        print(f"\n❌ Monitor error: {e}")

if __name__ == "__main__":
    main()