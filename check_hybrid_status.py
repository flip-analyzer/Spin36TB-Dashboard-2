#!/usr/bin/env python3
"""
Quick Hybrid System Status Check
Monitor the current status of the hybrid trading system
"""

import os
import json
from datetime import datetime
import glob

def check_hybrid_status():
    """Check current status of hybrid trading system"""
    print("🔍 HYBRID SYSTEM STATUS CHECK")
    print("=" * 50)
    print(f"⏰ Check time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check if hybrid trading log exists and get recent activity
    log_file = '/Users/jonspinogatti/Desktop/spin36TB/hybrid_trading.log'
    
    if os.path.exists(log_file):
        print("✅ Hybrid system log found")
        
        # Read last few lines
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
                
            if lines:
                recent_lines = lines[-10:]  # Last 10 log entries
                print(f"📊 Recent activity ({len(lines)} total log entries):")
                
                for line in recent_lines:
                    line = line.strip()
                    if 'INFO' in line:
                        # Extract timestamp and message
                        parts = line.split(' - INFO - ')
                        if len(parts) >= 2:
                            timestamp = parts[0].split(',')[0]
                            message = parts[1]
                            print(f"   {timestamp}: {message}")
                    elif 'ERROR' in line:
                        print(f"   ❌ {line}")
            else:
                print("📄 Log file is empty")
                
        except Exception as e:
            print(f"❌ Error reading log: {e}")
    else:
        print("❌ No hybrid trading log found")
    
    # Check for recent session files
    print("\n📁 SESSION FILES:")
    session_files = glob.glob('/Users/jonspinogatti/Desktop/spin36TB/hybrid_session_*.json')
    
    if session_files:
        # Sort by modification time, get most recent
        session_files.sort(key=os.path.getmtime, reverse=True)
        recent_session = session_files[0]
        
        try:
            with open(recent_session, 'r') as f:
                session_data = json.load(f)
            
            stats = session_data.get('session_summary', {})
            trades = session_data.get('trade_history', [])
            
            print(f"📊 Most recent session: {os.path.basename(recent_session)}")
            print(f"   💰 Final capital: ${stats.get('current_capital', 0):,.2f}")
            print(f"   📈 Total return: {stats.get('total_return', 0):+.2f}%")
            print(f"   🎯 Total decisions: {stats.get('total_decisions', 0)}")
            print(f"   📊 Trades executed: {len(trades)}")
            print(f"   ⚡ Max concurrent: {stats.get('max_concurrent_trades', 0)}")
            
            # Trade breakdown
            print(f"   🔄 Trade breakdown:")
            print(f"      Momentum: {stats.get('momentum_trades', 0)}")
            print(f"      Micro: {stats.get('micro_trades', 0)}")
            print(f"      Patterns: {stats.get('pattern_trades', 0)}")
            print(f"      Confluence: {stats.get('confluence_trades', 0)}")
            
        except Exception as e:
            print(f"❌ Error reading session file: {e}")
    else:
        print("📄 No session files found yet")
    
    # Check system processes
    print(f"\n🖥️  SYSTEM PROCESSES:")
    try:
        import subprocess
        
        # Check for python processes running our scripts
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        processes = result.stdout
        
        hybrid_processes = [line for line in processes.split('\n') 
                          if 'python' in line and any(script in line for script in 
                          ['launch_hybrid_trading.py', 'hybrid_live_trader.py'])]
        
        if hybrid_processes:
            print("✅ Hybrid system processes running:")
            for process in hybrid_processes:
                # Extract relevant parts
                parts = process.split()
                if len(parts) > 10:
                    pid = parts[1]
                    cpu = parts[2]
                    mem = parts[3]
                    script = next((part for part in parts if '.py' in part), 'unknown')
                    print(f"   PID {pid}: {script} (CPU: {cpu}%, MEM: {mem}%)")
        else:
            print("❌ No hybrid system processes found")
            
    except Exception as e:
        print(f"❌ Error checking processes: {e}")

if __name__ == "__main__":
    check_hybrid_status()