#!/usr/bin/env python3
"""
Export Live Trading Statistics for Cloud Dashboard
Reads actual trading logs and exports stats to JSON for cloud access
"""

import json
import re
from datetime import datetime, timedelta
from pathlib import Path

def parse_trading_logs():
    """Parse hybrid_trading.log and extract real statistics"""
    log_file = Path("hybrid_trading.log")
    
    if not log_file.exists():
        return None
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    # Initialize counters
    stats = {
        "last_updated": datetime.now().isoformat(),
        "total_decisions": 0,
        "total_trades": 0,
        "trades_opened": 0,
        "trades_closed": 0,
        "paper_pnl": 0.0,
        "winning_trades": 0,
        "losing_trades": 0,
        "system_active": True,
        "gmm_active": False,
        "latest_signals": [],
        "recent_trades": []
    }
    
    # Track PnL
    total_pnl = 0.0
    winning_trades = 0
    losing_trades = 0
    
    # Parse each line
    for line in lines:
        line = line.strip()
        
        # Count decisions
        if "⚡ Hybrid System Decisions:" in line:
            stats["total_decisions"] += 1
            
        # Count trades opened
        if "📄 PAPER TRADE OPENED:" in line:
            stats["trades_opened"] += 1
            stats["total_trades"] += 1
            
            # Extract trade details
            if "cluster_patterns" in line and "85.00%" in line:
                stats["gmm_active"] = True
        
        # Count trades closed and track PnL
        if "📄 PAPER TRADE CLOSED:" in line and "PnL:" in line:
            stats["trades_closed"] += 1
            
            # Extract PnL
            pnl_match = re.search(r'PnL: \$([+-]?[\d.]+)', line)
            if pnl_match:
                pnl = float(pnl_match.group(1))
                total_pnl += pnl
                
                if pnl > 0:
                    winning_trades += 1
                elif pnl < 0:
                    losing_trades += 1
        
        # Extract recent signals
        if "📊" in line and ("momentum_" in line or "cluster_patterns" in line or "hybrid_confluence" in line):
            signal_match = re.search(r'📊 (\w+): (\w+) \(confidence: ([\d.]+)%\)', line)
            if signal_match:
                signal_type, direction, confidence = signal_match.groups()
                stats["latest_signals"].append({
                    "type": signal_type,
                    "direction": direction,
                    "confidence": float(confidence)
                })
                
                # Keep only last 10 signals
                stats["latest_signals"] = stats["latest_signals"][-10:]
    
    # Calculate final stats
    stats["paper_pnl"] = round(total_pnl, 2)
    stats["winning_trades"] = winning_trades
    stats["losing_trades"] = losing_trades
    
    if stats["trades_closed"] > 0:
        stats["win_rate"] = round(winning_trades / stats["trades_closed"], 3)
    else:
        stats["win_rate"] = 0.0
    
    # System status
    recent_lines = lines[-50:] if len(lines) >= 50 else lines
    stats["system_active"] = any("🚀 Starting HYBRID trading session" in line for line in recent_lines)
    
    return stats

def export_stats():
    """Export trading statistics to JSON file"""
    stats = parse_trading_logs()
    
    if stats is None:
        print("❌ Could not find hybrid_trading.log file")
        return False
    
    # Save to JSON file
    output_file = Path("live_trading_stats.json")
    with open(output_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"✅ Exported live trading stats:")
    print(f"   📊 Decisions: {stats['total_decisions']}")
    print(f"   💼 Trades: {stats['total_trades']}")
    print(f"   💰 Paper P&L: ${stats['paper_pnl']}")
    print(f"   📈 Win Rate: {stats['win_rate']:.1%}")
    print(f"   🎯 GMM Active: {stats['gmm_active']}")
    print(f"   💾 Saved to: {output_file}")
    
    return True

if __name__ == "__main__":
    export_stats()