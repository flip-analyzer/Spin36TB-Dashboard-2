#!/usr/bin/env python3
"""
Spin36TB Trading Dashboard - Streamlit Cloud Entry Point
"""

import streamlit as st
import json
import os
from datetime import datetime

# Test imports to help debug cloud issues
try:
    import pandas as pd
    import numpy as np
    DEPS_OK = True
except ImportError as e:
    DEPS_OK = False
    IMPORT_ERROR = str(e)

# Page configuration
st.set_page_config(
    page_title="🚀 Spin36TB Dashboard", 
    page_icon="📊",
    layout="wide"
)

def load_trading_stats():
    """Load trading statistics from JSON file"""
    try:
        # Try to load the stats file
        if os.path.exists('live_trading_stats.json'):
            with open('live_trading_stats.json', 'r') as f:
                return json.load(f), True
        else:
            return None, False
    except Exception as e:
        st.error(f"Error loading stats: {e}")
        return None, False

def main():
    """Main dashboard function"""
    
    # Header
    st.title("🚀 Spin36TB Trading Dashboard")
    st.markdown("### Live Trading Performance")
    
    # Debug information for cloud deployment
    if not DEPS_OK:
        st.error(f"❌ **DEPENDENCY ERROR**: {IMPORT_ERROR}")
        st.stop()
    
    # Show environment info for debugging
    with st.expander("🔧 Debug Info"):
        st.write(f"**Python Version**: {st.__version__}")
        st.write(f"**Working Directory**: {os.getcwd()}")
        st.write(f"**Files in Directory**: {os.listdir('.')}")
        st.write(f"**Stats File Exists**: {os.path.exists('live_trading_stats.json')}")
    
    # Load statistics
    stats, success = load_trading_stats()
    
    if success and stats:
        # Show success indicator
        st.success("🔴 **LIVE DATA CONNECTED** - System is running")
        
        # Main metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Decisions", stats.get('total_decisions', 0))
        
        with col2:
            st.metric("💼 Trades", stats.get('total_trades', 0))
            
        with col3:
            pnl = stats.get('paper_pnl', 0)
            st.metric("💰 P&L", f"${pnl:.2f}")
            
        with col4:
            win_rate = stats.get('win_rate', 0)
            st.metric("📈 Win Rate", f"{win_rate:.1%}")
        
        # Detailed information
        st.markdown("---")
        st.markdown("## 📋 System Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Trades Opened:** {stats.get('trades_opened', 0)}")
            st.write(f"**Trades Closed:** {stats.get('trades_closed', 0)}")
            st.write(f"**Active Trades:** {stats.get('trades_opened', 0) - stats.get('trades_closed', 0)}")
            
        with col2:
            st.write(f"**Winning Trades:** {stats.get('winning_trades', 0)}")
            st.write(f"**Losing Trades:** {stats.get('losing_trades', 0)}")
            st.write(f"**System Status:** {'✅ Active' if stats.get('system_active', False) else '❌ Inactive'}")
        
        # Recent signals
        if 'latest_signals' in stats and stats['latest_signals']:
            st.markdown("## 📡 Latest Signals")
            
            # Show last 5 signals
            for signal in stats['latest_signals'][:5]:
                direction = signal.get('direction', 'N/A')
                confidence = signal.get('confidence', 0)
                signal_type = signal.get('type', 'Unknown')
                
                direction_emoji = "🔴" if direction == "DOWN" else "🟢" if direction == "UP" else "⚪"
                st.write(f"{direction_emoji} {signal_type}: {direction} ({confidence:.1f}% confidence)")
        
        # Show timestamp
        last_update = stats.get('last_updated', 'Unknown')
        st.caption(f"Last updated: {last_update}")
        
    else:
        # No data available - show sample/demo data
        st.warning("⚠️ **USING DEMO DATA** - Live stats file not found")
        
        # Demo data based on last known stats
        demo_stats = {
            "total_decisions": 74,
            "total_trades": 150, 
            "trades_opened": 150,
            "trades_closed": 77,
            "paper_pnl": -5.46,
            "winning_trades": 24,
            "losing_trades": 31,
            "win_rate": 0.312,
            "system_active": True,
            "latest_signals": [
                {"type": "momentum_standard", "direction": "DOWN", "confidence": 14.82},
                {"type": "momentum_standard", "direction": "DOWN", "confidence": 12.34},
                {"type": "6hour_momentum", "direction": "UP", "confidence": 65.44},
                {"type": "cluster_patterns", "direction": "UP", "confidence": 85.00},
                {"type": "hybrid_confluence", "direction": "UP", "confidence": 73.06}
            ]
        }
        
        # Show demo metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Decisions", demo_stats['total_decisions'])
        
        with col2:
            st.metric("💼 Trades", demo_stats['total_trades'])
            
        with col3:
            st.metric("💰 P&L", f"${demo_stats['paper_pnl']:.2f}")
            
        with col4:
            st.metric("📈 Win Rate", f"{demo_stats['win_rate']:.1%}")
        
        st.markdown("---")
        st.markdown("## 📋 System Details (Demo)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Trades Opened:** {demo_stats['trades_opened']}")
            st.write(f"**Trades Closed:** {demo_stats['trades_closed']}")
            st.write(f"**Active Trades:** {demo_stats['trades_opened'] - demo_stats['trades_closed']}")
            
        with col2:
            st.write(f"**Winning Trades:** {demo_stats['winning_trades']}")
            st.write(f"**Losing Trades:** {demo_stats['losing_trades']}")
            st.write(f"**System Status:** {'✅ Active' if demo_stats['system_active'] else '❌ Inactive'}")
        
        # Show demo signals
        st.markdown("## 📡 Latest Signals (Demo)")
        for signal in demo_stats['latest_signals']:
            direction = signal['direction']
            confidence = signal['confidence']
            signal_type = signal['type']
            
            direction_emoji = "🔴" if direction == "DOWN" else "🟢" if direction == "UP" else "⚪"
            st.write(f"{direction_emoji} {signal_type}: {direction} ({confidence:.1f}% confidence)")
        
        st.info("""
        📝 **Note**: This is demo data. For live data:
        1. Ensure the trading system is running locally
        2. Run the export script to generate live_trading_stats.json
        3. Commit and push the stats file to the repository
        """)
        
        stats = demo_stats  # Use demo stats for timestamp display
    
    # Footer
    st.markdown("---")
    st.caption(f"Dashboard rendered at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")

if __name__ == "__main__":
    main()