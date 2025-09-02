#!/usr/bin/env python3
"""
Spin36TB Trading Dashboard - Clean Streamlit Cloud Entry Point
"""

import streamlit as st
import json
import os
from datetime import datetime

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
        # No data available
        st.error("❌ **NO LIVE DATA**")
        st.markdown("""
        **Possible reasons:**
        - Trading system is not running
        - Statistics file not found
        - File not committed to repository
        
        **To fix:**
        1. Make sure the trading system is running locally
        2. Run the export script to generate stats
        3. Commit and push the stats file to the repository
        """)
    
    # Footer
    st.markdown("---")
    st.caption(f"Dashboard rendered at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")

if __name__ == "__main__":
    main()