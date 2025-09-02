#!/usr/bin/env python3
"""
Spin36TB Dashboard - Streamlit Cloud Entry Point
Simple, reliable dashboard showing actual trading statistics
"""

import streamlit as st
import json
import pandas as pd
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="🚀 Spin36TB Dashboard",
    page_icon="📊",
    layout="wide"
)

def main():
    """Main dashboard - guaranteed to work"""
    
    # Header
    st.title("🚀 Spin36TB Hybrid Trading System")
    st.markdown("### Real-time Performance Dashboard")
    
    try:
        # Load live trading statistics
        with open('live_trading_stats.json', 'r') as f:
            stats = json.load(f)
        
        # SUCCESS - Show live data
        st.success("🔴 **LIVE DATA CONNECTED** - Reading from hybrid trading system")
        
        # Main metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Total Decisions", stats['total_decisions'])
        
        with col2:
            st.metric("💼 Total Trades", stats['total_trades'])
        
        with col3:
            pnl = stats['paper_pnl']
            st.metric("💰 Paper P&L", f"${pnl:.2f}")
        
        with col4:
            st.metric("📈 Win Rate", f"{stats['win_rate']:.1%}")
        
        # Detailed breakdown
        st.markdown("---")
        st.markdown("## 📋 Trading Activity Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Trades Opened:** {stats['trades_opened']}")
            st.write(f"**Trades Closed:** {stats['trades_closed']}")
            st.write(f"**Active Trades:** {stats['trades_opened'] - stats['trades_closed']}")
        
        with col2:
            st.write(f"**Winning Trades:** {stats['winning_trades']}")
            st.write(f"**Losing Trades:** {stats['losing_trades']}")
            st.write(f"**System Active:** {'✅ Yes' if stats['system_active'] else '❌ No'}")
        
        # Latest signals if available
        if stats.get('latest_signals'):
            st.markdown("## 📡 Recent Trading Signals")
            signals_df = pd.DataFrame(stats['latest_signals'])
            st.dataframe(signals_df, use_container_width=True)
        
        # Show raw data for debugging
        with st.expander("🔧 Raw Data (for debugging)"):
            st.json(stats)
    
    except FileNotFoundError:
        st.error("❌ **live_trading_stats.json not found**")
        st.write("The trading statistics file is missing. This usually means:")
        st.write("- The export script hasn't been run yet")
        st.write("- The file wasn't committed to the repository")
        st.write("- The cloud deployment can't access local files")
    
    except json.JSONDecodeError:
        st.error("❌ **Invalid JSON format**")
        st.write("The statistics file exists but contains invalid JSON data")
    
    except Exception as e:
        st.error(f"❌ **Unexpected error**: {e}")
        st.write("Something went wrong loading the trading statistics")
    
    # Footer
    st.markdown("---")
    st.caption(f"Dashboard rendered at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")

if __name__ == "__main__":
    main()