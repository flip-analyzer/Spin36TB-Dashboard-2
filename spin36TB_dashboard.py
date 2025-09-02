#!/usr/bin/env python3
"""
Spin36TB Dashboard - Main Entry Point
Cloud-optimized dashboard showing real trading statistics
"""

import streamlit as st
import json
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="🚀 Spin36TB Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_live_stats():
    """Load actual trading statistics from JSON file"""
    try:
        with open('live_trading_stats.json', 'r') as f:
            data = json.load(f)
            return data
    except Exception as e:
        st.error(f"Could not load live stats: {e}")
        return None

def main():
    """Main dashboard"""
    
    # Header
    st.title("🚀 Spin36TB Hybrid Trading System")
    st.markdown("### Real-time Performance Dashboard")
    
    # Load live trading statistics
    live_stats = load_live_stats()
    
    if live_stats:
        # SUCCESS BANNER
        st.success("🔴 **LIVE DATA CONNECTED** - Reading from hybrid trading system")
        
        # Main metrics in prominent display
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                label="📊 Total Decisions",
                value=live_stats['total_decisions'],
                help="Number of trading decisions made by the hybrid system"
            )
        
        with col2:
            st.metric(
                label="💼 Total Trades", 
                value=live_stats['total_trades'],
                help="Number of paper trades executed"
            )
        
        with col3:
            pnl = live_stats['paper_pnl']
            pnl_delta = "↗️ Profit" if pnl > 0 else "↘️ Loss" if pnl < 0 else "➖ Breakeven"
            st.metric(
                label="💰 Paper P&L",
                value=f"${pnl:.2f}",
                delta=pnl_delta,
                help="Total profit/loss from paper trading"
            )
        
        with col4:
            win_rate = live_stats['win_rate']
            st.metric(
                label="📈 Win Rate",
                value=f"{win_rate:.1%}",
                help="Percentage of winning trades"
            )
        
        with col5:
            gmm_status = "✅ Active" if live_stats.get('gmm_active', False) else "⏸️ Inactive"
            st.metric(
                label="🎯 GMM Clustering",
                value=gmm_status,
                help="Advanced clustering system status"
            )
        
        # Detailed breakdown
        st.markdown("---")
        st.markdown("## 📋 Trading Activity Breakdown")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Trade Statistics")
            st.write(f"**Trades Opened:** {live_stats['trades_opened']}")
            st.write(f"**Trades Closed:** {live_stats['trades_closed']}")
            st.write(f"**Winning Trades:** {live_stats['winning_trades']}")
            st.write(f"**Losing Trades:** {live_stats['losing_trades']}")
            st.write(f"**Active Trades:** {live_stats['trades_opened'] - live_stats['trades_closed']}")
        
        with col2:
            st.markdown("### 🕐 System Status")
            st.write(f"**System Active:** {'✅ Yes' if live_stats['system_active'] else '❌ No'}")
            st.write(f"**Last Updated:** {live_stats['last_updated'][:19].replace('T', ' ')}")
            st.write(f"**GMM Clustering:** {'✅ Active' if live_stats.get('gmm_active', False) else '⏸️ Inactive'}")
        
        # Latest signals
        if live_stats.get('latest_signals'):
            st.markdown("### 📡 Latest Trading Signals")
            signals_df = pd.DataFrame(live_stats['latest_signals'])
            st.dataframe(signals_df, use_container_width=True)
        
        # Performance chart
        if live_stats['trades_closed'] > 0:
            st.markdown("### 📈 Performance Overview")
            
            # Create simple performance metrics chart
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=['Win Rate', 'Total Trades', 'Closed Trades'],
                y=[live_stats['win_rate'] * 100, live_stats['total_trades'], live_stats['trades_closed']],
                name='Trading Metrics',
                marker_color=['green', 'blue', 'orange']
            ))
            
            fig.update_layout(
                title="Key Trading Metrics",
                yaxis_title="Count / Percentage",
                xaxis_title="Metric"
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        # ERROR STATE
        st.error("❌ **NO LIVE DATA** - Could not connect to trading system")
        st.write("This may indicate:")
        st.write("- Trading system is not running")
        st.write("- live_trading_stats.json file is missing")
        st.write("- File permissions issue")
        
        st.info("💡 **To fix this:** Run `python export_live_stats.py` locally and push the updated JSON file")

    # Footer
    st.markdown("---")
    st.caption(f"🤖 Dashboard last rendered: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")

if __name__ == "__main__":
    main()