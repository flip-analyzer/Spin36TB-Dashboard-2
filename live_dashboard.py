#!/usr/bin/env python3
"""
Live Spin36TB PRADO Trading Dashboard

Reads real-time data from the automated trading daemon
"""

import streamlit as st
import json
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="Spin36TB LIVE PRADO Dashboard", 
    page_icon="🚀", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

@st.cache_data(ttl=5)  # Cache for 5 seconds
def load_dashboard_data():
    """Load live data from daemon"""
    try:
        dashboard_file = Path("dashboard_data.json")
        if dashboard_file.exists():
            with open(dashboard_file, 'r') as f:
                data = json.load(f)
            return data, None
        else:
            return None, "Dashboard data file not found. Is the auto trading daemon running?"
    except Exception as e:
        return None, f"Error loading data: {e}"

@st.cache_data(ttl=10)
def load_trade_history():
    """Load trade history"""
    try:
        trade_file = Path("trade_summary.json")
        if trade_file.exists():
            with open(trade_file, 'r') as f:
                data = json.load(f)
            return data.get('trade_history', [])
        return []
    except:
        return []

def format_currency(amount):
    """Format currency display"""
    return f"${amount:,.2f}"

def format_percentage(pct):
    """Format percentage display"""
    return f"{pct:+.2f}%"

def format_pips(pips):
    """Format pips display"""
    return f"{pips:+.1f}p"

def main():
    # Auto-refresh every 30 seconds
    st_autorefresh = st.empty()
    with st_autorefresh.container():
        st.markdown(
            '<meta http-equiv="refresh" content="30">',
            unsafe_allow_html=True
        )
    
    # Load live data
    data, error = load_dashboard_data()
    
    if error:
        st.error(f"🚨 {error}")
        st.info("💡 Start the auto trading daemon: `./start_auto_trading.sh start`")
        return
    
    if not data:
        st.warning("⏳ No data available yet. Auto trading daemon may be starting up...")
        return
    
    # Header with live status
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        status = data.get('system_status', 'UNKNOWN')
        if status == 'ACTIVE':
            st.success("**🟢 SYSTEM ACTIVE**")
        else:
            st.error("**🔴 SYSTEM STOPPED**")
    
    with col2:
        st.title("🚀 Spin36TB LIVE PRADO Dashboard")
    
    with col3:
        balance = data.get('account_balance', 0)
        st.metric("💰 Account Balance", format_currency(balance))
    
    st.markdown("---")
    
    # Last update info
    last_update = data.get('last_update', 'Unknown')
    st.caption(f"🔄 Last Updated: {last_update}")
    
    # PRADO Enhancement Status
    st.markdown("## 🎯 Project PRADO Status")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        accuracy = data.get('meta_labeling_accuracy', 0) * 100
        st.metric("Meta-Labeling", "ACTIVE", delta=f"{accuracy:.0f}% accuracy")
    
    with col2:
        st.metric("Triple Barriers", "ACTIVE", delta="Dynamic exits")
    
    with col3:
        filtered = data.get('prado_filtered', 0)
        enhanced = data.get('prado_enhanced', 0)
        filter_pct = (filtered / max(filtered + enhanced, 1)) * 100
        st.metric("Trades Filtered", str(filtered), delta=f"{filter_pct:.1f}%", delta_color="normal")
    
    with col4:
        prado_win_rate = data.get('prado_win_rate', 0) * 100
        st.metric("PRADO Win Rate", f"{prado_win_rate:.1f}%", delta="+enhanced")
    
    st.markdown("---")
    
    # Main trading metrics
    st.markdown("## 📊 Live Performance")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_trades = data.get('total_trades', 0)
        enhanced = data.get('prado_enhanced', 0)
        st.metric("Total Decisions", str(total_trades), delta=f"+{enhanced} enhanced")
    
    with col2:
        winning_trades = data.get('winning_trades', 0)
        st.metric("Winning Trades", str(winning_trades))
    
    with col3:
        total_return = data.get('total_return_pct', 0)
        starting_capital = 25000
        pnl = (total_return / 100) * starting_capital
        st.metric("Paper P&L", format_currency(pnl), delta=format_percentage(total_return), delta_color="normal")
    
    with col4:
        win_rate = data.get('win_rate', 0) * 100
        st.metric("Overall Win Rate", f"{win_rate:.1f}%", delta="Live tracking")
    
    st.markdown("---")
    
    # Performance by Coach Type
    st.markdown("## 🎯 Performance by Strategy")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        momentum_trades = data.get('momentum_trades', 0)
        st.metric("🏃‍♂️ Momentum", str(momentum_trades), delta="High-freq")
    
    with col2:
        academic_trades = data.get('academic_trades', 0)
        st.metric("🎓 Academic", str(academic_trades), delta="6-hour")
    
    with col3:
        traditional_trades = data.get('traditional_trades', 0)
        st.metric("📊 Portfolio", str(traditional_trades), delta="Risk-managed")
    
    with col4:
        gmm_trades = data.get('gmm_trades', 0)
        st.metric("🧠 GMM", str(gmm_trades), delta="ML-based")
    
    # Active Trades Section
    active_trades = data.get('active_trades', [])
    if active_trades:
        st.markdown("## 📈 Active Trades")
        
        active_df = pd.DataFrame(active_trades)
        
        # Display active trades table
        display_cols = ['trade_id', 'coach', 'direction', 'entry_price', 'entry_time', 
                       'profit_target', 'stop_loss', 'combined_confidence']
        
        if not active_df.empty:
            for col in display_cols:
                if col not in active_df.columns:
                    active_df[col] = 'N/A'
            
            active_display = active_df[display_cols].copy()
            active_display['entry_time'] = pd.to_datetime(active_display['entry_time']).dt.strftime('%H:%M:%S')
            active_display['combined_confidence'] = active_display['combined_confidence'].apply(lambda x: f"{float(x)*100:.1f}%" if str(x) != 'N/A' else 'N/A')
            
            st.dataframe(active_display, use_container_width=True)
        else:
            st.info("No active trades currently")
    
    else:
        st.markdown("## 📈 Active Trades")
        active_count = data.get('active_trades_count', 0)
        if active_count > 0:
            st.info(f"📊 {active_count} active trades (details loading...)")
        else:
            st.info("💤 No active trades currently")
    
    # Recent Trades Section
    recent_trades = data.get('recent_trades', [])
    if recent_trades:
        st.markdown("## 📋 Recent Trades (Last 10)")
        
        recent_df = pd.DataFrame(recent_trades)
        
        # Format for display
        display_cols = ['trade_id', 'coach', 'direction', 'pips', 'success', 
                       'exit_reason', 'hold_duration_minutes']
        
        if not recent_df.empty:
            for col in display_cols:
                if col not in recent_df.columns:
                    recent_df[col] = 'N/A'
            
            recent_display = recent_df[display_cols].copy()
            recent_display['pips'] = recent_display['pips'].apply(lambda x: format_pips(float(x)) if str(x) != 'N/A' else 'N/A')
            recent_display['success'] = recent_display['success'].apply(lambda x: '✅' if x else '❌')
            recent_display['hold_duration_minutes'] = recent_display['hold_duration_minutes'].apply(lambda x: f"{float(x):.0f}m" if str(x) != 'N/A' else 'N/A')
            
            st.dataframe(recent_display, use_container_width=True)
            
            # Quick stats on recent trades
            if len(recent_trades) > 0:
                recent_pips = sum([t.get('pips', 0) for t in recent_trades])
                recent_wins = len([t for t in recent_trades if t.get('success', False)])
                recent_win_rate = recent_wins / len(recent_trades) * 100
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Recent Pips", format_pips(recent_pips))
                with col2:
                    st.metric("Recent Win Rate", f"{recent_win_rate:.1f}%")
                with col3:
                    st.metric("Recent Trades", f"{recent_wins}/{len(recent_trades)}")
    
    # Performance Chart (if we have trade history)
    trade_history = load_trade_history()
    if trade_history:
        st.markdown("## 📈 Equity Curve")
        
        # Create equity curve
        running_pnl = 0
        equity_data = []
        
        for trade in trade_history:
            if 'pnl' in trade and 'exit_time' in trade:
                running_pnl += trade.get('pnl', 0)
                equity_data.append({
                    'time': trade['exit_time'],
                    'equity': 25000 + running_pnl,  # Starting capital + P&L
                    'trade_pnl': trade.get('pnl', 0)
                })
        
        if equity_data:
            df = pd.DataFrame(equity_data)
            df['time'] = pd.to_datetime(df['time'])
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df['time'],
                y=df['equity'],
                mode='lines',
                name='Account Equity',
                line=dict(color='#00ff00' if df['equity'].iloc[-1] > 25000 else '#ff0000')
            ))
            
            fig.update_layout(
                title="Account Equity Over Time",
                xaxis_title="Time",
                yaxis_title="Account Value ($)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "🤖 **Auto PRADO Trading Daemon** | "
        "🧠 **Adaptive Learning Active** | "
        f"🔄 **Auto-refresh every 30s**"
    )

if __name__ == "__main__":
    main()