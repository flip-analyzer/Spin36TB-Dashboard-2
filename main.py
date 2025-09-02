import streamlit as st
import json

st.set_page_config(page_title="Spin36TB Dashboard", layout="wide")

st.title("🚀 Spin36TB Trading Dashboard")

try:
    with open('live_trading_stats.json', 'r') as f:
        stats = json.load(f)
    
    st.success("🔴 LIVE DATA CONNECTED")
    
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("📊 Decisions", stats['total_decisions'])
    col2.metric("💼 Trades", stats['total_trades']) 
    col3.metric("💰 P&L", f"${stats['paper_pnl']:.2f}")
    col4.metric("📈 Win Rate", f"{stats['win_rate']:.1%}")
    
    st.json(stats)
    
except Exception as e:
    st.error("❌ NO LIVE DATA")
    st.write(f"Error: {e}")
    st.write("Using demo data instead")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📊 Decisions", "Demo")
    col2.metric("💼 Trades", "Demo") 
    col3.metric("💰 P&L", "Demo")
    col4.metric("📈 Win Rate", "Demo")