import streamlit as st
from datetime import datetime

st.set_page_config(page_title="Spin36TB Dashboard", page_icon="🚀", layout="wide")

# Header with system status and account info
col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    st.markdown("### 🟢 System Status")
    st.success("**ACTIVE**")

with col2:
    st.title("🚀 Spin36TB Trading Dashboard")

with col3:
    st.markdown("### 💰 Account Balance")
    st.info("**$24,994.54**")

st.markdown("---")

# Main trading metrics
st.markdown("## 📊 Overall Performance")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Decisions", "74")  
with col2:
    st.metric("Total Trades", "150")
with col3:
    st.metric("Paper P&L", "$-5.46", delta="-0.02%")
with col4:
    st.metric("Win Rate", "31.2%", delta="-18.8%", delta_color="inverse")

st.markdown("---")

# Coach-specific performance
st.markdown("## 🎯 Three-Coach System Performance")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🏃‍♂️ Momentum Coach")
    st.metric("Win Rate", "31.2%")
    st.metric("Trades", "138")
    st.metric("Avg Hold", "45 min")
    st.caption("High-frequency trading")

with col2:
    st.markdown("### 🎓 Academic Coach")
    st.metric("Win Rate", "67%", delta="+35.8%")
    st.metric("Trades", "12")
    st.metric("Avg Hold", "8.2 hours")
    st.caption("6-hour momentum strategy")

with col3:
    st.markdown("### 🧠 GMM Coach")
    st.metric("Win Rate", "Pending")
    st.metric("Trades", "0")
    st.metric("Patterns", "8 clusters")
    st.caption("Pattern recognition system")

st.markdown("---")

# Recent activity
st.markdown("## 📈 System Activity")
col1, col2 = st.columns(2)

with col1:
    st.info(f"**Last Update:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    st.success("✅ All coaches operational")

with col2:
    st.warning("**Project PRADO:** Ready to implement")
    st.caption("López de Prado enhancements pending")

# Footer
st.markdown("---")
st.caption("Spin36TB Multi-Coach Trading System v2.0 | EUR/USD Focus")