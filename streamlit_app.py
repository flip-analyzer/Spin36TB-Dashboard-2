import streamlit as st

st.set_page_config(page_title="Spin36TB Dashboard", page_icon="🚀")

st.title("🚀 Spin36TB Trading Dashboard")
st.write("Dashboard is working! 🎉")

# Minimal demo data
st.metric("📊 Total Decisions", "74")  
st.metric("💼 Total Trades", "150")
st.metric("💰 Paper P&L", "$-5.46")
st.metric("📈 Win Rate", "31.2%")

st.success("✅ Carson's 6-hour momentum system integrated!")
st.info("This is a simplified version to test Streamlit Cloud deployment")