import streamlit as st

# This is the multi-page app entry point
st.set_page_config(
    page_title="Spin36TB Dashboard",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 Spin36TB Trading Dashboard")
st.write("Dashboard is working! 🎉")

# Demo metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("📊 Total Decisions", "74")
with col2:  
    st.metric("💼 Total Trades", "150")
with col3:
    st.metric("💰 Paper P&L", "$-5.46")  
with col4:
    st.metric("📈 Win Rate", "31.2%")

st.success("✅ Carson's 6-hour momentum system integrated!")
st.info("✅ Multi-page app structure - this should work!")

st.markdown("---")
st.markdown("### Dashboard Status")
st.markdown("- ✅ Repository cleaned up")
st.markdown("- ✅ Multiple entry points available")
st.markdown("- ✅ Streamlit configuration added")
st.markdown("- ✅ Using Home.py multipage format")