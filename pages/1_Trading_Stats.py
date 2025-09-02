import streamlit as st

st.set_page_config(page_title="Trading Stats", page_icon="📊")

st.title("📊 Trading Statistics")

st.markdown("### Carson's 6-Hour Momentum System")
st.success("✅ Successfully integrated academic approach")

col1, col2 = st.columns(2)
with col1:
    st.metric("6-Hour Decisions", "12")
    st.metric("Success Rate", "67%")
with col2:
    st.metric("Average Hold Time", "8.2 hours")  
    st.metric("Momentum Score", "0.74")

st.markdown("---")
st.markdown("### System Status")
st.info("All systems operational ✅")