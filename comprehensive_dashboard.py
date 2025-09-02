#!/usr/bin/env python3
"""
Comprehensive Spin36TB Live Trading Dashboard
All essential monitors in one real-time interface
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import time
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import subprocess

# Page config
st.set_page_config(
    page_title="🚀 Spin36TB Live Trading Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

class LiveTradingDashboard:
    def __init__(self):
        # OANDA Configuration
        self.api_token = "fdf72f4b5166602e39e001d1c1cb38b0-524c6662baa3187260b34c006eeab1fd"
        self.account_id = "101-001-31365224-001"
        self.api_url = 'https://api-fxpractice.oanda.com'
        self.headers = {
            'Authorization': f'Bearer {self.api_token}',
            'Content-Type': 'application/json'
        }
        
        # Initialize session state
        if 'trading_data' not in st.session_state:
            st.session_state.trading_data = {
                'prices': [],
                'trades': [],
                'decisions': [],
                'performance': {'capital': 25000, 'return': 0.0}
            }
    
    def load_live_stats(self):
        """Load actual trading statistics from JSON file"""
        try:
            with open('live_trading_stats.json', 'r') as f:
                return json.load(f)
        except:
            # Fallback to simulated data if file doesn't exist
            return None
    
    def get_live_market_data(self, candles=50):
        """Get live EURUSD data"""
        try:
            url = f"{self.api_url}/v3/instruments/EUR_USD/candles"
            params = {'granularity': 'M1', 'count': candles, 'price': 'MBA'}
            response = requests.get(url, headers=self.headers, params=params, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                candles_data = []
                
                for candle in data['candles'][-candles:]:
                    if candle['complete']:
                        candles_data.append({
                            'time': pd.to_datetime(candle['time']),
                            'open': float(candle['mid']['o']),
                            'high': float(candle['mid']['h']),
                            'low': float(candle['mid']['l']),
                            'close': float(candle['mid']['c']),
                            'volume': int(candle['volume'])
                        })
                
                return pd.DataFrame(candles_data)
            return pd.DataFrame()
        except:
            return pd.DataFrame()
    
    def get_account_info(self):
        """Get OANDA account information"""
        try:
            url = f"{self.api_url}/v3/accounts/{self.account_id}"
            response = requests.get(url, headers=self.headers, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                account = data['account']
                return {
                    'balance': float(account['balance']),
                    'currency': account['currency'],
                    'unrealized_pl': float(account.get('unrealizedPL', 0)),
                    'margin_used': float(account.get('marginUsed', 0)),
                    'margin_available': float(account.get('marginAvailable', 0))
                }
            return None
        except:
            return None
    
    def read_trading_log(self):
        """Read and parse trading log"""
        try:
            log_file = '/Users/jonspinogatti/Desktop/spin36TB/hybrid_trading.log'
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                return [line.strip() for line in lines if line.strip()]
            return []
        except:
            return []
    
    def parse_recent_activity(self, log_lines, max_entries=20):
        """Parse recent trading activity from logs"""
        activities = []
        for line in log_lines[-max_entries:]:
            if ' - INFO - ' in line:
                parts = line.split(' - INFO - ', 1)
                if len(parts) == 2:
                    timestamp_str = parts[0]
                    message = parts[1]
                    
                    try:
                        timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S,%f')
                    except:
                        timestamp = datetime.now()
                    
                    # Categorize activities
                    if 'PAPER TRADE' in message:
                        activity_type = '🔵 Trade Opened'
                    elif 'TRADE CLOSED' in message:
                        activity_type = '🟢 Trade Closed'
                    elif 'Hybrid System Decisions' in message:
                        activity_type = '🎯 Decision Made'
                    elif 'Starting live trading' in message:
                        activity_type = '🚀 Session Started'
                    else:
                        activity_type = 'ℹ️ System Info'
                    
                    activities.append({
                        'time': timestamp,
                        'type': activity_type,
                        'message': message
                    })
        
        return sorted(activities, key=lambda x: x['time'], reverse=True)
    
    def check_trading_process(self):
        """Check if trading process is running"""
        try:
            result = subprocess.run(['pgrep', '-f', 'live_paper_trader.py'], 
                                  capture_output=True, text=True)
            return len(result.stdout.strip()) > 0
        except:
            return False
    
    def load_session_results(self):
        """Load latest session results if available"""
        try:
            session_dir = '/Users/jonspinogatti/Desktop/spin36TB/'
            session_files = [f for f in os.listdir(session_dir) if f.startswith('session_') and f.endswith('.json')]
            
            if session_files:
                latest_session = sorted(session_files)[-1]
                with open(os.path.join(session_dir, latest_session), 'r') as f:
                    return json.load(f)
            return None
        except:
            return None
    
    def create_price_chart(self, df):
        """Create real-time price chart"""
        if df.empty:
            return go.Figure()
        
        fig = go.Figure()
        
        # Candlestick chart
        fig.add_trace(go.Candlestick(
            x=df['time'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name="EURUSD",
            increasing_line_color='#00ff88',
            decreasing_line_color='#ff4444'
        ))
        
        fig.update_layout(
            title="📈 Live EURUSD Price Action",
            yaxis_title="Price",
            xaxis_title="Time",
            template="plotly_dark",
            height=400,
            showlegend=False
        )
        
        return fig
    
    def create_performance_gauge(self, current_return):
        """Create performance gauge"""
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = current_return * 100,
            delta = {'reference': 0, 'valueformat': '.2f'},
            title = {'text': "Portfolio Return %"},
            gauge = {
                'axis': {'range': [None, 5]},
                'bar': {'color': "darkgreen" if current_return >= 0 else "darkred"},
                'steps': [
                    {'range': [0, 1], 'color': "lightgray"},
                    {'range': [1, 3], 'color': "gray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 2
                }
            }
        ))
        
        fig.update_layout(
            template="plotly_dark",
            height=300
        )
        
        return fig
    
    def run_dashboard(self):
        """Main dashboard interface"""
        
        # Header
        st.markdown("# 🚀 Spin36TB Live Trading Dashboard")
        st.markdown("### Real-time monitoring of your optimized dual-strategy system")
        
        # Load and display live stats
        live_stats = self.load_live_stats()
        if live_stats:
            st.success("🔴 LIVE DATA - Connected to hybrid trading system")
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("📊 Decisions", live_stats['total_decisions'])
            with col2:
                st.metric("💼 Trades", live_stats['total_trades'])
            with col3:
                st.metric("💰 Paper P&L", f"${live_stats['paper_pnl']:.2f}")
            with col4:
                st.metric("📈 Win Rate", f"{live_stats['win_rate']:.1%}")
            with col5:
                st.metric("🎯 GMM Active", "✅" if live_stats['gmm_active'] else "⏸️")
            
            st.markdown("---")
        else:
            st.warning("⚫ DEMO DATA - No live connection")
        
        # Auto-refresh
        placeholder = st.empty()
        
        with placeholder.container():
            # Check trading process status
            is_trading = self.check_trading_process()
            
            # Status indicator
            col1, col2, col3 = st.columns([1, 1, 2])
            
            with col1:
                if is_trading:
                    st.success("🟢 Trading System: ACTIVE")
                else:
                    st.error("🔴 Trading System: INACTIVE")
            
            with col2:
                current_time = datetime.now().strftime("%H:%M:%S")
                st.info(f"🕐 Current Time: {current_time}")
            
            with col3:
                account_info = self.get_account_info()
                if account_info:
                    st.metric("💰 Account Balance", f"${account_info['balance']:,.2f}", 
                             f"{account_info['unrealized_pl']:+.2f}")
                else:
                    st.warning("⚠️ Unable to connect to OANDA")
            
            # Decision Analysis Section
            st.markdown("## 🎯 Live Decision Analysis")
            
            # Get activities data
            log_lines = self.read_trading_log()
            activities = self.parse_recent_activity(log_lines)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Decision timeline
                st.markdown("**📊 Recent Trading Decisions:**")
                
                decision_activities = [a for a in activities if 'Decision Made' in a['type']]
                if decision_activities:
                    decision_data = []
                    for i, decision in enumerate(decision_activities[-10:], 1):
                        decision_data.append({
                            'Decision #': len(decision_activities) - len(decision_activities[-10:]) + i,
                            'Time': decision['time'].strftime('%H:%M:%S'),
                            'Actions': '2 strategies evaluated',
                            'Result': 'HOLD (waiting for quality signals)' if 'PAPER TRADE' not in str(activities) else 'TRADES EXECUTED'
                        })
                    
                    decision_df = pd.DataFrame(decision_data)
                    st.dataframe(decision_df, use_container_width=True, hide_index=True)
                    
                    # Decision frequency
                    if len(decision_activities) >= 2:
                        time_diff = (decision_activities[-1]['time'] - decision_activities[-2]['time']).total_seconds() / 60
                        st.success(f"✅ Decision Frequency: Every {time_diff:.1f} minutes (target: 5 minutes)")
                else:
                    st.info("No decisions recorded yet - system starting up...")
            
            with col2:
                # Current market conditions
                market_data = self.get_live_market_data()
                if not market_data.empty:
                    current_price = market_data['close'].iloc[-1]
                    price_change = current_price - market_data['close'].iloc[-2] if len(market_data) > 1 else 0
                    
                    st.metric("Current EURUSD", f"{current_price:.4f}", f"{price_change:+.4f}")
                    
                    # Market analysis for trading decisions
                    st.markdown("**🔍 Market Conditions:**")
                    
                    # Calculate current volatility
                    if len(market_data) > 10:
                        volatility = (market_data['high'] - market_data['low']).tail(10).mean() * 10000
                        st.text(f"10-min Volatility: {volatility:.1f} pips")
                        
                        # Price momentum
                        momentum = abs(market_data['close'].iloc[-1] - market_data['close'].iloc[-6]) / market_data['close'].iloc[-6] * 10000
                        st.text(f"5-min Momentum: {momentum:.1f} pips")
                        
                        # Trading conditions assessment
                        st.markdown("**⚖️  Trading Readiness:**")
                        if momentum > 6:  # Our 6-pip threshold
                            st.success("🟢 Momentum: Strong enough")
                        else:
                            st.warning(f"🟡 Momentum: {momentum:.1f} pips (need >6)")
                        
                        if volatility > 2:  # 2 pip minimum volatility
                            st.success("🟢 Volatility: Sufficient")
                        else:
                            st.warning(f"🟡 Volatility: {volatility:.1f} pips (need >2)")
                        
                        # Time filter check
                        current_hour = datetime.now().hour
                        if 7 <= current_hour <= 11 or 13 <= current_hour <= 17:
                            st.success("🟢 Time: Optimal hours")
                        elif 22 <= current_hour or current_hour <= 6:
                            st.error("🔴 Time: Asian session pause")
                        else:
                            st.warning("🟡 Time: Off-peak hours")
            
            # Why No Trades Section
            if not any('PAPER TRADE' in line for line in log_lines):
                st.markdown("## 🤔 Why No Trades Yet?")
                
                reasons_col1, reasons_col2 = st.columns(2)
                
                with reasons_col1:
                    st.markdown("**🎯 Our Quality Thresholds:**")
                    st.text("• Confidence: >67% (very high)")
                    st.text("• Momentum: >6 pips minimum")
                    st.text("• Volatility: >2 pips minimum")
                    st.text("• Time: Optimal hours only")
                    st.text("• Regime: No ranging markets")
                
                with reasons_col2:
                    st.markdown("**✅ This is GOOD Trading:**")
                    st.text("• System being selective")
                    st.text("• Avoiding poor setups") 
                    st.text("• Waiting for quality signals")
                    st.text("• Professional discipline")
                    st.text("• Better than over-trading")
            
            # Trading Activity
            st.markdown("## 🎯 Live Trading Activity")
            
            if activities:
                # Recent activity table
                activity_data = []
                for activity in activities[:10]:  # Last 10 activities
                    activity_data.append({
                        'Time': activity['time'].strftime('%H:%M:%S'),
                        'Type': activity['type'],
                        'Details': activity['message'][:80] + "..." if len(activity['message']) > 80 else activity['message']
                    })
                
                activity_df = pd.DataFrame(activity_data)
                st.dataframe(activity_df, use_container_width=True, hide_index=True)
                
                # Activity summary
                col1, col2, col3, col4 = st.columns(4)
                
                # Use live stats if available, otherwise parse from activities  
                if live_stats:
                    decision_count = live_stats['total_decisions']
                    trade_open_count = live_stats['trades_opened'] 
                    trade_close_count = live_stats['trades_closed']
                else:
                    decision_count = len([a for a in activities if 'Decision Made' in a['type']])
                    trade_open_count = len([a for a in activities if 'Trade Opened' in a['type']])
                    trade_close_count = len([a for a in activities if 'Trade Closed' in a['type']])
                
                col1.metric("🎯 Decisions Made", decision_count)
                col2.metric("🔵 Trades Opened", trade_open_count)
                col3.metric("🟢 Trades Closed", trade_close_count)
                col4.metric("📊 Active Trades", max(0, trade_open_count - trade_close_count))
            
            else:
                st.info("No trading activity detected yet. System may be starting up or waiting for optimal conditions.")
            
            # Performance Section
            st.markdown("## 📈 Performance Tracking")
            
            session_data = self.load_session_results()
            if session_data:
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    summary = session_data.get('session_summary', {})
                    current_return = summary.get('total_return', 0)
                    
                    gauge_chart = self.create_performance_gauge(current_return)
                    st.plotly_chart(gauge_chart, use_container_width=True)
                
                with col2:
                    st.markdown("**📊 Session Summary:**")
                    if summary:
                        st.text(f"Starting Capital: ${summary.get('starting_capital', 25000):,.2f}")
                        st.text(f"Current Capital: ${summary.get('final_capital', 25000):,.2f}")
                        st.text(f"Total Return: {summary.get('total_return', 0)*100:+.3f}%")
                        st.text(f"Total Trades: {summary.get('total_trades', 0)}")
                        st.text(f"Decisions Made: {summary.get('total_decisions', 0)}")
                    
                    # Strategy performance
                    strategy_perf = session_data.get('strategy_performance', {})
                    if strategy_perf:
                        st.markdown("**🎭 Strategy Performance:**")
                        momentum_wr = strategy_perf.get('momentum_wins', 0) / max(1, strategy_perf.get('momentum_trades', 1)) * 100
                        mr_wr = strategy_perf.get('mr_wins', 0) / max(1, strategy_perf.get('mr_trades', 1)) * 100
                        
                        st.text(f"Momentum: {momentum_wr:.1f}% win rate")
                        st.text(f"Mean Rev: {mr_wr:.1f}% win rate")
            
            # System Controls
            st.markdown("## 🎮 System Controls")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🔄 Refresh Data", type="primary"):
                    st.rerun()
            
            with col2:
                if st.button("📋 View Full Log"):
                    with st.expander("Complete Trading Log", expanded=True):
                        for line in log_lines[-50:]:  # Last 50 lines
                            st.code(line, language=None)
            
            with col3:
                if not is_trading:
                    if st.button("🚀 Start Trading", type="secondary"):
                        st.info("To start trading, run: `python live_paper_trader.py` in terminal")
        
        # Auto-refresh every 5 seconds
        time.sleep(5)
        st.rerun()

def main():
    """Main dashboard entry point"""
    dashboard = LiveTradingDashboard()
    dashboard.run_dashboard()

if __name__ == "__main__":
    main()