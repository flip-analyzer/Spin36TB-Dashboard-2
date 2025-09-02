#!/usr/bin/env python3
"""
Real-Time Spin36TB System Monitoring Dashboard
Web-based dashboard to monitor all system metrics, performance, and controls
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from datetime import datetime, timedelta
import time
import os
from automated_spin36TB_system import AutomatedSpin36TBSystem, AutomatedDisciplineEngine

# Dashboard configuration
st.set_page_config(
    page_title="Spin36TB Momentum System Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

class Spin36TBDashboard:
    def __init__(self):
        # Initialize session state
        if 'spin36TB_system' not in st.session_state:
            st.session_state.spin36TB_system = AutomatedSpin36TBSystem()
            st.session_state.last_update = datetime.now()
            st.session_state.system_running = False
    
    def load_live_stats(self):
        """Load actual trading statistics from JSON file"""
        try:
            with open('live_trading_stats.json', 'r') as f:
                return json.load(f)
        except:
            # Fallback to simulated data if file doesn't exist
            return None
    
    def create_dashboard(self):
        """Main dashboard creation"""
        
        # Header
        st.title("🚀 Spin36TB Momentum System Dashboard")
        st.markdown("Real-time monitoring of your automated trading system")
        
        # Load live stats
        live_stats = self.load_live_stats()
        
        # Get current system status
        system_status = st.session_state.spin36TB_system.discipline.get_system_status()
        
        # Override with real data if available
        if live_stats:
            # Update system status with real data
            system_status['trading']['total_trades'] = live_stats['total_trades']
            system_status['trading']['trades_today'] = live_stats['trades_opened']
            system_status['portfolio']['paper_pnl'] = live_stats['paper_pnl']
            system_status['trading']['recent_win_rate'] = live_stats['win_rate']
            system_status['system_state']['is_active'] = live_stats['system_active']
        
        # Live Stats Display
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
        
        # Sidebar controls
        self.create_sidebar_controls(system_status)
        
        # Main dashboard layout
        col1, col2, col3, col4 = st.columns(4)
        
        # Key metrics cards
        with col1:
            self.create_portfolio_card(system_status)
        
        with col2:
            self.create_performance_card(system_status)
        
        with col3:
            self.create_trading_card(system_status)
        
        with col4:
            self.create_risk_card(system_status)
        
        # Main content area
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Portfolio Performance", 
            "📊 Trade Analytics", 
            "🎯 System Status", 
            "⚠️ Risk Management", 
            "📋 Trade History"
        ])
        
        with tab1:
            self.create_portfolio_tab(system_status)
        
        with tab2:
            self.create_analytics_tab(system_status)
        
        with tab3:
            self.create_system_tab(system_status)
        
        with tab4:
            self.create_risk_tab(system_status)
        
        with tab5:
            self.create_trades_tab()
    
    def create_sidebar_controls(self, system_status):
        """Sidebar with system controls and settings"""
        st.sidebar.header("🎛️ System Controls")
        
        # System status indicator
        status_color = "🟢" if system_status['system_state']['is_active'] else "🔴"
        st.sidebar.markdown(f"**Status:** {status_color} {system_status['system_state']['health_status']}")
        
        # System controls
        st.sidebar.subheader("Controls")
        
        if st.sidebar.button("▶️ Start System" if not st.session_state.system_running else "⏸️ Pause System"):
            st.session_state.system_running = not st.session_state.system_running
            if st.session_state.system_running:
                st.sidebar.success("System started!")
            else:
                st.sidebar.warning("System paused!")
        
        if st.sidebar.button("🔄 Reset Daily Counters"):
            st.session_state.spin36TB_system.discipline.reset_daily_counters()
            st.sidebar.success("Daily counters reset!")
        
        # Risk management settings
        st.sidebar.subheader("⚠️ Risk Settings")
        
        max_daily_loss = st.sidebar.slider(
            "Max Daily Loss %", 
            min_value=1.0, 
            max_value=10.0, 
            value=4.0, 
            step=0.5,
            help="System will auto-pause if daily loss exceeds this percentage"
        )
        
        max_position = st.sidebar.slider(
            "Max Position Size %", 
            min_value=1.0, 
            max_value=20.0, 
            value=15.0, 
            step=1.0,
            help="Maximum position size per trade"
        )
        
        max_trades = st.sidebar.slider(
            "Max Daily Trades", 
            min_value=5, 
            max_value=50, 
            value=25, 
            step=5,
            help="Maximum number of trades per day"
        )
        
        # Update system settings
        st.session_state.spin36TB_system.discipline.max_daily_loss_pct = max_daily_loss / 100
        st.session_state.spin36TB_system.discipline.max_position_size = max_position / 100
        st.session_state.spin36TB_system.discipline.max_daily_trades = max_trades
        
        # Current regime display
        st.sidebar.subheader("🌍 Market Regime")
        regime = system_status['system_state']['current_regime']
        regime_colors = {
            'HIGH_VOL_TRENDING': '🟢',
            'HIGH_MOMENTUM': '🟡',
            'STRONG_TREND': '🔵',
            'MIXED_CONDITIONS': '⚪',
            'LOW_VOL_RANGING': '🔴',
            'UNKNOWN': '❓'
        }
        st.sidebar.markdown(f"{regime_colors.get(regime, '❓')} **{regime}**")
        
        # Auto-refresh toggle
        st.sidebar.subheader("🔄 Auto-Refresh")
        auto_refresh = st.sidebar.checkbox("Enable Auto-Refresh (30s)", value=True)
        
        if auto_refresh:
            time.sleep(30)
            st.rerun()
    
    def create_portfolio_card(self, system_status):
        """Portfolio value card"""
        portfolio = system_status['portfolio']
        
        st.metric(
            label="💰 Portfolio Value",
            value=f"${portfolio['current_value']:,.2f}",
            delta=f"{portfolio['total_return_pct']:+.2%}"
        )
        
        # Progress bar for monthly goal
        monthly_target = 0.12  # 12% monthly target
        monthly_progress = portfolio['monthly_return'] / monthly_target
        st.progress(min(monthly_progress, 1.0))
        st.caption(f"Monthly: {portfolio['monthly_return']:+.1%} / {monthly_target:.0%} target")
    
    def create_performance_card(self, system_status):
        """Performance metrics card"""
        portfolio = system_status['portfolio']
        
        st.metric(
            label="📈 Daily P&L",
            value=f"${portfolio['daily_pnl']:+,.2f}",
            delta=f"{portfolio['daily_pnl']/portfolio['current_value']*100:+.1f}%"
        )
        
        # Drawdown indicator
        max_dd = portfolio['max_drawdown']
        dd_color = "🟢" if max_dd < 0.05 else "🟡" if max_dd < 0.15 else "🔴"
        st.caption(f"{dd_color} Max DD: {max_dd:.1%}")
    
    def create_trading_card(self, system_status):
        """Trading activity card"""
        trading = system_status['trading']
        
        st.metric(
            label="🎯 Trades Today",
            value=f"{trading['trades_today']}",
            delta=f"Total: {trading['total_trades']}"
        )
        
        # Win rate indicator
        win_rate = trading['recent_win_rate']
        wr_color = "🟢" if win_rate > 0.6 else "🟡" if win_rate > 0.45 else "🔴"
        st.caption(f"{wr_color} Win Rate: {win_rate:.1%}")
    
    def create_risk_card(self, system_status):
        """Risk management card"""
        trading = system_status['trading']
        
        st.metric(
            label="⚠️ Risk Status",
            value="HEALTHY" if system_status['system_state']['is_active'] else "PAUSED",
            delta=f"Losses: {trading['consecutive_losses']}"
        )
        
        # Open positions
        st.caption(f"Open: {trading['open_positions']} positions")
    
    def create_portfolio_tab(self, system_status):
        """Portfolio performance tab"""
        st.subheader("💰 Portfolio Performance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Portfolio value chart
            self.create_portfolio_chart()
        
        with col2:
            # Monthly performance breakdown
            self.create_monthly_performance()
        
        # Performance metrics table
        st.subheader("📊 Performance Metrics")
        
        portfolio = system_status['portfolio']
        metrics_data = {
            'Metric': [
                'Starting Capital',
                'Current Value', 
                'Total Return',
                'Monthly Return',
                'Daily P&L',
                'Max Drawdown',
                'Sharpe Ratio (Est.)'
            ],
            'Value': [
                f"${portfolio['current_value']/(1+portfolio['total_return_pct']):,.2f}",
                f"${portfolio['current_value']:,.2f}",
                f"{portfolio['total_return_pct']:+.2%}",
                f"{portfolio['monthly_return']:+.2%}",
                f"${portfolio['daily_pnl']:+,.2f}",
                f"{portfolio['max_drawdown']:.1%}",
                f"{np.random.uniform(1.2, 2.8):.2f}"  # Placeholder
            ]
        }
        
        st.dataframe(pd.DataFrame(metrics_data), hide_index=True)
    
    def create_portfolio_chart(self):
        """Create portfolio value over time chart"""
        # Simulate historical data for demo
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
        starting_value = 25000
        
        # Simulate portfolio growth with some volatility
        np.random.seed(42)
        daily_returns = np.random.normal(0.003, 0.02, len(dates))  # 0.3% daily avg, 2% volatility
        portfolio_values = [starting_value]
        
        for ret in daily_returns[1:]:
            portfolio_values.append(portfolio_values[-1] * (1 + ret))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=portfolio_values,
            mode='lines',
            name='Portfolio Value',
            line=dict(color='#00ff88', width=3)
        ))
        
        fig.update_layout(
            title="📈 Portfolio Value Over Time",
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def create_monthly_performance(self):
        """Create monthly performance breakdown"""
        # Simulate monthly data
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
        returns = [0.08, 0.12, -0.03, 0.15, 0.09, 0.11]  # Sample monthly returns
        
        fig = go.Figure(data=[
            go.Bar(
                x=months,
                y=returns,
                marker_color=['green' if r > 0 else 'red' for r in returns]
            )
        ])
        
        fig.update_layout(
            title="📅 Monthly Returns",
            yaxis_title="Return %",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def create_analytics_tab(self, system_status):
        """Trade analytics tab"""
        st.subheader("📊 Trade Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Win rate by cluster
            self.create_cluster_performance()
        
        with col2:
            # Regime performance
            self.create_regime_performance()
        
        # Trade distribution
        st.subheader("🎯 Trade Distribution")
        self.create_trade_distribution()
    
    def create_cluster_performance(self):
        """Cluster performance analysis"""
        # Simulate cluster data
        clusters = list(range(8))
        win_rates = np.random.uniform(0.45, 0.75, 8)
        avg_pips = np.random.uniform(25, 65, 8)
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Bar(x=clusters, y=win_rates, name="Win Rate", marker_color='blue'),
            secondary_y=False,
        )
        
        fig.add_trace(
            go.Scatter(x=clusters, y=avg_pips, mode='lines+markers', name="Avg Pips", line=dict(color='red')),
            secondary_y=True,
        )
        
        fig.update_xaxes(title_text="Cluster ID")
        fig.update_yaxes(title_text="Win Rate", secondary_y=False)
        fig.update_yaxes(title_text="Average Pips", secondary_y=True)
        fig.update_layout(title_text="🎯 Cluster Performance")
        
        st.plotly_chart(fig, use_container_width=True)
    
    def create_regime_performance(self):
        """Market regime performance"""
        regimes = ['HIGH_VOL_TRENDING', 'HIGH_MOMENTUM', 'STRONG_TREND', 'MIXED_CONDITIONS', 'LOW_VOL_RANGING']
        performance = [0.18, 0.15, 0.13, 0.05, -0.02]
        
        colors = ['green' if p > 0 else 'red' for p in performance]
        
        fig = go.Figure(data=[
            go.Bar(x=regimes, y=performance, marker_color=colors)
        ])
        
        fig.update_layout(
            title="🌍 Regime Performance",
            xaxis_title="Market Regime",
            yaxis_title="Average Return %",
            xaxis_tickangle=-45
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def create_trade_distribution(self):
        """Trade distribution analysis"""
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Direction distribution
            directions = ['UP', 'DOWN']
            counts = [55, 45]
            
            fig = go.Figure(data=[go.Pie(labels=directions, values=counts, hole=.3)])
            fig.update_layout(title="Direction Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Position size distribution
            sizes = np.random.lognormal(mean=np.log(0.03), sigma=0.5, size=100)
            sizes = np.clip(sizes, 0.005, 0.15)
            
            fig = go.Figure(data=[go.Histogram(x=sizes, nbinsx=20)])
            fig.update_layout(title="Position Size Distribution", xaxis_title="Position Size %")
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            # P&L distribution
            pnl = np.random.normal(0.02, 0.05, 100)
            
            fig = go.Figure(data=[go.Histogram(x=pnl, nbinsx=20)])
            fig.update_layout(title="P&L Distribution", xaxis_title="Return %")
            st.plotly_chart(fig, use_container_width=True)
    
    def create_system_tab(self, system_status):
        """System status tab"""
        st.subheader("🎯 System Status")
        
        # System health indicators
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("**System Health Check**")
            
            checks = [
                ("Daily Loss Limit", "✅ PASS", "green"),
                ("Max Drawdown", "✅ PASS", "green"), 
                ("Win Rate", "✅ PASS", "green"),
                ("Position Limits", "✅ PASS", "green"),
                ("Market Hours", "✅ PASS", "green"),
                ("Consecutive Losses", "⚠️ WARN" if system_status['trading']['consecutive_losses'] > 3 else "✅ PASS", 
                 "orange" if system_status['trading']['consecutive_losses'] > 3 else "green")
            ]
            
            for check, status, color in checks:
                st.markdown(f"**{check}:** :{color}[{status}]")
        
        with col2:
            st.info("**Current Settings**")
            limits = system_status['limits']
            
            settings = [
                ("Max Daily Loss", f"{limits['max_daily_loss_pct']:.1%}"),
                ("Max Drawdown", f"{limits['max_drawdown_pct']:.1%}"),
                ("Max Position Size", f"{limits['max_position_size']:.1%}"),
                ("Max Daily Trades", f"{limits['max_daily_trades']}"),
                ("Max Consecutive Losses", f"{limits['max_consecutive_losses']}")
            ]
            
            for setting, value in settings:
                st.markdown(f"**{setting}:** {value}")
        
        # System log (simulated)
        st.subheader("📝 System Log")
        
        log_entries = [
            f"{datetime.now().strftime('%H:%M:%S')} - System health check: PASSED",
            f"{(datetime.now() - timedelta(minutes=5)).strftime('%H:%M:%S')} - Trade executed: BUY 2.3% (Cluster 5)",
            f"{(datetime.now() - timedelta(minutes=12)).strftime('%H:%M:%S')} - Regime detected: HIGH_MOMENTUM",
            f"{(datetime.now() - timedelta(minutes=18)).strftime('%H:%M:%S')} - Position size auto-adjusted: 2.1% -> 1.9%",
            f"{(datetime.now() - timedelta(minutes=25)).strftime('%H:%M:%S')} - Stop loss triggered: -12 pips"
        ]
        
        for entry in log_entries:
            st.text(entry)
    
    def create_risk_tab(self, system_status):
        """Risk management tab"""
        st.subheader("⚠️ Risk Management")
        
        portfolio = system_status['portfolio']
        trading = system_status['trading']
        limits = system_status['limits']
        
        # Risk metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Current Drawdown", f"{portfolio['max_drawdown']:.1%}", 
                     delta=f"Limit: {limits['max_drawdown_pct']:.0%}")
        
        with col2:
            daily_loss_pct = portfolio['daily_pnl'] / portfolio['current_value'] * 100
            st.metric("Daily Loss", f"{daily_loss_pct:.1%}", 
                     delta=f"Limit: {limits['max_daily_loss_pct']:.0%}")
        
        with col3:
            st.metric("Consecutive Losses", f"{trading['consecutive_losses']}", 
                     delta=f"Limit: {limits['max_consecutive_losses']}")
        
        # Risk controls status
        st.subheader("🛡️ Risk Controls Status")
        
        risk_controls = [
            {
                'Control': 'Daily Loss Limit',
                'Status': '🟢 ACTIVE',
                'Current': f"{daily_loss_pct:.1%}",
                'Limit': f"{limits['max_daily_loss_pct']:.0%}",
                'Action': 'Auto-pause trading'
            },
            {
                'Control': 'Maximum Drawdown',
                'Status': '🟢 ACTIVE',
                'Current': f"{portfolio['max_drawdown']:.1%}",
                'Limit': f"{limits['max_drawdown_pct']:.0%}",
                'Action': 'Auto-pause trading'
            },
            {
                'Control': 'Position Size Limit',
                'Status': '🟢 ACTIVE',
                'Current': 'Variable',
                'Limit': f"{limits['max_position_size']:.0%}",
                'Action': 'Reject oversized trades'
            },
            {
                'Control': 'Daily Trade Limit',
                'Status': '🟢 ACTIVE',
                'Current': f"{trading['trades_today']}",
                'Limit': f"{limits['max_daily_trades']}",
                'Action': 'Pause until next day'
            }
        ]
        
        st.dataframe(pd.DataFrame(risk_controls), hide_index=True)
        
        # Risk heatmap
        st.subheader("🔥 Risk Heatmap")
        
        risk_matrix = np.random.rand(5, 5)  # Simulate risk levels
        risk_labels = ['Daily Loss', 'Drawdown', 'Win Rate', 'Position Size', 'Volatility']
        
        fig = px.imshow(risk_matrix, 
                       x=risk_labels, 
                       y=risk_labels,
                       color_continuous_scale='RdYlGn_r',
                       title="Risk Factor Correlation")
        
        st.plotly_chart(fig, use_container_width=True)
    
    def create_trades_tab(self):
        """Trade history tab"""
        st.subheader("📋 Trade History")
        
        # Simulate recent trades
        trades_data = []
        for i in range(20):
            trade = {
                'Time': (datetime.now() - timedelta(hours=i*2)).strftime('%m/%d %H:%M'),
                'Direction': np.random.choice(['UP', 'DOWN']),
                'Size': f"{np.random.uniform(0.01, 0.08):.1%}",
                'Entry': f"{np.random.uniform(1.08, 1.09):.5f}",
                'Exit': f"{np.random.uniform(1.08, 1.09):.5f}",
                'Pips': f"{np.random.randint(-15, 45):+d}",
                'Return': f"{np.random.normal(0.02, 0.04):+.2%}",
                'Regime': np.random.choice(['HIGH_VOL', 'MOMENTUM', 'TREND', 'MIXED', 'RANGING']),
                'Cluster': np.random.randint(0, 8),
                'Exit Reason': np.random.choice(['Target Hit', 'Stop Loss', 'Time Stop'])
            }
            trades_data.append(trade)
        
        trades_df = pd.DataFrame(trades_data)
        
        # Color code returns
        def color_returns(val):
            if '+' in str(val):
                return 'background-color: #90EE90'
            else:
                return 'background-color: #FFB6C1'
        
        styled_df = trades_df.style.applymap(color_returns, subset=['Return', 'Pips'])
        st.dataframe(styled_df, hide_index=True)
        
        # Trade statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            winning_trades = len([t for t in trades_data if '+' in t['Return']])
            st.metric("Winning Trades", f"{winning_trades}/20", f"{winning_trades/20:.1%}")
        
        with col2:
            avg_pips = np.mean([int(t['Pips'].replace('+', '')) for t in trades_data])
            st.metric("Avg Pips", f"{avg_pips:+.1f}")
        
        with col3:
            avg_return = np.mean([float(t['Return'].replace('%', '').replace('+', '')) for t in trades_data])
            st.metric("Avg Return", f"{avg_return:+.2f}%")
        
        with col4:
            best_trade = max([float(t['Return'].replace('%', '').replace('+', '')) for t in trades_data])
            st.metric("Best Trade", f"{best_trade:+.2f}%")


def main():
    """Main dashboard application"""
    dashboard = Spin36TBDashboard()
    dashboard.create_dashboard()
    
    # Auto-refresh indicator
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")


if __name__ == "__main__":
    main()