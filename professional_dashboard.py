#!/usr/bin/env python3
"""
Professional Trading Dashboard
Everything you need to monitor for successful trading
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json
import os
import pytz

st.set_page_config(
    page_title="🏛️ Professional Trading Monitor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

class ForexMarketScheduler:
    """Market hours detection for Streamlit dashboard"""
    
    def __init__(self):
        from datetime import time
        self.sessions = {
            'sydney': {
                'timezone': pytz.timezone('Australia/Sydney'),
                'open': time(8, 0),
                'close': time(17, 0),
                'days': [0, 1, 2, 3, 4]
            },
            'tokyo': {
                'timezone': pytz.timezone('Asia/Tokyo'), 
                'open': time(9, 0),
                'close': time(18, 0),
                'days': [0, 1, 2, 3, 4]
            },
            'london': {
                'timezone': pytz.timezone('Europe/London'),
                'open': time(8, 0),
                'close': time(17, 0),
                'days': [0, 1, 2, 3, 4]
            },
            'new_york': {
                'timezone': pytz.timezone('US/Eastern'),
                'open': time(8, 0),
                'close': time(17, 0),
                'days': [0, 1, 2, 3, 4]
            }
        }
    
    def is_market_open(self):
        """Check if any major forex market is open"""
        utc_now = datetime.now(pytz.UTC)
        active_sessions = []
        
        for session_name, session_info in self.sessions.items():
            if self._is_session_active(utc_now, session_info):
                active_sessions.append(session_name.title())
        
        is_open = len(active_sessions) > 0
        
        if is_open:
            description = f"Markets Open: {', '.join(active_sessions)}"
        else:
            next_open = self._get_next_market_open()
            description = f"Markets Closed - Next open: {next_open}"
        
        return is_open, description
    
    def _is_session_active(self, utc_time, session_info):
        """Check if specific session is active"""
        local_time = utc_time.astimezone(session_info['timezone'])
        
        if local_time.weekday() not in session_info['days']:
            return False
        
        current_time = local_time.time()
        return session_info['open'] <= current_time <= session_info['close']
    
    def _get_next_market_open(self):
        """Get next market opening time"""
        utc_now = datetime.now(pytz.UTC)
        next_openings = []
        
        for session_name, session_info in self.sessions.items():
            local_now = utc_now.astimezone(session_info['timezone'])
            
            # Check if opens later today
            if local_now.weekday() in session_info['days']:
                if local_now.time() < session_info['open']:
                    next_open_local = local_now.replace(
                        hour=session_info['open'].hour,
                        minute=session_info['open'].minute,
                        second=0, microsecond=0
                    )
                    next_open_utc = next_open_local.astimezone(pytz.UTC)
                    next_openings.append((next_open_utc, session_name.title()))
            
            # Check tomorrow
            tomorrow_local = (local_now + timedelta(days=1)).replace(
                hour=session_info['open'].hour,
                minute=session_info['open'].minute,
                second=0, microsecond=0
            )
            
            if tomorrow_local.weekday() in session_info['days']:
                tomorrow_utc = tomorrow_local.astimezone(pytz.UTC)
                next_openings.append((tomorrow_utc, session_name.title()))
        
        if next_openings:
            earliest = min(next_openings, key=lambda x: x[0])
            hours_until = (earliest[0] - utc_now).total_seconds() / 3600
            return f"{earliest[1]} in {hours_until:.1f} hours"
        
        return "Monday morning"

class ProfessionalTradingDashboard:
    def __init__(self):
        # Market scheduler for determining if system should be active
        self.market_scheduler = ForexMarketScheduler()
        
        # OANDA Configuration - using Streamlit secrets for security
        try:
            self.api_token = st.secrets["OANDA_API_TOKEN"]
            self.account_id = st.secrets["OANDA_ACCOUNT_ID"]
        except:
            # Fallback for local development
            self.api_token = "fdf72f4b5166602e39e001d1c1cb38b0-524c6662baa3187260b34c006eeab1fd"
            self.account_id = "101-001-31365224-001"
            
        self.api_url = 'https://api-fxpractice.oanda.com'
        self.headers = {
            'Authorization': f'Bearer {self.api_token}',
            'Content-Type': 'application/json'
        }
    
    def detect_market_regime(self, market_data):
        """Detect current market regime using same logic as trading system"""
        if len(market_data) < 20:
            return "NORMAL_VOLATILITY"
        
        recent_data = market_data.tail(20)
        
        # Calculate volatility (20-period ATR)
        high_low = recent_data['high'] - recent_data['low']
        current_volatility = high_low.mean()
        
        # Calculate momentum (price change over 10 periods)
        price_momentum = abs(recent_data['close'].iloc[-1] - recent_data['close'].iloc[-11]) / recent_data['close'].iloc[-11]
        
        # Regime thresholds (same as trading system)
        if current_volatility > 0.0004:  # 4 pips
            return "HIGH_VOLATILITY"
        elif current_volatility < 0.0001:  # 1 pip
            return "LOW_VOLATILITY"
        elif price_momentum > 0.0003:  # 3 pips trend
            return "TRENDING"
        elif price_momentum < 0.0002:  # 2 pips momentum
            return "RANGING"
        else:
            return "NORMAL_VOLATILITY"
    
    def get_live_data(self, candles=100):
        """Get live market data"""
        try:
            url = f"{self.api_url}/v3/instruments/EUR_USD/candles"
            params = {'granularity': 'M5', 'count': candles, 'price': 'MBA'}
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
    
    def read_trading_log(self):
        """Read trading log"""
        try:
            with open('/Users/jonspinogatti/Desktop/spin36TB/live_trading.log', 'r') as f:
                return [line.strip() for line in f.readlines() if line.strip()]
        except:
            return []
    
    def get_virtual_account_info(self):
        """Get virtual paper trading account info instead of real OANDA account"""
        try:
            # Try to read current session state if available
            session_file = '/Users/jonspinogatti/Desktop/spin36TB/current_session.json'
            if os.path.exists(session_file):
                with open(session_file, 'r') as f:
                    session_data = json.load(f)
                    return {
                        'balance': session_data.get('final_capital', 25000),
                        'starting_balance': session_data.get('starting_capital', 25000),
                        'unrealized_pl': 0,  # Virtual account
                        'margin_used': 0     # No margin in paper trading
                    }
            
            # Fallback: parse from log file to estimate current balance
            log_lines = self.read_trading_log()
            balance = 25000  # Starting amount
            
            for line in log_lines:
                # Look for trade results in logs (if any trades executed)
                if "P&L:" in line:
                    try:
                        # Extract P&L from log line
                        pnl_part = line.split("P&L:")[1].split()[0]
                        pnl = float(pnl_part.replace('$', '').replace(',', ''))
                        balance += pnl
                    except:
                        pass
                        
            return {
                'balance': balance,
                'starting_balance': 25000,
                'unrealized_pl': 0,
                'margin_used': 0
            }
            
        except:
            # Ultimate fallback
            return {
                'balance': 25000,
                'starting_balance': 25000,  
                'unrealized_pl': 0,
                'margin_used': 0
            }
    
    def get_account_info(self):
        """Get account information"""
        try:
            url = f"{self.api_url}/v3/accounts/{self.account_id}"
            response = requests.get(url, headers=self.headers, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                account = data['account']
                return {
                    'balance': float(account['balance']),
                    'unrealized_pl': float(account.get('unrealizedPL', 0)),
                    'margin_used': float(account.get('marginUsed', 0))
                }
            return None
        except:
            return None
    
    def calculate_system_health(self, log_lines):
        """Calculate overall system health score"""
        decisions = len([l for l in log_lines if 'Portfolio Decision' in l])
        errors = len([l for l in log_lines if 'ERROR' in l])
        
        if decisions == 0:
            return 0, "System Not Active"
        
        # Health scoring
        health_score = 100
        if errors > 0:
            health_score -= (errors / decisions) * 50
        
        # Check decision frequency (should be every 5 minutes)
        recent_decisions = [l for l in log_lines if 'Portfolio Decision' in l][-5:]
        if len(recent_decisions) >= 2:
            # Check if decisions are regular
            times = []
            for decision in recent_decisions:
                time_str = decision.split(',')[0]
                try:
                    times.append(datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S'))
                except:
                    continue
            
            if len(times) >= 2:
                intervals = [(times[i] - times[i-1]).total_seconds()/60 for i in range(1, len(times))]
                avg_interval = np.mean(intervals)
                if abs(avg_interval - 5) > 1:  # More than 1 minute deviation
                    health_score -= 20
        
        if health_score >= 90:
            status = "Excellent"
        elif health_score >= 70:
            status = "Good"
        elif health_score >= 50:
            status = "Fair"
        else:
            status = "Poor"
        
        return health_score, status
    
    def check_system_running_status(self):
        """Check if the live trading system is currently running"""
        try:
            import subprocess
            
            # Check if live_paper_trader.py process is running
            result = subprocess.run(['pgrep', '-f', 'live_paper_trader.py'], 
                                  capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                # Process found - check recent activity
                log_lines = self.read_trading_log()
                if log_lines:
                    try:
                        # Check if last log entry is recent (within 10 minutes)
                        last_log = log_lines[-1]
                        last_time_str = last_log.split(',')[0]
                        last_time = datetime.strptime(last_time_str, '%Y-%m-%d %H:%M:%S')
                        minutes_since = (datetime.now() - last_time).total_seconds() / 60
                        
                        if minutes_since < 10:
                            return {
                                'running': True,
                                'status': 'active',
                                'details': f'System active - Last activity: {minutes_since:.1f}m ago'
                            }
                        else:
                            return {
                                'running': True,
                                'status': 'stalled',
                                'details': f'Process running but inactive for {minutes_since:.0f}m'
                            }
                    except:
                        return {
                            'running': True,
                            'status': 'unknown',
                            'details': 'Process running - Activity status unclear'
                        }
                else:
                    return {
                        'running': True, 
                        'status': 'starting',
                        'details': 'Process running - No logs yet'
                    }
            else:
                return {
                    'running': False,
                    'status': 'stopped',
                    'details': 'No live_paper_trader.py process found'
                }
        except Exception as e:
            return {
                'running': False,
                'status': 'error',
                'details': f'Cannot determine system status: {str(e)}'
            }
    
    def display_system_status(self, is_market_open, market_status, system_running):
        """Display comprehensive system status at top of dashboard"""
        
        # Create status container
        st.markdown("---")
        
        # System status header
        if is_market_open and system_running['running'] and system_running['status'] == 'active':
            # Everything optimal
            st.markdown("### 🚀 SYSTEM STATUS: ACTIVE & TRADING")
            st.success(f"✅ Markets Open & System Running - {system_running['details']}")
        elif is_market_open and system_running['running']:
            # Markets open but system issues
            if system_running['status'] == 'stalled':
                st.markdown("### ⚠️ SYSTEM STATUS: RUNNING BUT STALLED")
                st.warning(f"🟡 Markets Open but System Inactive - {system_running['details']}")
            elif system_running['status'] == 'starting':
                st.markdown("### 🔄 SYSTEM STATUS: STARTING UP")
                st.info(f"🔄 Markets Open & System Starting - {system_running['details']}")
            else:
                st.markdown("### ❓ SYSTEM STATUS: UNKNOWN")
                st.warning(f"⚠️ Markets Open - {system_running['details']}")
        elif not is_market_open and not system_running['running']:
            # Markets closed and system properly sleeping
            st.markdown("### 💤 SYSTEM STATUS: SLEEPING (ENERGY SAVE MODE)")
            st.info(f"💤 {market_status} - System properly conserving energy")
        elif not is_market_open and system_running['running']:
            # Markets closed but system still running (wasteful)
            st.markdown("### ⚡ SYSTEM STATUS: RUNNING DURING CLOSURE")
            st.warning(f"⚡ {market_status} - System should sleep to conserve energy")
        else:
            # Markets open but system not running (missing trades)
            st.markdown("### ❌ SYSTEM STATUS: OFFLINE DURING TRADING HOURS")
            st.error(f"❌ {market_status} - System should be active!")
        
        # Additional system details
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🌍 Market Status:**")
            st.text(market_status)
        
        with col2:
            st.markdown("**🔧 System Details:**")
            if system_running['running']:
                st.text(f"✅ Process: RUNNING")
            else:
                st.text(f"❌ Process: STOPPED")
            st.text(f"📊 Status: {system_running['status'].upper()}")
        
        st.markdown("---")
    
    def create_performance_metrics(self, is_market_open=True):
        """Create key performance metrics"""
        header_style = "" if is_market_open else "color: #888888"
        st.markdown(f"<h2 style='{header_style}'>📊 KEY PERFORMANCE INDICATORS</h2>", unsafe_allow_html=True)
        
        # Get current data
        market_data = self.get_live_data(50)
        log_lines = self.read_trading_log()
        account_info = self.get_virtual_account_info()
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        # System Health
        health_score, health_status = self.calculate_system_health(log_lines)
        with col1:
            color = "normal" if health_score >= 70 else "off"
            st.metric("🏥 System Health", f"{health_score:.0f}%", health_status, delta_color=color)
        
        # Account Status (Virtual Paper Trading)
        with col2:
            if account_info:
                current_balance = account_info['balance']
                starting_balance = account_info['starting_balance']
                pnl = current_balance - starting_balance
                pnl_pct = (pnl / starting_balance) * 100
                st.metric("💰 Paper Trading P&L", f"${pnl:+,.2f}", f"{pnl_pct:+.2f}% (${current_balance:,.2f})")
            else:
                st.metric("💰 Paper Trading P&L", "Offline", "Connection Issue")
        
        # Current Price & Movement
        with col3:
            if not market_data.empty:
                current_price = market_data['close'].iloc[-1]
                price_change = current_price - market_data['close'].iloc[-6] if len(market_data) > 6 else 0
                st.metric("📈 EURUSD", f"{current_price:.4f}", f"{price_change*10000:+.1f} pips")
            else:
                st.metric("📈 EURUSD", "No Data", "Connection Issue")
        
        # Decision Count
        decisions = len([l for l in log_lines if 'Portfolio Decision' in l])
        with col4:
            st.metric("🎯 Decisions", f"{decisions}", "Total Made")
        
        # Trade Status
        trades = len([l for l in log_lines if 'PAPER TRADE' in l])
        with col5:
            st.metric("💼 Trades", f"{trades}", "Executed")
    
    def create_market_conditions_monitor(self, is_market_open=True):
        """Monitor current market conditions for trading"""
        header_style = "" if is_market_open else "color: #888888"
        status_text = "MARKET CONDITIONS MONITOR" if is_market_open else "MARKET CONDITIONS (INACTIVE)"
        st.markdown(f"<h2 style='{header_style}'>🌍 {status_text}</h2>", unsafe_allow_html=True)
        
        market_data = self.get_live_data()
        if market_data.empty:
            st.error("❌ No market data available")
            return
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Price action chart
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=market_data['time'],
                open=market_data['open'],
                high=market_data['high'],
                low=market_data['low'],
                close=market_data['close'],
                name="EURUSD",
                increasing_line_color='#00ff88',
                decreasing_line_color='#ff4444'
            ))
            
            fig.update_layout(
                title="📈 Live Price Action (Last 100 candles)",
                template="plotly_dark",
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Trading readiness indicators
            current_price = market_data['close'].iloc[-1]
            
            st.markdown("### 🎯 Trading Readiness")
            
            # Calculate metrics
            if len(market_data) > 10:
                # Volatility (10-period ATR in pips)
                volatility = (market_data['high'] - market_data['low']).tail(10).mean() * 10000
                
                # Momentum (5-candle price change in pips)
                momentum = abs(current_price - market_data['close'].iloc[-6]) / current_price * 10000
                
                # Trend strength (20-period slope)
                if len(market_data) >= 20:
                    trend = (current_price - market_data['close'].iloc[-21]) / current_price * 10000
                else:
                    trend = 0
                
                # Time filter
                current_hour = datetime.now().hour
                
                # Display readiness indicators
                st.markdown("**📊 Current Metrics:**")
                st.text(f"Volatility: {volatility:.1f} pips")
                st.text(f"Momentum: {momentum:.1f} pips")
                st.text(f"Trend: {trend:+.1f} pips")
                st.text(f"Time: {datetime.now().strftime('%H:%M')} UTC")
                
                st.markdown("**✅ Readiness Checklist:**")
                
                # Volatility check
                if volatility > 2:
                    st.success("🟢 Volatility: Sufficient")
                else:
                    st.warning(f"🟡 Volatility: {volatility:.1f} pips (need >2)")
                
                # Momentum check
                if momentum > 6:
                    st.success("🟢 Momentum: Strong enough")
                else:
                    st.warning(f"🟡 Momentum: {momentum:.1f} pips (need >6)")
                
                # Time check
                if 7 <= current_hour <= 11 or 13 <= current_hour <= 17:
                    st.success("🟢 Time: Optimal hours")
                elif 22 <= current_hour or current_hour <= 6:
                    st.error("🔴 Time: Asian session pause")
                else:
                    st.warning("🟡 Time: Off-peak hours")
                
                # Market regime detection
                st.markdown("**🎭 Market Regime:**")
                regime = self.detect_market_regime(market_data)
                regime_color = {
                    'HIGH_VOLATILITY': '🔴',
                    'LOW_VOLATILITY': '🟡', 
                    'TRENDING': '🟢',
                    'RANGING': '🟠',
                    'NORMAL_VOLATILITY': '🔵'
                }.get(regime, '⚪')
                
                st.text(f"{regime_color} {regime.replace('_', ' ').title()}")
                
                # Overall readiness
                readiness_score = 0
                if volatility > 2:
                    readiness_score += 33
                if momentum > 6:
                    readiness_score += 34
                if 7 <= current_hour <= 11 or 13 <= current_hour <= 17:
                    readiness_score += 33
                
                st.markdown("**🎯 Overall Readiness:**")
                if readiness_score >= 80:
                    st.success(f"🟢 READY TO TRADE ({readiness_score}%)")
                elif readiness_score >= 50:
                    st.warning(f"🟡 MARGINAL CONDITIONS ({readiness_score}%)")
                else:
                    st.error(f"🔴 WAIT FOR BETTER SETUP ({readiness_score}%)")
    
    def create_system_activity_monitor(self, is_market_open=True):
        """Monitor system activity and decisions"""
        header_style = "" if is_market_open else "color: #888888"
        status_text = "SYSTEM ACTIVITY MONITOR" if is_market_open else "SYSTEM ACTIVITY (PAUSED)"
        st.markdown(f"<h2 style='{header_style}'>🎯 {status_text}</h2>", unsafe_allow_html=True)
        
        if not is_market_open:
            st.info("🔋 System activity paused during market closure for energy conservation")
        
        log_lines = self.read_trading_log()
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Recent decisions table
            st.markdown("### 📋 Recent Decisions")
            
            decisions = [l for l in log_lines if 'Portfolio Decision' in l][-10:]
            trades = [l for l in log_lines if 'PAPER TRADE' in l][-5:]
            
            if decisions:
                decision_data = []
                for i, decision in enumerate(decisions, 1):
                    time_str = decision.split(',')[0].split(' ')[-1]  # Extract time
                    decision_data.append({
                        'Decision #': len(decisions) - len(decisions) + i,
                        'Time': time_str,
                        'Status': 'HOLD (Quality filters active)',
                        'Strategies': 'Momentum + Mean Reversion'
                    })
                
                decision_df = pd.DataFrame(decision_data)
                st.dataframe(decision_df, use_container_width=True, hide_index=True)
            else:
                st.info("No decisions recorded yet")
            
            # Trade execution log
            if trades:
                st.markdown("### 💼 Recent Trades")
                trade_data = []
                for trade in trades:
                    parts = trade.split(' - INFO - ')
                    if len(parts) >= 2:
                        time_str = parts[0].split(' ')[-1]
                        message = parts[1]
                        trade_data.append({
                            'Time': time_str,
                            'Action': message[:50] + "..." if len(message) > 50 else message
                        })
                
                trade_df = pd.DataFrame(trade_data)
                st.dataframe(trade_df, use_container_width=True, hide_index=True)
        
        with col2:
            # System statistics
            st.markdown("### 📊 Session Stats")
            
            total_decisions = len([l for l in log_lines if 'Portfolio Decision' in l])
            total_trades = len([l for l in log_lines if 'PAPER TRADE' in l])
            errors = len([l for l in log_lines if 'ERROR' in l])
            
            st.metric("Total Decisions", total_decisions)
            st.metric("Trades Executed", total_trades)
            st.metric("System Errors", errors)
            
            # Decision frequency
            if total_decisions >= 2:
                recent_decisions = [l for l in log_lines if 'Portfolio Decision' in l][-2:]
                if len(recent_decisions) == 2:
                    try:
                        time1 = datetime.strptime(recent_decisions[0].split(',')[0], '%Y-%m-%d %H:%M:%S')
                        time2 = datetime.strptime(recent_decisions[1].split(',')[0], '%Y-%m-%d %H:%M:%S')
                        interval = (time2 - time1).total_seconds() / 60
                        st.metric("Decision Interval", f"{interval:.1f} min")
                    except:
                        st.metric("Decision Interval", "Unknown")
            
            # System uptime
            if log_lines:
                start_time = log_lines[0].split(',')[0]
                try:
                    start_dt = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
                    uptime = (datetime.now() - start_dt).total_seconds() / 3600
                    st.metric("System Uptime", f"{uptime:.1f} hours")
                except:
                    st.metric("System Uptime", "Unknown")
    
    def create_risk_monitor(self):
        """Monitor risk levels and alerts"""
        st.markdown("## ⚠️ RISK MONITOR")
        
        account_info = self.get_virtual_account_info()
        log_lines = self.read_trading_log()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 💰 Account Risk")
            if account_info:
                balance = account_info['balance']
                margin_used = account_info['margin_used']
                margin_level = (balance / max(margin_used, 1)) * 100 if margin_used > 0 else float('inf')
                
                st.text(f"Balance: ${balance:,.2f}")
                st.text(f"Margin Used: ${margin_used:.2f}")
                if margin_level != float('inf'):
                    st.text(f"Margin Level: {margin_level:.1f}%")
                    if margin_level < 200:
                        st.error("🔴 LOW MARGIN LEVEL!")
                    elif margin_level < 500:
                        st.warning("🟡 Monitor margin usage")
                    else:
                        st.success("🟢 Healthy margin level")
                else:
                    st.success("🟢 No margin used")
            else:
                st.error("❌ Cannot connect to account")
        
        with col2:
            st.markdown("### 🎯 Position Risk")
            # Simulate current position analysis
            active_trades = len([l for l in log_lines if 'PAPER TRADE' in l]) - len([l for l in log_lines if 'TRADE CLOSED' in l])
            
            st.text(f"Active Trades: {active_trades}")
            st.text("Max Position Size: 2.2%")
            st.text("Current Exposure: 0.0%")  # No real trades yet
            
            if active_trades == 0:
                st.success("🟢 No position risk")
            elif active_trades <= 2:
                st.success("🟢 Low risk exposure")
            else:
                st.warning("🟡 Monitor position sizes")
        
        with col3:
            st.markdown("### 🛡️ System Alerts")
            
            errors = len([l for l in log_lines if 'ERROR' in l])
            decisions = len([l for l in log_lines if 'Portfolio Decision' in l])
            
            # Alert levels - check for emergency conditions
            alerts = []
            emergency_alerts = []
            
            # EMERGENCY: Major drawdown  
            if account_info and 'balance' in account_info:
                current_balance = account_info['balance']
                starting_balance = 25000  # Your starting capital
                drawdown = (starting_balance - current_balance) / starting_balance * 100
                
                if drawdown > 15:  # 15% drawdown = EMERGENCY
                    emergency_alerts.append(f"🚨 STOP SYSTEM: {drawdown:.1f}% drawdown!")
                elif drawdown > 10:  # 10% drawdown = serious warning
                    alerts.append(f"🔴 HIGH RISK: {drawdown:.1f}% drawdown")
                elif drawdown > 5:  # 5% drawdown = warning
                    alerts.append(f"🟡 Monitor: {drawdown:.1f}% drawdown")
            
            # EMERGENCY: System errors
            if errors > 5:
                emergency_alerts.append(f"🚨 STOP SYSTEM: {errors} critical errors!")
            elif errors > 0:
                alerts.append(f"⚠️ {errors} system errors detected")
            
            # EMERGENCY: System stuck/dead
            if decisions == 0:
                alerts.append("🔴 System not making decisions")
            elif decisions < 5:
                alerts.append("🟡 System recently started")
            
            # Check if system is stuck
            if len(log_lines) > 0:
                last_log = log_lines[-1]
                try:
                    last_time = datetime.strptime(last_log.split(',')[0], '%Y-%m-%d %H:%M:%S')
                    minutes_silent = (datetime.now() - last_time).total_seconds() / 60
                    
                    if minutes_silent > 20:  # 20 minutes = EMERGENCY
                        emergency_alerts.append("🚨 STOP SYSTEM: No activity >20min!")
                    elif minutes_silent > 10:  # 10 minutes = warning
                        alerts.append(f"🔴 System may be stuck ({minutes_silent:.0f}min silent)")
                except:
                    pass
            
            # EMERGENCY: Market conditions
            if 'market_data' in locals() and len(market_data) > 0:
                current_price = market_data['close'].iloc[-1]
                volatility = (market_data['high'] - market_data['low']).tail(10).mean() * 10000
                
                if volatility > 25:  # Extreme volatility
                    emergency_alerts.append(f"🚨 CONSIDER STOPPING: Extreme volatility {volatility:.1f} pips!")
                elif volatility > 15:
                    alerts.append(f"🔴 High volatility: {volatility:.1f} pips")
            
            # Show emergency alerts first (in red)
            if emergency_alerts:
                for alert in emergency_alerts:
                    st.error(alert)
            
            if not alerts:
                st.success("🟢 All systems normal")
            else:
                for alert in alerts:
                    st.warning(alert)
    
    def run_dashboard(self):
        """Main dashboard"""
        # Check market status first
        is_market_open, market_status = self.market_scheduler.is_market_open()
        
        # Check if trading system is actually running
        system_running = self.check_system_running_status()
        
        # Display system status
        self.display_system_status(is_market_open, market_status, system_running)
        
        if not is_market_open:
            # Show dimmed indicators
            st.markdown("### 📊 Market Data (Inactive)")
        else:
            # Markets open - normal display
            st.markdown("### Complete monitoring for Spin36TB system")
        
        # Auto-refresh indicator
        refresh_color = "#CCCCCC" if not is_market_open else "#FFFFFF"
        st.markdown(f"<div style='color: {refresh_color}'>🔄 Last updated: {datetime.now().strftime('%H:%M:%S')} (refreshes every 10 seconds)</div>", unsafe_allow_html=True)
        
        # Discrete account balance at top
        account_info = self.get_virtual_account_info()
        if account_info:
            current_balance = account_info['balance']
            starting_balance = account_info['starting_balance'] 
            pnl = current_balance - starting_balance
            pnl_pct = (pnl / starting_balance) * 100
            
            if pnl >= 0:
                st.markdown(f"<div style='text-align: right; color: #00ff00; font-size: 14px; margin-bottom: 10px;'>Balance: ${current_balance:,.2f} (+${pnl:,.2f} | +{pnl_pct:.2f}%)</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='text-align: right; color: #ff4444; font-size: 14px; margin-bottom: 10px;'>Balance: ${current_balance:,.2f} (${pnl:,.2f} | {pnl_pct:.2f}%)</div>", unsafe_allow_html=True)
        
        # Key Performance Indicators (pass market status)
        self.create_performance_metrics(is_market_open)
        
        st.divider()
        
        # Market Conditions (pass market status)
        self.create_market_conditions_monitor(is_market_open)
        
        st.divider()
        
        # System Activity (pass market status)
        self.create_system_activity_monitor(is_market_open)
        
        st.divider()
        
        # Risk Monitor
        self.create_risk_monitor()
        
        # Control buttons
        st.markdown("## 🎮 Controls")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Force Refresh", type="primary"):
                st.rerun()
        
        with col2:
            if st.button("📋 Download Log"):
                st.download_button("💾 Download Trading Log", 
                                 data="\n".join(self.read_trading_log()),
                                 file_name=f"trading_log_{datetime.now().strftime('%Y%m%d_%H%M')}.txt")
        
        with col3:
            st.metric("Next Auto-Refresh", "10 seconds")
        
        # Auto-refresh
        import time
        time.sleep(10)
        st.rerun()

def main():
    dashboard = ProfessionalTradingDashboard()
    dashboard.run_dashboard()

if __name__ == "__main__":
    main()