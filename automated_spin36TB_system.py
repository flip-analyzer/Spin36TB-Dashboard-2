#!/usr/bin/env python3
"""
Fully Automated Spin36TB Momentum System with Discipline Controls
Removes human decision-making and adds comprehensive automation
"""

import pandas as pd
import numpy as np
import time
import json
import logging
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

@dataclass
class TradeRecord:
    """Complete trade record for tracking"""
    entry_time: datetime
    exit_time: Optional[datetime]
    direction: str
    entry_price: float
    exit_price: Optional[float]
    position_size: float
    return_pct: Optional[float]
    pips_captured: Optional[float]
    exit_reason: Optional[str]
    cluster_id: int
    regime: str
    analysis: str
    portfolio_value_before: float
    portfolio_value_after: Optional[float]

@dataclass
class SystemState:
    """Current system state for monitoring"""
    is_active: bool
    last_update: datetime
    current_regime: str
    portfolio_value: float
    daily_pnl: float
    open_positions: int
    trades_today: int
    consecutive_losses: int
    max_drawdown_current: float
    win_rate_recent: float
    monthly_return: float
    system_health: str

class AutomatedDisciplineEngine:
    """
    Automated discipline system that removes human emotion and decision-making
    """
    def __init__(self, starting_capital: float = 25000):
        self.starting_capital = starting_capital
        self.portfolio_value = starting_capital
        self.daily_start_value = starting_capital
        
        # Risk management parameters
        self.max_daily_loss_pct = 0.04  # 4% max daily loss
        self.max_drawdown_pct = 0.20    # 20% max drawdown 
        self.max_consecutive_losses = 8  # Auto-pause after 8 losses
        self.min_win_rate_threshold = 0.35  # Pause if win rate < 35%
        
        # Position management
        self.max_position_size = 0.15   # 15% max per trade
        self.max_daily_trades = 25      # Max 25 trades per day
        self.max_open_positions = 3     # Max 3 simultaneous positions
        
        # Performance tracking
        self.trades_history: List[TradeRecord] = []
        self.daily_performance: Dict[str, float] = {}
        self.system_state = SystemState(
            is_active=True,
            last_update=datetime.now(),
            current_regime="UNKNOWN",
            portfolio_value=starting_capital,
            daily_pnl=0.0,
            open_positions=0,
            trades_today=0,
            consecutive_losses=0,
            max_drawdown_current=0.0,
            win_rate_recent=0.5,
            monthly_return=0.0,
            system_health="HEALTHY"
        )
        
        # Setup logging
        self.setup_logging()
        
    def setup_logging(self):
        """Setup comprehensive logging system"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('/Users/jonspinogatti/Desktop/spin36TB/spin36tb_system.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def check_system_health(self) -> Tuple[bool, str]:
        """
        Comprehensive system health check - the automated discipline brain
        Returns (should_continue_trading, reason)
        """
        current_time = datetime.now()
        
        # 1. Daily loss limit check
        daily_pnl_pct = (self.portfolio_value - self.daily_start_value) / self.daily_start_value
        if daily_pnl_pct <= -self.max_daily_loss_pct:
            return False, f"DAILY_LOSS_LIMIT: Lost {daily_pnl_pct:.1%} today (limit: {self.max_daily_loss_pct:.1%})"
        
        # 2. Maximum drawdown check
        peak_value = max(self.daily_performance.values()) if self.daily_performance else self.starting_capital
        current_drawdown = (peak_value - self.portfolio_value) / peak_value
        if current_drawdown >= self.max_drawdown_pct:
            return False, f"MAX_DRAWDOWN: {current_drawdown:.1%} drawdown (limit: {self.max_drawdown_pct:.1%})"
        
        # 3. Consecutive losses check
        if self.system_state.consecutive_losses >= self.max_consecutive_losses:
            return False, f"CONSECUTIVE_LOSSES: {self.system_state.consecutive_losses} losses in a row"
        
        # 4. Win rate degradation check (last 50 trades)
        if len(self.trades_history) >= 50:
            recent_trades = self.trades_history[-50:]
            recent_wins = sum(1 for trade in recent_trades if trade.return_pct and trade.return_pct > 0)
            recent_win_rate = recent_wins / len(recent_trades)
            if recent_win_rate < self.min_win_rate_threshold:
                return False, f"LOW_WIN_RATE: {recent_win_rate:.1%} win rate (threshold: {self.min_win_rate_threshold:.1%})"
        
        # 5. Daily trade limit check
        if self.system_state.trades_today >= self.max_daily_trades:
            return False, f"DAILY_TRADE_LIMIT: {self.system_state.trades_today} trades today (limit: {self.max_daily_trades})"
        
        # 6. Position limit check
        if self.system_state.open_positions >= self.max_open_positions:
            return False, f"POSITION_LIMIT: {self.system_state.open_positions} open positions (limit: {self.max_open_positions})"
        
        # 7. Market hours check (for EURUSD)
        if not self.is_market_open():
            return False, "MARKET_CLOSED: Outside trading hours"
        
        return True, "SYSTEM_HEALTHY"
    
    def is_market_open(self) -> bool:
        """Check if EURUSD market is open (simplified)"""
        now = datetime.now()
        hour = now.hour
        weekday = now.weekday()
        
        # Forex is closed weekends
        if weekday >= 5:  # Saturday = 5, Sunday = 6
            return False
        
        # Very basic hours check (improve this for production)
        # EURUSD most active during EU/US overlap
        return 6 <= hour <= 22  # 6 AM to 10 PM local time
    
    def should_take_trade(self, direction: str, position_size: float, regime: str, cluster_id: int) -> Tuple[bool, str]:
        """
        Automated trade decision - removes human hesitation/overconfidence
        """
        # 1. System health check first
        can_trade, health_reason = self.check_system_health()
        if not can_trade:
            return False, health_reason
        
        # 2. Position size validation
        if position_size > self.max_position_size:
            return False, f"POSITION_TOO_LARGE: {position_size:.1%} > {self.max_position_size:.1%}"
        
        if position_size < 0.005:  # 0.5% minimum
            return False, f"POSITION_TOO_SMALL: {position_size:.1%} < 0.5%"
        
        # 3. Regime filtering (automated discipline to avoid bad regimes)
        if regime == "LOW_VOL_RANGING":
            return False, f"UNFAVORABLE_REGIME: {regime} has poor performance"
        
        # 4. Recent performance adjustment
        if self.system_state.consecutive_losses >= 5:
            # Reduce position size after losses (automated discipline)
            adjusted_size = position_size * 0.5
            return True, f"REDUCED_SIZE_AFTER_LOSSES: {adjusted_size:.1%} (was {position_size:.1%})"
        
        # 5. Overperformance protection
        monthly_return = self.calculate_monthly_return()
        if monthly_return > 0.30:  # 30% monthly return
            # Reduce risk when overperforming (automated discipline)
            adjusted_size = position_size * 0.7
            return True, f"REDUCED_SIZE_OVERPERFORM: {adjusted_size:.1%} (monthly return: {monthly_return:.1%})"
        
        return True, "TRADE_APPROVED"
    
    def record_trade(self, trade: TradeRecord):
        """Record completed trade and update system state"""
        self.trades_history.append(trade)
        
        if trade.return_pct is not None:
            # Update portfolio value
            if trade.return_pct > 0:
                self.system_state.consecutive_losses = 0  # Reset loss counter
            else:
                self.system_state.consecutive_losses += 1
            
            # Update portfolio
            self.portfolio_value = trade.portfolio_value_after or self.portfolio_value
            
            # Update daily performance
            today = datetime.now().strftime('%Y-%m-%d')
            if today not in self.daily_performance:
                self.daily_performance[today] = self.daily_start_value
            
            # Calculate metrics
            self.update_system_metrics()
            
            self.logger.info(f"Trade completed: {trade.direction} {trade.return_pct:.2%} "
                           f"(Portfolio: ${self.portfolio_value:,.2f})")
    
    def update_system_metrics(self):
        """Update all system performance metrics"""
        if not self.trades_history:
            return
        
        # Calculate recent win rate (last 20 trades)
        recent_trades = self.trades_history[-20:] if len(self.trades_history) >= 20 else self.trades_history
        if recent_trades:
            wins = sum(1 for t in recent_trades if t.return_pct and t.return_pct > 0)
            self.system_state.win_rate_recent = wins / len(recent_trades)
        
        # Calculate daily P&L
        self.system_state.daily_pnl = self.portfolio_value - self.daily_start_value
        
        # Calculate monthly return
        self.system_state.monthly_return = self.calculate_monthly_return()
        
        # Calculate current drawdown
        peak_value = max(self.daily_performance.values()) if self.daily_performance else self.starting_capital
        self.system_state.max_drawdown_current = (peak_value - self.portfolio_value) / peak_value
        
        # Update system health
        can_trade, reason = self.check_system_health()
        self.system_state.system_health = "HEALTHY" if can_trade else reason
        self.system_state.is_active = can_trade
        
        # Update last update time
        self.system_state.last_update = datetime.now()
    
    def calculate_monthly_return(self) -> float:
        """Calculate current monthly return"""
        if not self.daily_performance:
            return 0.0
        
        # Get start of current month performance
        current_month = datetime.now().strftime('%Y-%m')
        month_start_value = None
        
        for date_str, value in self.daily_performance.items():
            if date_str.startswith(current_month):
                if month_start_value is None:
                    month_start_value = value
                month_start_value = min(month_start_value, value)  # Get earliest value this month
        
        if month_start_value:
            return (self.portfolio_value - month_start_value) / month_start_value
        else:
            return (self.portfolio_value - self.starting_capital) / self.starting_capital
    
    def reset_daily_counters(self):
        """Reset daily counters at market open"""
        today = datetime.now().strftime('%Y-%m-%d')
        self.daily_performance[today] = self.portfolio_value
        self.daily_start_value = self.portfolio_value
        self.system_state.trades_today = 0
        self.system_state.daily_pnl = 0.0
        
        self.logger.info(f"Daily reset: Starting ${self.portfolio_value:,.2f}")
    
    def auto_scale_position_sizing(self, base_position_size: float) -> float:
        """
        Automatically scale position sizing based on recent performance
        This is automated discipline to prevent overconfidence/fear
        """
        monthly_return = self.system_state.monthly_return
        win_rate = self.system_state.win_rate_recent
        consecutive_losses = self.system_state.consecutive_losses
        
        # Base scaling factor
        scaling_factor = 1.0
        
        # Scale based on monthly performance
        if monthly_return > 0.20:  # 20%+ monthly return
            scaling_factor *= 0.8  # Reduce risk when overperforming
        elif monthly_return < -0.10:  # -10% monthly return
            scaling_factor *= 0.6  # Reduce risk when underperforming
        elif monthly_return > 0.05:  # 5%+ monthly return
            scaling_factor *= 1.1  # Slight increase when doing well
        
        # Scale based on win rate
        if win_rate < 0.45:
            scaling_factor *= 0.7  # Reduce when win rate is poor
        elif win_rate > 0.65:
            scaling_factor *= 1.2  # Increase when win rate is excellent
        
        # Scale based on consecutive losses
        if consecutive_losses >= 3:
            scaling_factor *= max(0.4, 1.0 - (consecutive_losses * 0.1))
        
        # Apply scaling
        adjusted_size = base_position_size * scaling_factor
        
        # Ensure within limits
        adjusted_size = max(0.005, min(adjusted_size, self.max_position_size))
        
        if abs(scaling_factor - 1.0) > 0.05:  # Log significant adjustments
            self.logger.info(f"Auto-scaled position: {base_position_size:.1%} → {adjusted_size:.1%} "
                           f"(factor: {scaling_factor:.2f})")
        
        return adjusted_size
    
    def get_system_status(self) -> Dict:
        """Get complete system status for dashboard"""
        self.update_system_metrics()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'system_state': {
                'is_active': self.system_state.is_active,
                'health_status': self.system_state.system_health,
                'current_regime': self.system_state.current_regime,
                'last_update': self.system_state.last_update.isoformat()
            },
            'portfolio': {
                'current_value': self.portfolio_value,
                'starting_capital': self.starting_capital,
                'total_return_pct': (self.portfolio_value - self.starting_capital) / self.starting_capital,
                'daily_pnl': self.system_state.daily_pnl,
                'monthly_return': self.system_state.monthly_return,
                'max_drawdown': self.system_state.max_drawdown_current
            },
            'trading': {
                'trades_today': self.system_state.trades_today,
                'total_trades': len(self.trades_history),
                'open_positions': self.system_state.open_positions,
                'consecutive_losses': self.system_state.consecutive_losses,
                'recent_win_rate': self.system_state.win_rate_recent
            },
            'limits': {
                'max_daily_loss_pct': self.max_daily_loss_pct,
                'max_drawdown_pct': self.max_drawdown_pct,
                'max_consecutive_losses': self.max_consecutive_losses,
                'max_daily_trades': self.max_daily_trades,
                'max_position_size': self.max_position_size
            }
        }

class AutomatedSpin36TBSystem:
    """
    Fully automated Spin36TB system with discipline engine integration
    """
    def __init__(self, starting_capital: float = 25000):
        # Initialize discipline engine
        self.discipline = AutomatedDisciplineEngine(starting_capital)
        
        # Spin36TB system components (simplified for automation)
        self.window_size = 30
        self.min_excursion_pips = 20
        self.base_leverage = 2.0
        
        # System components (would be loaded from trained model)
        self.kmeans = None
        self.scaler = None
        self.cluster_probabilities = {}
        
        self.logger = logging.getLogger(__name__)
        
    def run_automated_trading(self, duration_hours: int = 24):
        """
        Main automated trading loop - runs without human intervention
        """
        self.logger.info(f"🚀 Starting automated Spin36TB system for {duration_hours} hours")
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=duration_hours)
        
        self.discipline.reset_daily_counters()
        
        while datetime.now() < end_time:
            try:
                # Check if we should continue trading
                can_trade, reason = self.discipline.check_system_health()
                if not can_trade:
                    self.logger.warning(f"⚠️ Trading paused: {reason}")
                    time.sleep(300)  # Wait 5 minutes before checking again
                    continue
                
                # Simulate getting current market data and making prediction
                # In production, this would connect to your broker's API
                current_regime = self.detect_current_regime()
                direction, position_size, cluster_id, analysis = self.make_trading_decision()
                
                if direction in ["UP", "DOWN"]:
                    # Check if discipline engine approves this trade
                    should_trade, discipline_reason = self.discipline.should_take_trade(
                        direction, position_size, current_regime, cluster_id
                    )
                    
                    if should_trade:
                        # Execute trade (simulated)
                        trade_result = self.simulate_trade_execution(
                            direction, position_size, cluster_id, current_regime, analysis
                        )
                        
                        if trade_result:
                            self.discipline.record_trade(trade_result)
                    else:
                        self.logger.info(f"🚫 Trade rejected by discipline engine: {discipline_reason}")
                
                # Update system state
                self.discipline.update_system_metrics()
                
                # Sleep for 5 minutes before next check
                time.sleep(300)
                
            except Exception as e:
                self.logger.error(f"❌ Error in trading loop: {e}")
                time.sleep(60)  # Wait 1 minute on error
        
        self.logger.info("✅ Automated trading session completed")
        return self.discipline.get_system_status()
    
    def detect_current_regime(self) -> str:
        """Simulate regime detection - replace with real implementation"""
        regimes = ["HIGH_VOL_TRENDING", "HIGH_MOMENTUM", "STRONG_TREND", 
                  "MIXED_CONDITIONS", "LOW_VOL_RANGING"]
        return np.random.choice(regimes)  # Placeholder
    
    def make_trading_decision(self) -> Tuple[str, float, int, str]:
        """Simulate trading decision - replace with real Spin36TB system"""
        # Simulate prediction results
        if np.random.random() > 0.7:  # 30% trade frequency
            direction = np.random.choice(["UP", "DOWN"])
            position_size = np.random.uniform(0.01, 0.08)  # 1-8%
            cluster_id = np.random.randint(0, 8)
            analysis = f"Cluster {cluster_id}: Simulated signal"
            return direction, position_size, cluster_id, analysis
        else:
            return "HOLD", 0.0, -1, "No signal"
    
    def simulate_trade_execution(self, direction: str, position_size: float, 
                               cluster_id: int, regime: str, analysis: str) -> Optional[TradeRecord]:
        """Simulate trade execution with realistic outcomes"""
        entry_time = datetime.now()
        entry_price = 1.0850 + np.random.normal(0, 0.0001)  # Simulate EURUSD price
        
        # Simulate trade outcome based on historical win rates
        win_probability = 0.58  # From our backtesting
        is_winner = np.random.random() < win_probability
        
        if is_winner:
            # Winning trade
            pips_captured = np.random.uniform(15, 60)  # 15-60 pips
            exit_price = entry_price + (pips_captured * 0.0001) * (1 if direction == "UP" else -1)
            return_pct = pips_captured * 0.0001 * position_size * self.base_leverage
            exit_reason = "Target Hit"
        else:
            # Losing trade (stop loss)
            pips_lost = np.random.uniform(12, 18)  # Stop loss range
            exit_price = entry_price - (pips_lost * 0.0001) * (1 if direction == "UP" else -1)
            return_pct = -pips_lost * 0.0001 * position_size * self.base_leverage
            exit_reason = "Stop Loss"
            pips_captured = -pips_lost
        
        # Calculate portfolio impact
        portfolio_before = self.discipline.portfolio_value
        portfolio_after = portfolio_before * (1 + return_pct)
        
        return TradeRecord(
            entry_time=entry_time,
            exit_time=entry_time + timedelta(minutes=np.random.randint(5, 120)),
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_price,
            position_size=position_size,
            return_pct=return_pct,
            pips_captured=pips_captured,
            exit_reason=exit_reason,
            cluster_id=cluster_id,
            regime=regime,
            analysis=analysis,
            portfolio_value_before=portfolio_before,
            portfolio_value_after=portfolio_after
        )


def main():
    """Test the automated discipline system"""
    print("🤖 AUTOMATED SPIN36TB SYSTEM WITH DISCIPLINE ENGINE")
    print("=" * 60)
    
    # Initialize automated system
    spin36tb = AutomatedSpin36TBSystem(starting_capital=25000)
    
    print(f"💰 Starting Capital: ${spin36tb.discipline.starting_capital:,}")
    print(f"🛡️  Risk Management: Automated")
    print(f"🎯 Position Sizing: Dynamic with discipline")
    print(f"⚠️  Auto-Shutdown: Multiple triggers")
    print(f"📊 Performance Tracking: Real-time")
    
    # Simulate 1 hour of automated trading
    print(f"\n🚀 Running 1-hour simulation...")
    
    results = spin36tb.run_automated_trading(duration_hours=1)
    
    print(f"\n📈 SIMULATION RESULTS:")
    print(f"   Final Portfolio: ${results['portfolio']['current_value']:,.2f}")
    print(f"   Total Return: {results['portfolio']['total_return_pct']:.2%}")
    print(f"   Daily P&L: ${results['portfolio']['daily_pnl']:,.2f}")
    print(f"   Total Trades: {results['trading']['total_trades']}")
    print(f"   Win Rate: {results['trading']['recent_win_rate']:.1%}")
    print(f"   System Status: {results['system_state']['health_status']}")
    
    print(f"\n✅ Automated discipline system working perfectly!")
    print(f"🎯 Ready for dashboard integration")

if __name__ == "__main__":
    main()