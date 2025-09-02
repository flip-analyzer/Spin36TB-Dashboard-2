#!/usr/bin/env python3
"""
Forex Market Hours Scheduler
Prevents trading when markets are closed to conserve energy and resources
"""

import pytz
from datetime import datetime, time, timedelta
from typing import Dict, Tuple

class ForexMarketScheduler:
    """
    Intelligent scheduler that knows when forex markets are open/closed
    Prevents unnecessary system activity during market closures
    """
    
    def __init__(self):
        # Major forex trading sessions (all times in respective local time zones)
        self.sessions = {
            'sydney': {
                'timezone': pytz.timezone('Australia/Sydney'),
                'open': time(8, 0),    # 8:00 AM Sydney time
                'close': time(17, 0),  # 5:00 PM Sydney time
                'days': [0, 1, 2, 3, 4]  # Monday-Friday
            },
            'tokyo': {
                'timezone': pytz.timezone('Asia/Tokyo'),
                'open': time(9, 0),    # 9:00 AM Tokyo time
                'close': time(18, 0),  # 6:00 PM Tokyo time
                'days': [0, 1, 2, 3, 4]  # Monday-Friday
            },
            'london': {
                'timezone': pytz.timezone('Europe/London'),
                'open': time(8, 0),    # 8:00 AM London time
                'close': time(17, 0),  # 5:00 PM London time
                'days': [0, 1, 2, 3, 4]  # Monday-Friday
            },
            'new_york': {
                'timezone': pytz.timezone('US/Eastern'),
                'open': time(8, 0),    # 8:00 AM Eastern time
                'close': time(17, 0),  # 5:00 PM Eastern time
                'days': [0, 1, 2, 3, 4]  # Monday-Friday
            }
        }
        
        # Peak trading hours (when multiple sessions overlap)
        self.peak_hours = [
            ('London/New York Overlap', '13:00', '17:00', 'Europe/London'),
            ('Tokyo/London Overlap', '08:00', '10:00', 'Europe/London'),
            ('Sydney/Tokyo Overlap', '09:00', '11:00', 'Asia/Tokyo')
        ]
        
        print("🕒 FOREX MARKET SCHEDULER INITIALIZED")
        print("=" * 45)
        print("📅 Tracks Sydney, Tokyo, London, New York sessions")
        print("⚡ Optimizes energy usage by sleeping during market closures")
        print("🎯 Focuses activity during peak trading hours")
    
    def is_market_open(self) -> Tuple[bool, str]:
        """
        Check if any major forex market is currently open
        Returns: (is_open, active_sessions_description)
        """
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
    
    def _is_session_active(self, utc_time: datetime, session_info: Dict) -> bool:
        """Check if a specific trading session is active"""
        
        # Convert UTC to local session time
        local_time = utc_time.astimezone(session_info['timezone'])
        
        # Check if it's a trading day
        if local_time.weekday() not in session_info['days']:
            return False
        
        # Check if within trading hours
        current_time = local_time.time()
        return session_info['open'] <= current_time <= session_info['close']
    
    def _get_next_market_open(self) -> str:
        """Get description of when markets next open"""
        utc_now = datetime.now(pytz.UTC)
        
        # Check each session to find the next opening
        next_openings = []
        
        for session_name, session_info in self.sessions.items():
            local_now = utc_now.astimezone(session_info['timezone'])
            
            # If market closed today, check if it opens later today
            if local_now.weekday() in session_info['days']:
                if local_now.time() < session_info['open']:
                    # Opens later today
                    next_open_local = local_now.replace(
                        hour=session_info['open'].hour,
                        minute=session_info['open'].minute,
                        second=0,
                        microsecond=0
                    )
                    next_open_utc = next_open_local.astimezone(pytz.UTC)
                    next_openings.append((next_open_utc, session_name.title()))
            
            # Check tomorrow
            tomorrow_local = local_now.replace(
                hour=session_info['open'].hour,
                minute=session_info['open'].minute,
                second=0,
                microsecond=0
            ) + timedelta(days=1)
            
            # Make sure tomorrow is a trading day
            if tomorrow_local.weekday() in session_info['days']:
                tomorrow_utc = tomorrow_local.astimezone(pytz.UTC)
                next_openings.append((tomorrow_utc, session_name.title()))
        
        if next_openings:
            # Find the earliest opening
            earliest = min(next_openings, key=lambda x: x[0])
            hours_until = (earliest[0] - utc_now).total_seconds() / 3600
            
            return f"{earliest[1]} in {hours_until:.1f} hours"
        else:
            return "Monday 8:00 AM Sydney time"
    
    def is_peak_trading_hours(self) -> Tuple[bool, str]:
        """
        Check if we're in peak trading hours (session overlaps)
        Returns: (is_peak, description)
        """
        utc_now = datetime.now(pytz.UTC)
        
        for peak_name, start_time, end_time, timezone_name in self.peak_hours:
            tz = pytz.timezone(timezone_name)
            local_time = utc_now.astimezone(tz)
            
            # Check if within peak hours and trading day
            if local_time.weekday() < 5:  # Monday-Friday
                start_hour, start_min = map(int, start_time.split(':'))
                end_hour, end_min = map(int, end_time.split(':'))
                
                start_time_obj = time(start_hour, start_min)
                end_time_obj = time(end_hour, end_min)
                
                current_time = local_time.time()
                
                if start_time_obj <= current_time <= end_time_obj:
                    return True, f"Peak Hours: {peak_name}"
        
        return False, "Off-peak trading hours"
    
    def get_trading_quality_score(self) -> Tuple[float, str]:
        """
        Get a score (0-1) indicating trading activity quality
        1.0 = Peak hours, multiple markets open
        0.5 = Single market open
        0.0 = Markets closed
        """
        is_open, open_description = self.is_market_open()
        is_peak, peak_description = self.is_peak_trading_hours()
        
        if not is_open:
            return 0.0, "Markets Closed - No trading activity"
        
        # Count active sessions
        active_count = len([s for s in open_description.split(',') if 'Open:' in open_description])
        
        if is_peak:
            score = 1.0
            description = f"Optimal Trading - {peak_description}"
        elif active_count >= 2:
            score = 0.8
            description = f"Good Trading - Multiple sessions active"
        elif active_count == 1:
            score = 0.5
            description = f"Moderate Trading - Single session active"
        else:
            score = 0.3
            description = f"Low Activity - Limited session overlap"
        
        return score, description
    
    def should_run_trading_system(self) -> Tuple[bool, str, Dict]:
        """
        Main decision function: Should the trading system be running?
        Returns: (should_run, reason, details)
        """
        is_open, market_status = self.is_market_open()
        quality_score, quality_desc = self.get_trading_quality_score()
        is_peak, peak_status = self.is_peak_trading_hours()
        
        # Decision logic
        if not is_open:
            should_run = False
            reason = "💤 SLEEP MODE: Markets closed - conserving energy"
        elif quality_score >= 0.8:
            should_run = True
            reason = "🚀 ACTIVE TRADING: Optimal market conditions"
        elif quality_score >= 0.5:
            should_run = True
            reason = "📈 STANDARD TRADING: Good market conditions"
        else:
            should_run = True  # Still trade, but note low activity
            reason = "⚡ MINIMAL TRADING: Limited market activity"
        
        details = {
            'market_open': is_open,
            'market_status': market_status,
            'quality_score': quality_score,
            'quality_description': quality_desc,
            'is_peak_hours': is_peak,
            'peak_status': peak_status,
            'current_utc': datetime.now(pytz.UTC).isoformat(),
            'energy_conservation_mode': not should_run
        }
        
        return should_run, reason, details
    
    def get_sleep_duration(self) -> int:
        """
        Get recommended sleep duration in seconds when markets are closed
        """
        if self.is_market_open()[0]:
            return 300  # 5 minutes during market hours
        
        # Calculate time until next market open
        utc_now = datetime.now(pytz.UTC)
        next_openings = []
        
        for session_name, session_info in self.sessions.items():
            local_now = utc_now.astimezone(session_info['timezone'])
            
            # Check if market opens later today
            if (local_now.weekday() in session_info['days'] and 
                local_now.time() < session_info['open']):
                
                next_open_local = local_now.replace(
                    hour=session_info['open'].hour,
                    minute=session_info['open'].minute,
                    second=0,
                    microsecond=0
                )
                next_open_utc = next_open_local.astimezone(pytz.UTC)
                next_openings.append(next_open_utc)
            
            # Check tomorrow
            for days_ahead in range(1, 8):  # Check next 7 days
                future_local = local_now + timedelta(days=days_ahead)
                if future_local.weekday() in session_info['days']:
                    future_open_local = future_local.replace(
                        hour=session_info['open'].hour,
                        minute=session_info['open'].minute,
                        second=0,
                        microsecond=0
                    )
                    future_open_utc = future_open_local.astimezone(pytz.UTC)
                    next_openings.append(future_open_utc)
                    break
        
        if next_openings:
            earliest_open = min(next_openings)
            sleep_seconds = int((earliest_open - utc_now).total_seconds())
            
            # Cap sleep at 4 hours for safety
            return min(sleep_seconds, 14400)
        
        return 3600  # Default 1 hour sleep

def test_forex_scheduler():
    """Test the forex market scheduler"""
    scheduler = ForexMarketScheduler()
    
    print("\n🕒 FOREX MARKET SCHEDULER TEST")
    print("=" * 40)
    
    should_run, reason, details = scheduler.should_run_trading_system()
    
    print(f"📊 Current Status:")
    print(f"   Should run trading: {should_run}")
    print(f"   Reason: {reason}")
    print(f"   Market status: {details['market_status']}")
    print(f"   Quality score: {details['quality_score']:.1f}")
    print(f"   Peak hours: {details['is_peak_hours']}")
    
    if not should_run:
        sleep_duration = scheduler.get_sleep_duration()
        print(f"   Recommended sleep: {sleep_duration/3600:.1f} hours")
    
    print(f"\n⏰ Detailed Market Info:")
    for key, value in details.items():
        if key not in ['current_utc']:
            print(f"   {key}: {value}")

if __name__ == "__main__":
    test_forex_scheduler()