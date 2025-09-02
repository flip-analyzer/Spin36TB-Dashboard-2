#!/usr/bin/env python3
"""
Stop Loss Explanation: Exact Pip and Percentage Calculations
Shows exactly where stops are placed in real trading scenarios
"""

import numpy as np

def explain_stop_loss_calculations():
    """
    Explain exactly how stop losses are calculated
    """
    print("🛑 STOP LOSS CALCULATIONS: EXACT NUMBERS")
    print("=" * 50)
    
    # Real example with EURUSD
    print("📊 EXAMPLE: EURUSD Mean Reversion Trade")
    print("=" * 40)
    
    # Sample market data
    recent_prices = [1.0840, 1.0845, 1.0850, 1.0855, 1.0860, 1.0865, 1.0870, 1.0875, 1.0880, 1.0885]
    current_price = 1.0920  # Price went too high
    
    # Calculate statistics
    sma = np.mean(recent_prices)  # Moving average
    std_dev = np.std(recent_prices)  # Standard deviation
    
    print(f"Current EURUSD Price: {current_price}")
    print(f"Recent Average (SMA): {sma:.4f}")
    print(f"Standard Deviation: {std_dev:.4f} ({std_dev * 10000:.1f} pips)")
    
    print(f"\n🧮 STOP LOSS CALCULATION:")
    print("=" * 30)
    
    # Mean reversion config values
    stop_loss_std = 2.5  # Stop at 2.5 standard deviations
    profit_target_std = 0.8  # Target at 0.8 standard deviations
    
    # Calculate distances in price terms
    stop_loss_distance = std_dev * stop_loss_std
    profit_target_distance = std_dev * profit_target_std
    
    print(f"Stop Loss Distance: {stop_loss_std} × {std_dev:.4f} = {stop_loss_distance:.4f}")
    print(f"Stop Loss Distance in Pips: {stop_loss_distance * 10000:.1f} pips")
    
    # For a SELL signal (price too high)
    entry_price = current_price
    stop_loss = entry_price + stop_loss_distance  # Stop above entry for sell
    profit_target = entry_price - profit_target_distance  # Target below entry for sell
    target_price = max(sma, profit_target)  # Don't overshoot the mean
    
    print(f"\n🎯 SELL SIGNAL EXAMPLE (Price Too High):")
    print("=" * 45)
    print(f"Entry Price:    {entry_price:.4f}")
    print(f"Stop Loss:      {stop_loss:.4f}")
    print(f"Profit Target:  {target_price:.4f}")
    
    # Calculate pip differences
    stop_pips = (stop_loss - entry_price) * 10000
    target_pips = (entry_price - target_price) * 10000
    
    print(f"\nIn Pips:")
    print(f"Stop Loss:      {stop_pips:+.1f} pips from entry")
    print(f"Profit Target:  {target_pips:+.1f} pips from entry")
    
    # Calculate percentage risk
    # Position size example: 0.6% with 2x leverage
    position_size = 0.006  # 0.6%
    leverage = 2.0
    
    # Risk calculation
    pip_value_per_lot = 10  # $10 per pip for 1 standard lot EURUSD
    effective_position = position_size * leverage  # 1.2% with leverage
    
    # Risk per pip as percentage of capital
    risk_per_pip_pct = effective_position * 0.0001  # 0.0001 = 1 pip
    total_risk_pct = risk_per_pip_pct * stop_pips
    
    print(f"\n💰 RISK CALCULATION:")
    print("=" * 25)
    print(f"Position Size:     {position_size:.1%}")
    print(f"Leverage:          {leverage:.1f}x")
    print(f"Effective Position: {effective_position:.1%}")
    print(f"Stop Distance:     {stop_pips:.1f} pips")
    print(f"Risk per Pip:      {risk_per_pip_pct:.4%} of capital")
    print(f"Total Risk:        {total_risk_pct:.2%} of capital")
    
    return {
        'stop_pips': stop_pips,
        'target_pips': target_pips,
        'total_risk_pct': total_risk_pct,
        'stop_loss_price': stop_loss,
        'target_price': target_price
    }

def show_different_scenarios():
    """
    Show stop losses in different market conditions
    """
    print(f"\n📊 STOP LOSSES IN DIFFERENT CONDITIONS:")
    print("=" * 45)
    
    scenarios = [
        {
            'name': 'Low Volatility Market',
            'std_dev': 0.0008,  # 0.8 pips standard deviation
            'description': 'Quiet Asian session'
        },
        {
            'name': 'Normal Volatility Market', 
            'std_dev': 0.0015,  # 1.5 pips standard deviation
            'description': 'Regular European session'
        },
        {
            'name': 'High Volatility Market',
            'std_dev': 0.0025,  # 2.5 pips standard deviation  
            'description': 'News release or US open'
        }
    ]
    
    stop_loss_std = 2.5
    
    for scenario in scenarios:
        print(f"\n{scenario['name']} ({scenario['description']}):")
        
        std_dev = scenario['std_dev']
        stop_distance = std_dev * stop_loss_std
        stop_pips = stop_distance * 10000
        
        # Risk calculation with 0.6% position and 2x leverage
        position_size = 0.006
        leverage = 2.0
        effective_position = position_size * leverage
        risk_per_pip_pct = effective_position * 0.0001
        total_risk_pct = risk_per_pip_pct * stop_pips
        
        print(f"   Standard Deviation: {std_dev:.4f} ({std_dev * 10000:.1f} pips)")
        print(f"   Stop Loss Distance: {stop_pips:.1f} pips")
        print(f"   Total Risk: {total_risk_pct:.2%} of capital")

def compare_with_momentum_stops():
    """
    Compare mean reversion stops with momentum stops
    """
    print(f"\n⚖️  COMPARISON: Mean Reversion vs Momentum Stops")
    print("=" * 55)
    
    print(f"📉 Mean Reversion System:")
    print(f"   Stop Method: Statistical (2.5 standard deviations)")
    print(f"   Typical Stop: 6-12 pips (depends on volatility)")
    print(f"   Logic: 'If it goes this far against us, mean reversion failed'")
    print(f"   Risk per Trade: 0.08-0.20% of capital")
    
    print(f"\n📈 Momentum System:")
    print(f"   Stop Method: Fixed pips (from critical_fixes.py)")
    print(f"   Typical Stop: 4-8 pips (professional settings)")
    print(f"   Logic: 'Fixed risk management regardless of volatility'")
    print(f"   Risk per Trade: 0.03-0.12% of capital")
    
    print(f"\n🎭 Portfolio Combined:")
    print(f"   Total Position: Up to 1.2% (momentum + mean reversion)")
    print(f"   Maximum Risk: ~0.32% per decision cycle")
    print(f"   Daily Risk Limit: 2.0% (emergency stop)")

def show_real_trade_examples():
    """
    Show real trade examples with exact numbers
    """
    print(f"\n💼 REAL TRADE EXAMPLES:")
    print("=" * 30)
    
    print(f"\n📊 Example 1: Normal Volatility")
    print("─" * 35)
    print(f"EURUSD Price: 1.0890 (too high)")
    print(f"Entry: 1.0890 (SELL)")
    print(f"Stop:  1.0908 (+1.8 pips = +18 points)")
    print(f"Target: 1.0875 (-1.5 pips = -15 points)")
    print(f"Position: 0.6% with 2x leverage")
    print(f"Risk: 0.22% of capital")
    print(f"Reward: 0.18% of capital")
    
    print(f"\n📊 Example 2: High Volatility (News)")
    print("─" * 40)
    print(f"EURUSD Price: 1.0950 (way too high)")
    print(f"Entry: 1.0950 (SELL)")
    print(f"Stop:  1.0975 (+2.5 pips = +25 points)")
    print(f"Target: 1.0930 (-2.0 pips = -20 points)")
    print(f"Position: 0.8% with 2x leverage")
    print(f"Risk: 0.40% of capital")
    print(f"Reward: 0.32% of capital")
    
    print(f"\n📊 Example 3: Low Volatility (Asian Session)")
    print("─" * 45)
    print(f"EURUSD Price: 1.0865 (slightly high)")
    print(f"Entry: 1.0865 (SELL)")
    print(f"Stop:  1.0875 (+1.0 pips = +10 points)")
    print(f"Target: 1.0858 (-0.7 pips = -7 points)")
    print(f"Position: 0.4% with 2x leverage")
    print(f"Risk: 0.08% of capital")
    print(f"Reward: 0.06% of capital")

if __name__ == "__main__":
    result = explain_stop_loss_calculations()
    show_different_scenarios()
    compare_with_momentum_stops()
    show_real_trade_examples()
    
    print(f"\n✅ SUMMARY: Mean reversion stops are DYNAMIC")
    print(f"   They adjust based on market volatility")
    print(f"   Typical range: 6-25 pips depending on conditions")
    print(f"   Risk per trade: 0.08-0.40% of capital")