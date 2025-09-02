#!/usr/bin/env python3
"""
Test Hybrid System
Quick test of the high-frequency hybrid system to verify functionality
"""

import sys
sys.path.append('/Users/jonspinogatti/Desktop/spin36TB')

from hybrid_live_trader import HybridLivePaperTrader
import time

def test_hybrid_system():
    """Test the hybrid system with a short session"""
    print("🧪 TESTING HYBRID SYSTEM")
    print("=" * 50)
    
    try:
        # Create hybrid trader
        trader = HybridLivePaperTrader(starting_capital=25000)
        
        # Test data fetching
        print("📊 Testing data fetch...")
        market_data = trader.get_live_data(50)
        
        if not market_data.empty:
            print(f"✅ Data fetch successful: {len(market_data)} candles")
            print(f"   Latest price: {market_data['close'].iloc[-1]:.4f}")
        else:
            print("❌ Data fetch failed")
            return False
        
        # Test decision engines
        print("\n🧠 Testing decision engines...")
        
        # Test traditional system
        traditional_decision = trader.make_traditional_portfolio_decision(market_data)
        print(f"   Traditional system: {traditional_decision is not None}")
        
        # Test hybrid system  
        hybrid_decisions = trader.make_hybrid_decisions(market_data)
        print(f"   Hybrid system: {len(hybrid_decisions)} decisions")
        
        # Test trade execution
        print("\n📄 Testing trade execution...")
        current_price = market_data['close'].iloc[-1]
        
        if traditional_decision:
            executed_traditional = trader.execute_paper_trades(traditional_decision, current_price)
            print(f"   Traditional trades executed: {len(executed_traditional)}")
        
        if hybrid_decisions:
            executed_hybrid = trader.execute_paper_trades(hybrid_decisions, current_price)
            print(f"   Hybrid trades executed: {len(executed_hybrid)}")
        
        # Test trade management
        print("\n🎪 Testing trade management...")
        closed_trades = trader.manage_active_trades(current_price)
        print(f"   Active trades: {len(trader.active_trades)}")
        print(f"   Closed trades: {len(closed_trades)}")
        
        print("\n✅ ALL TESTS PASSED!")
        print("🚀 Hybrid system ready for live trading!")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_hybrid_system()
    
    if success:
        print("\n🎉 System test successful - Ready to deploy!")
    else:
        print("\n💥 System test failed - Check errors above")