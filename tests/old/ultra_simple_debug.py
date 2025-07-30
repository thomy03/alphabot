#!/usr/bin/env python3
"""
Ultra Simple Debug - Version sans erreurs pandas
"""

import pandas as pd
import numpy as np
import yfinance as yf

def ultra_simple_test():
    """Test ultra-simple sans erreurs pandas"""
    
    print("🔍 ULTRA SIMPLE REGIME DEBUG")
    print("="*50)
    
    # Download SPY
    print("📊 Downloading SPY...")
    spy_data = yf.download('SPY', start="2019-01-01", end="2024-01-01", progress=False)
    print(f"✅ SPY downloaded: {len(spy_data)} days")
    
    # Test sur COVID crash
    test_date = pd.to_datetime("2020-03-20")
    print(f"\n📅 Testing COVID crash: 2020-03-20")
    
    # Get data
    historical_spy = spy_data[spy_data.index <= test_date]
    closes = historical_spy['Close'].values  # Convert to numpy array!
    
    print(f"📊 Historical data: {len(closes)} days")
    print(f"📈 Current price: ${closes[-1]:.2f}")
    
    # Calculate with numpy (no pandas issues)
    print(f"\n🔢 CALCULATING INDICATORS:")
    
    # 1. Moving averages
    ma_10 = np.mean(closes[-10:])
    ma_30 = np.mean(closes[-30:])
    
    print(f"  MA10: ${ma_10:.2f}")
    print(f"  MA30: ${ma_30:.2f}")
    print(f"  Trend: {'UP' if ma_10 > ma_30 else 'DOWN'}")
    
    # 2. Volatility (20-day)
    returns = np.diff(closes) / closes[:-1]  # Daily returns
    volatility = np.std(returns[-20:]) * np.sqrt(252)
    
    print(f"  Volatility (20d): {volatility:.1%}")
    
    # 3. Momentum (10-day)
    if len(closes) >= 11:
        momentum = (closes[-1] / closes[-11]) - 1
        print(f"  Momentum (10d): {momentum:.1%}")
    else:
        momentum = 0
        print(f"  Momentum (10d): N/A")
    
    # 4. Drawdown (30-day)
    if len(closes) >= 30:
        rolling_max = np.max(closes[-30:])
        drawdown = (closes[-1] / rolling_max) - 1
        print(f"  Max 30d: ${rolling_max:.2f}")
        print(f"  Drawdown: {drawdown:.1%}")
    else:
        drawdown = 0
        print(f"  Drawdown: N/A")
    
    # Test conditions
    print(f"\n🧠 TESTING CONDITIONS:")
    
    # Current thresholds
    is_uptrend = ma_10 > ma_30
    is_high_vol = volatility > 0.18  # 18%
    is_strong_momentum = abs(momentum) > 0.015  # 1.5%
    is_crisis = drawdown < -0.10  # -10%
    
    print(f"  is_uptrend: {is_uptrend}")
    print(f"  is_high_vol: {is_high_vol} (vol {volatility:.1%} > 18%)")
    print(f"  is_strong_momentum: {is_strong_momentum} (|{momentum:.1%}| > 1.5%)")
    print(f"  is_crisis: {is_crisis} ({drawdown:.1%} < -10%)")
    
    # Current regime logic
    if is_crisis:
        regime = "BEAR_TREND (crisis)"
    elif is_uptrend and is_strong_momentum:
        regime = "BULL_TREND"
    elif not is_uptrend and is_strong_momentum:
        regime = "BEAR_TREND"
    elif is_high_vol:
        regime = "HIGH_VOLATILITY"
    else:
        regime = "CONSOLIDATION"
    
    print(f"  🎯 CURRENT REGIME: {regime}")
    
    # Test progressively looser thresholds
    print(f"\n🔥 TESTING LOOSER THRESHOLDS:")
    
    # Level 1: Slightly looser
    l1_high_vol = volatility > 0.15  # 15%
    l1_momentum = abs(momentum) > 0.01  # 1%
    l1_crisis = drawdown < -0.08  # -8%
    
    print(f"  Level 1 (15%, 1%, -8%):")
    print(f"    high_vol: {l1_high_vol}, momentum: {l1_momentum}, crisis: {l1_crisis}")
    
    # Level 2: Much looser  
    l2_high_vol = volatility > 0.12  # 12%
    l2_momentum = abs(momentum) > 0.005  # 0.5%
    l2_crisis = drawdown < -0.05  # -5%
    
    print(f"  Level 2 (12%, 0.5%, -5%):")
    print(f"    high_vol: {l2_high_vol}, momentum: {l2_momentum}, crisis: {l2_crisis}")
    
    # Level 3: Very loose
    l3_high_vol = volatility > 0.08  # 8%
    l3_momentum = abs(momentum) > 0.002  # 0.2%
    l3_crisis = drawdown < -0.03  # -3%
    
    print(f"  Level 3 (8%, 0.2%, -3%):")
    print(f"    high_vol: {l3_high_vol}, momentum: {l3_momentum}, crisis: {l3_crisis}")
    
    # Test which level would trigger
    for level, (vol_thresh, mom_thresh, crisis_thresh, vol_test, mom_test, crisis_test) in enumerate([
        (0.15, 0.01, -0.08, l1_high_vol, l1_momentum, l1_crisis),
        (0.12, 0.005, -0.05, l2_high_vol, l2_momentum, l2_crisis),
        (0.08, 0.002, -0.03, l3_high_vol, l3_momentum, l3_crisis)
    ], 1):
        
        if crisis_test:
            regime_l = "BEAR_TREND (crisis)"
        elif not is_uptrend and mom_test:
            regime_l = "BEAR_TREND"
        elif vol_test:
            regime_l = "HIGH_VOLATILITY"
        else:
            regime_l = "CONSOLIDATION"
        
        print(f"    → Level {level} regime: {regime_l}")
    
    # Test multiple COVID dates
    print(f"\n📅 TESTING COVID TIMELINE:")
    
    covid_dates = [
        ("2020-02-20", "Pre-crash"),
        ("2020-03-01", "Start decline"),
        ("2020-03-15", "Mid crash"),
        ("2020-03-20", "Near bottom"),
        ("2020-03-23", "Actual bottom"),
        ("2020-04-01", "Early recovery"),
        ("2020-04-15", "Recovery")
    ]
    
    for date_str, desc in covid_dates:
        test_date = pd.to_datetime(date_str)
        historical = spy_data[spy_data.index <= test_date]
        
        if len(historical) < 30:
            continue
            
        closes_test = historical['Close'].values
        
        # Quick calcs
        current_price = closes_test[-1]
        
        if len(closes_test) >= 30:
            rolling_max = np.max(closes_test[-30:])
            drawdown_test = (current_price / rolling_max) - 1
        else:
            drawdown_test = 0
            
        if len(closes_test) >= 20:
            returns_test = np.diff(closes_test) / closes_test[:-1]
            vol_test = np.std(returns_test[-20:]) * np.sqrt(252)
        else:
            vol_test = 0.15
        
        print(f"  {date_str} ({desc}):")
        print(f"    Price: ${current_price:.2f}, Drawdown: {drawdown_test:.1%}, Vol: {vol_test:.1%}")
        
        # Test which triggers
        triggers = []
        if drawdown_test < -0.03:  # -3% threshold
            triggers.append("crisis")
        if vol_test > 0.08:  # 8% threshold
            triggers.append("high_vol")
        if len(closes_test) >= 11:
            momentum_test = (closes_test[-1] / closes_test[-11]) - 1
            if abs(momentum_test) > 0.002:  # 0.2% threshold
                triggers.append("momentum")
        
        print(f"    Triggers (loose): {triggers if triggers else 'NONE'}")
    
    # Final recommendations
    print(f"\n💡 FINAL RECOMMENDATIONS:")
    
    if drawdown < -0.03:
        print(f"  ✅ COVID crash SHOULD trigger with -3% drawdown threshold")
    else:
        print(f"  ❌ Even -3% drawdown doesn't trigger on COVID crash!")
    
    if volatility > 0.08:
        print(f"  ✅ COVID volatility SHOULD trigger with 8% threshold")
    else:
        print(f"  ❌ Even 8% volatility doesn't trigger on COVID crash!")
    
    print(f"\n🎯 SUGGESTED NEW THRESHOLDS:")
    print(f"  - Volatility: 18% → 8%")
    print(f"  - Momentum: 1.5% → 0.5%")
    print(f"  - Crisis: -10% → -3%")
    print(f"  - This should capture COVID crash and 2022 bear market")


if __name__ == "__main__":
    ultra_simple_test()