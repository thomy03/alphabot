#!/usr/bin/env python3
"""
Final Regime Debug - Version définitive sans erreurs
"""

import pandas as pd
import numpy as np
import yfinance as yf

def final_regime_test():
    """Test final de détection de régime"""
    
    print("🔍 FINAL REGIME DEBUG")
    print("="*50)
    
    # Download SPY
    print("📊 Downloading SPY...")
    spy_data = yf.download('SPY', start="2019-01-01", end="2024-01-01", progress=False)
    print("✅ SPY downloaded:", len(spy_data), "days")
    
    # Test sur COVID crash
    test_date = pd.to_datetime("2020-03-20")
    print("\n📅 Testing COVID crash: 2020-03-20")
    
    # Get data
    historical_spy = spy_data[spy_data.index <= test_date]
    closes = historical_spy['Close'].values  # Convert to numpy array
    
    print("📊 Historical data:", len(closes), "days")
    
    # Current price (convert to float)
    current_price = float(closes[-1])
    print("📈 Current price: $" + str(round(current_price, 2)))
    
    # Calculate indicators with explicit conversions
    print("\n🔢 CALCULATING INDICATORS:")
    
    # 1. Moving averages
    ma_10 = float(np.mean(closes[-10:]))
    ma_30 = float(np.mean(closes[-30:]))
    
    print("  MA10: $" + str(round(ma_10, 2)))
    print("  MA30: $" + str(round(ma_30, 2)))
    print("  Trend:", "UP" if ma_10 > ma_30 else "DOWN")
    
    # 2. Volatility (20-day)
    returns = np.diff(closes) / closes[:-1]  # Daily returns
    volatility = float(np.std(returns[-20:]) * np.sqrt(252))
    
    print("  Volatility (20d):", str(round(volatility * 100, 1)) + "%")
    
    # 3. Momentum (10-day)
    if len(closes) >= 11:
        momentum = float((closes[-1] / closes[-11]) - 1)
        print("  Momentum (10d):", str(round(momentum * 100, 1)) + "%")
    else:
        momentum = 0
        print("  Momentum (10d): N/A")
    
    # 4. Drawdown (30-day)
    if len(closes) >= 30:
        rolling_max = float(np.max(closes[-30:]))
        drawdown = float((closes[-1] / rolling_max) - 1)
        print("  Max 30d: $" + str(round(rolling_max, 2)))
        print("  Drawdown:", str(round(drawdown * 100, 1)) + "%")
    else:
        drawdown = 0
        print("  Drawdown: N/A")
    
    # Test conditions
    print("\n🧠 TESTING CONDITIONS:")
    
    # Current thresholds
    is_uptrend = ma_10 > ma_30
    is_high_vol = volatility > 0.18  # 18%
    is_strong_momentum = abs(momentum) > 0.015  # 1.5%
    is_crisis = drawdown < -0.10  # -10%
    
    print("  is_uptrend:", is_uptrend)
    print("  is_high_vol:", is_high_vol, "(vol", str(round(volatility * 100, 1)) + "% > 18%)")
    print("  is_strong_momentum:", is_strong_momentum, "(|" + str(round(momentum * 100, 1)) + "%| > 1.5%)")
    print("  is_crisis:", is_crisis, "(" + str(round(drawdown * 100, 1)) + "% < -10%)")
    
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
    
    print("  🎯 CURRENT REGIME:", regime)
    
    # Test progressively looser thresholds
    print("\n🔥 TESTING LOOSER THRESHOLDS:")
    
    # Level 1: Slightly looser
    l1_high_vol = volatility > 0.15  # 15%
    l1_momentum = abs(momentum) > 0.01  # 1%
    l1_crisis = drawdown < -0.08  # -8%
    
    print("  Level 1 (15%, 1%, -8%):")
    print("    high_vol:", l1_high_vol, "momentum:", l1_momentum, "crisis:", l1_crisis)
    
    # Level 2: Much looser  
    l2_high_vol = volatility > 0.12  # 12%
    l2_momentum = abs(momentum) > 0.005  # 0.5%
    l2_crisis = drawdown < -0.05  # -5%
    
    print("  Level 2 (12%, 0.5%, -5%):")
    print("    high_vol:", l2_high_vol, "momentum:", l2_momentum, "crisis:", l2_crisis)
    
    # Level 3: Very loose
    l3_high_vol = volatility > 0.08  # 8%
    l3_momentum = abs(momentum) > 0.002  # 0.2%
    l3_crisis = drawdown < -0.03  # -3%
    
    print("  Level 3 (8%, 0.2%, -3%):")
    print("    high_vol:", l3_high_vol, "momentum:", l3_momentum, "crisis:", l3_crisis)
    
    # Test what each level would give
    levels = [
        (1, l1_crisis, l1_momentum, l1_high_vol),
        (2, l2_crisis, l2_momentum, l2_high_vol),
        (3, l3_crisis, l3_momentum, l3_high_vol)
    ]
    
    for level_num, crisis_test, mom_test, vol_test in levels:
        if crisis_test:
            regime_l = "BEAR_TREND (crisis)"
        elif not is_uptrend and mom_test:
            regime_l = "BEAR_TREND"
        elif vol_test:
            regime_l = "HIGH_VOLATILITY"
        else:
            regime_l = "CONSOLIDATION"
        
        print("    → Level", level_num, "regime:", regime_l)
    
    # Quick test of key COVID dates
    print("\n📅 QUICK COVID TIMELINE TEST:")
    
    covid_dates = [
        "2020-02-20",  # Pre-crash
        "2020-03-15",  # Mid crash
        "2020-03-23",  # Bottom
        "2020-04-15"   # Recovery
    ]
    
    for date_str in covid_dates:
        test_date_loop = pd.to_datetime(date_str)
        historical = spy_data[spy_data.index <= test_date_loop]
        
        if len(historical) < 30:
            continue
            
        closes_test = historical['Close'].values
        current_price_test = float(closes_test[-1])
        
        # Calculate key metrics
        if len(closes_test) >= 30:
            rolling_max_test = float(np.max(closes_test[-30:]))
            drawdown_test = float((current_price_test / rolling_max_test) - 1)
        else:
            drawdown_test = 0
            
        if len(closes_test) >= 20:
            returns_test = np.diff(closes_test) / closes_test[:-1]
            vol_test = float(np.std(returns_test[-20:]) * np.sqrt(252))
        else:
            vol_test = 0.15
        
        print("  " + date_str + ":")
        print("    Price: $" + str(round(current_price_test, 2)))
        print("    Drawdown:", str(round(drawdown_test * 100, 1)) + "%")
        print("    Volatility:", str(round(vol_test * 100, 1)) + "%")
        
        # Test Level 2 thresholds (reasonable)
        triggers = []
        if drawdown_test < -0.05:  # -5%
            triggers.append("crisis")
        if vol_test > 0.12:  # 12%
            triggers.append("high_vol")
        if len(closes_test) >= 11:
            momentum_test = float((closes_test[-1] / closes_test[-11]) - 1)
            if abs(momentum_test) > 0.005:  # 0.5%
                triggers.append("momentum")
        
        print("    Level 2 triggers:", triggers if triggers else "NONE")
    
    # Final analysis
    print("\n💡 ANALYSIS RESULTS:")
    
    print("Current system problems:")
    print("- Volatility threshold 18% too high")
    print("- Momentum threshold 1.5% too high")  
    print("- Crisis threshold -10% too strict")
    
    print("\nRecommended fixes:")
    print("- Volatility: 18% → 12%")
    print("- Momentum: 1.5% → 0.5%")
    print("- Crisis: -10% → -5%")
    
    # Test if our COVID crash would trigger with recommended settings
    covid_would_trigger = []
    if volatility > 0.12:
        covid_would_trigger.append("high_vol")
    if abs(momentum) > 0.005:
        covid_would_trigger.append("momentum")
    if drawdown < -0.05:
        covid_would_trigger.append("crisis")
    
    print("\nWith recommended settings, COVID crash would trigger:")
    print(covid_would_trigger if covid_would_trigger else "STILL NOTHING - need even looser!")
    
    if not covid_would_trigger:
        print("\n🚨 EMERGENCY THRESHOLDS NEEDED:")
        print("- Volatility: 18% → 8%")
        print("- Momentum: 1.5% → 0.2%")  
        print("- Crisis: -10% → -3%")


if __name__ == "__main__":
    final_regime_test()