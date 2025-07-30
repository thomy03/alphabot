#!/usr/bin/env python3
"""
Working Regime Debug - Version qui fonctionne vraiment
"""

import pandas as pd
import numpy as np
import yfinance as yf

def working_regime_test():
    """Test de détection de régime qui fonctionne"""
    
    print("🔍 WORKING REGIME DEBUG")
    print("="*50)
    
    # Download SPY
    print("📊 Downloading SPY...")
    spy_data = yf.download('SPY', start="2019-01-01", end="2024-01-01", progress=False)
    print("✅ SPY downloaded:", len(spy_data), "days")
    
    # Test sur COVID crash - utilisons pandas directement
    test_date = pd.to_datetime("2020-03-20")
    print("\n📅 Testing COVID crash: 2020-03-20")
    
    # Get data using pandas (plus safe)
    historical_spy = spy_data[spy_data.index <= test_date]
    closes = historical_spy['Close']  # Keep as pandas Series
    
    print("📊 Historical data:", len(closes), "days")
    print("📈 Current price: $", round(closes.iloc[-1], 2))
    
    # Calculate indicators using pandas (safer)
    print("\n🔢 CALCULATING INDICATORS:")
    
    # 1. Moving averages
    ma_10 = closes.rolling(10).mean().iloc[-1]
    ma_30 = closes.rolling(30).mean().iloc[-1]
    
    print("  MA10: $", round(ma_10, 2))
    print("  MA30: $", round(ma_30, 2))
    print("  Trend:", "UP" if ma_10 > ma_30 else "DOWN")
    
    # 2. Volatility (20-day) using pandas
    returns = closes.pct_change().dropna()
    volatility = returns.tail(20).std() * np.sqrt(252)
    
    print("  Volatility (20d):", str(round(volatility * 100, 1)) + "%")
    
    # 3. Momentum (10-day)
    if len(closes) >= 11:
        momentum = (closes.iloc[-1] / closes.iloc[-11]) - 1
        print("  Momentum (10d):", str(round(momentum * 100, 1)) + "%")
    else:
        momentum = 0
        print("  Momentum (10d): N/A")
    
    # 4. Drawdown (30-day)
    if len(closes) >= 30:
        rolling_max = closes.rolling(30).max().iloc[-1]
        drawdown = (closes.iloc[-1] / rolling_max) - 1
        print("  Max 30d: $", round(rolling_max, 2))
        print("  Drawdown:", str(round(drawdown * 100, 1)) + "%")
    else:
        drawdown = 0
        print("  Drawdown: N/A")
    
    # Test conditions with CURRENT thresholds
    print("\n🧠 TESTING CURRENT THRESHOLDS:")
    
    is_uptrend = ma_10 > ma_30
    is_high_vol = volatility > 0.18  # 18%
    is_strong_momentum = abs(momentum) > 0.015  # 1.5%
    is_crisis = drawdown < -0.10  # -10%
    
    print("  is_uptrend:", is_uptrend)
    print("  is_high_vol:", is_high_vol, "(", str(round(volatility * 100, 1)) + "% > 18%)")
    print("  is_strong_momentum:", is_strong_momentum, "(|" + str(round(momentum * 100, 1)) + "%| > 1.5%)")
    print("  is_crisis:", is_crisis, "(" + str(round(drawdown * 100, 1)) + "% < -10%)")
    
    # Current regime
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
    
    # Test different threshold levels
    print("\n🔥 TESTING DIFFERENT THRESHOLDS:")
    
    threshold_tests = [
        ("LEVEL 1", 0.15, 0.01, -0.08),   # Slightly looser
        ("LEVEL 2", 0.12, 0.005, -0.05),  # Moderately loose
        ("LEVEL 3", 0.08, 0.002, -0.03),  # Very loose
        ("EMERGENCY", 0.05, 0.001, -0.01) # Ultra loose
    ]
    
    for level_name, vol_thresh, mom_thresh, crisis_thresh in threshold_tests:
        l_high_vol = volatility > vol_thresh
        l_momentum = abs(momentum) > mom_thresh
        l_crisis = drawdown < crisis_thresh
        
        print(f"  {level_name} ({vol_thresh*100:.0f}%, {mom_thresh*100:.1f}%, {crisis_thresh*100:.0f}%):")
        print(f"    vol: {l_high_vol}, mom: {l_momentum}, crisis: {l_crisis}")
        
        # Determine regime for this level
        if l_crisis:
            l_regime = "BEAR_TREND"
        elif not is_uptrend and l_momentum:
            l_regime = "BEAR_TREND"  
        elif l_high_vol:
            l_regime = "HIGH_VOLATILITY"
        else:
            l_regime = "CONSOLIDATION"
        
        print(f"    → {level_name} regime: {l_regime}")
    
    # Test key COVID dates
    print("\n📅 COVID TIMELINE ANALYSIS:")
    
    covid_dates = [
        ("2020-02-19", "Pre-crash (SPY peak)"),
        ("2020-03-01", "Decline begins"),
        ("2020-03-15", "Major drop"),
        ("2020-03-20", "Near bottom"),
        ("2020-03-23", "Actual bottom"),
        ("2020-04-01", "Bounce starts"),
        ("2020-04-15", "Recovery underway")
    ]
    
    for date_str, description in covid_dates:
        try:
            test_date_specific = pd.to_datetime(date_str)
            historical_specific = spy_data[spy_data.index <= test_date_specific]
            
            if len(historical_specific) < 30:
                continue
                
            closes_specific = historical_specific['Close']
            current_price_specific = closes_specific.iloc[-1]
            
            # Key metrics
            if len(closes_specific) >= 30:
                rolling_max_specific = closes_specific.rolling(30).max().iloc[-1]
                drawdown_specific = (current_price_specific / rolling_max_specific) - 1
            else:
                drawdown_specific = 0
                
            if len(closes_specific) >= 20:
                returns_specific = closes_specific.pct_change().dropna()
                vol_specific = returns_specific.tail(20).std() * np.sqrt(252)
            else:
                vol_specific = 0.15
            
            print(f"  {date_str} - {description}:")
            print(f"    Price: ${round(current_price_specific, 2)}")
            print(f"    Drawdown: {round(drawdown_specific * 100, 1)}%")
            print(f"    Volatility: {round(vol_specific * 100, 1)}%")
            
            # Test Level 2 triggers (reasonable thresholds)
            triggers = []
            if drawdown_specific < -0.05:  # -5%
                triggers.append("crisis")
            if vol_specific > 0.12:  # 12%
                triggers.append("high_vol")
            if len(closes_specific) >= 11:
                momentum_specific = (closes_specific.iloc[-1] / closes_specific.iloc[-11]) - 1
                if abs(momentum_specific) > 0.005:  # 0.5%
                    triggers.append("momentum")
            
            print(f"    Level 2 triggers: {triggers if triggers else 'NONE'}")
            
        except Exception as e:
            print(f"  {date_str}: Error - {e}")
    
    # Final recommendations
    print("\n💡 FINAL ANALYSIS:")
    
    print("CURRENT PROBLEMS:")
    print("- Current thresholds are TOO CONSERVATIVE")
    print("- COVID crash should definitely trigger regime change")
    print("- Volatility", str(round(volatility * 100, 1)) + "% during crash")
    print("- Drawdown", str(round(drawdown * 100, 1)) + "% during crash")
    
    print("\nRECOMMENDED SOLUTION:")
    if volatility > 0.12 or abs(momentum) > 0.005 or drawdown < -0.05:
        print("✅ Use LEVEL 2 thresholds:")
        print("  - Volatility: 18% → 12%")
        print("  - Momentum: 1.5% → 0.5%")
        print("  - Crisis: -10% → -5%")
    else:
        print("🚨 Use EMERGENCY thresholds:")
        print("  - Volatility: 18% → 5%")
        print("  - Momentum: 1.5% → 0.1%")
        print("  - Crisis: -10% → -1%")
    
    print("\nThis should finally capture:")
    print("- COVID crash March 2020")
    print("- Bear market 2022")  
    print("- High volatility periods")
    print("- Recovery phases")


if __name__ == "__main__":
    working_regime_test()