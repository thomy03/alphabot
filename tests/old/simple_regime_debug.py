#!/usr/bin/env python3
"""
Simple Regime Debug - Version simplifiée pour identifier le problème
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

def simple_regime_test():
    """Test simple de détection de régime"""
    
    print("🔍 SIMPLE REGIME DEBUG")
    print("="*50)
    
    # Download SPY
    print("📊 Downloading SPY...")
    spy_data = yf.download('SPY', start="2019-01-01", end="2024-01-01", progress=False)
    print(f"✅ SPY downloaded: {len(spy_data)} days")
    
    # Test sur une date COVID spécifique
    test_date = pd.to_datetime("2020-03-20")  # COVID crash
    print(f"\n📅 Testing COVID crash date: {test_date.strftime('%Y-%m-%d')}")
    
    # Get historical data
    historical_spy = spy_data[spy_data.index <= test_date]
    closes = historical_spy['Close']
    
    print(f"📊 Historical data: {len(closes)} days")
    print(f"📈 Current price: ${closes.iloc[-1]:.2f}")
    
    # Calculate indicators step by step
    print(f"\n🔢 CALCULATING INDICATORS:")
    
    # 1. Moving averages
    ma_10_series = closes.rolling(10).mean()
    ma_30_series = closes.rolling(30).mean()
    
    ma_10 = ma_10_series.iloc[-1]
    ma_30 = ma_30_series.iloc[-1]
    
    print(f"  MA10: ${ma_10:.2f}")
    print(f"  MA30: ${ma_30:.2f}")
    print(f"  Trend: {'UP' if ma_10 > ma_30 else 'DOWN'}")
    
    # 2. Volatility
    returns = closes.pct_change().dropna()
    vol_series = returns.tail(20).std() * np.sqrt(252)
    volatility = vol_series
    
    print(f"  Volatility (20d): {volatility:.1%}")
    
    # 3. Momentum  
    if len(closes) >= 11:
        momentum = (closes.iloc[-1] / closes.iloc[-11]) - 1
        print(f"  Momentum (10d): {momentum:.1%}")
    else:
        momentum = 0
        print(f"  Momentum (10d): N/A")
    
    # 4. Drawdown
    if len(closes) >= 30:
        rolling_max_series = closes.rolling(30).max()
        rolling_max = rolling_max_series.iloc[-1]
        drawdown = (closes.iloc[-1] / rolling_max) - 1
        print(f"  Max 30d: ${rolling_max:.2f}")
        print(f"  Drawdown: {drawdown:.1%}")
    else:
        drawdown = 0
        print(f"  Drawdown: N/A")
    
    # Test conditions with CURRENT thresholds
    print(f"\n🧠 TESTING CONDITIONS (Current thresholds):")
    
    is_uptrend = ma_10 > ma_30
    is_high_vol = volatility > 0.18  # 18%
    is_strong_momentum = abs(momentum) > 0.015  # 1.5%
    is_crisis = drawdown < -0.10  # -10%
    
    print(f"  is_uptrend: {is_uptrend}")
    print(f"  is_high_vol: {is_high_vol} (vol {volatility:.1%} > 18%)")
    print(f"  is_strong_momentum: {is_strong_momentum} (|{momentum:.1%}| > 1.5%)")
    print(f"  is_crisis: {is_crisis} ({drawdown:.1%} < -10%)")
    
    # Determine regime
    if is_crisis:
        regime = "bear_trend (crisis)"
    elif is_uptrend and is_strong_momentum:
        regime = "bull_trend"
    elif not is_uptrend and is_strong_momentum:
        regime = "bear_trend"
    elif is_high_vol:
        regime = "high_volatility"
    else:
        regime = "consolidation"
    
    print(f"  🎯 REGIME: {regime}")
    
    # Test ULTRA-AGGRESSIVE thresholds
    print(f"\n🔥 TESTING ULTRA-AGGRESSIVE thresholds:")
    
    ultra_high_vol = volatility > 0.12  # 12%
    ultra_momentum = abs(momentum) > 0.008  # 0.8%
    ultra_crisis = drawdown < -0.05  # -5%
    
    print(f"  ultra_high_vol: {ultra_high_vol} (vol {volatility:.1%} > 12%)")
    print(f"  ultra_momentum: {ultra_momentum} (|{momentum:.1%}| > 0.8%)")
    print(f"  ultra_crisis: {ultra_crisis} ({drawdown:.1%} < -5%)")
    
    # Ultra regime
    if ultra_crisis:
        ultra_regime = "bear_trend (ultra crisis)"
    elif is_uptrend and ultra_momentum:
        ultra_regime = "bull_trend (ultra)"
    elif not is_uptrend and ultra_momentum:
        ultra_regime = "bear_trend (ultra)"
    elif ultra_high_vol:
        ultra_regime = "high_volatility (ultra)"
    else:
        ultra_regime = "consolidation (ultra)"
    
    print(f"  🎯 ULTRA REGIME: {ultra_regime}")
    
    # Test multiple COVID dates
    print(f"\n📅 TESTING MULTIPLE COVID DATES:")
    
    covid_dates = [
        "2020-02-20",  # Pre-crash
        "2020-03-01",  # Start decline
        "2020-03-15",  # Mid crash
        "2020-03-23",  # Bottom
        "2020-04-15"   # Recovery
    ]
    
    for date_str in covid_dates:
        test_date = pd.to_datetime(date_str)
        historical = spy_data[spy_data.index <= test_date]
        if len(historical) < 30:
            continue
            
        closes = historical['Close']
        
        # Quick calculations
        current_price = closes.iloc[-1]
        rolling_max = closes.rolling(30).max().iloc[-1] 
        drawdown = (current_price / rolling_max) - 1
        
        returns = closes.pct_change().dropna()
        vol = returns.tail(10).std() * np.sqrt(252)
        
        print(f"  {date_str}: Price=${current_price:.2f}, DD={drawdown:.1%}, Vol={vol:.1%}")
        
        # Test if ANY threshold triggers
        triggers = []
        if drawdown < -0.05:
            triggers.append("crisis")
        if vol > 0.12:
            triggers.append("high_vol")
        if len(closes) >= 11:
            momentum = (closes.iloc[-1] / closes.iloc[-11]) - 1
            if abs(momentum) > 0.008:
                triggers.append("momentum")
        
        print(f"    Triggers: {triggers if triggers else 'NONE'}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    print(f"  If COVID crash doesn't trigger regime change, try:")
    print(f"  1. Volatility: 18% → 10%")
    print(f"  2. Momentum: 1.5% → 0.5%") 
    print(f"  3. Crisis: -10% → -3%")
    print(f"  4. Use 5-day periods instead of 10-day")


if __name__ == "__main__":
    simple_regime_test()