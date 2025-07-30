#!/usr/bin/env python3
"""
Quick Validation - Test rapide performance système simplifié
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

def quick_performance_test():
    """Test rapide de performance"""
    
    print("🚀 QUICK VALIDATION - AlphaBot Simplified System")
    print("="*60)
    
    # Symboles de test
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'SPY']
    period = "2y"  # 2 ans seulement
    
    print(f"📊 Testing {len(symbols)} symbols over {period}")
    
    # 1. Télécharger données
    print("\n📥 Downloading data...")
    data = {}
    for symbol in symbols:
        try:
            ticker_data = yf.download(symbol, period=period, progress=False)
            if len(ticker_data) > 100:
                data[symbol] = ticker_data['Close']
                print(f"  ✅ {symbol}: {len(ticker_data)} days")
            else:
                print(f"  ❌ {symbol}: Insufficient data")
        except Exception as e:
            print(f"  ❌ {symbol}: Error - {str(e)[:30]}...")
    
    if len(data) < 3:
        print("❌ Insufficient data for validation")
        return
    
    # 2. Calculer signaux EMA+RSI pour chaque actif
    print("\n🎯 Calculating signals...")
    signals = {}
    
    for symbol, prices in data.items():
        try:
            # EMA
            ema_20 = prices.ewm(span=20).mean()
            ema_50 = prices.ewm(span=50).mean()
            ema_current = float(ema_20.iloc[-1])
            ema_long = float(ema_50.iloc[-1])
            ema_signal = 1 if ema_current > ema_long else 0
            
            # RSI
            delta = prices.diff()
            gains = delta.where(delta > 0, 0.0)
            losses = -delta.where(delta < 0, 0.0)
            avg_gains = gains.ewm(alpha=1/14).mean()
            avg_losses = losses.ewm(alpha=1/14).mean()
            rs = avg_gains / avg_losses
            rsi = 100 - (100 / (1 + rs))
            rsi_current = float(rsi.iloc[-1])
            rsi_signal = 1 if rsi_current < 70 else 0  # Not overbought
            
            # Score combiné
            score = 0.6 * ema_signal + 0.4 * rsi_signal
            
            current_price = float(prices.iloc[-1])
            first_price = float(prices.iloc[0])
            return_2y = (current_price / first_price) - 1
            
            signals[symbol] = {
                'score': score,
                'ema_bullish': ema_signal,
                'rsi_ok': rsi_signal,
                'current_price': current_price,
                'return_2y': return_2y,
                'ema_current': ema_current,
                'ema_long': ema_long,
                'rsi_current': rsi_current
            }
            
            print(f"  📈 {symbol}: Score={score:.2f}, EMA={'Bull' if ema_signal else 'Bear'}, RSI={rsi_current:.1f}{'✅' if rsi_signal else '⚠️'}, 2Y Return={return_2y:.1%}")
            
        except Exception as e:
            print(f"  ❌ {symbol} signal failed: {e}")
    
    # 3. Simulation simple portfolio équi-pondéré
    print("\n💼 Portfolio simulation...")
    
    # Sélectionner top 3 actifs (excluding SPY benchmark)
    non_spy_signals = {k: v for k, v in signals.items() if k != 'SPY'}
    top_assets = sorted(non_spy_signals.items(), key=lambda x: x[1]['score'], reverse=True)[:3]
    
    print(f"  🎯 Selected assets:")
    for symbol, signal in top_assets:
        print(f"    {symbol}: Score {signal['score']:.2f}, 2Y Return {signal['return_2y']:.1%}")
    
    # Portfolio performance (equal weight)
    portfolio_return = np.mean([signal['return_2y'] for symbol, signal in top_assets])
    
    # Benchmark performance (SPY)
    spy_return = signals.get('SPY', {}).get('return_2y', 0)
    
    # Annualisé sur 2 ans
    portfolio_annual = (1 + portfolio_return) ** (1/2) - 1
    spy_annual = (1 + spy_return) ** (1/2) - 1
    
    print(f"\n📊 RESULTS:")
    print(f"  Portfolio 2Y Return:   {portfolio_return:>8.1%}")
    print(f"  Portfolio Annual:      {portfolio_annual:>8.1%}")
    print(f"  SPY 2Y Return:         {spy_return:>8.1%}")
    print(f"  SPY Annual:            {spy_annual:>8.1%}")
    print(f"  Alpha vs SPY:          {portfolio_annual - spy_annual:>8.1%}")
    print(f"  Outperformance:        {'✅ YES' if portfolio_annual > spy_annual else '❌ NO'}")
    
    # 4. Validation système
    print(f"\n💡 SYSTEM VALIDATION:")
    
    # Signaux générés avec succès
    success_rate = len(signals) / len(symbols)
    print(f"  Signal Success Rate:   {success_rate:>8.1%}")
    
    # Performance acceptable
    if portfolio_annual > 0.05:  # >5% annual
        print("  ✅ Return threshold met")
    else:
        print("  ⚠️ Return below threshold")
    
    # Outperformance
    if portfolio_annual > spy_annual:
        print("  ✅ Benchmark outperformance")
    else:
        print("  ⚠️ Benchmark underperformance")
    
    # EMA+RSI logic working
    ema_signals = [s['ema_bullish'] for s in signals.values()]
    if any(ema_signals):
        print("  ✅ EMA signals functional")
    
    rsi_signals = [s['rsi_ok'] for s in signals.values()]
    if any(rsi_signals):
        print("  ✅ RSI signals functional")
    
    # Verdict final
    system_working = (success_rate >= 0.75 and 
                     portfolio_annual > 0.05 and 
                     any(ema_signals) and 
                     any(rsi_signals))
    
    print(f"\n🎯 FINAL VERDICT:")
    if system_working:
        print("  🎉 SYSTÈME SIMPLIFIÉ VALIDÉ!")
        print("  ✅ Prêt pour Sprint 35-36 optimisations")
    else:
        print("  ⚠️ Optimisations nécessaires")
        print("  🔧 Vérifier logique signaux")
    
    return system_working

if __name__ == "__main__":
    quick_performance_test()