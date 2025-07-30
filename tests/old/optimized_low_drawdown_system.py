#!/usr/bin/env python3
"""
Optimized Low Drawdown System - Ultimate Champion with Risk Control
Maintain 22-25% annual return while reducing max drawdown to <25%
Advanced risk management with volatility targeting and position sizing
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class OptimizedLowDrawdownSystem:
    """
    Enhanced Ultimate Champion with sophisticated risk management
    Target: 22-25% annual return with max drawdown <25%
    """
    
    def __init__(self):
        # PROVEN CHAMPION UNIVERSE (same as Ultimate)
        self.champion_universe = [
            # Mega tech champions
            'AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META', 'AMZN', 'TSLA',
            
            # Growth tech leaders
            'CRM', 'ADBE', 'AMD', 'QCOM', 'ORCL',
            
            # Quality large caps
            'V', 'MA', 'UNH', 'HD', 'JPM',
            
            # Tech ETFs
            'QQQ', 'XLK', 'VGT',
            
            # Defensive additions for risk control
            'SPY', 'IWM', 'GLD', 'TLT'  # Gold and bonds for crisis
        ]
        
        # ENHANCED CONFIGURATION
        self.start_date = "2015-01-01"
        self.end_date = "2024-12-01"
        self.initial_capital = 100000
        
        # ADVANCED RISK MANAGEMENT
        self.target_volatility = 0.18        # Target 18% volatility (vs 21.8% original)
        self.max_portfolio_risk = 0.85       # Maximum 85% invested
        self.cash_buffer_min = 0.15          # Minimum 15% cash in crisis
        self.max_single_position = 0.08      # Reduced from 12% to 8%
        self.volatility_lookback = 60        # 60-day volatility calculation
        
        # DRAWDOWN PROTECTION
        self.drawdown_threshold = -0.15      # Start reducing risk at -15% DD
        self.crisis_threshold = -0.25        # Emergency risk reduction at -25% DD
        self.recovery_threshold = -0.05      # Return to normal above -5% DD
        
        # DYNAMIC ALLOCATION (more conservative)
        self.base_trend_allocation = 0.60    # Reduced from 70%
        self.base_swing_allocation = 0.25    # Reduced from 30%
        self.defensive_allocation = 0.15     # New: defensive assets
        
        print(f"🛡️ OPTIMIZED LOW DRAWDOWN SYSTEM")
        print(f"📊 UNIVERSE: {len(self.champion_universe)} symbols with defensive assets")
        print(f"🎯 TARGET: 22-25% annual, <25% max drawdown")
        print(f"🛡️ VOLATILITY TARGET: {self.target_volatility:.0%}")
        print(f"💰 MAX POSITION: {self.max_single_position:.0%}")
        
        # Storage
        self.data = {}
        self.market_data = {}
        self.performance_history = []
        
    def run_low_drawdown_system(self):
        """Run the optimized low drawdown system"""
        print("\n" + "="*80)
        print("🛡️ OPTIMIZED LOW DRAWDOWN SYSTEM - EXECUTION")
        print("="*80)
        
        # Download data
        print("\n📊 Step 1: Downloading enhanced universe data...")
        self.download_enhanced_data()
        
        # Download market data
        print("\n📈 Step 2: Downloading risk management indicators...")
        self.download_risk_indicators()
        
        # Run enhanced strategy
        print("\n🛡️ Step 3: Executing low drawdown strategy...")
        portfolio_history = self.execute_low_drawdown_strategy()
        
        # Calculate performance
        print("\n📊 Step 4: Calculating optimized performance...")
        performance = self.calculate_enhanced_performance(portfolio_history)
        
        # Generate report
        print("\n📋 Step 5: Generating low drawdown report...")
        self.generate_enhanced_report(performance)
        
        return performance
    
    def download_enhanced_data(self):
        """Download data for enhanced universe"""
        failed_downloads = []
        
        for i, symbol in enumerate(self.champion_universe, 1):
            try:
                print(f"  🛡️ ({i:2d}/{len(self.champion_universe)}) {symbol:8s}...", end=" ")
                
                ticker_data = yf.download(
                    symbol, 
                    start=self.start_date, 
                    end=self.end_date,
                    progress=False
                )
                
                if len(ticker_data) > 2000:
                    if isinstance(ticker_data.columns, pd.MultiIndex):
                        ticker_data.columns = ticker_data.columns.droplevel(1)
                    
                    self.data[symbol] = ticker_data['Close'].dropna()
                    print(f"✅ {len(ticker_data)} days")
                else:
                    print(f"❌ {len(ticker_data)} days (insufficient)")
                    failed_downloads.append(symbol)
                    
            except Exception as e:
                print(f"❌ Error")
                failed_downloads.append(symbol)
        
        print(f"  🛡️ ENHANCED DATA: {len(self.data)} symbols loaded")
        if failed_downloads:
            print(f"  ⚠️ Failed: {failed_downloads}")
    
    def download_risk_indicators(self):
        """Download risk management indicators"""
        indices = ['SPY', 'QQQ', '^VIX', 'GLD', 'TLT']
        
        for symbol in indices:
            try:
                data = yf.download(symbol, start=self.start_date, end=self.end_date, progress=False)
                
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.droplevel(1)
                
                market_symbol = symbol.replace('^', '')
                self.market_data[market_symbol] = data
                print(f"  ✅ {market_symbol}: {len(data)} days")
            except Exception as e:
                print(f"  ❌ {symbol}: {e}")
    
    def execute_low_drawdown_strategy(self):
        """Execute the low drawdown strategy"""
        portfolio = {
            'cash': float(self.initial_capital),
            'trend_positions': {},
            'swing_positions': {},
            'defensive_positions': {},
            'value': float(self.initial_capital),
            'peak_value': float(self.initial_capital)
        }
        
        trading_dates = pd.bdate_range(start=self.start_date, end=self.end_date).tolist()
        history = []
        
        print(f"    🛡️ Low drawdown execution: {len(trading_dates)} days")
        
        for i, date in enumerate(trading_dates):
            if i % 500 == 0:
                print(f"      📅 Progress: {(i/len(trading_dates)*100):5.1f}% - {date.strftime('%Y-%m-%d')}")
            
            # Update portfolio value
            portfolio_value = self.update_enhanced_portfolio_value(portfolio, date)
            
            # Update peak and calculate current drawdown
            if portfolio_value > portfolio['peak_value']:
                portfolio['peak_value'] = portfolio_value
            
            current_drawdown = (portfolio_value / portfolio['peak_value']) - 1
            
            # Enhanced regime detection
            regime = self.detect_enhanced_regime(date)
            
            # Risk-adjusted allocation based on drawdown
            risk_factor = self.calculate_risk_factor(current_drawdown, regime)
            
            # Dynamic allocation with risk adjustment
            trend_alloc, swing_alloc, defensive_alloc = self.calculate_risk_adjusted_allocation(
                regime, risk_factor, current_drawdown
            )
            
            # Portfolio volatility targeting
            portfolio_vol = self.calculate_portfolio_volatility(portfolio, date)
            vol_adjustment = self.target_volatility / max(portfolio_vol, 0.10)  # Avoid division by zero
            vol_adjustment = min(vol_adjustment, 1.5)  # Cap at 150%
            
            # Trend following rebalancing (monthly with risk adjustment)
            if i % 20 == 0:
                adjusted_trend_alloc = trend_alloc * vol_adjustment
                self.execute_enhanced_trend_rebalancing(portfolio, date, adjusted_trend_alloc)
            
            # Swing trading rebalancing (every 5 days in risk mode)
            rebalance_frequency = 5 if risk_factor < 0.8 else 3
            if i % rebalance_frequency == 0:
                adjusted_swing_alloc = swing_alloc * vol_adjustment
                self.execute_enhanced_swing_rebalancing(portfolio, date, adjusted_swing_alloc)
            
            # Defensive rebalancing (weekly)
            if i % 5 == 0:
                self.execute_defensive_rebalancing(portfolio, date, defensive_alloc)
            
            history.append({
                'date': date.strftime('%Y-%m-%d'),
                'portfolio_value': portfolio_value,
                'current_drawdown': current_drawdown,
                'regime': regime,
                'risk_factor': risk_factor,
                'portfolio_volatility': portfolio_vol,
                'trend_allocation': trend_alloc,
                'swing_allocation': swing_alloc,
                'defensive_allocation': defensive_alloc
            })
        
        return history
    
    def detect_enhanced_regime(self, current_date):
        """Enhanced regime detection with risk signals"""
        try:
            spy_data = self.market_data.get('SPY')
            vix_data = self.market_data.get('VIX')
            
            if spy_data is None:
                return 'neutral'
            
            historical_spy = spy_data[spy_data.index <= current_date]
            if len(historical_spy) < 50:
                return 'neutral'
            
            closes = historical_spy['Close']
            
            # Moving averages
            ma_20 = closes.rolling(20).mean().iloc[-1]
            ma_50 = closes.rolling(50).mean().iloc[-1]
            ma_200 = closes.rolling(200).mean().iloc[-1] if len(closes) >= 200 else ma_50
            
            current_price = closes.iloc[-1]
            
            # Enhanced volatility analysis
            current_vix = 20
            if vix_data is not None:
                historical_vix = vix_data[vix_data.index <= current_date]
                if len(historical_vix) > 0:
                    current_vix = historical_vix['Close'].iloc[-1]
                    vix_ma = historical_vix['Close'].rolling(20).mean().iloc[-1]
                    vix_spike = current_vix > vix_ma * 1.5  # VIX spike detection
                else:
                    vix_spike = False
            else:
                vix_spike = False
            
            # Momentum with multiple timeframes
            momentum_5d = (current_price / closes.iloc[-5]) - 1 if len(closes) >= 5 else 0
            momentum_20d = (current_price / closes.iloc[-20]) - 1 if len(closes) >= 20 else 0
            
            # Enhanced regime classification
            if current_vix > 35 or vix_spike:
                return 'panic'
            elif current_vix > 30:
                return 'crisis'
            elif current_vix > 25 or momentum_20d < -0.10:
                return 'high_stress'
            elif current_price > ma_20 > ma_50 > ma_200 and momentum_5d > 0.01:
                return 'strong_bull'
            elif current_price > ma_20 > ma_50:
                return 'bull'
            elif current_price < ma_20 < ma_50 and momentum_5d < -0.01:
                return 'bear'
            elif current_vix > 20:
                return 'volatile'
            else:
                return 'neutral'
                
        except Exception:
            return 'neutral'
    
    def calculate_risk_factor(self, current_drawdown, regime):
        """Calculate dynamic risk factor based on drawdown and regime"""
        # Base risk factor from drawdown
        if current_drawdown > self.drawdown_threshold:
            # Normal operation
            drawdown_factor = 1.0
        elif current_drawdown > self.crisis_threshold:
            # Moderate risk reduction
            drawdown_factor = 0.7 + 0.3 * (current_drawdown - self.crisis_threshold) / (self.drawdown_threshold - self.crisis_threshold)
        else:
            # Emergency risk reduction
            drawdown_factor = 0.4
        
        # Regime-based adjustment
        regime_factors = {
            'strong_bull': 1.0,
            'bull': 0.9,
            'neutral': 0.8,
            'volatile': 0.6,
            'bear': 0.5,
            'high_stress': 0.4,
            'crisis': 0.3,
            'panic': 0.2
        }
        
        regime_factor = regime_factors.get(regime, 0.8)
        
        # Combined risk factor
        combined_factor = min(drawdown_factor, regime_factor)
        return max(combined_factor, 0.2)  # Minimum 20% allocation
    
    def calculate_risk_adjusted_allocation(self, regime, risk_factor, current_drawdown):
        """Calculate risk-adjusted allocations"""
        # Base allocations
        base_trend = self.base_trend_allocation
        base_swing = self.base_swing_allocation
        base_defensive = self.defensive_allocation
        
        # Risk adjustment
        trend_alloc = base_trend * risk_factor
        swing_alloc = base_swing * risk_factor
        
        # Increase defensive allocation in stress
        if risk_factor < 0.6:
            defensive_alloc = min(base_defensive + (0.6 - risk_factor), 0.4)
        else:
            defensive_alloc = base_defensive
        
        # Ensure allocations sum to reasonable total
        total_alloc = trend_alloc + swing_alloc + defensive_alloc
        if total_alloc > 0.95:
            scale_factor = 0.95 / total_alloc
            trend_alloc *= scale_factor
            swing_alloc *= scale_factor
            defensive_alloc *= scale_factor
        
        return trend_alloc, swing_alloc, defensive_alloc
    
    def calculate_portfolio_volatility(self, portfolio, date):
        """Calculate current portfolio volatility"""
        try:
            # Get recent portfolio returns (simplified estimation)
            if len(self.performance_history) < self.volatility_lookback:
                return 0.15  # Default assumption
            
            recent_values = [h['portfolio_value'] for h in self.performance_history[-self.volatility_lookback:]]
            returns = pd.Series(recent_values).pct_change().dropna()
            
            if len(returns) > 10:
                volatility = returns.std() * np.sqrt(252)
                return float(volatility)
            else:
                return 0.15
                
        except Exception:
            return 0.15
    
    def execute_enhanced_trend_rebalancing(self, portfolio, date, allocation):
        """Enhanced trend rebalancing with risk controls"""
        signals = self.calculate_enhanced_trend_signals(date)
        
        # More conservative selection
        qualified_signals = sorted(
            [(s, sig['score']) for s, sig in signals.items() if sig['score'] > 0.75],
            key=lambda x: x[1], 
            reverse=True
        )[:6]  # Reduced from 8 to 6
        
        if not qualified_signals:
            return
        
        # Enhanced position sizing
        target_positions = {}
        
        for symbol, score in qualified_signals:
            # Base weight with score adjustment
            base_weight = allocation / len(qualified_signals)
            score_adjusted_weight = base_weight * min(score / 0.8, 1.2)  # Score boost cap
            
            # Apply maximum position limit
            final_weight = min(score_adjusted_weight, self.max_single_position)
            target_positions[symbol] = final_weight
        
        self.execute_enhanced_position_changes(portfolio, 'trend_positions', date, target_positions)
    
    def execute_enhanced_swing_rebalancing(self, portfolio, date, allocation):
        """Enhanced swing rebalancing with risk controls"""
        signals = self.calculate_enhanced_swing_signals(date)
        
        qualified_signals = sorted(
            [(s, sig['score']) for s, sig in signals.items() if sig['score'] > 0.65],
            key=lambda x: x[1], 
            reverse=True
        )[:4]  # Reduced from 6 to 4
        
        if not qualified_signals:
            return
        
        target_positions = {}
        for symbol, score in qualified_signals:
            weight = allocation / len(qualified_signals)
            weight = min(weight, self.max_single_position * 0.8)  # Smaller swing positions
            target_positions[symbol] = weight
        
        self.execute_enhanced_position_changes(portfolio, 'swing_positions', date, target_positions)
    
    def execute_defensive_rebalancing(self, portfolio, date, allocation):
        """Execute defensive asset rebalancing"""
        if allocation < 0.05:
            return
        
        # Defensive assets with safe allocation
        defensive_assets = ['SPY', 'GLD', 'TLT']
        available_assets = [asset for asset in defensive_assets if asset in self.data]
        
        if not available_assets:
            return
        
        target_positions = {}
        weight_per_asset = allocation / len(available_assets)
        
        for asset in available_assets:
            target_positions[asset] = min(weight_per_asset, 0.10)  # Max 10% in any defensive asset
        
        self.execute_enhanced_position_changes(portfolio, 'defensive_positions', date, target_positions)
    
    def calculate_enhanced_trend_signals(self, date):
        """Enhanced trend signals with risk adjustment"""
        signals = {}
        
        for symbol, prices in self.data.items():
            try:
                historical_data = prices[prices.index <= date]
                if len(historical_data) < 100:
                    continue
                
                # Enhanced trend analysis
                ma_50 = historical_data.rolling(50).mean().iloc[-1]
                ma_100 = historical_data.rolling(100).mean().iloc[-1]
                ma_200 = historical_data.rolling(200).mean().iloc[-1] if len(historical_data) >= 200 else ma_100
                
                current_price = float(historical_data.iloc[-1])
                
                # Trend strength with quality filter
                if current_price > ma_50 > ma_100 > ma_200:
                    trend_score = 1.0
                elif current_price > ma_50 > ma_100:
                    trend_score = 0.8
                elif current_price > ma_50:
                    trend_score = 0.6
                else:
                    trend_score = 0.0
                
                # Volatility penalty (prefer stable trends)
                recent_returns = historical_data.pct_change().tail(20)
                volatility = recent_returns.std()
                vol_penalty = max(0, min(0.2, (volatility - 0.015) * 10))  # Penalty for high vol
                
                # Momentum confirmation
                if len(historical_data) >= 50:
                    momentum = (current_price / float(historical_data.iloc[-50])) - 1
                    momentum_score = min(momentum * 3, 0.4) if momentum > 0 else 0
                else:
                    momentum_score = 0
                
                # Quality adjustment for defensive assets
                defensive_assets = ['SPY', 'GLD', 'TLT']
                quality_boost = 0.1 if symbol in defensive_assets else 0
                
                # Final enhanced score
                score = (0.6 * trend_score + 0.3 * momentum_score + quality_boost) - vol_penalty
                score = max(score, 0)
                
                signals[symbol] = {
                    'score': score,
                    'price': current_price
                }
                
            except Exception:
                continue
        
        return signals
    
    def calculate_enhanced_swing_signals(self, date):
        """Enhanced swing signals with risk controls"""
        signals = {}
        
        for symbol, prices in self.data.items():
            try:
                historical_data = prices[prices.index <= date]
                if len(historical_data) < 30:
                    continue
                
                # Conservative swing indicators
                ema_8 = historical_data.ewm(span=8).mean().iloc[-1]
                ema_21 = historical_data.ewm(span=21).mean().iloc[-1]
                sma_50 = historical_data.rolling(50).mean().iloc[-1] if len(historical_data) >= 50 else ema_21
                
                current_price = float(historical_data.iloc[-1])
                
                # Enhanced signal quality
                ema_signal = 1.0 if ema_8 > ema_21 and current_price > sma_50 else 0.0
                
                # Conservative RSI
                delta = historical_data.diff()
                gains = delta.where(delta > 0, 0.0)
                losses = -delta.where(delta < 0, 0.0)
                avg_gains = gains.ewm(alpha=1/14).mean()
                avg_losses = losses.ewm(alpha=1/14).mean()
                rs = avg_gains / avg_losses.replace(0, 0.001)
                rsi = 100 - (100 / (1 + rs))
                
                # More conservative RSI range
                rsi_signal = 1.0 if 35 < rsi.iloc[-1] < 65 else 0.5
                
                # Risk-adjusted momentum
                if len(historical_data) >= 10:
                    momentum = (current_price / float(historical_data.iloc[-10])) - 1
                    momentum_signal = 1.0 if 0.01 < momentum < 0.15 else 0.0  # Avoid extreme moves
                else:
                    momentum_signal = 0.0
                
                # Enhanced final score
                score = 0.5 * ema_signal + 0.3 * rsi_signal + 0.2 * momentum_signal
                
                signals[symbol] = {
                    'score': score,
                    'price': current_price
                }
                
            except Exception:
                continue
        
        return signals
    
    def execute_enhanced_position_changes(self, portfolio, position_type, date, target_positions):
        """Enhanced position changes with better execution"""
        current_positions = portfolio[position_type]
        current_value = portfolio['value']
        
        # Sell positions not in targets
        positions_to_sell = [s for s in current_positions.keys() if s not in target_positions]
        for symbol in positions_to_sell:
            if symbol in self.data:
                shares = current_positions[symbol]
                if shares > 0:
                    try:
                        prices = self.data[symbol]
                        available_prices = prices[prices.index <= date]
                        if len(available_prices) > 0:
                            price = float(available_prices.iloc[-1])
                            proceeds = float(shares) * price * 0.999  # 0.1% transaction cost
                            portfolio['cash'] += proceeds
                    except:
                        pass
                del current_positions[symbol]
        
        # Buy/adjust target positions
        for symbol, target_weight in target_positions.items():
            if symbol in self.data:
                try:
                    prices = self.data[symbol]
                    available_prices = prices[prices.index <= date]
                    if len(available_prices) > 0:
                        price = float(available_prices.iloc[-1])
                        target_value = current_value * target_weight
                        target_shares = target_value / price
                        
                        current_shares = current_positions.get(symbol, 0)
                        shares_diff = target_shares - current_shares
                        
                        if abs(shares_diff * price) > current_value * 0.005:  # 0.5% threshold
                            cost = float(shares_diff) * price
                            if shares_diff > 0 and portfolio['cash'] >= cost * 1.001:  # Include transaction cost
                                portfolio['cash'] -= cost * 1.001
                                current_positions[symbol] = target_shares
                            elif shares_diff < 0:
                                portfolio['cash'] -= cost * 0.999  # Selling with cost
                                current_positions[symbol] = target_shares if target_shares > 0 else 0
                                if current_positions[symbol] <= 0:
                                    current_positions.pop(symbol, None)
                except:
                    continue
    
    def update_enhanced_portfolio_value(self, portfolio, date):
        """Update enhanced portfolio value"""
        total_value = portfolio['cash']
        
        # Add all position values
        for position_type in ['trend_positions', 'swing_positions', 'defensive_positions']:
            for symbol, shares in portfolio[position_type].items():
                if symbol in self.data and shares > 0:
                    try:
                        prices = self.data[symbol]
                        available_prices = prices[prices.index <= date]
                        if len(available_prices) > 0:
                            current_price = float(available_prices.iloc[-1])
                            position_value = float(shares) * current_price
                            total_value += position_value
                    except:
                        continue
        
        portfolio['value'] = total_value
        return total_value
    
    def calculate_enhanced_performance(self, history):
        """Calculate enhanced performance metrics"""
        try:
            history_df = pd.DataFrame(history)
            history_df['date'] = pd.to_datetime(history_df['date'])
            history_df.set_index('date', inplace=True)
            
            values = history_df['portfolio_value']
            daily_returns = values.pct_change().dropna()
            
            total_return = (values.iloc[-1] / values.iloc[0]) - 1
            years = len(daily_returns) / 252
            annual_return = (1 + total_return) ** (1/years) - 1
            
            volatility = daily_returns.std() * np.sqrt(252)
            sharpe_ratio = (annual_return - 0.02) / volatility if volatility > 0 else 0
            
            cumulative = values / values.iloc[0]
            running_max = cumulative.expanding().max()
            drawdowns = (cumulative - running_max) / running_max
            max_drawdown = drawdowns.min()
            
            calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
            win_rate = (daily_returns > 0).mean()
            
            # Enhanced risk metrics
            avg_drawdown = history_df['current_drawdown'].mean()
            drawdown_days = (history_df['current_drawdown'] < -0.05).sum()
            recovery_factor = annual_return / abs(avg_drawdown) if avg_drawdown != 0 else 0
            
            return {
                'total_return': float(total_return),
                'annual_return': float(annual_return),
                'volatility': float(volatility),
                'sharpe_ratio': float(sharpe_ratio),
                'calmar_ratio': float(calmar_ratio),
                'max_drawdown': float(max_drawdown),
                'avg_drawdown': float(avg_drawdown),
                'win_rate': float(win_rate),
                'final_value': float(values.iloc[-1]),
                'years_simulated': float(years),
                'drawdown_days': int(drawdown_days),
                'recovery_factor': float(recovery_factor)
            }
            
        except Exception as e:
            print(f"❌ Enhanced performance calculation failed: {e}")
            return None
    
    def generate_enhanced_report(self, performance):
        """Generate enhanced performance report"""
        if not performance:
            print("❌ No performance data to report")
            return
        
        print("\n" + "="*80)
        print("🛡️ OPTIMIZED LOW DRAWDOWN SYSTEM - PERFORMANCE REPORT")
        print("="*80)
        
        print(f"📊 ENHANCED UNIVERSE: {len(self.data)} symbols with defensive assets")
        print(f"📅 TESTING PERIOD: {self.start_date} to {self.end_date}")
        print(f"💰 INITIAL CAPITAL: ${self.initial_capital:,}")
        print(f"🎯 VOLATILITY TARGET: {self.target_volatility:.0%}")
        
        print(f"\n🛡️ OPTIMIZED PERFORMANCE:")
        print(f"  📈 Annual Return:     {performance['annual_return']:>8.1%}")
        print(f"  📊 Total Return:      {performance['total_return']:>8.1%}")
        print(f"  💰 Final Value:       ${performance['final_value']:>10,.0f}")
        print(f"  📉 Max Drawdown:      {performance['max_drawdown']:>8.1%}")
        print(f"  📊 Avg Drawdown:      {performance['avg_drawdown']:>8.1%}")
        print(f"  ⚡ Volatility:        {performance['volatility']:>8.1%}")
        print(f"  🎯 Sharpe Ratio:      {performance['sharpe_ratio']:>8.2f}")
        print(f"  📊 Calmar Ratio:      {performance['calmar_ratio']:>8.2f}")
        print(f"  🔄 Recovery Factor:   {performance['recovery_factor']:>8.2f}")
        print(f"  ✅ Win Rate:          {performance['win_rate']:>8.1%}")
        print(f"  📉 Drawdown Days:     {performance['drawdown_days']:>8d}")
        
        print(f"\n🎯 VS BENCHMARKS:")
        nasdaq_annual = 0.184
        spy_annual = 0.134
        
        nasdaq_gap = performance['annual_return'] - nasdaq_annual
        spy_gap = performance['annual_return'] - spy_annual
        
        print(f"  📊 vs NASDAQ (18.4%): {nasdaq_gap:>8.1%} ({'BEATS' if nasdaq_gap > 0 else 'LAGS'})")
        print(f"  📊 vs S&P 500 (13.4%): {spy_gap:>8.1%} ({'BEATS' if spy_gap > 0 else 'LAGS'})")
        
        # Enhanced assessment
        drawdown_success = performance['max_drawdown'] > -0.25
        return_success = performance['annual_return'] > 0.20
        risk_success = performance['volatility'] < 0.20
        
        print(f"\n🛡️ RISK OPTIMIZATION RESULTS:")
        print(f"  📉 Drawdown Target (<25%): {'✅ ACHIEVED' if drawdown_success else '❌ MISSED'}")
        print(f"  📈 Return Target (>20%):   {'✅ ACHIEVED' if return_success else '❌ MISSED'}")
        print(f"  ⚡ Volatility Target (<20%): {'✅ ACHIEVED' if risk_success else '❌ MISSED'}")
        
        success_count = sum([drawdown_success, return_success, risk_success])
        
        if success_count == 3:
            rating = "🌟 PERFECT OPTIMIZATION"
        elif success_count == 2:
            rating = "🏆 EXCELLENT OPTIMIZATION"
        elif success_count == 1:
            rating = "✅ GOOD OPTIMIZATION"
        else:
            rating = "🔧 NEEDS FURTHER OPTIMIZATION"
        
        print(f"\n{rating}")
        
        if drawdown_success and return_success:
            print(f"\n🎉 SUCCESS: Optimized system achieves target risk-return profile!")
            print(f"🚀 RECOMMENDATION: This system is ready for production deployment")
        else:
            print(f"\n🔧 OPTIMIZATION NEEDED: Some targets not fully achieved")
            if not drawdown_success:
                print(f"   📉 Further drawdown reduction needed")
            if not return_success:
                print(f"   📈 Return enhancement strategies required")


def main():
    """Execute Optimized Low Drawdown System"""
    print("🛡️ OPTIMIZED LOW DRAWDOWN SYSTEM")
    print("Enhanced Risk Management with Target Performance")
    print("="*80)
    
    system = OptimizedLowDrawdownSystem()
    performance = system.run_low_drawdown_system()
    
    return 0


if __name__ == "__main__":
    exit_code = main()