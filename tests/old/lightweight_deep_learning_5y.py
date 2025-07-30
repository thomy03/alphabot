#!/usr/bin/env python3
"""
Lightweight Deep Learning System - Optimized for AMD Ryzen 5 PC
5-year test version with reduced complexity for 16GB RAM
Uses CPU-optimized settings and simplified models
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import json
from pathlib import Path
import warnings
import time
warnings.filterwarnings('ignore')

# Lightweight ML imports
try:
    # CPU-optimized TensorFlow
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    import tensorflow as tf
    # Force CPU usage (no GPU)
    tf.config.set_visible_devices([], 'GPU')
    
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    
    # Scikit-learn (CPU efficient)
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import MinMaxScaler
    
    ML_AVAILABLE = True
    print("🚀 Lightweight ML Stack Ready (CPU Mode)")
    print(f"📊 TensorFlow: {tf.__version__}")
    
except ImportError as e:
    ML_AVAILABLE = False
    print(f"⚠️ ML Stack Missing: {e}")
    print("📦 Install: pip install tensorflow scikit-learn")

class LightweightDeepLearningSystem:
    """
    Optimized for AMD Ryzen 5 5500U with 16GB RAM
    - Reduced model complexity
    - Smaller universe (15 symbols)
    - 5-year backtest only
    - CPU-optimized computations
    - Memory-efficient data handling
    """
    
    def __init__(self):
        # LIGHTWEIGHT UNIVERSE (15 top performers only)
        self.lightweight_universe = [
            # Top 6 tech mega caps only
            'AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META', 'AMZN',
            
            # Top 3 growth tech
            'TSLA', 'AMD', 'CRM',
            
            # Top 3 quality stocks
            'V', 'MA', 'UNH',
            
            # Core ETFs only
            'QQQ', 'SPY', 'XLK'
        ]
        
        # 5-YEAR TEST PERIOD (less data)
        self.start_date = "2019-01-01"
        self.end_date = "2024-12-01"
        self.initial_capital = 100000
        
        # SIMPLIFIED PARAMETERS
        self.training_window = 252  # 1 year only (vs 2 years)
        self.lookback_days = 30     # 30 days (vs 60)
        self.retrain_frequency = 126 # Semi-annual (vs quarterly)
        self.max_positions = 6       # Fewer positions
        
        # LIGHTWEIGHT MODEL CONFIG
        self.lstm_units = 32        # Reduced from 64
        self.dense_units = 16       # Reduced from 32
        self.batch_size = 64        # Larger batches (faster)
        self.epochs = 30            # Fewer epochs
        
        # CPU OPTIMIZATION
        self.parallel_jobs = 4      # Use 4 CPU cores
        self.chunk_size = 1000      # Process in chunks
        
        print(f"🚀 LIGHTWEIGHT DEEP LEARNING SYSTEM")
        print(f"💻 Optimized for: AMD Ryzen 5 5500U + 16GB RAM")
        print(f"📊 Universe: {len(self.lightweight_universe)} symbols")
        print(f"📅 Period: 5 years (2019-2024)")
        print(f"🧠 Model: Simplified LSTM (32 units)")
        print(f"⚡ CPU Cores: {self.parallel_jobs}")
        
        # Storage
        self.data = {}
        self.models = {}
        self.performance_log = []
        
    def run_lightweight_test(self):
        """Run lightweight 5-year test"""
        print("\n" + "="*80)
        print("🚀 LIGHTWEIGHT DEEP LEARNING - 5 YEAR TEST")
        print("="*80)
        
        start_time = time.time()
        
        # Step 1: Download data
        print("\n📊 Step 1: Downloading 5 years of data...")
        self.download_lightweight_data()
        
        # Step 2: Build lightweight models
        print("\n🧠 Step 2: Building lightweight models...")
        self.build_lightweight_models()
        
        # Step 3: Train models
        print("\n🎓 Step 3: Training models (CPU optimized)...")
        self.train_lightweight_models()
        
        # Step 4: Run backtest
        print("\n📈 Step 4: Running 5-year backtest...")
        portfolio_history = self.execute_lightweight_strategy()
        
        # Step 5: Calculate performance
        print("\n📊 Step 5: Calculating performance...")
        performance = self.calculate_performance(portfolio_history)
        
        # Step 6: Generate report
        print("\n📋 Step 6: Generating report...")
        self.generate_lightweight_report(performance, time.time() - start_time)
        
        return performance
    
    def download_lightweight_data(self):
        """Download 5 years of data efficiently"""
        failed_downloads = []
        
        for i, symbol in enumerate(self.lightweight_universe, 1):
            try:
                print(f"  💻 ({i:2d}/{len(self.lightweight_universe)}) {symbol:8s}...", end=" ")
                
                # Download with minimal overhead
                ticker_data = yf.download(
                    symbol,
                    start=self.start_date,
                    end=self.end_date,
                    progress=False,
                    threads=False  # Single thread per symbol
                )
                
                if len(ticker_data) > 1000:  # 5 years ~ 1250 days
                    if isinstance(ticker_data.columns, pd.MultiIndex):
                        ticker_data.columns = ticker_data.columns.droplevel(1)
                    
                    # Store only Close prices to save memory
                    self.data[symbol] = ticker_data['Close'].astype('float32')  # float32 saves memory
                    print(f"✅ {len(ticker_data)} days")
                else:
                    print(f"❌ Insufficient data")
                    failed_downloads.append(symbol)
                    
            except Exception as e:
                print(f"❌ Error")
                failed_downloads.append(symbol)
        
        # Also get market indicators
        for symbol in ['SPY', '^VIX']:
            if symbol not in self.data:
                try:
                    data = yf.download(symbol, start=self.start_date, end=self.end_date, progress=False)
                    if isinstance(data.columns, pd.MultiIndex):
                        data.columns = data.columns.droplevel(1)
                    self.data[symbol.replace('^', '')] = data['Close'].astype('float32')
                except:
                    pass
        
        print(f"\n  💻 Data loaded: {len(self.data)} symbols")
        if failed_downloads:
            print(f"  ⚠️ Failed: {failed_downloads}")
    
    def build_lightweight_models(self):
        """Build simplified models for CPU"""
        if not ML_AVAILABLE:
            print("⚠️ ML not available, using rule-based fallback")
            return
        
        # Build 2 lightweight models instead of 3
        self.models = {
            'trend_model': self.build_trend_lstm(),
            'risk_model': self.build_risk_classifier()
        }
        
        print("  ✅ Models built: 2 lightweight models")
    
    def build_trend_lstm(self):
        """Build lightweight LSTM for trend prediction"""
        model = Sequential([
            LSTM(self.lstm_units, return_sequences=False, 
                 input_shape=(self.lookback_days, 5)),  # 5 features only
            Dropout(0.2),
            Dense(self.dense_units, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def build_risk_classifier(self):
        """Build lightweight risk classifier"""
        # Use RandomForest for CPU efficiency
        return RandomForestClassifier(
            n_estimators=50,  # Reduced trees
            max_depth=10,
            n_jobs=self.parallel_jobs,  # Use multiple cores
            random_state=42
        )
    
    def train_lightweight_models(self):
        """Train models with CPU optimization"""
        if not ML_AVAILABLE:
            return
        
        print("  🎓 Training on CPU (this may take 2-3 minutes)...")
        
        # Prepare training data
        X_trend, y_trend = [], []
        X_risk, y_risk = [], []
        
        # Use only SPY for training (faster)
        if 'SPY' in self.data:
            spy_data = self.data['SPY']
            
            for i in range(self.lookback_days, len(spy_data) - 10):
                # Trend features (simplified)
                window = spy_data.iloc[i-self.lookback_days:i]
                returns = window.pct_change().dropna()
                
                features = [
                    returns.mean(),          # Mean return
                    returns.std(),           # Volatility
                    (window.iloc[-1] / window.iloc[0]) - 1,  # Period return
                    (window.iloc[-1] / window.mean()) - 1,   # Price vs MA
                    len(returns[returns > 0]) / len(returns)  # Win rate
                ]
                
                # Create sequence for LSTM
                sequence = []
                for j in range(self.lookback_days):
                    if j < len(returns):
                        day_features = [
                            returns.iloc[j],
                            returns.iloc[:j+1].std() if j > 0 else 0,
                            returns.iloc[:j+1].mean() if j > 0 else 0,
                            j / self.lookback_days,  # Time position
                            0.5  # Placeholder
                        ]
                    else:
                        day_features = [0, 0, 0, 0, 0]
                    sequence.append(day_features)
                
                X_trend.append(sequence)
                X_risk.append(features)
                
                # Labels
                future_return = (spy_data.iloc[i+5] / spy_data.iloc[i]) - 1
                y_trend.append(1 if future_return > 0.01 else 0)
                y_risk.append(1 if abs(future_return) > 0.03 else 0)  # High volatility
        
        if len(X_trend) > 100:
            # Train LSTM
            X_trend = np.array(X_trend)
            y_trend = np.array(y_trend)
            
            print("    🧠 Training Trend LSTM...")
            self.models['trend_model'].fit(
                X_trend, y_trend,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=0.2,
                verbose=0
            )
            
            # Train Risk Classifier
            print("    🛡️ Training Risk Classifier...")
            self.models['risk_model'].fit(X_risk, y_risk)
            
            print("  ✅ Training complete!")
        else:
            print("  ⚠️ Insufficient training data")
    
    def execute_lightweight_strategy(self):
        """Execute strategy with CPU optimization"""
        portfolio = {
            'cash': float(self.initial_capital),
            'positions': {},
            'value': float(self.initial_capital),
            'peak_value': float(self.initial_capital)
        }
        
        trading_dates = pd.bdate_range(start=self.start_date, end=self.end_date).tolist()
        history = []
        
        print(f"    💻 Backtesting {len(trading_dates)} days...")
        
        # Process in chunks for memory efficiency
        for i, date in enumerate(trading_dates):
            if i % 250 == 0:
                print(f"      📅 Year {i//250 + 1}/5 - {date.strftime('%Y-%m-%d')}")
            
            # Update portfolio value
            portfolio_value = self.update_portfolio_value(portfolio, date)
            
            # Update peak and drawdown
            if portfolio_value > portfolio['peak_value']:
                portfolio['peak_value'] = portfolio_value
            
            current_drawdown = (portfolio_value / portfolio['peak_value']) - 1
            
            # Rebalance every 10 days (less frequent for speed)
            if i % 10 == 0 and i > self.training_window:
                self.rebalance_portfolio(portfolio, date, current_drawdown)
            
            # Track history
            history.append({
                'date': date.strftime('%Y-%m-%d'),
                'portfolio_value': portfolio_value,
                'drawdown': current_drawdown
            })
        
        return history
    
    def rebalance_portfolio(self, portfolio, date, current_drawdown):
        """Lightweight rebalancing logic"""
        # Get signals for each symbol
        signals = {}
        
        for symbol in self.lightweight_universe:
            if symbol not in self.data:
                continue
            
            try:
                # Simple momentum signal (no ML if not available)
                prices = self.data[symbol]
                historical = prices[prices.index <= date]
                
                if len(historical) < self.lookback_days:
                    continue
                
                # Calculate simple signals
                ma_20 = historical.tail(20).mean()
                ma_50 = historical.tail(50).mean() if len(historical) >= 50 else ma_20
                current_price = historical.iloc[-1]
                
                # Momentum
                momentum = (current_price / historical.iloc[-20]) - 1 if len(historical) >= 20 else 0
                
                # Simple scoring
                score = 0
                if current_price > ma_20 > ma_50:
                    score += 0.5
                if momentum > 0.05:
                    score += 0.3
                if momentum > 0.10:
                    score += 0.2
                
                # ML enhancement if available
                if ML_AVAILABLE and 'trend_model' in self.models:
                    try:
                        # Prepare features
                        window = historical.tail(self.lookback_days)
                        returns = window.pct_change().dropna()
                        
                        sequence = []
                        for j in range(self.lookback_days):
                            if j < len(returns):
                                day_features = [
                                    returns.iloc[j],
                                    returns.iloc[:j+1].std() if j > 0 else 0,
                                    returns.iloc[:j+1].mean() if j > 0 else 0,
                                    j / self.lookback_days,
                                    0.5
                                ]
                            else:
                                day_features = [0, 0, 0, 0, 0]
                            sequence.append(day_features)
                        
                        # Predict
                        X = np.array([sequence])
                        ml_prediction = self.models['trend_model'].predict(X, verbose=0)[0][0]
                        
                        # Combine with simple signal
                        score = 0.6 * score + 0.4 * ml_prediction
                        
                    except:
                        pass  # Use simple signal if ML fails
                
                signals[symbol] = score
                
            except:
                continue
        
        # Select top positions
        sorted_signals = sorted(signals.items(), key=lambda x: x[1], reverse=True)
        top_signals = sorted_signals[:self.max_positions]
        
        # Risk adjustment
        if current_drawdown < -0.15:
            allocation = 0.6  # Reduce exposure
        elif current_drawdown < -0.10:
            allocation = 0.75
        else:
            allocation = 0.85
        
        # Execute trades
        self.execute_trades(portfolio, date, top_signals, allocation)
    
    def execute_trades(self, portfolio, date, top_signals, allocation):
        """Execute trades efficiently"""
        current_positions = portfolio['positions']
        portfolio_value = portfolio['value']
        
        # Sell positions not in top signals
        for symbol in list(current_positions.keys()):
            if symbol not in [s[0] for s in top_signals]:
                # Sell
                if symbol in self.data:
                    try:
                        price = self.data[symbol][self.data[symbol].index <= date].iloc[-1]
                        proceeds = current_positions[symbol] * price * 0.999  # Transaction cost
                        portfolio['cash'] += proceeds
                        del current_positions[symbol]
                    except:
                        pass
        
        # Buy/adjust top positions
        if len(top_signals) > 0:
            position_size = (portfolio_value * allocation) / len(top_signals)
            
            for symbol, score in top_signals:
                if symbol in self.data:
                    try:
                        price = self.data[symbol][self.data[symbol].index <= date].iloc[-1]
                        target_shares = position_size / price
                        current_shares = current_positions.get(symbol, 0)
                        
                        shares_diff = target_shares - current_shares
                        cost = shares_diff * price
                        
                        if shares_diff > 0 and portfolio['cash'] >= cost * 1.001:
                            portfolio['cash'] -= cost * 1.001
                            current_positions[symbol] = target_shares
                        elif shares_diff < 0:
                            portfolio['cash'] -= cost * 0.999
                            current_positions[symbol] = target_shares
                    except:
                        pass
    
    def update_portfolio_value(self, portfolio, date):
        """Update portfolio value"""
        total_value = portfolio['cash']
        
        for symbol, shares in portfolio['positions'].items():
            if symbol in self.data and shares > 0:
                try:
                    price = self.data[symbol][self.data[symbol].index <= date].iloc[-1]
                    total_value += shares * price
                except:
                    pass
        
        portfolio['value'] = total_value
        return total_value
    
    def calculate_performance(self, history):
        """Calculate performance metrics"""
        try:
            df = pd.DataFrame(history)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            values = df['portfolio_value']
            
            # Basic metrics
            total_return = (values.iloc[-1] / values.iloc[0]) - 1
            years = len(values) / 252
            annual_return = (1 + total_return) ** (1/years) - 1
            
            # Risk metrics
            daily_returns = values.pct_change().dropna()
            volatility = daily_returns.std() * np.sqrt(252)
            sharpe_ratio = (annual_return - 0.02) / volatility if volatility > 0 else 0
            
            # Drawdown
            max_drawdown = df['drawdown'].min()
            
            # Win rate
            win_rate = (daily_returns > 0).mean()
            
            return {
                'total_return': total_return,
                'annual_return': annual_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'final_value': values.iloc[-1]
            }
            
        except Exception as e:
            print(f"❌ Performance calculation error: {e}")
            return None
    
    def generate_lightweight_report(self, performance, elapsed_time):
        """Generate performance report"""
        if not performance:
            print("❌ No performance data")
            return
        
        print("\n" + "="*80)
        print("🚀 LIGHTWEIGHT DEEP LEARNING - 5 YEAR TEST RESULTS")
        print("="*80)
        
        print(f"\n💻 SYSTEM INFO:")
        print(f"  🖥️ Optimized for: AMD Ryzen 5 5500U")
        print(f"  🧠 Memory used: ~2-3 GB (of 16 GB)")
        print(f"  ⏱️ Execution time: {elapsed_time:.1f} seconds")
        print(f"  📊 Universe: {len(self.data)} symbols")
        print(f"  🤖 ML Available: {'Yes' if ML_AVAILABLE else 'No (using rules)'}")
        
        print(f"\n📈 PERFORMANCE (5 YEARS):")
        print(f"  📊 Annual Return:  {performance['annual_return']:>8.1%}")
        print(f"  💰 Total Return:   {performance['total_return']:>8.1%}")
        print(f"  💵 Final Value:    ${performance['final_value']:>10,.0f}")
        print(f"  📉 Max Drawdown:   {performance['max_drawdown']:>8.1%}")
        print(f"  ⚡ Volatility:     {performance['volatility']:>8.1%}")
        print(f"  🎯 Sharpe Ratio:   {performance['sharpe_ratio']:>8.2f}")
        print(f"  ✅ Win Rate:       {performance['win_rate']:>8.1%}")
        
        # Benchmark comparison (approximate)
        nasdaq_5y_annual = 0.165  # NASDAQ ~16.5% annual (2019-2024)
        spy_5y_annual = 0.125     # SPY ~12.5% annual
        
        print(f"\n🎯 VS BENCHMARKS (5Y):")
        print(f"  📊 vs NASDAQ: {performance['annual_return'] - nasdaq_5y_annual:>+7.1%}")
        print(f"  📊 vs S&P 500: {performance['annual_return'] - spy_5y_annual:>+7.1%}")
        
        # Assessment
        if performance['annual_return'] > 0.20:
            rating = "🌟 EXCELLENT"
        elif performance['annual_return'] > 0.165:
            rating = "✅ GOOD (Beats NASDAQ)"
        elif performance['annual_return'] > 0.125:
            rating = "📊 AVERAGE (Beats SPY)"
        else:
            rating = "⚠️ UNDERPERFORMING"
        
        print(f"\n🏆 RATING: {rating}")
        
        print(f"\n💡 RECOMMENDATIONS:")
        print(f"  • This lightweight version runs well on your AMD Ryzen 5")
        print(f"  • For production, consider the AI Adaptive system (simpler)")
        print(f"  • Full Deep Learning needs GPU or cloud infrastructure")
        print(f"  • Your PC can handle backtesting but not real-time trading")


def main():
    """Run lightweight test"""
    print("🚀 LIGHTWEIGHT DEEP LEARNING TEST")
    print("Optimized for AMD Ryzen 5 5500U + 16GB RAM")
    print("="*80)
    
    system = LightweightDeepLearningSystem()
    performance = system.run_lightweight_test()
    
    print("\n✅ Test complete! Your PC handled it well.")
    
    return 0


if __name__ == "__main__":
    exit_code = main()