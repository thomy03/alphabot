#!/usr/bin/env python3
"""
AlphaBot Performance Dashboard - Interactive Web Visualization
Compare trading systems performance vs market benchmarks
Charts: Portfolio value, drawdowns, regime detection, sector allocation
"""

import pandas as pd
import numpy as np
import yfinance as yf
from flask import Flask, render_template, jsonify, request
import plotly.graph_objs as go
import plotly.utils
import json
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path to import our systems
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import our trading systems
try:
    from scripts.balanced_swing_system import BalancedSwingSystem
    from scripts.adaptive_risk_swing_system import AdaptiveRiskSwingSystem
    from scripts.optimized_daily_system import OptimizedDailySystem
except ImportError as e:
    print(f"Import error: {e}")
    print("Running in standalone mode...")

app = Flask(__name__)

class PerformanceDashboard:
    """
    Dashboard for visualizing trading system performance
    """
    
    def __init__(self):
        self.systems_data = {}
        self.benchmark_data = {}
        self.start_date = "2019-01-01"
        self.end_date = "2024-01-01"
        
    def run_system_comparison(self):
        """Run multiple systems and compare performance"""
        print("🔄 Running systems for dashboard comparison...")
        
        # Run Balanced Swing System
        try:
            balanced_system = BalancedSwingSystem()
            balanced_results = balanced_system.run_balanced_backtest()
            self.systems_data['Balanced Swing'] = self.process_system_results(balanced_results)
            print("✅ Balanced Swing System completed")
        except Exception as e:
            print(f"❌ Balanced Swing System failed: {e}")
        
        # Run Optimized Daily System (our 23.5% champion)
        try:
            daily_system = OptimizedDailySystem()
            daily_results = daily_system.run_optimized_backtest()
            self.systems_data['Optimized Daily'] = self.process_system_results(daily_results)
            print("✅ Optimized Daily System completed")
        except Exception as e:
            print(f"❌ Optimized Daily System failed: {e}")
        
        # Download benchmark data
        self.download_benchmark_data()
        
        print(f"📊 Dashboard ready with {len(self.systems_data)} systems")
        
    def process_system_results(self, results):
        """Process system results for dashboard"""
        if not results or 'performance' not in results:
            return None
            
        # Extract history data
        history_data = None
        if 'history' in results:
            history_data = results['history']
        elif hasattr(results, 'history'):
            history_data = results.history
            
        processed = {
            'performance': results['performance'],
            'config': results.get('config', {}),
            'history': history_data,
            'trades_summary': results.get('trades_summary', {})
        }
        
        return processed
    
    def download_benchmark_data(self):
        """Download benchmark data for comparison"""
        benchmarks = ['SPY', 'QQQ', 'IWM', 'VTI']
        
        for symbol in benchmarks:
            try:
                data = yf.download(symbol, start=self.start_date, end=self.end_date, progress=False)
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.droplevel(1)
                
                closes = data['Close']
                
                # Calculate cumulative returns
                initial_value = 100000  # Same starting value
                cumulative_values = (closes / closes.iloc[0]) * initial_value
                
                # Calculate performance metrics
                total_return = (closes.iloc[-1] / closes.iloc[0]) - 1
                years = len(closes) / 252
                annual_return = (1 + total_return) ** (1/years) - 1
                
                daily_returns = closes.pct_change().dropna()
                volatility = daily_returns.std() * np.sqrt(252)
                
                # Calculate drawdown
                cumulative = closes / closes.iloc[0]
                running_max = cumulative.expanding().max()
                drawdowns = (cumulative - running_max) / running_max
                max_drawdown = drawdowns.min()
                
                self.benchmark_data[symbol] = {
                    'values': cumulative_values.tolist(),
                    'dates': closes.index.strftime('%Y-%m-%d').tolist(),
                    'performance': {
                        'annual_return': float(annual_return),
                        'total_return': float(total_return),
                        'volatility': float(volatility),
                        'max_drawdown': float(max_drawdown),
                        'sharpe_ratio': float((annual_return - 0.02) / volatility) if volatility > 0 else 0
                    }
                }
                
                print(f"✅ {symbol}: {annual_return:.1%} annual")
                
            except Exception as e:
                print(f"❌ {symbol}: {e}")
    
    def create_performance_chart(self):
        """Create interactive performance comparison chart"""
        fig = go.Figure()
        
        # Add system performance lines
        for system_name, system_data in self.systems_data.items():
            if system_data and system_data['history']:
                history_df = pd.DataFrame(system_data['history'])
                history_df['date'] = pd.to_datetime(history_df['date'])
                
                fig.add_trace(go.Scatter(
                    x=history_df['date'],
                    y=history_df['portfolio_value'],
                    mode='lines',
                    name=f"{system_name} ({system_data['performance']['annual_return']:.1%})",
                    line=dict(width=3),
                    hovertemplate=f'<b>{system_name}</b><br>' +
                                 'Date: %{x}<br>' +
                                 'Value: $%{y:,.0f}<br>' +
                                 '<extra></extra>'
                ))
        
        # Add benchmark lines
        for benchmark, bench_data in self.benchmark_data.items():
            fig.add_trace(go.Scatter(
                x=pd.to_datetime(bench_data['dates']),
                y=bench_data['values'],
                mode='lines',
                name=f"{benchmark} ({bench_data['performance']['annual_return']:.1%})",
                line=dict(dash='dash', width=2),
                hovertemplate=f'<b>{benchmark}</b><br>' +
                             'Date: %{x}<br>' +
                             'Value: $%{y:,.0f}<br>' +
                             '<extra></extra>'
            ))
        
        fig.update_layout(
            title={
                'text': '📊 AlphaBot Performance vs Market Benchmarks',
                'x': 0.5,
                'font': {'size': 20}
            },
            xaxis_title='Date',
            yaxis_title='Portfolio Value ($)',
            template='plotly_white',
            height=600,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            hovermode='x unified'
        )
        
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    def create_drawdown_chart(self):
        """Create drawdown comparison chart"""
        fig = go.Figure()
        
        # Add system drawdowns
        for system_name, system_data in self.systems_data.items():
            if system_data and system_data['history']:
                history_df = pd.DataFrame(system_data['history'])
                history_df['date'] = pd.to_datetime(history_df['date'])
                
                # Calculate drawdown
                values = history_df['portfolio_value']
                cumulative = values / values.iloc[0]
                running_max = cumulative.expanding().max()
                drawdowns = (cumulative - running_max) / running_max
                
                fig.add_trace(go.Scatter(
                    x=history_df['date'],
                    y=drawdowns * 100,  # Convert to percentage
                    mode='lines',
                    name=f"{system_name} (Max: {drawdowns.min():.1%})",
                    fill='tonexty' if len(fig.data) == 0 else None,
                    line=dict(width=2),
                    hovertemplate=f'<b>{system_name}</b><br>' +
                                 'Date: %{x}<br>' +
                                 'Drawdown: %{y:.1f}%<br>' +
                                 '<extra></extra>'
                ))
        
        # Add benchmark drawdowns
        for benchmark, bench_data in self.benchmark_data.items():
            # Calculate benchmark drawdown
            values = np.array(bench_data['values'])
            cumulative = values / values[0]
            running_max = np.maximum.accumulate(cumulative)
            drawdowns = (cumulative - running_max) / running_max
            
            fig.add_trace(go.Scatter(
                x=pd.to_datetime(bench_data['dates']),
                y=drawdowns * 100,
                mode='lines',
                name=f"{benchmark} (Max: {bench_data['performance']['max_drawdown']:.1%})",
                line=dict(dash='dash', width=2),
                hovertemplate=f'<b>{benchmark}</b><br>' +
                             'Date: %{x}<br>' +
                             'Drawdown: %{y:.1f}%<br>' +
                             '<extra></extra>'
            ))
        
        fig.update_layout(
            title={
                'text': '📉 Drawdown Analysis - Risk Management Effectiveness',
                'x': 0.5,
                'font': {'size': 18}
            },
            xaxis_title='Date',
            yaxis_title='Drawdown (%)',
            template='plotly_white',
            height=500,
            yaxis=dict(range=[min(-50, -30), 5]),  # Set reasonable range
            hovermode='x unified'
        )
        
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    def create_metrics_table(self):
        """Create performance metrics comparison table"""
        metrics_data = []
        
        # Add systems data
        for system_name, system_data in self.systems_data.items():
            if system_data:
                perf = system_data['performance']
                metrics_data.append({
                    'System': system_name,
                    'Type': 'Trading System',
                    'Annual Return': f"{perf['annual_return']:.1%}",
                    'Total Return': f"{perf['total_return']:.1%}",
                    'Volatility': f"{perf['volatility']:.1%}",
                    'Sharpe Ratio': f"{perf['sharpe_ratio']:.2f}",
                    'Max Drawdown': f"{perf['max_drawdown']:.1%}",
                    'Win Rate': f"{perf.get('win_rate', 0):.1%}",
                    'Final Value': f"${perf['final_value']:,.0f}"
                })
        
        # Add benchmark data
        for benchmark, bench_data in self.benchmark_data.items():
            perf = bench_data['performance']
            metrics_data.append({
                'System': benchmark,
                'Type': 'Market Benchmark',
                'Annual Return': f"{perf['annual_return']:.1%}",
                'Total Return': f"{perf['total_return']:.1%}",
                'Volatility': f"{perf['volatility']:.1%}",
                'Sharpe Ratio': f"{perf['sharpe_ratio']:.2f}",
                'Max Drawdown': f"{perf['max_drawdown']:.1%}",
                'Win Rate': 'N/A',
                'Final Value': f"${bench_data['values'][-1]:,.0f}"
            })
        
        return metrics_data
    
    def create_monthly_returns_heatmap(self, system_name):
        """Create monthly returns heatmap for a specific system"""
        if system_name not in self.systems_data or not self.systems_data[system_name]:
            return None
            
        system_data = self.systems_data[system_name]
        if not system_data['history']:
            return None
            
        # Process monthly returns
        history_df = pd.DataFrame(system_data['history'])
        history_df['date'] = pd.to_datetime(history_df['date'])
        history_df.set_index('date', inplace=True)
        
        # Calculate monthly returns
        monthly_values = history_df['portfolio_value'].resample('M').last()
        monthly_returns = monthly_values.pct_change().dropna()
        
        # Create pivot table for heatmap
        monthly_returns_df = monthly_returns.to_frame('return')
        monthly_returns_df['year'] = monthly_returns_df.index.year
        monthly_returns_df['month'] = monthly_returns_df.index.month
        
        pivot_table = monthly_returns_df.pivot(index='year', columns='month', values='return')
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=pivot_table.values * 100,  # Convert to percentage
            x=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
            y=pivot_table.index,
            colorscale='RdYlGn',
            zmid=0,
            hovertemplate='Year: %{y}<br>Month: %{x}<br>Return: %{z:.1f}%<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'📅 Monthly Returns Heatmap - {system_name}',
            xaxis_title='Month',
            yaxis_title='Year',
            height=400
        )
        
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

# Initialize dashboard
dashboard = PerformanceDashboard()

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/run_analysis')
def run_analysis():
    """API endpoint to run system analysis"""
    try:
        dashboard.run_system_comparison()
        return jsonify({'status': 'success', 'message': 'Analysis completed'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/performance_chart')
def performance_chart():
    """API endpoint for performance chart"""
    try:
        chart_json = dashboard.create_performance_chart()
        return jsonify({'chart': chart_json})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/drawdown_chart')
def drawdown_chart():
    """API endpoint for drawdown chart"""
    try:
        chart_json = dashboard.create_drawdown_chart()
        return jsonify({'chart': chart_json})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/metrics_table')
def metrics_table():
    """API endpoint for metrics table"""
    try:
        table_data = dashboard.create_metrics_table()
        return jsonify({'data': table_data})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/monthly_heatmap/<system_name>')
def monthly_heatmap(system_name):
    """API endpoint for monthly returns heatmap"""
    try:
        chart_json = dashboard.create_monthly_returns_heatmap(system_name)
        if chart_json:
            return jsonify({'chart': chart_json})
        else:
            return jsonify({'error': 'No data available'})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/systems_list')
def systems_list():
    """API endpoint to get available systems"""
    return jsonify({'systems': list(dashboard.systems_data.keys())})

if __name__ == '__main__':
    print("🚀 Starting AlphaBot Performance Dashboard...")
    print("📊 Dashboard will be available at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)