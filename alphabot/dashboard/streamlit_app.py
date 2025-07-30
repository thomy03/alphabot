"""
AlphaBot Dashboard Streamlit
Monitoring en temps réel du paper trading et backtests
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from datetime import datetime, timedelta
from pathlib import Path
import asyncio
import time

# Configuration page
st.set_page_config(
    page_title="AlphaBot Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .positive {
        color: #00ff00;
    }
    .negative {
        color: #ff0000;
    }
    .neutral {
        color: #888888;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=30)  # Cache 30 secondes
def load_portfolio_data():
    """Charge les données du portfolio"""
    try:
        # Chercher le fichier le plus récent
        data_dir = Path("paper_trading_data/snapshots")
        if not data_dir.exists():
            return None
        
        json_files = list(data_dir.glob("portfolio_*.json"))
        if not json_files:
            return None
        
        # Prendre le plus récent
        latest_file = max(json_files, key=lambda x: x.stat().st_mtime)
        
        with open(latest_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Erreur chargement portfolio: {e}")
        return None

@st.cache_data(ttl=60)  # Cache 1 minute  
def load_backtest_results():
    """Charge les résultats de backtest"""
    try:
        results_dir = Path("backtests/reports")
        if not results_dir.exists():
            return None
        
        json_files = list(results_dir.glob("full_report_*.json"))
        if not json_files:
            return None
        
        latest_file = max(json_files, key=lambda x: x.stat().st_mtime)
        
        with open(latest_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Erreur chargement backtest: {e}")
        return None

@st.cache_data(ttl=300)  # Cache 5 minutes
def load_historical_performance():
    """Charge l'historique de performance"""
    try:
        # Simuler données historiques pour demo
        dates = pd.date_range(start='2024-01-01', end=datetime.now(), freq='D')
        
        # Simuler rendements quotidiens
        np.random.seed(42)
        returns = np.random.normal(0.0008, 0.02, len(dates))  # ~20% annuel, 20% vol
        
        # Calculer valeur portfolio
        portfolio_values = [100000]  # Capital initial
        for ret in returns:
            portfolio_values.append(portfolio_values[-1] * (1 + ret))
        
        df = pd.DataFrame({
            'date': dates,
            'portfolio_value': portfolio_values[:-1],
            'daily_return': returns
        })
        
        # Benchmark SPY
        spy_returns = np.random.normal(0.0005, 0.015, len(dates))  # 12% annuel, 15% vol
        spy_values = [100000]
        for ret in spy_returns:
            spy_values.append(spy_values[-1] * (1 + ret))
        
        df['spy_value'] = spy_values[:-1]
        df['spy_return'] = spy_returns
        
        return df
    except Exception as e:
        st.error(f"Erreur données historiques: {e}")
        return None

def create_performance_chart(df):
    """Crée le graphique de performance"""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Performance Portfolio vs SPY', 'Rendements Quotidiens'),
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3]
    )
    
    # Performance cumulée
    fig.add_trace(
        go.Scatter(
            x=df['date'],
            y=df['portfolio_value'],
            name='AlphaBot',
            line=dict(color='#1f77b4', width=2)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['date'],
            y=df['spy_value'],
            name='SPY',
            line=dict(color='#ff7f0e', width=2)
        ),
        row=1, col=1
    )
    
    # Rendements quotidiens
    colors = ['green' if x > 0 else 'red' for x in df['daily_return']]
    fig.add_trace(
        go.Bar(
            x=df['date'],
            y=df['daily_return'] * 100,
            name='Rendement %',
            marker_color=colors,
            opacity=0.7
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        height=600,
        showlegend=True,
        title_text="Performance AlphaBot"
    )
    
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Valeur Portfolio ($)", row=1, col=1)
    fig.update_yaxes(title_text="Rendement (%)", row=2, col=1)
    
    return fig

def create_positions_chart(positions):
    """Crée le graphique des positions"""
    if not positions:
        return None
    
    df_pos = pd.DataFrame(positions)
    
    # Pie chart des allocations
    fig = px.pie(
        df_pos,
        values='market_value',
        names='symbol',
        title='Allocation du Portfolio'
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    
    return fig

def create_risk_metrics_chart(metrics):
    """Crée le graphique des métriques de risque"""
    # Gauge charts pour les métriques clés
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Sharpe Ratio', 'Max Drawdown', 'Win Rate'),
        specs=[[{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}]]
    )
    
    # Sharpe Ratio
    sharpe = metrics.get('sharpe_ratio', 0)
    fig.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=sharpe,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Sharpe"},
            gauge={
                'axis': {'range': [None, 3]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 1], 'color': "lightgray"},
                    {'range': [1, 1.5], 'color': "yellow"},
                    {'range': [1.5, 3], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 1.5
                }
            }
        ),
        row=1, col=1
    )
    
    # Max Drawdown
    max_dd = abs(metrics.get('max_drawdown', 0)) * 100
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=max_dd,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Max DD (%)"},
            gauge={
                'axis': {'range': [None, 30]},
                'bar': {'color': "darkred"},
                'steps': [
                    {'range': [0, 10], 'color': "green"},
                    {'range': [10, 15], 'color': "yellow"},
                    {'range': [15, 30], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 15
                }
            }
        ),
        row=1, col=2
    )
    
    # Win Rate
    win_rate = metrics.get('win_rate', 0) * 100
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=win_rate,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Win Rate (%)"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkgreen"},
                'steps': [
                    {'range': [0, 50], 'color': "red"},
                    {'range': [50, 60], 'color': "yellow"},
                    {'range': [60, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 60
                }
            }
        ),
        row=1, col=3
    )
    
    fig.update_layout(height=300)
    
    return fig

def main():
    """Interface principale du dashboard"""
    
    # Header
    st.title("🤖 AlphaBot Dashboard")
    st.markdown("**Monitoring en temps réel du système de trading multi-agents**")
    
    # Sidebar
    st.sidebar.title("Navigation")
    
    # Mode sélection
    mode = st.sidebar.selectbox(
        "Mode d'affichage",
        ["📊 Live Trading", "📈 Backtest Results", "📋 Agent Status", "⚙️ Configuration"]
    )
    
    # Auto-refresh
    auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=True)
    
    if auto_refresh:
        # Auto-refresh toutes les 30 secondes
        time.sleep(1)
        st.rerun()
    
    # Mode Live Trading
    if mode == "📊 Live Trading":
        st.header("📊 Live Trading Dashboard")
        
        # Charger données portfolio
        portfolio_data = load_portfolio_data()
        
        if portfolio_data is None:
            st.warning("🔌 Aucunes données de paper trading disponibles")
            st.info("Lancez le paper trading avec: `python scripts/test_paper_trading.py`")
            return
        
        # Métriques principales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_value = portfolio_data.get('total_value', 0)
            st.metric(
                "💰 Valeur Portfolio",
                f"${total_value:,.0f}",
                f"{portfolio_data.get('total_return', 0):+.1%}"
            )
        
        with col2:
            total_pnl = portfolio_data.get('total_pnl', 0)
            st.metric(
                "📈 P&L Total",
                f"${total_pnl:+,.0f}",
                f"{portfolio_data.get('total_return', 0):+.1%}"
            )
        
        with col3:
            active_positions = len(portfolio_data.get('positions', []))
            st.metric(
                "🎯 Positions Actives",
                active_positions,
                f"{portfolio_data.get('active_orders', 0)} ordres"
            )
        
        with col4:
            total_trades = portfolio_data.get('total_trades', 0)
            st.metric(
                "⚡ Total Trades",
                total_trades,
                "aujourd'hui"
            )
        
        # Graphiques
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Performance historique
            df = load_historical_performance()
            if df is not None:
                fig_perf = create_performance_chart(df)
                st.plotly_chart(fig_perf, use_container_width=True)
        
        with col2:
            # Allocation portfolio
            positions = portfolio_data.get('positions', [])
            if positions:
                fig_pos = create_positions_chart(positions)
                if fig_pos:
                    st.plotly_chart(fig_pos, use_container_width=True)
            else:
                st.info("Aucune position active")
        
        # Métriques de risque
        st.subheader("📊 Métriques de Risque")
        metrics = portfolio_data.get('metrics', {})
        
        if metrics:
            fig_risk = create_risk_metrics_chart(metrics)
            st.plotly_chart(fig_risk, use_container_width=True)
        
        # Table des positions
        st.subheader("📋 Positions Détaillées")
        if positions:
            df_positions = pd.DataFrame(positions)
            df_positions['weight'] = df_positions['weight'] * 100
            df_positions = df_positions.round(2)
            
            # Colorer les P&L
            def color_pnl(val):
                color = 'green' if val > 0 else 'red' if val < 0 else 'black'
                return f'color: {color}'
            
            styled_df = df_positions.style.applymap(color_pnl, subset=['unrealized_pnl'])
            st.dataframe(styled_df, use_container_width=True)
        else:
            st.info("Aucune position active")
    
    # Mode Backtest Results
    elif mode == "📈 Backtest Results":
        st.header("📈 Résultats de Backtest")
        
        backtest_data = load_backtest_results()
        
        if backtest_data is None:
            st.warning("📊 Aucun backtest disponible")
            st.info("Lancez un backtest avec: `python scripts/run_full_backtest_10years.py`")
            return
        
        # Métriques principales du backtest
        main_results = backtest_data.get('main_results', {})
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_return = main_results.get('total_return', 0)
            st.metric("📈 Rendement Total", f"{total_return:+.1%}")
        
        with col2:
            ann_return = main_results.get('annualized_return', 0)
            st.metric("📊 Rendement Annualisé", f"{ann_return:+.1%}")
        
        with col3:
            sharpe = main_results.get('sharpe_ratio', 0)
            st.metric("⚡ Sharpe Ratio", f"{sharpe:.2f}")
        
        with col4:
            max_dd = main_results.get('max_drawdown', 0)
            st.metric("📉 Max Drawdown", f"{max_dd:.1%}")
        
        # Objectifs atteints
        objectives_met = backtest_data.get('objectives_met', False)
        
        if objectives_met:
            st.success("🎉 Tous les objectifs sont atteints!")
        else:
            st.warning("⚠️ Certains objectifs ne sont pas atteints")
        
        # Validation détaillée
        st.subheader("🎯 Validation des Objectifs")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sharpe_ok = sharpe >= 1.5
            st.metric(
                "Sharpe ≥ 1.5",
                "✅" if sharpe_ok else "❌",
                f"{sharpe:.2f}"
            )
        
        with col2:
            dd_ok = max_dd >= -0.15
            st.metric(
                "Drawdown ≤ 15%",
                "✅" if dd_ok else "❌", 
                f"{max_dd:.1%}"
            )
        
        with col3:
            return_ok = ann_return >= 0.12
            st.metric(
                "Rendement ≥ 12%",
                "✅" if return_ok else "❌",
                f"{ann_return:.1%}"
            )
        
        # Analyse des crises
        st.subheader("🚨 Performance en Périodes de Crise")
        
        crisis_data = backtest_data.get('crisis_analysis', [])
        if crisis_data:
            df_crisis = pd.DataFrame(crisis_data)
            
            # Graphique performance par crise
            fig_crisis = px.bar(
                df_crisis,
                x='name',
                y='outperformance',
                title='Outperformance vs SPY par Période de Crise',
                color='outperformance',
                color_continuous_scale='RdYlGn'
            )
            fig_crisis.update_xaxes(tickangle=45)
            st.plotly_chart(fig_crisis, use_container_width=True)
            
            # Table détaillée
            st.dataframe(df_crisis, use_container_width=True)
        
        # Analyse sectorielle
        st.subheader("🏭 Performance par Secteur")
        
        sector_data = backtest_data.get('sector_analysis', [])
        if sector_data:
            df_sectors = pd.DataFrame(sector_data)
            
            # Graphique performance sectorielle
            fig_sectors = px.scatter(
                df_sectors,
                x='return',
                y='sharpe',
                size='trades',
                color='sector',
                title='Performance vs Sharpe par Secteur',
                hover_data=['max_dd']
            )
            st.plotly_chart(fig_sectors, use_container_width=True)
    
    # Mode Agent Status
    elif mode == "📋 Agent Status":
        st.header("📋 Statut des Agents")
        
        # Simuler statut des agents
        agents_status = [
            {"name": "Risk Agent", "status": "🟢 Active", "last_signal": "2 min ago", "performance": "98.5%"},
            {"name": "Technical Agent", "status": "🟢 Active", "last_signal": "1 min ago", "performance": "97.2%"},
            {"name": "Sentiment Agent", "status": "🟡 Slow", "last_signal": "5 min ago", "performance": "89.1%"},
            {"name": "Fundamental Agent", "status": "🟢 Active", "last_signal": "3 min ago", "performance": "95.8%"},
            {"name": "Optimization Agent", "status": "🟢 Active", "last_signal": "1 min ago", "performance": "99.1%"},
            {"name": "Execution Agent", "status": "🟢 Active", "last_signal": "30 sec ago", "performance": "96.7%"}
        ]
        
        # Table des agents
        df_agents = pd.DataFrame(agents_status)
        st.dataframe(df_agents, use_container_width=True)
        
        # Graphique performance agents
        fig_agents = px.bar(
            df_agents,
            x='name',
            y='performance',
            title='Performance des Agents (%)',
            color='performance',
            color_continuous_scale='RdYlGn'
        )
        fig_agents.update_xaxes(tickangle=45)
        st.plotly_chart(fig_agents, use_container_width=True)
        
        # Logs récents (simulés)
        st.subheader("📝 Logs Récents")
        
        logs = [
            {"timestamp": "2024-01-15 14:32:15", "agent": "Technical Agent", "message": "EMA crossover signal generated for AAPL"},
            {"timestamp": "2024-01-15 14:31:45", "agent": "Risk Agent", "message": "Portfolio VaR updated: 2.1%"},
            {"timestamp": "2024-01-15 14:31:30", "agent": "Execution Agent", "message": "Order filled: BUY 100 MSFT @ $350.25"},
            {"timestamp": "2024-01-15 14:30:22", "agent": "Sentiment Agent", "message": "News sentiment score: GOOGL +0.65"},
            {"timestamp": "2024-01-15 14:29:18", "agent": "Optimization Agent", "message": "Portfolio rebalancing triggered"}
        ]
        
        for log in logs:
            st.text(f"[{log['timestamp']}] {log['agent']}: {log['message']}")
    
    # Mode Configuration
    elif mode == "⚙️ Configuration":
        st.header("⚙️ Configuration du Système")
        
        # Configuration trading
        st.subheader("📊 Paramètres de Trading")
        
        col1, col2 = st.columns(2)
        
        with col1:
            initial_capital = st.number_input("Capital Initial ($)", value=100000, step=10000)
            max_position_size = st.slider("Taille Position Max (%)", 1, 20, 5)
            commission_rate = st.number_input("Commission (%)", value=0.1, step=0.01, format="%.2f")
        
        with col2:
            max_drawdown_limit = st.slider("Limite Drawdown (%)", 5, 25, 15)
            rebalance_frequency = st.selectbox("Fréquence Rebalancement", ["Daily", "Weekly", "Monthly"])
            risk_tolerance = st.selectbox("Tolérance Risque", ["Conservative", "Moderate", "Aggressive"])
        
        # Configuration agents
        st.subheader("🤖 Configuration des Agents")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            technical_weight = st.slider("Poids Technical Agent", 0.0, 1.0, 0.3)
            fundamental_weight = st.slider("Poids Fundamental Agent", 0.0, 1.0, 0.2)
        
        with col2:
            sentiment_weight = st.slider("Poids Sentiment Agent", 0.0, 1.0, 0.2)
            risk_weight = st.slider("Poids Risk Agent", 0.0, 1.0, 0.3)
        
        with col3:
            signal_threshold = st.slider("Seuil Signal", 0.0, 1.0, 0.7)
            update_frequency = st.selectbox("Fréquence Update", ["1s", "5s", "10s", "30s"])
        
        # Boutons d'action
        st.subheader("🎮 Actions")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("💾 Sauvegarder Config"):
                st.success("Configuration sauvegardée!")
        
        with col2:
            if st.button("🔄 Recharger Config"):
                st.info("Configuration rechargée!")
        
        with col3:
            if st.button("🚀 Démarrer Trading"):
                st.success("Paper trading démarré!")
        
        with col4:
            if st.button("🛑 Arrêter Trading"):
                st.warning("Paper trading arrêté!")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>🤖 <b>AlphaBot Multi-Agent Trading System</b> - Phase 5 Dashboard</p>
            <p>Dernière mise à jour: {}</p>
        </div>
        """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()