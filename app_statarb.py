# Fichier streamlit pour l'application de démonstration


# Imports de bibliothèques pour l'interface utilisateur
import streamlit as st # Bibliothèque pour créer des applications web interactives
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Imports du backend
from stat_arb_core import MarketData, StatTests
from stat_arb_spread import AdvancedStatTests, SpreadDynamics
from stat_arb_kalman import KalmanFilterStatArb
from stat_arb_portfolio import InstitutionalPortfolioSizer

# Configuration de la page
st.set_page_config(page_title="StatArb Quant Platform", layout="wide", page_icon="📈")


# EN-TÊTE DE L'APPLICATION
st.title("📈 Statistical Arbitrage Platform (StatArb)")
st.markdown("""
**Market-Neutral Quantitative Engine via Dynamic Cointegration**  
This application demonstrates the production deployment of a *Pairs Trading* strategy. 
It models the hedge ratio using a Kalman Filter and allocates capital by integrating real market frictions.
""")


# BARRE LATÉRALE : PARAMÉTRAGE DU MODÈLE
with st.sidebar:
    st.header("⚙️ Model Configuration")
    
    st.subheader("1. Investment Universe")
    col_y, col_x = st.columns(2)
    with col_y:
        ticker_y = st.text_input("Asset Y", value="KO", help="Dependent asset (e.g., Coca-Cola)")
    with col_x:
        ticker_x = st.text_input("Asset X", value="PEP", help="Independent asset (e.g., PepsiCo)")
    
    start_date = st.date_input("Start Date", pd.to_datetime("2020-01-01"))
    end_date = st.date_input("End Date", pd.to_datetime("2024-01-01"))

    st.subheader("2. Frictions & Capital")
    capital = st.number_input("Allocated Capital ($)", value=10000.0, step=1000.0, help="Nominal capital committed per trade.")
    tc_bps = st.number_input("Commissions (bps)", value=5.0, help="Brokerage fees in basis points.")
    borrow_fee = st.number_input("Short Borrow Fee (%)", value=2.0, help="Annualized rate for short selling.") / 100

    st.subheader("3. Execution Signals")
    entry_z = st.slider("Z-Score Threshold (Entry)", min_value=1.0, max_value=4.0, value=2.0, step=0.1, 
                        help="Number of standard deviations to trigger an arbitrage order.")

    run_engine = st.button("🚀 Run Quantitative Engine", type="primary", use_container_width=True)


# EXÉCUTION DU PIPELINE QUANTITATIF
if run_engine:
    with st.spinner("Fetching data and performing matrix calculations..."):
        
        # 1. Ingestion
        df_prices = MarketData.fetch_pair_data(ticker_y, ticker_x, str(start_date), str(end_date))
        
        # 2. Moteur Kalman
        dyn_beta, dyn_spread = KalmanFilterStatArb.calculate_dynamic_beta(df_prices[ticker_y], df_prices[ticker_x])
        dyn_beta = dyn_beta.iloc[20:]
        dyn_spread = dyn_spread.iloc[20:]
        df_prices_clean = df_prices.iloc[20:]
        
        # 3. Z-Score Dynamique (OU)
        half_life = AdvancedStatTests.estimate_half_life(dyn_spread)
        lookback = max(int(np.round(half_life * 1.5)), 20)
        df_dynamics = SpreadDynamics.compute_rolling_zscore(dyn_spread, lookback_window=lookback)
        
        # 4. Sizer Institutionnel
        sizer = InstitutionalPortfolioSizer(capital=capital, tc_bps=tc_bps, slippage_bps=2.0, borrow_fee_ann=borrow_fee)
        idx = df_dynamics.index
        df_results = sizer.run_sized_backtest(
            df_prices=df_prices_clean.loc[idx],
            ticker_y=ticker_y, ticker_x=ticker_x,
            dyn_beta=dyn_beta.loc[idx],
            zscore=df_dynamics.loc[idx]['zscore'],
            entry_z=entry_z, exit_z=0.0
        )
        metrics = sizer.calculate_metrics(df_results)


        # STRUCTURE EN ONGLETS
        tab1, tab2, tab3 = st.tabs(["📊 Global Dashboard", "🧠 Mathematical Mechanics", "💼 Risk & Performance"])

        # ONGLET 1 : VUE D'ENSEMBLE & SIGNAUX
        with tab1:
            st.subheader(f"Arbitrage on the {ticker_y} / {ticker_x} pair")
            
            # KPIs Métiers
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            kpi1.metric("Net PnL", f"{metrics['PnL Net ($)']:.2f} $", help="Net profit after all market frictions.")
            kpi2.metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.2f}", help="Risk-adjusted return (Annualized).")
            kpi3.metric("Win Rate", f"{metrics['Win Rate (%)']:.1f} %", help="Percentage of days spent in position with a positive PnL.")
            kpi4.metric("Max Drawdown", f"{metrics['Max Drawdown (%)']:.2f} %", help="Worst observed capital drop.")

            # Explication interactive
            with st.expander("💡 How to read this chart?"):
                st.write("""
                This chart represents the **normalized deviation** of the spread (Z-Score). 
                - The purple curve is the spread between the two assets, adjusted for the dynamic hedge ratio.
                - When the curve crosses the red/green dashed lines, the model detects a statistical anomaly and opens a position.
                - The position is held until mean reversion (gray dotted line at 0).
                """)

            # Graphe Z-Score
            fig_z = go.Figure()
            fig_z.add_trace(go.Scatter(x=df_results.index, y=df_results['zscore'], name="Z-Score", line=dict(color='purple', width=1.5)))
            fig_z.add_hline(y=entry_z, line_dash="dash", line_color="red", annotation_text="Short Spread Threshold")
            fig_z.add_hline(y=-entry_z, line_dash="dash", line_color="green", annotation_text="Long Spread Threshold")
            fig_z.add_hline(y=0.0, line_dash="dot", line_color="gray", annotation_text="Mean Reversion")
            
            longs = df_results[df_results['signal'] == 1]
            shorts = df_results[df_results['signal'] == -1]
            fig_z.add_trace(go.Scatter(x=longs.index, y=longs['zscore'], mode='markers', marker=dict(color='green', size=8), name="Long Entry"))
            fig_z.add_trace(go.Scatter(x=shorts.index, y=shorts['zscore'], mode='markers', marker=dict(color='red', size=8), name="Short Entry"))
            
            fig_z.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0), hovermode="x unified")
            st.plotly_chart(fig_z, use_container_width=True)

        # ONGLET 2 : THÉORIE ET MÉTHODOLOGIE QUANT
        with tab2:
            st.subheader("Noise/Signal Filtering and Stochastic Dynamics")
            
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.info(f"**Half-Life:** {half_life:.2f} days")
                st.write("""
                Calculated via the **Ornstein-Uhlenbeck** process, the half-life represents the time required for the spread to mean-revert 50% of its deviation after a shock. 
                *The shorter it is, the less the strategy suffers from carry costs.*
                """)
            with col_m2:
                st.info(f"**Standardization window:** {lookback} days")
                st.write("""
                The Z-score is not calculated on a biased global mean, but on a rolling window calibrated to $1.5 \\times t_{1/2}$. 
                This ensures an adaptive model **without any lookahead bias**.
                """)

            st.markdown("---")
            st.markdown("### 🔄 Hedge Ratio (Kalman Filter)")
            st.write("""
            Unlike classic regression (OLS) which fixes the relationship between the two assets, the **Kalman Filter** updates the $\\beta_t$ recursively. 
            It models the relationship's ability to drift over time.
            """)
            
            fig_beta = go.Figure()
            fig_beta.add_trace(go.Scatter(x=df_results.index, y=df_results['beta'], name="Hedge Ratio (Beta)", line=dict(color='orange', width=2)))
            fig_beta.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_beta, use_container_width=True)

        # ONGLET 3 : FINANCIALS & BACKTEST
        with tab3:
            st.subheader("Institutional Portfolio Simulation")
            
            with st.expander("📋 Modeled costs details (Frictions)"):
                st.write(f"""
                - **Commissions:** {tc_bps} basis points per transaction.
                - **Slippage:** 2.0 basis points (simulating execution degradation).
                - **Borrow Fee:** {borrow_fee*100:.1f} % annualized (Cost paid daily to maintain the *short* leg).
                """)

            fig_pnl = go.Figure()
            fig_pnl.add_trace(go.Scatter(x=df_results.index, y=df_results['portfolio_value'], 
                                     name="Net Capital", fill='tozeroy', line=dict(color='teal', width=2)))
            fig_pnl.add_hline(y=capital, line_dash="dot", line_color="black")
            fig_pnl.update_layout(height=400, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_pnl, use_container_width=True)
            
            st.markdown("### 🔍 Log of latest operations (Audit Log)")
            cols_to_show = ['zscore', 'signal', 'locked_beta', 'locked_units', 'pos_y', 'pos_x', 'net_pnl']
            st.dataframe(df_results[cols_to_show].tail(10).style.format("{:.2f}"))
else:
    
    # État initial avant le lancement
    st.info("👈 Configure your assets and click 'Run Quantitative Engine' in the sidebar to generate the analysis.")
