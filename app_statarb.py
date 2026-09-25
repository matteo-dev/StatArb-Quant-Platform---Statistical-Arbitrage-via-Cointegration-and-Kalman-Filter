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
st.title("📈 Plateforme d'Arbitrage Statistique (StatArb)")
st.markdown("""
**Moteur Quantitatif Market-Neutral par Cointégration Dynamique**  
Cette application démontre la mise en production d'une stratégie de *Pairs Trading*. 
Elle modélise le ratio de couverture par Filtre de Kalman et alloue le capital en intégrant les frictions réelles du marché.
""")


# BARRE LATÉRALE : PARAMÉTRAGE DU MODÈLE
with st.sidebar:
    st.header("⚙️ Paramétrage du Modèle")
    
    st.subheader("1. Univers d'Investissement")
    col_y, col_x = st.columns(2)
    with col_y:
        ticker_y = st.text_input("Actif Y", value="KO", help="Actif dépendant (ex: Coca-Cola)")
    with col_x:
        ticker_x = st.text_input("Actif X", value="PEP", help="Actif indépendant (ex: PepsiCo)")
    
    start_date = st.date_input("Date de début", pd.to_datetime("2020-01-01"))
    end_date = st.date_input("Date de fin", pd.to_datetime("2024-01-01"))

    st.subheader("2. Frictions & Capital")
    capital = st.number_input("Capital Alloué ($)", value=10000.0, step=1000.0, help="Capital nominal engagé par trade.")
    tc_bps = st.number_input("Commissions (bps)", value=5.0, help="Frais de courtage en points de base.")
    borrow_fee = st.number_input("Emprunt Short (%)", value=2.0, help="Taux annualisé pour la vente à découvert.") / 100

    st.subheader("3. Signaux d'Exécution")
    entry_z = st.slider("Seuil Z-Score (Entrée)", min_value=1.0, max_value=4.0, value=2.0, step=0.1, 
                        help="Nombre d'écarts-types pour déclencher un ordre d'arbitrage.")

    run_engine = st.button("🚀 Lancer le Moteur Quantitatif", type="primary", use_container_width=True)


# EXÉCUTION DU PIPELINE QUANTITATIF
if run_engine:
    with st.spinner("Acquisition des données et calculs matriciels en cours..."):
        
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
        tab1, tab2, tab3 = st.tabs(["📊 Tableau de Bord Global", "🧠 Mécanique Mathématique", "💼 Risque & Performance"])

        # ONGLET 1 : VUE D'ENSEMBLE & SIGNAUX
        with tab1:
            st.subheader(f"Arbitrage sur la paire {ticker_y} / {ticker_x}")
            
            # KPIs Métiers
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            kpi1.metric("PnL Net", f"{metrics['PnL Net ($)']:.2f} $", help="Profit net de toutes les frictions de marché.")
            kpi2.metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.2f}", help="Rendement ajusté au risque (Annualisé).")
            kpi3.metric("Win Rate", f"{metrics['Win Rate (%)']:.1f} %", help="Pourcentage de jours passés en position avec un PnL positif.")
            kpi4.metric("Max Drawdown", f"{metrics['Max Drawdown (%)']:.2f} %", help="Pire chute du capital observée.")

            # Explication interactive
            with st.expander("💡 Comment lire ce graphique ?"):
                st.write("""
                Ce graphique représente la **déviation normalisée** du spread (Z-Score). 
                - La courbe violette est l'écart entre les deux actifs, ajusté du ratio de couverture dynamique.
                - Lorsque la courbe franchit les lignes pointillées rouges/vertes, le modèle détecte une anomalie statistique et ouvre une position.
                - La position est conservée jusqu'au retour à la moyenne (ligne pointillée grise à 0).
                """)

            # Graphe Z-Score
            fig_z = go.Figure()
            fig_z.add_trace(go.Scatter(x=df_results.index, y=df_results['zscore'], name="Z-Score", line=dict(color='purple', width=1.5)))
            fig_z.add_hline(y=entry_z, line_dash="dash", line_color="red", annotation_text="Seuil Short Spread")
            fig_z.add_hline(y=-entry_z, line_dash="dash", line_color="green", annotation_text="Seuil Long Spread")
            fig_z.add_hline(y=0.0, line_dash="dot", line_color="gray", annotation_text="Retour à la moyenne")
            
            longs = df_results[df_results['signal'] == 1]
            shorts = df_results[df_results['signal'] == -1]
            fig_z.add_trace(go.Scatter(x=longs.index, y=longs['zscore'], mode='markers', marker=dict(color='green', size=8), name="Entrée Long"))
            fig_z.add_trace(go.Scatter(x=shorts.index, y=shorts['zscore'], mode='markers', marker=dict(color='red', size=8), name="Entrée Short"))
            
            fig_z.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0), hovermode="x unified")
            st.plotly_chart(fig_z, use_container_width=True)

        # ONGLET 2 : THÉORIE ET MÉTHODOLOGIE QUANT
        with tab2:
            st.subheader("Filtrage Bruit/Signal et Dynamique Stochastique")
            
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.info(f"**Demi-vie (Half-Life) :** {half_life:.2f} jours")
                st.write("""
                Calculée via le processus d'**Ornstein-Uhlenbeck**, la demi-vie représente le temps nécessaire pour que le spread résorbe 50% de son écart à la moyenne après un choc. 
                *Plus elle est courte, moins la stratégie souffre des coûts de portage.*
                """)
            with col_m2:
                st.info(f"**Fenêtre de standardisation :** {lookback} jours")
                st.write("""
                Le Z-score n'est pas calculé sur une moyenne globale biaisée, mais sur une fenêtre glissante calibrée sur $1.5 \\times t_{1/2}$. 
                Cela garantit un modèle adaptatif et **sans aucun lookahead bias**.
                """)

            st.markdown("---")
            st.markdown("### 🔄 Le Ratio de Couverture (Filtre de Kalman)")
            st.write("""
            Contrairement à la régression classique (OLS) qui fige la relation entre les deux actifs, le **Filtre de Kalman** met à jour le $\\beta_t$ récursivement. 
            Il modélise la capacité de la relation à dériver dans le temps.
            """)
            
            fig_beta = go.Figure()
            fig_beta.add_trace(go.Scatter(x=df_results.index, y=df_results['beta'], name="Hedge Ratio (Beta)", line=dict(color='orange', width=2)))
            fig_beta.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_beta, use_container_width=True)

        # ONGLET 3 : FINANCIALS & BACKTEST
        with tab3:
            st.subheader("Simulation du Portefeuille Institutionnel")
            
            with st.expander("📋 Détail des coûts modélisés (Frictions)"):
                st.write(f"""
                - **Commissions :** {tc_bps} points de base par transaction.
                - **Slippage :** 2.0 points de base (simulation de la dégradation d'exécution).
                - **Borrow Fee :** {borrow_fee*100:.1f} % annualisé (Coût payé chaque jour pour maintenir la jambe *short*).
                """)

            fig_pnl = go.Figure()
            fig_pnl.add_trace(go.Scatter(x=df_results.index, y=df_results['portfolio_value'], 
                                     name="Capital Net", fill='tozeroy', line=dict(color='teal', width=2)))
            fig_pnl.add_hline(y=capital, line_dash="dot", line_color="black")
            fig_pnl.update_layout(height=400, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_pnl, use_container_width=True)
            
            st.markdown("### 🔍 Journal des dernières opérations (Audit Log)")
            cols_to_show = ['zscore', 'signal', 'locked_beta', 'locked_units', 'pos_y', 'pos_x', 'net_pnl']
            st.dataframe(df_results[cols_to_show].tail(10).style.format("{:.2f}"))
else:
    
    # État initial avant le lancement
    st.info("👈 Paramètre tes actifs et clique sur 'Lancer le Moteur Quantitatif' dans la barre latérale pour générer l'analyse.")