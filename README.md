# 📈 StatArb Quant Platform (Pairs Trading & Kalman Filter)

Plateforme quantitative complète de trading paires (*Pairs Trading*) market-neutral développée en Python et Streamlit. Ce projet implémente une chaîne complète de recherche, de modélisation stochastique et de backtesting institutionnel intégrant les frictions réelles du marché (commissions, slippage, coûts d'emprunt à découvert).

## 🚀 Fonctionnalités Clés par Module

1. **Ingestion & Cointégration (`stat_arb_core.py`) :**
   - Téléchargement robuste des prix ajustés via l'API `yfinance`.
   - Application de la méthodologie en deux étapes d'Engle-Granger (Régression OLS et test de stationnarité ADF sur le spread).

2. **Dynamique du Spread & Half-Life (`stat_arb_spread.py`) :**
   - Double confirmation de stationnarité couplant les tests ADF et KPSS.
   - Modélisation du retour à la moyenne via le processus d'Ornstein-Uhlenbeck pour estimer la demi-vie (*Half-Life*).
   - Calcul d'un Z-Score glissant calibré sans aucun biais d'anticipation (*Lookahead Bias*).

3. **Filtre de Kalman Dynamique (`stat_arb_kalman.py`) :**
   - Estimation récursive du ratio de couverture ($\beta_t$) en temps réel pour s'adapter aux changements de régimes macroéconomiques, s'affranchissant des limites d'une régression statique.

4. **Sizer Institutionnel & Frictions (`stat_arb_portfolio.py` & `stat_arb_backtest.py`) :**
   - Allocation dynamique du capital nominal engagé.
   - Verrouillage des positions à l'entrée (*Anti-Churning*) pour éviter la sur-rotation et l'érosion des profits par les frais de courtage.
   - Modélisation fine des coûts de transaction en points de base (*bps*), du slippage et des taux d'emprunt journaliers sur la jambe *short*.

5. **Interface Interactive (`app_statarb.py`) :**
   - Tableau de bord complet sous Streamlit pour piloter les paires d'actifs, visualiser les signaux de trading et analyser la performance nette.

## English Below

Comprehensive market-neutral pairs trading quantitative platform developed in Python and Streamlit. This project implements an end-to-end pipeline for research, stochastic modeling, and institutional backtesting incorporating real market frictions (commissions, slippage, short-borrowing costs).

## 🚀 Key Features by Module

1. **Ingestion & Cointegration (`stat_arb_core.py`):**
   - Robust downloading of adjusted prices via the `yfinance` API.
   - Application of the Engle-Granger two-step methodology (OLS regression and ADF stationarity test on the spread).

2. **Spread Dynamics & Half-Life (`stat_arb_spread.py`):**
   - Dual stationarity confirmation combining ADF and KPSS tests.
   - Mean-reversion modeling via the Ornstein-Uhlenbeck process to estimate the half-life.
   - Calculation of a rolling Z-score calibrated without any lookahead bias.

3. **Dynamic Kalman Filter (`stat_arb_kalman.py`):**
   - Recursive real-time estimation of the hedge ratio ($\beta_t$) to adapt to macroeconomic regime shifts, overcoming the limitations of static regression.

4. **Institutional Sizer & Frictions (`stat_arb_portfolio.py` & `stat_arb_backtest.py`):**
   - Dynamic allocation of committed notional capital.
   - Position entry locking (*Anti-Churning*) to prevent over-rotation and profit erosion from brokerage fees.
   - Detailed modeling of transaction costs in basis points (*bps*), slippage, and daily borrowing rates on the short leg.

5. **Interactive Interface (`app_statarb.py`):**
   - Comprehensive Streamlit dashboard to monitor asset pairs, visualize trading signals, and analyze net performance.
   - 
---

## 🛠️ Installation et Utilisation

1. **Cloner le dépôt / Clone the Reposit :**
   ```bash
   git clone [https://github.com/votre-nom-d-utilisateur/statarb-quant-platform.git](https://github.com/votre-nom-d-utilisateur/statarb-quant-platform.git)
   cd statarb-quant-platform
2. **Installer les dépendances et lancer le frontend / Install requirements and run frontend :**
   ```bash
   pip install -r requirements.txt
   streamlit run app_statarb.py
