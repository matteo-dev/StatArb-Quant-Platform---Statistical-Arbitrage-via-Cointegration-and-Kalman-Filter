# Fichier python pour l'implémentation du Filtre de Kalman pour l'estimation dynamique du Hedge Ratio 


# Importation des bibliothèques nécessaires
import numpy as np
import pandas as pd
from typing import Tuple
import logging

# Configuration du logger
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


# Classe pour l'implémentation du Filtre de Kalman pour l'arbitrage statistique
class KalmanFilterStatArb:
    
    @staticmethod

    # Fonction pour estimer le beta dynamique et le spread dynamique via le Filtre de Kalman
    def calculate_dynamic_beta(y: pd.Series, x: pd.Series, 
                               trans_cov: float = 1e-5, 
                               obs_cov: float = 1e-3) -> Tuple[pd.Series, pd.Series]:

        n = len(y)
        beta = np.zeros(n)
        P = np.zeros(n)  # Covariance de l'erreur d'estimation de l'état
        
        # Initialisation (les premiers pas de temps vont converger rapidement)
        beta[0] = 0.0
        P[0] = 1.0
        
        # Vectorisation des accès pour optimiser la boucle d'état
        y_vals = y.values
        x_vals = x.values
        
        for t in range(1, n):

            # 1. Étape de Prédiction (Prior)
            beta_prior = beta[t-1]
            P_prior = P[t-1] + trans_cov
            
            # 2. Étape de Mise à jour (Posterior)
            # Prédiction de l'observation et calcul de l'innovation
            y_pred = beta_prior * x_vals[t]
            innovation = y_vals[t] - y_pred
            
            # Variance de l'innovation (S) et Gain de Kalman (K)
            S = P_prior * (x_vals[t]**2) + obs_cov
            K = (P_prior * x_vals[t]) / S if S != 0 else 0.0
            
            # Mise à jour de l'état
            beta[t] = beta_prior + K * innovation
            P[t] = (1 - K * x_vals[t]) * P_prior
            
        beta_series = pd.Series(beta, index=y.index, name="dynamic_beta")
        
        # Calcul du spread (résidu) : Y_t - Beta_t * X_t
        # Lookahead Bias = 0 car beta[t] n'utilise que les prix jusqu'en t.
        dynamic_spread = y - beta_series * x
        dynamic_spread.name = "dynamic_spread"
        
        return beta_series, dynamic_spread


# BLOC DE TESTS ET COMPARAISON (Étape 4)
if __name__ == "__main__":
    from stat_arb_core import MarketData
    from stat_arb_spread import AdvancedStatTests, SpreadDynamics
    from stat_arb_backtest import VectorizedBacktester

    # 1. Ingestion
    df_prices = MarketData.fetch_pair_data("KO", "PEP", "2020-01-01", "2024-01-01")
    y = df_prices["KO"]
    x = df_prices["PEP"]

    # 2. Application du Filtre de Kalman
    logging.info("Exécution du Filtre de Kalman...")
    dyn_beta, dyn_spread = KalmanFilterStatArb.calculate_dynamic_beta(y, x)
    
    # Exclure les premiers jours de chauffe (convergence initiale du filtre)
    burn_in = 20
    dyn_spread = dyn_spread.iloc[burn_in:]
    
    # 3. Métriques Ornstein-Uhlenbeck sur le spread dynamique
    half_life = AdvancedStatTests.estimate_half_life(dyn_spread)
    logging.info(f"Half-Life (Kalman Spread) : {half_life:.2f} jours")
    
    # 4. Calibration du Z-score
    lookback = max(int(np.round(half_life * 1.5)), 20)
    df_dynamics = SpreadDynamics.compute_rolling_zscore(dyn_spread, lookback_window=lookback)
    
    # 5. Backtest sur le signal dynamique
    backtester = VectorizedBacktester(capital=10000.0)
    df_bt = backtester.generate_signals_and_pnl(df_dynamics, entry_z=2.0, exit_z=0.0)
    metrics = backtester.calculate_metrics(df_bt)
    
    print("\n--- RAPPORT DE PERFORMANCE AVEC KALMAN ---")
    for key, value in metrics.items():
        if "Ratio" in key or "Trades" in key:
            print(f"{key:20} : {value:.2f}")
        else:
            print(f"{key:20} : {value:.2f} %" if "%" in key else f"{key:20} : {value:.2f} $")