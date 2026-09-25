# Fichier python pour la gestion des tests statistiques avancés et la génération du spread.


# Importation des bibliothèques nécessaires
import numpy as np
import pandas as pd
import statsmodels.api as sm # Bibliothèque pour la régression OLS
from statsmodels.tsa.stattools import adfuller, kpss # Bibliothèque pour les tests de stationnarité
from typing import Dict, Tuple, Optional
import logging


# Classe pour les tests statistiques avancés
class AdvancedStatTests:

    @staticmethod

    # Fonction pour confirmer la stationnarité d'une série temporelle en combinant les tests ADF et KPSS
    def stationarity_confirmation(series: pd.Series, alpha: float = 0.05) -> Dict[str, any]:

        # 1. Test ADF
        adf_res = adfuller(series, autolag='AIC')
        adf_pvalue = adf_res[1]
        is_adf_stationary = adf_pvalue < alpha

        # 2. Test KPSS
        # regression='c' teste la stationnarité autour d'une constante
        kpss_res = kpss(series, regression='c', nlags='auto')
        kpss_pvalue = kpss_res[1]
        is_kpss_stationary = kpss_pvalue >= alpha

        # Grille de décision
        if is_adf_stationary and is_kpss_stationary:
            conclusion = "STATIONNAIRE_CONFIRME"
        elif is_adf_stationary and not is_kpss_stationary:
            conclusion = "DIFFERENCE_STATIONARY_OU_BRUIT"
        elif not is_adf_stationary and is_kpss_stationary:
            conclusion = "TREND_STATIONARY_OU_FAIBLE_PUISSANCE"
        else:
            conclusion = "NON_STATIONNAIRE"

        return {
            "adf_pvalue": float(adf_pvalue),
            "kpss_pvalue": float(kpss_pvalue),
            "is_stationary": is_adf_stationary and is_kpss_stationary,
            "regime": conclusion
        }

    @staticmethod

    # Fonction pour estimer la demi-vie d'un processus de type Ornstein-Uhlenbeck à partir du spread
    def estimate_half_life(spread: pd.Series) -> float:

        # Nettoyage des valeurs manquantes et calcul des différences
        spread_clean = spread.dropna()
        lagged_spread = spread_clean.shift(1).dropna()
        delta_spread = spread_clean.diff().dropna()

        # Alignement des index
        idx = lagged_spread.index.intersection(delta_spread.index)
        y = delta_spread.loc[idx]
        x = sm.add_constant(lagged_spread.loc[idx])

        # Régression OLS
        model = sm.OLS(y, x).fit()
        beta = model.params.iloc[1]

        # Condition de stabilité du processus OU : beta doit être strictement négatif
        if beta >= 0:
            logging.warning("Le coefficient beta d'Ornstein-Uhlenbeck est positif : pas de retour à la moyenne.")
            return np.inf

        # Calcul exact : lambda = -ln(1 + beta), half_life = ln(2) / lambda
        # Approximation d'Euler : lambda ~ -beta => half_life ~ -ln(2) / beta
        half_life = -np.log(2) / np.log(1 + beta) if (1 + beta) > 0 else -np.log(2) / beta
        return float(half_life)


# Classe pour la dynamique du spread et le calcul du Z-score glissant
class SpreadDynamics:

    @staticmethod

    # Fonction pour calculer le Z-score glissant du spread avec un lookback défini
    def compute_rolling_zscore(spread: pd.Series, lookback_window: int) -> pd.DataFrame:

        # Validation de la taille de la fenêtre pour éviter les biais et assurer la significativité
        if lookback_window < 5:
            raise ValueError("Le lookback_window doit être supérieur à 5 jours pour assurer la significativité.")

        # Moyenne et écart-type calculés sur les données disponibles jusqu'à t-1
        rolling_mean = spread.rolling(window=lookback_window).mean().shift(1)
        rolling_std = spread.rolling(window=lookback_window).std(ddof=1).shift(1)

        zscore = (spread - rolling_mean) / rolling_std

        df_spread = pd.DataFrame({
            'spread': spread,
            'rolling_mean': rolling_mean,
            'rolling_std': rolling_std,
            'zscore': zscore
        }, index=spread.index)

        return df_spread.dropna()


# BLOC DE TESTS (Étape 2)
if __name__ == "__main__":
    from stat_arb_core import MarketData, StatTests

    TICKER_Y = "KO"
    TICKER_X = "PEP"
    START = "2020-01-01"
    END = "2024-01-01"

    print("--- 1. RÉCUPÉRATION DES SÉRIES ET CASCADES STATISTIQUES ---")
    df_prices = MarketData.fetch_pair_data(TICKER_Y, TICKER_X, START, END)
    
    # Estimation OLS statique du hedge ratio
    coint_res = StatTests.engle_granger_cointegration(df_prices[TICKER_Y], df_prices[TICKER_X])
    spread = coint_res['spread']
    hedge_ratio = coint_res['hedge_ratio']

    # 2. Confirmation ADF + KPSS
    validation = AdvancedStatTests.stationarity_confirmation(spread)
    print(f"Paire                 : {TICKER_Y} / {TICKER_X}")
    print(f"Hedge Ratio           : {hedge_ratio:.4f}")
    print(f"P-Value ADF           : {validation['adf_pvalue']:.4f}")
    print(f"P-Value KPSS          : {validation['kpss_pvalue']:.4f}")
    print(f"Régime confirmé       : {validation['regime']}")

    # 3. Estimation Ornstein-Uhlenbeck
    half_life = AdvancedStatTests.estimate_half_life(spread)
    print(f"Half-Life (OU)        : {half_life:.2f} jours de bourse")

    # 4. Calibration de la fenêtre de rolling et Z-score
    # Règle heuristique quant : lookback optimal souvent calibré entre 1x et 2x la half-life
    optimal_lookback = max(int(np.round(half_life * 1.5)), 20)
    print(f"Lookback fenêtre dynamique retenu : {optimal_lookback} jours")

    df_dynamics = SpreadDynamics.compute_rolling_zscore(spread, lookback_window=optimal_lookback)
    print("\nAperçu des dernières métriques dynamiques (sans lookahead bias) :")
    print(df_dynamics.tail(5)[['spread', 'zscore']])