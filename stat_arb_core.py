# Fichier python pour la gestion des données de marché et l'analyse de cointégration.


# Importation des bibliothèques nécessaires
import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm # Bibliothèque pour la régression OLS
from statsmodels.tsa.stattools import adfuller # Bibliothèque pour le test augmenté de Dickey-Fuller
from typing import Tuple, Dict, Optional
import logging

# Configuration basique du logger pour suivre l'exécution
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Classe pour la gestion des données de marché
class MarketData:
    
    @staticmethod

    # Fonction robuste pour télécharger les données de marché pour une paire d'actifs
    def fetch_pair_data(ticker_y: str, ticker_x: str, start_date: str, end_date: str) -> pd.DataFrame:
        logging.info(f"Téléchargement des données pour {ticker_y} et {ticker_x}...")
        try:

            # auto_adjust=True applique les splits et dividendes directement.
            tickers_str = f"{ticker_y} {ticker_x}"
            data = yf.download(tickers_str, start=start_date, end=end_date, auto_adjust=True)
            
            # Extraction robuste selon la structure retournée par yfinance
            if isinstance(data.columns, pd.MultiIndex):
                if 'Close' in data.columns.levels[0]:
                    prices = data['Close']
                elif 'Adj Close' in data.columns.levels[0]:
                    prices = data['Adj Close']
                else:
                    raise KeyError("Ni 'Close' ni 'Adj Close' trouvés dans les colonnes.")
            else:
                prices = data

            # Nettoyage en enlevant les lignes avec des valeurs manquantes
            prices = prices.dropna()
            
            if prices.empty:
                raise ValueError("Le DataFrame est vide après le téléchargement.")
                
            # On s'assure de l'ordre des colonnes
            return prices[[ticker_y, ticker_x]]
            
        except Exception as e:
            logging.error(f"Erreur lors du téléchargement des données : {e}")
            raise

# Classe pour les tests statistiques liés à la cointégration
class StatTests:
    
    @staticmethod

    # Fonction pour effectuer le test ADF sur une série temporelle
    def adf_test(series: pd.Series, significance_level: float = 0.05) -> Tuple[bool, float, float]:

        # Effectue le test ADF et retourne si la série est stationnaire, la p-value et la statistique ADF
        result = adfuller(series)
        adf_stat = result[0]
        p_value = result[1]
        
        # Détermine si la série est stationnaire en comparant la p-value au niveau de signification
        is_stationary = p_value < significance_level
        return is_stationary, p_value, adf_stat

    @staticmethod

    # Fonction pour appliquer la méthode d'Engle-Granger en deux étapes pour tester la cointégration
    def engle_granger_cointegration(y: pd.Series, x: pd.Series) -> Dict[str, float]:

        # Étape 1 : Régression OLS (Y = beta * X + alpha)
        x_with_const = sm.add_constant(x)
        ols_model = sm.OLS(y, x_with_const).fit()
        
        hedge_ratio = ols_model.params.iloc[1] # Le beta
        
        # Étape 2 : Extraction du spread (résidus)
        spread = y - hedge_ratio * x
        
        # Étape 3 : Test ADF sur le spread
        is_stat, p_value, adf_stat = StatTests.adf_test(spread)
        
        return {
            "hedge_ratio": hedge_ratio,
            "is_cointegrated": is_stat,
            "p_value": p_value,
            "adf_statistic": adf_stat,
            "spread": spread
        }

# BLOC DE TESTS (À exécuter pour vérifier que le module fonctionne)
if __name__ == "__main__":
    # Test avec un grand classique : Coca-Cola (KO) et PepsiCo (PEP)
    TICKER_Y = "KO"
    TICKER_X = "PEP"
    START = "2020-01-01"
    END = "2024-01-01"
    
    try:
        
        # 1. Récupération des données
        df_prices = MarketData.fetch_pair_data(TICKER_Y, TICKER_X, START, END)
        logging.info(f"Données récupérées avec succès. Shape: {df_prices.shape}")
        
        # 2. Test de cointégration statique
        coint_results = StatTests.engle_granger_cointegration(df_prices[TICKER_Y], df_prices[TICKER_X])
        
        print("\n--- RÉSULTATS DU TEST DE COINTÉGRATION ---")
        print(f"Paire : {TICKER_Y} (Y) / {TICKER_X} (X)")
        print(f"Hedge Ratio (Beta) statique : {coint_results['hedge_ratio']:.4f}")
        print(f"P-Value ADF du spread     : {coint_results['p_value']:.4f}")
        print(f"Cointégration validée (5%): {coint_results['is_cointegrated']}")
        
    except Exception as e:
        logging.error(f"Le test a échoué : {e}")