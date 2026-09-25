# Fichier python pour le backtesting vectorisé d'une stratégie d'arbitrage de paires avec prise en compte des frictions de marché.


# Importation des bibliothèques nécessaires
import numpy as np
import pandas as pd
from typing import Dict
import logging


# Classe pour le backtesting vectorisé d'une stratégie d'arbitrage de paires
class VectorizedBacktester:

    # Fonction d'initialisation avec les paramètres de capital et de frictions de marché
    def __init__(self, capital: float = 10000.0, tc_bps: float = 5.0, slippage_bps: float = 2.0, borrow_fee_ann: float = 0.02):

        # Conversion des bps en pourcentage et calcul du coût d'emprunt journalier
        self.capital = capital
        self.tc = tc_bps / 10000
        self.slippage = slippage_bps / 10000
        self.borrow_fee_daily = borrow_fee_ann / 252

    # Fonction principale pour générer les signaux et calculer le PnL net
    def generate_signals_and_pnl(self, df_dynamics: pd.DataFrame, entry_z: float = 2.0, exit_z: float = 0.0) -> pd.DataFrame:

        df = df_dynamics.copy()
        
        # 1. Génération des signaux cibles (1 = Long Spread, -1 = Short Spread, 0 = Flat)
        df['target_position'] = np.nan
        df.loc[df['zscore'] < -entry_z, 'target_position'] = 1   
        df.loc[df['zscore'] > entry_z, 'target_position'] = -1   
        df.loc[(df['zscore'] >= -exit_z) & (df['zscore'] <= exit_z), 'target_position'] = 0 
        
        # Forward fill pour maintenir la position entre l'entrée et la sortie
        df['target_position'] = df['target_position'].ffill().fillna(0)
        
        # 2. Décalage d'exécution (Lookahead bias protection)
        # Le signal généré en t est appliqué comme position détenue en t+1
        df['position'] = df['target_position'].shift(1).fillna(0)
        
        # 3. Calcul du PnL Brut
        # Pour une exposition unitaire au spread
        df['spread_diff'] = df['spread'].diff()
        df['gross_pnl'] = df['position'] * df['spread_diff']
        
        # 4. Frictions de marché
        # Un 'trade' a lieu chaque fois que la position change
        df['trade'] = df['position'].diff().fillna(0)
        
        # Coûts de transaction et slippage (appliqués au nominal du spread à chaque mouvement)
        df['tx_costs'] = np.abs(df['trade']) * (self.tc + self.slippage) * df['spread'].abs()
        
        # Coûts d'emprunt (appliqués chaque jour où une position est ouverte)
        df['borrow_costs'] = np.where(df['position'] != 0, self.borrow_fee_daily * df['spread'].abs(), 0)
        
        # 5. PnL Net et Rendement
        df['net_pnl'] = df['gross_pnl'] - df['tx_costs'] - df['borrow_costs']
        df['cum_pnl'] = df['net_pnl'].cumsum()
        
        # Rendement quotidien par rapport au capital alloué
        df['strategy_return'] = df['net_pnl'] / self.capital
        
        return df

    # Fonction pour calculer les métriques de performance du backtest
    def calculate_metrics(self, df_results: pd.DataFrame) -> Dict[str, float]:
        returns = df_results['strategy_return'].dropna()
        
        # Sharpe Ratio (annualisé, risk-free rate supposé à 0 pour le spread)
        mean_ret = returns.mean()
        std_ret = returns.std()
        sharpe = (mean_ret / std_ret) * np.sqrt(252) if std_ret > 0 else 0.0
        
        # Max Drawdown
        cum_ret = (1 + returns).cumprod()
        peak = cum_ret.cummax()
        drawdown = (cum_ret - peak) / peak
        max_dd = drawdown.min()
        
        # Win Rate (Jours gagnants / Jours investis)
        active_days = df_results[df_results['position'] != 0]
        win_rate = len(active_days[active_days['net_pnl'] > 0]) / len(active_days) if not active_days.empty else 0.0
        
        # Nombre de trades aller-retour
        total_trades = np.abs(df_results['trade']).sum() / 2
        
        return {
            "PnL Net (Absolu)": df_results['net_pnl'].sum(),
            "Sharpe Ratio": sharpe,
            "Max Drawdown (%)": max_dd * 100,
            "Win Rate (%)": win_rate * 100,
            "Total Round Trades": total_trades
        }


# BLOC DE TESTS (Étape 3)
if __name__ == "__main__":
    from stat_arb_core import MarketData, StatTests
    from stat_arb_spread import AdvancedStatTests, SpreadDynamics

    # 1. Pipeline d'ingestion et modélisation (repris de l'Étape 2)
    df_prices = MarketData.fetch_pair_data("KO", "PEP", "2020-01-01", "2024-01-01")
    coint_res = StatTests.engle_granger_cointegration(df_prices["KO"], df_prices["PEP"])
    half_life = AdvancedStatTests.estimate_half_life(coint_res['spread'])
    lookback = max(int(np.round(half_life * 1.5)), 20)
    df_dynamics = SpreadDynamics.compute_rolling_zscore(coint_res['spread'], lookback_window=lookback)

    # 2. Exécution du Backtest
    backtester = VectorizedBacktester(capital=10000.0)
    # Entrée si le Z-score dépasse 2.0 (anomalie de 2 écarts-types), sortie au retour à la moyenne (0.0)
    df_bt = backtester.generate_signals_and_pnl(df_dynamics, entry_z=2.0, exit_z=0.0)
    
    # 3. Métrologie
    metrics = backtester.calculate_metrics(df_bt)
    
    print("\n--- RAPPORT DE PERFORMANCE DU BACKTEST ---")
    for key, value in metrics.items():
        if "Ratio" in key or "Trades" in key:
            print(f"{key:20} : {value:.2f}")
        else:
            print(f"{key:20} : {value:.2f} %" if "%" in key else f"{key:20} : {value:.2f} $")