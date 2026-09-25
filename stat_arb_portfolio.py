# Fichier python pour l'allocation de capital institutionnelle et le calcul du PnL net avec frictions de marché.


# Importation des bibliothèques nécessaires
import numpy as np
import pandas as pd
from typing import Dict
import logging


# Classe pour l'allocation de capital institutionnelle et le calcul du PnL net avec frictions de marché
class InstitutionalPortfolioSizer:

    # Fonction d'initialisation avec les paramètres de capital et de frictions de marché
    def __init__(self, capital: float = 10000.0, tc_bps: float = 5.0, slippage_bps: float = 2.0, borrow_fee_ann: float = 0.02):
        self.capital = capital
        self.tc = tc_bps / 10000
        self.slippage = slippage_bps / 10000
        self.borrow_fee_daily = borrow_fee_ann / 252

    # Fonction principale pour exécuter le backtest avec allocation de capital et frictions de marché
    def run_sized_backtest(self, df_prices: pd.DataFrame, ticker_y: str, ticker_x: str, 
                           dyn_beta: pd.Series, zscore: pd.Series, 
                           entry_z: float = 2.0, exit_z: float = 0.0) -> pd.DataFrame:
        
        df = pd.DataFrame(index=df_prices.index)
        df['y'] = df_prices[ticker_y]
        df['x'] = df_prices[ticker_x]
        df['beta'] = dyn_beta
        df['zscore'] = zscore
        
        # 1. Génération du signal cible
        df['signal'] = np.nan
        df.loc[df['zscore'] < -entry_z, 'signal'] = 1   
        df.loc[df['zscore'] > entry_z, 'signal'] = -1   
        df.loc[(df['zscore'] >= -exit_z) & (df['zscore'] <= exit_z), 'signal'] = 0 
        df['signal'] = df['signal'].ffill().fillna(0)
        
        # 2. Capacité d'allocation
        # Notionnel requis pour 1 unité de spread = Prix(Y) + |Beta| * Prix(X)
        df['notional_1_unit'] = df['y'] + df['beta'].abs() * df['x']
        df['max_units'] = np.floor(self.capital / df['notional_1_unit'])
        
        # 3. Verrouillage à l'entrée (Anti-Churning)
        # On recalcule les unités uniquement lors d'un changement de signal
        df['signal_changed'] = df['signal'].diff().fillna(0) != 0
        
        # Capture du Beta et des unités exactes au moment de l'entrée
        df['locked_units'] = np.where(df['signal_changed'], df['max_units'] * df['signal'], np.nan)
        df['locked_units'] = df['locked_units'].ffill().fillna(0)
        
        df['locked_beta'] = np.where(df['signal_changed'], df['beta'], np.nan)
        df['locked_beta'] = df['locked_beta'].ffill().fillna(0)
        
        # 4. Protection Lookahead Bias
        # L'ordre calculé à la clôture t-1 est exécuté et maintenu sur la journée t
        df['pos_y'] = df['locked_units'].shift(1).fillna(0)
        df['pos_x'] = -df['locked_units'].shift(1).fillna(0) * df['locked_beta'].shift(1).fillna(0)
        
        # 5. Calcul vectorisé du PnL
        df['dy'] = df['y'].diff()
        df['dx'] = df['x'].diff()
        
        # PnL Brut (Mark-to-Market de la position détenue)
        df['gross_pnl'] = (df['pos_y'] * df['dy']) + (df['pos_x'] * df['dx'])
        
        # 6. Frictions : appliquées sur les variations nominales des deux jambes
        df['trade_y'] = df['pos_y'].diff().fillna(0)
        df['trade_x'] = df['pos_x'].diff().fillna(0)
        
        df['notional_traded'] = df['trade_y'].abs() * df['y'] + df['trade_x'].abs() * df['x']
        df['tx_costs'] = df['notional_traded'] * (self.tc + self.slippage)
        
        # Coût d'emprunt uniquement sur les positions vendeuses
        short_y = np.where(df['pos_y'] < 0, df['pos_y'].abs() * df['y'], 0)
        short_x = np.where(df['pos_x'] < 0, df['pos_x'].abs() * df['x'], 0)
        df['borrow_costs'] = (short_y + short_x) * self.borrow_fee_daily
        
        # 7. Agrégation finale
        df['net_pnl'] = df['gross_pnl'] - df['tx_costs'] - df['borrow_costs']
        df['cum_pnl'] = df['net_pnl'].cumsum()
        df['portfolio_value'] = self.capital + df['cum_pnl']
        df['strategy_return'] = df['portfolio_value'].pct_change().fillna(0)
        
        return df

    # Fonction pour calculer les métriques de performance du backtest
    def calculate_metrics(self, df_results: pd.DataFrame) -> Dict[str, float]:

        # 1. Rendement moyen et volatilité
        returns = df_results['strategy_return']
        mean_ret = returns.mean()
        std_ret = returns.std()
        sharpe = (mean_ret / std_ret) * np.sqrt(252) if std_ret > 0 else 0.0
        
        # 2. Max Drawdown
        cum_ret = df_results['portfolio_value'] / self.capital
        max_dd = ((cum_ret - cum_ret.cummax()) / cum_ret.cummax()).min()
        
        # 3. Win Rate (jours actifs uniquement)
        active_days = df_results[df_results['pos_y'] != 0]
        win_rate = len(active_days[active_days['net_pnl'] > 0]) / len(active_days) if not active_days.empty else 0.0
        
        return {
            "Capital Final ($)": df_results['portfolio_value'].iloc[-1],
            "PnL Net ($)": df_results['net_pnl'].sum(),
            "Sharpe Ratio": sharpe,
            "Max Drawdown (%)": max_dd * 100,
            "Win Rate (%)": win_rate * 100
        }
    

# BLOC DE TESTS (Étape 5 : Exécution du Sizer Institutionnel)
if __name__ == "__main__":
    from stat_arb_core import MarketData
    from stat_arb_spread import AdvancedStatTests, SpreadDynamics
    from stat_arb_kalman import KalmanFilterStatArb
    import logging
    
    # Réduire la verbosité pour l'affichage final
    logging.getLogger().setLevel(logging.WARNING)

    # 1. Pipeline de données
    TICKER_Y = "KO"
    TICKER_X = "PEP"
    START = "2020-01-01"
    END = "2024-01-01"
    
    print("1. Ingestion des données historiques...")
    df_prices = MarketData.fetch_pair_data(TICKER_Y, TICKER_X, START, END)
    y = df_prices[TICKER_Y]
    x = df_prices[TICKER_X]

    # 2. Filtre de Kalman et dynamique du spread
    print("2. Calibration du Filtre de Kalman...")
    dyn_beta, dyn_spread = KalmanFilterStatArb.calculate_dynamic_beta(y, x)
    
    # Exclure la période de convergence initiale du filtre (20 premiers jours)
    dyn_beta = dyn_beta.iloc[20:]
    dyn_spread = dyn_spread.iloc[20:]
    df_prices_clean = df_prices.iloc[20:]

    half_life = AdvancedStatTests.estimate_half_life(dyn_spread)
    lookback = max(int(np.round(half_life * 1.5)), 20)
    
    print(f"3. Calcul du Z-Score dynamique (Lookback: {lookback} jours)...")
    df_dynamics = SpreadDynamics.compute_rolling_zscore(dyn_spread, lookback_window=lookback)

    # 4. Application du Sizer Institutionnel
    print("4. Allocation du capital et calcul des frictions de marché...")
    sizer = InstitutionalPortfolioSizer(capital=10000.0, tc_bps=5.0, slippage_bps=2.0, borrow_fee_ann=0.02)
    
    # Alignement strict des index pour le backtest
    idx = df_dynamics.index
    df_results = sizer.run_sized_backtest(
        df_prices=df_prices_clean.loc[idx],
        ticker_y=TICKER_Y,
        ticker_x=TICKER_X,
        dyn_beta=dyn_beta.loc[idx],
        zscore=df_dynamics.loc[idx]['zscore'],
        entry_z=2.0,
        exit_z=0.0
    )

    # 5. Métrologie
    metrics = sizer.calculate_metrics(df_results)
    
    print("\n" + "="*50)
    print("RÉSULTATS DE L'ALLOCATION INSTITUTIONNELLE (10 000 $)".center(50))
    print("="*50)
    for key, value in metrics.items():
        if "Ratio" in key:
            print(f"{key:25} : {value:.2f}")
        else:
            print(f"{key:25} : {value:.2f} %" if "%" in key else f"{key:25} : {value:.2f} $")

    print("\n--- Diagnostic des positions récentes (Lookahead Bias Check) ---")
    cols_to_show = ['zscore', 'signal', 'locked_units', 'pos_y', 'pos_x', 'net_pnl', 'cum_pnl']
    print(df_results[cols_to_show].tail(5))