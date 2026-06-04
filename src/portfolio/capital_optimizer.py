import numpy as np
import pandas as pd
from typing import Dict, List, Any
from src.sa_ccr.regulatory import SACCRCalculator, compute_rwa, compute_capital_requirement
from src.xva.kva import KVAEngine

class CapitalOptimizer:
    """Computes capital metrics and Return on Capital (RoC) to rank trades."""
    
    def __init__(self, ois_curve, counterparties_df: pd.DataFrame):
        self.ois_curve = ois_curve
        self.cptys = counterparties_df.set_index('Counterparty').to_dict(orient='index')
        self.saccr = SACCRCalculator()
        self.kva_engine = KVAEngine(ois_curve)
        
    def evaluate_trade(self, trade: Dict[str, Any], mtm_val: float) -> Dict[str, Any]:
        """Calculates Capital and KVA for a single trade."""
        notional = float(trade['Notional'])
        maturity = float(trade['Maturity'])
        cpty_name = trade['Counterparty']
        direction = trade['Direction']
        
        # 1. SA-CCR EAD
        # Simplify to unmargined EAD for trade level
        rc = max(mtm_val, 0)
        
        # PFE
        sf = 0.005 if maturity <= 5 else 0.015
        mf = 1.0  # unmargined MF
        delta = 1.0 if direction == 'Receive Fixed' else -1.0
        
        supervisory_duration = (1.0 - np.exp(-0.05 * maturity)) / 0.05
        adjusted_notional = notional * supervisory_duration
        
        pfe = sf * abs(adjusted_notional * mf * delta)
        ead = 1.4 * (rc + pfe)
        
        # 2. Capital & KVA
        risk_weight = self.cptys.get(cpty_name, {}).get('RiskWeight', 0.50)
        rwa = compute_rwa(ead, risk_weight)
        capital = compute_capital_requirement(rwa)
        
        # Approximate average KVA (using a flat profile proxy for speed)
        # KVA = cost_of_capital * EAD_profile * DF
        cost_of_capital = 0.12
        # Rough proxy for KVA: capital * duration * cost
        kva_proxy = capital * supervisory_duration * cost_of_capital
        
        # 3. Profit / Return on Capital
        # Revenue proxy: MTM (if positive) or some spread
        revenue = max(mtm_val, 0) + (notional * 0.001 * maturity)  # assume 10bps embedded margin
        
        roc = (revenue - kva_proxy) / capital if capital > 0 else 0
        
        return {
            'TradeID': trade['TradeID'],
            'Counterparty': cpty_name,
            'MTM': mtm_val,
            'EAD': ead,
            'RWA': rwa,
            'Capital': capital,
            'KVA': kva_proxy,
            'Revenue': revenue,
            'RoC': roc
        }
        
    def rank_portfolio(self, trades: List[Dict[str, Any]], mtm_vals: Dict[int, float]) -> pd.DataFrame:
        """Evaluates all trades and returns a ranked dataframe by RoC."""
        import numpy as np
        
        results = []
        for trade in trades:
            tid = trade['TradeID']
            mtm = mtm_vals.get(tid, 0.0)
            res = self.evaluate_trade(trade, mtm)
            results.append(res)
            
        df = pd.DataFrame(results)
        if not df.empty:
            df = df.sort_values(by='RoC', ascending=False).reset_index(drop=True)
            
        return df
