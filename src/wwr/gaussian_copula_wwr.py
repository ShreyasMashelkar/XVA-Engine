"""
Gaussian Copula Wrong-Way Risk Model.

Portfolio-level WWR using a Gaussian copula to model the joint
distribution of counterparty defaults. Captures contagion between
Indian bank counterparties.

Asset Correlation (Basel IRB one-factor ASRF model — free BIS text):
    ρ = 0.12 for financials / PSU banks
    ρ = 0.15 for corporates
    ρ = 0.20 for NBFCs

References:
    - Li (2000), "On Default Correlation: A Copula Function Approach"
    - Basel II IRB correlation parameters (BIS free text)
    - ISDA "Copula Methods in Finance" (free working paper)
"""

import numpy as np
from scipy.stats import norm
from typing import Dict, List, Optional
from src.curves.ois_curve import OISCurve
from src.xva.cva import CreditCurve

# Basel IRB asset correlations (BIS Basel II §272, free)
BASEL_ASSET_CORRELATIONS = {
    'PSU_Bank':     0.12,
    'Private_Bank': 0.12,
    'Financial':    0.12,
    'Corporate_IG': 0.15,
    'Corporate_HY': 0.12,
    'NBFC':         0.20,
    'Sovereign':    0.08,
}


def get_asset_correlation(sector: str) -> float:
    """Get Basel IRB asset correlation for a given counterparty sector."""
    for key, val in BASEL_ASSET_CORRELATIONS.items():
        if key.lower() in sector.lower():
            return val
    return 0.15


def build_empirical_correlation_matrix(sectors: List[str]) -> np.ndarray:
    """
    Build counterparty asset correlation matrix using Basel IRB one-factor model.

    corr(i,j) = sqrt(ρ_i × ρ_j) × same_sector_boost

    Args:
        sectors: List of counterparty sector strings.

    Returns:
        (n×n) correlation matrix (symmetric, diagonal = 1).
    """
    n    = len(sectors)
    corr = np.eye(n)
    for i in range(n):
        for j in range(i+1, n):
            rho_i = get_asset_correlation(sectors[i])
            rho_j = get_asset_correlation(sectors[j])
            base  = np.sqrt(rho_i * rho_j)
            if sectors[i].split('_')[0] == sectors[j].split('_')[0]:
                base = min(base * 1.5, 0.90)
            corr[i,j] = corr[j,i] = base
    return corr


class GaussianCopulaWWR:
    """
    Portfolio-level Wrong-Way Risk using Gaussian Copula.

    Models correlated defaults across counterparties and estimates
    portfolio CVA accounting for joint default scenarios with adverse
    rate moves.
    """

    def __init__(self, ois_curve: OISCurve, n_paths: int = 10000, seed: int = 42):
        self.ois_curve = ois_curve
        self.n_paths   = n_paths
        self.seed      = seed

    def simulate_correlated_defaults(
        self,
        counterparties: List[str],
        credit_curves:  List[CreditCurve],
        sectors:        List[str],
        horizon:        float = 10.0,
        n_time_steps:   int   = 60,
        rate_paths:     Optional[np.ndarray] = None,
        time_grid:      Optional[np.ndarray] = None,
        rate_correlation: float = 0.40,
    ) -> Dict:
        """
        Simulate correlated default times using Gaussian copula.

        Steps:
            1. Generate correlated N(0,1) via Cholesky of asset corr matrix
            2. Transform to U[0,1] via normal CDF
            3. Invert survival function: T_default = -ln(U) / hazard_rate
            4. Optionally correlate systematic factor with rate paths (WWR)

        Args:
            counterparties:   List of counterparty names.
            credit_curves:    List of CreditCurve objects.
            sectors:          Counterparty sectors (for correlation).
            horizon:          Simulation horizon in years.
            n_time_steps:     Number of time steps.
            rate_paths:       Optional (n_paths, n_steps+1) for WWR.
            time_grid:        Time grid for rate_paths.
            rate_correlation: Corr between systematic default factor and rates.

        Returns:
            Dict with default_times, time_grid, corr_matrix.
        """
        rng = np.random.default_rng(self.seed)
        n   = len(counterparties)

        # corr is the Basel one-factor asset-correlation matrix; it is returned
        # for display. We sample directly from the SAME one-factor structure
        # (loadings b_i = sqrt(asset_corr_i), so corr(i,j)=b_i*b_j matches the
        # matrix's off-diagonals) rather than applying a Cholesky on top of an
        # already-correlated draw — the latter double-counted correlation and
        # inflated Var(Z) above 1, biasing default times toward t=0.
        corr = build_empirical_correlation_matrix(sectors)

        # Systematic factor (~N(0,1)) optionally tilted toward the rate paths so
        # default timing co-moves with rates (the WWR channel).
        z_common = rng.standard_normal(self.n_paths)
        if rate_paths is not None:
            final_idx   = min(n_time_steps, rate_paths.shape[1]-1)
            rate_factor = rate_paths[:, final_idx]
            rf_norm     = (rate_factor - rate_factor.mean()) / (rate_factor.std() + 1e-10)
            rho_r       = float(np.clip(rate_correlation, -0.999, 0.999))
            Z_sys       = rho_r * rf_norm + np.sqrt(1.0 - rho_r**2) * z_common
        else:
            Z_sys = z_common

        # One-factor copula draws: each U[:,i] is exactly uniform (no variance
        # inflation), so marginal default-time distributions are unbiased.
        U = np.zeros((self.n_paths, n))
        for i in range(n):
            a_i    = get_asset_correlation(sectors[i] if i < len(sectors) else 'Financial')
            b_i    = np.sqrt(max(min(a_i, 1.0), 0.0))          # factor loading
            z_idio = rng.standard_normal(self.n_paths)
            Z_i    = b_i * Z_sys + np.sqrt(max(1.0 - b_i**2, 0.0)) * z_idio
            U[:,i] = norm.cdf(Z_i)

        default_times = np.zeros((self.n_paths, n))
        for i, curve in enumerate(credit_curves):
            h = max(curve.hazard_rate, 1e-6)
            default_times[:,i] = -np.log(np.maximum(U[:,i], 1e-12)) / h

        return {
            'default_times': default_times,
            'time_grid':     np.linspace(0, horizon, n_time_steps+1),
            'corr_matrix':   corr,
        }

    def compute_portfolio_cva_copula(
        self,
        counterparties: List[str],
        credit_curves:  List[CreditCurve],
        sectors:        List[str],
        ee_profiles:    List[np.ndarray],
        time_grid:      np.ndarray,
        recovery_rate:  float = 0.40,
        rate_paths:     Optional[np.ndarray] = None,
        rate_correlation: float = 0.40,
        mtm_cubes:      Optional[List[np.ndarray]] = None,
    ) -> Dict:
        """
        Compute portfolio CVA under Gaussian copula WWR model.

        Accounts for simultaneous defaults and WWR (correlated defaults
        with adverse rate moves). Compare to sum of standalone CVAs.

        Args:
            counterparties: Counterparty names.
            credit_curves:  Credit curves.
            sectors:        Counterparty sectors.
            ee_profiles:    List of EE profiles (one per counterparty).
            time_grid:      Shared time grid.
            recovery_rate:  Recovery rate (uniform).
            rate_paths:     Optional rate paths for WWR.
            rate_correlation: Rate-default correlation.

        Returns:
            Dict with standalone CVAs, portfolio CVA, WWR multiplier.
        """
        lgd = 1.0 - recovery_rate

        default_sim   = self.simulate_correlated_defaults(
            counterparties=counterparties, credit_curves=credit_curves,
            sectors=sectors, horizon=float(time_grid[-1]),
            n_time_steps=len(time_grid)-1,
            rate_paths=rate_paths, time_grid=time_grid,
            rate_correlation=rate_correlation,
        )
        default_times = default_sim['default_times']

        # Standalone analytical CVAs
        standalone_cvas = {}
        for name, curve, ee in zip(counterparties, credit_curves, ee_profiles):
            h   = curve.hazard_rate
            cva = 0.0
            for k in range(1, len(time_grid)):
                dt      = time_grid[k] - time_grid[k-1]
                ee_mid  = 0.5 * (ee[k-1] + ee[k])
                dp      = h * np.exp(-h * time_grid[k]) * dt
                df      = self.ois_curve.df(time_grid[k])
                cva    += lgd * ee_mid * dp * df
            standalone_cvas[name] = cva

        # Portfolio CVA via simulation.
        #
        # When path-level MTM cubes are supplied, exposure at default is read
        # from the SAME path's scenario, so default timing (correlated with
        # rates via rate_correlation) and exposure are jointly sampled — this is
        # what lets the rate-default correlation produce genuine wrong-way /
        # right-way risk. Falling back to the deterministic average-EE profile
        # (no cube) leaves rate_correlation essentially inert, since exposure no
        # longer depends on the path.
        use_cubes = mtm_cubes is not None
        portfolio_cva_paths = np.zeros(self.n_paths)
        for i, (name, ee) in enumerate(zip(counterparties, ee_profiles)):
            t_defs = default_times[:,i]
            pos_cube = np.maximum(mtm_cubes[i], 0.0) if use_cubes else None
            for path_idx in range(self.n_paths):
                t_def = t_defs[path_idx]
                if t_def > time_grid[-1]:
                    continue
                if use_cubes:
                    ee_at_def = float(np.interp(t_def, time_grid, pos_cube[path_idx]))
                else:
                    ee_at_def = float(np.interp(t_def, time_grid, ee))
                df_at_def = self.ois_curve.df(min(t_def, time_grid[-1]))
                portfolio_cva_paths[path_idx] += lgd * max(ee_at_def, 0.0) * df_at_def

        portfolio_cva = float(np.mean(portfolio_cva_paths))
        sum_standalone = sum(standalone_cvas.values())

        return {
            'standalone_cvas':        standalone_cvas,
            'sum_standalone_cva':     sum_standalone,
            'portfolio_cva_copula':   portfolio_cva,
            'wwr_multiplier':         portfolio_cva / max(sum_standalone, 1e-10),
            'diversification_benefit':sum_standalone - portfolio_cva,
            'corr_matrix':            default_sim['corr_matrix'],
        }
