"""
CSA (Credit Support Annex) Collateral Engine.

Models the impact of collateral agreements on counterparty exposure.
Supports multiple CSA configurations:
- Uncollateralised
- Partially collateralised (with threshold and MTA)
- Fully collateralised (ISDA standard)
- CCP-cleared (with initial margin)

Key parameter: MPOR (Margin Period of Risk) — the closeout period
during which exposure persists even with perfect collateralisation.
"""

import numpy as np
from typing import Dict, Optional


class CSAEngine:
    """
    CSA Collateral Engine for exposure modelling.

    Applies collateral rules to simulated MTM paths to produce
    collateralised exposure profiles.

    Attributes:
        threshold: Exposure above which collateral must be posted (₹ Cr).
        mta: Minimum Transfer Amount (₹ Cr).
        mpor_days: Margin Period of Risk in business days.
        margin_frequency: 'daily', 'weekly', or 'none'.
        independent_amount: Upfront collateral regardless of MTM (₹ Cr).
    """

    def __init__(self, threshold: float = 0.0, mta: float = 0.0,
                 mpor_days: int = 10, margin_frequency: str = 'daily',
                 independent_amount: float = 0.0):
        """
        Initialise the CSA engine.

        Args:
            threshold: Exposure threshold in ₹ Crores.
            mta: Minimum Transfer Amount in ₹ Crores.
            mpor_days: Margin Period of Risk in business days.
            margin_frequency: 'daily', 'weekly', or 'none'.
            independent_amount: Independent Amount (IA) in ₹ Crores.
        """
        self.threshold = threshold
        self.mta = mta
        self.mpor_days = mpor_days
        self.margin_frequency = margin_frequency
        self.independent_amount = independent_amount

    def compute_collateral(self, mtm_paths: np.ndarray,
                           time_grid: np.ndarray) -> np.ndarray:
        """
        Compute the collateral held at each time step.

        Collateral(t) = max(MTM(t) - Threshold, 0) if MTM > Threshold + MTA
                       = existing collateral otherwise (sticky below MTA)

        With MPOR: collateral reflects the MTM from mpor_steps ago.

        Args:
            mtm_paths: MTM paths of shape (n_paths, n_steps+1).
            time_grid: Time grid of shape (n_steps+1,).

        Returns:
            Collateral paths of shape (n_paths, n_steps+1).
        """
        n_paths, n_steps_plus_1 = mtm_paths.shape

        if self.margin_frequency == 'none' or np.isinf(self.threshold):
            return np.zeros_like(mtm_paths)

        # Convert MPOR days to time steps
        dt = time_grid[1] - time_grid[0] if len(time_grid) > 1 else 1/12
        mpor_years = self.mpor_days / 252  # Business days to years
        mpor_steps = max(1, int(round(mpor_years / dt)))

        collateral = np.zeros_like(mtm_paths)

        # Margin frequency in steps
        if self.margin_frequency == 'daily':
            margin_every = 1
        elif self.margin_frequency == 'weekly':
            margin_every = max(1, int(round((5/252) / dt)))
        else:
            margin_every = n_steps_plus_1  # Never call

        for i in range(1, n_steps_plus_1):
            # Use lagged MTM (MPOR effect)
            lag_idx = max(0, i - mpor_steps)
            mtm_for_call = mtm_paths[:, lag_idx]

            if i % margin_every == 0:
                # Compute desired collateral
                desired = np.maximum(mtm_for_call - self.threshold, 0.0)

                # Apply MTA: only transfer if change exceeds MTA
                change = desired - collateral[:, i-1]
                transfer = np.where(np.abs(change) >= self.mta,
                                    change, 0.0)
                collateral[:, i] = collateral[:, i-1] + transfer
            else:
                collateral[:, i] = collateral[:, i-1]

            # Add independent amount
            collateral[:, i] += self.independent_amount

        return collateral

    def compute_collateralised_exposure(self, mtm_paths: np.ndarray,
                                         time_grid: np.ndarray) -> np.ndarray:
        """
        Compute residual (collateralised) exposure.

        Residual Exposure = max(MTM(t) - Collateral(t), 0)

        For uncollateralised counterparties:
        Exposure = max(MTM(t), 0)

        Args:
            mtm_paths: MTM paths of shape (n_paths, n_steps+1).
            time_grid: Time grid of shape (n_steps+1,).

        Returns:
            Collateralised exposure paths of shape (n_paths, n_steps+1).
        """
        if self.margin_frequency == 'none' or np.isinf(self.threshold):
            # Uncollateralised: exposure = max(MTM, 0)
            return np.maximum(mtm_paths, 0.0)

        collateral = self.compute_collateral(mtm_paths, time_grid)
        residual = mtm_paths - collateral
        return np.maximum(residual, 0.0)

    def compute_exposure_metrics(self, mtm_paths: np.ndarray,
                                  time_grid: np.ndarray,
                                  percentile: float = 95.0) -> Dict[str, np.ndarray]:
        """
        Compute exposure metrics under this CSA's collateral rules.

        EPE: Time-weighted average of EE over the first year (Basel III 1-year cap).

        Args:
            mtm_paths: Raw MTM paths.
            time_grid: Time grid.
            percentile: PFE percentile.

        Returns:
            Dictionary with EE, PFE, collateral_held.
        """
        coll_exposure = self.compute_collateralised_exposure(
            mtm_paths, time_grid
        )

        ee = np.mean(coll_exposure, axis=0)
        pfe = np.percentile(coll_exposure, percentile, axis=0)
        collateral = self.compute_collateral(mtm_paths, time_grid)
        avg_collateral = np.mean(collateral, axis=0)

        # EPE — Basel III: time-weighted average of EE over first year only
        dt_array = np.diff(time_grid)
        reg_horizon = min(time_grid[-1], 1.0)
        mask = time_grid[1:] <= reg_horizon + 1e-9
        if mask.any() and reg_horizon > 0:
            epe = np.sum(ee[1:][mask] * dt_array[mask]) / reg_horizon
        else:
            epe = float(ee[1]) if len(ee) > 1 else 0.0

        # EEPE: Effective EPE — time-weighted average of running-max EE over the first year
        effective_ee = np.maximum.accumulate(ee)
        if mask.any() and reg_horizon > 0:
            eepe = np.sum(effective_ee[1:][mask] * dt_array[mask]) / reg_horizon
        else:
            eepe = float(effective_ee[1]) if len(effective_ee) > 1 else 0.0

        return {
            'time_grid': time_grid,
            'EE': ee,
            'PFE': pfe,
            'EPE': epe,
            'EEPE': eepe,
            'avg_collateral': avg_collateral,
        }


def compare_csa_scenarios(mtm_paths: np.ndarray,
                          time_grid: np.ndarray,
                          scenarios: Dict[str, Dict]) -> Dict[str, Dict]:
    """
    Compare exposure profiles across multiple CSA configurations.

    Args:
        mtm_paths: Raw MTM paths from Monte Carlo.
        time_grid: Time grid.
        scenarios: Dictionary of scenario_name → CSA parameter dict.

    Returns:
        Dictionary mapping scenario names to exposure metrics.
    """
    results = {}

    for name, params in scenarios.items():
        threshold = params.get('threshold_cr', float('inf'))
        mta = params.get('mta_cr', 0.0)
        mpor = params.get('mpor_days', 10)
        freq = params.get('margin_frequency', 'daily')
        ia = params.get('independent_amount_cr', 0.0)

        engine = CSAEngine(
            threshold=threshold, mta=mta,
            mpor_days=mpor, margin_frequency=freq,
            independent_amount=ia
        )

        metrics = engine.compute_exposure_metrics(mtm_paths, time_grid)
        results[name] = metrics

    return results

def is_ccp_cleared(csa_type: str) -> bool:
    """Returns True if the CSA type represents a CCP-cleared trade."""
    return str(csa_type).strip().upper() in ('CCP-CLEARED', 'CCP_CLEARED', 'CCP')

def get_ccp_mpor() -> int:
    """Standard MPOR for CCP-cleared trades: 5 business days (Basel III)."""
    return 5

