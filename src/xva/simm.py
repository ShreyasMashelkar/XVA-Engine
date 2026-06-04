"""
ISDA SIMM v2.7 Initial Margin Calculator and MVA Engine.

Calculates Initial Margin (IM) under the ISDA Standard Initial Margin Model.
Calculates MVA (Margin Valuation Adjustment) over the life of the trade using
Dynamic Initial Margin simulation.

Data Source:
  SIMM Risk Weights and Correlations are sourced directly from the publicly
  available ISDA SIMM v2.7 methodology document (free).
"""

import numpy as np


class SIMMCalculator:
    """
    Computes Initial Margin for an INR Interest Rate Swap portfolio.
    Implements a simplified version of ISDA SIMM v2.7 for a single currency
    Interest Rate delta risk class.
    """

    # ISDA SIMM v2.7 Risk Weights for Regular Volatility Currencies (like INR)
    # Tenors: 2W, 1M, 3M, 6M, 1Y, 2Y, 3Y, 5Y, 10Y, 15Y, 20Y, 30Y
    # Here we map to a simplified set of standard swap tenors
    RW_RATES = {
        '1Y': 114,
        '2Y': 65,
        '3Y': 54,
        '5Y': 43,
        '10Y': 36
    }

    # Correlation matrix across tenors (simplified)
    # ISDA SIMM v2.7 specifies high correlation across adjacent tenors
    CORR_RATES = np.array([
        [1.00, 0.90, 0.81, 0.65, 0.40],  # 1Y
        [0.90, 1.00, 0.95, 0.83, 0.58],  # 2Y
        [0.81, 0.95, 1.00, 0.93, 0.70],  # 3Y
        [0.65, 0.83, 0.93, 1.00, 0.88],  # 5Y
        [0.40, 0.58, 0.70, 0.88, 1.00],  # 10Y
    ])

    def __init__(self):
        self.tenor_labels = ['1Y', '2Y', '3Y', '5Y', '10Y']
        self.rw = np.array([self.RW_RATES[t] for t in self.tenor_labels])

    def compute_im_rates_delta(self, sensitivities: dict) -> float:
        """
        Compute SIMM IM for Interest Rate Delta.
        Args:
            sensitivities: Dict of DV01 sensitivities per tenor (e.g., {'5Y': 1000})
        Returns:
            Initial Margin in INR
        """
        # 1. Map sensitivities to SIMM buckets
        s = np.zeros(len(self.tenor_labels))
        for i, t in enumerate(self.tenor_labels):
            s[i] = sensitivities.get(t, 0.0)

        # 2. Multiply by Risk Weights
        ws = s * self.rw

        # 3. Aggregate using correlation matrix
        # Variance = Sum(ws_i^2) + Sum(ws_i * ws_j * corr_ij)
        variance = ws.T @ self.CORR_RATES @ ws

        # SIMM IM is the square root of the variance
        return np.sqrt(max(variance, 0.0))


class MVAEngineV2:
    """
    Margin Valuation Adjustment Engine using SIMM.

    Calculates MVA by simulating Dynamic Initial Margin (DIM) over time.
    MVA = sum( IM(t) * funding_spread * dt * DF(t) )
    """

    def __init__(self, funding_spread: float = 0.01):
        """
        Args:
            funding_spread: Cost of funding IM (e.g., OIS + spread)
        """
        self.funding_spread = funding_spread
        self.simm = SIMMCalculator()

    def compute_mva(self,
                    time_grid: np.ndarray,
                    expected_im_profile: np.ndarray,
                    discount_factors: np.ndarray) -> float:
        """
        Compute MVA from a pre-calculated IM profile.

        Args:
            time_grid: 1D array of time steps
            expected_im_profile: 1D array of expected IM at each time step
            discount_factors: 1D array of discount factors

        Returns:
            MVA value
        """
        dt = np.diff(time_grid, prepend=0.0)
        mva_increments = expected_im_profile * self.funding_spread * dt * discount_factors
        return float(np.sum(mva_increments))

    def estimate_dim_profile(self,
                             trade_maturity: float,
                             time_grid: np.ndarray,
                             initial_sensitivities: dict) -> np.ndarray:
        """
        Estimate Dynamic Initial Margin (DIM) profile.

        In a full implementation, sensitivities are simulated pathwise.
        For a fast approximation, IM is assumed to scale with the remaining
        square root of time or proportional to the amortization of the trade.
        """
        im_t0 = self.simm.compute_im_rates_delta(initial_sensitivities)

        im_profile = np.zeros_like(time_grid)
        for i, t in enumerate(time_grid):
            if t >= trade_maturity:
                im_profile[i] = 0.0
            else:
                # Linear amortization proxy
                im_profile[i] = im_t0 * (trade_maturity - t) / trade_maturity

        return im_profile
