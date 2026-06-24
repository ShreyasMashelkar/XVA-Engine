"""
Equity exposure Monte Carlo.

Simulates index (Nifty / Bank Nifty) paths under risk-neutral GBM with a
continuous dividend yield, optionally correlated to the interest-rate factor,
and computes the CCR exposure profile (EE / ENE / PFE) of equity forwards,
European options and total-return swaps.

Equity dynamics (risk-neutral):
    dS = (r - q) S dt + σ S dW_S
so log-spot is a correlated Brownian; the correlation to the rate factor's
Brownian drives equity-rate wrong-way effects in the hybrid XVA.

Pure NumPy. Designed to share its equity Brownian with the rate simulation in
src/xva/hybrid_xva.py so the two asset classes are jointly consistent.
"""

import numpy as np
from typing import Dict, Optional
from src.curves.ois_curve import OISCurve
from src.pricing.equity_options import bsm_price, EquityVolSmile


class EquityGBM:
    """Risk-neutral GBM index simulator with dividends."""

    def __init__(self, spot: float, vol: float, div_yield: float = 0.0):
        self.spot = spot
        self.vol = vol
        self.q = div_yield

    def simulate(self, time_grid: np.ndarray, n_paths: int,
                 ois_curve: OISCurve,
                 equity_normals: Optional[np.ndarray] = None,
                 seed: int = 42) -> np.ndarray:
        """
        Simulate spot paths S(t).

        Args:
            time_grid:      (n_steps+1,) times from 0.
            n_paths:        number of paths.
            ois_curve:      domestic curve (per-step forward rate for drift).
            equity_normals: optional (n_paths, n_steps) standard normals — pass
                            correlated draws from the hybrid engine; otherwise
                            generated independently.
            seed:           RNG seed (used only if equity_normals is None).

        Returns:
            (n_paths, n_steps+1) spot paths with S[:,0] = spot.
        """
        n_steps = len(time_grid) - 1
        if equity_normals is None:
            rng = np.random.default_rng(seed)
            equity_normals = rng.standard_normal((n_paths, n_steps))

        # per-step instantaneous forward rate for the risk-neutral drift
        fwd = np.array([ois_curve.instantaneous_forward(max(t, 1e-6)) for t in time_grid])

        logS = np.zeros((n_paths, n_steps + 1))
        logS[:, 0] = np.log(self.spot)
        for i in range(n_steps):
            dt = time_grid[i + 1] - time_grid[i]
            drift = (fwd[i] - self.q - 0.5 * self.vol ** 2) * dt
            logS[:, i + 1] = logS[:, i] + drift + self.vol * np.sqrt(dt) * equity_normals[:, i]
        return np.exp(logS)

    # ── instrument MTM profiles ───────────────────────────────────────────
    def forward_mtm_paths(self, spot_paths: np.ndarray, time_grid: np.ndarray,
                          ois_curve: OISCurve, strike: float, maturity: float,
                          units: float, long: bool = True) -> np.ndarray:
        """
        Equity forward MTM: units · (S_t·e^{-q(T-t)} - K·P(t,T)).
        """
        n_paths, n_time = spot_paths.shape
        mtm = np.zeros((n_paths, n_time))
        for ti in range(n_time):
            t = time_grid[ti]
            if t > maturity:
                continue
            tau = maturity - t
            df = ois_curve.df(maturity) / max(ois_curve.df(t), 1e-12)
            val = units * (spot_paths[:, ti] * np.exp(-self.q * tau) - strike * df)
            mtm[:, ti] = val if long else -val
        return mtm

    def option_mtm_paths(self, spot_paths: np.ndarray, time_grid: np.ndarray,
                         ois_curve: OISCurve, strike: float, maturity: float,
                         units: float, call: bool = True,
                         smile: Optional[EquityVolSmile] = None) -> np.ndarray:
        """
        European option MTM repriced along each path with remaining maturity,
        using the smile vol at the option's strike.
        """
        n_paths, n_time = spot_paths.shape
        mtm = np.zeros((n_paths, n_time))
        for ti in range(n_time):
            t = time_grid[ti]
            tau = max(maturity - t, 0.0)
            r_t = ois_curve.zero_rate(maturity) if maturity > 1e-6 else 0.0
            S_t = spot_paths[:, ti]
            if smile is not None:
                fwd = S_t * np.exp((r_t - self.q) * tau)
                # Vectorised smile vol (matches EquityVolSmile.vol element-wise):
                # max(atm + skew·k + curv·k², 0.01), k = log(K/forward).
                k = np.log(strike / fwd)
                vol = np.maximum(
                    smile.atm_vol + smile.skew * k + smile.curv * k ** 2, 0.01)
            else:
                vol = np.full(n_paths, self.vol)
            if tau <= 0:
                payoff = np.maximum(S_t - strike, 0.0) if call \
                    else np.maximum(strike - S_t, 0.0)
                mtm[:, ti] = units * payoff
            else:
                # bsm_price is written with NumPy ops, so it broadcasts over the
                # full path slice at once (164k scalar calls → one vector op).
                mtm[:, ti] = units * bsm_price(S_t, strike, tau, r_t, self.q,
                                               vol, call)
        return mtm

    def trs_mtm_paths(self, spot_paths: np.ndarray, time_grid: np.ndarray,
                      notional: float, funding_spread: float = 0.0,
                      receive_equity: bool = True) -> np.ndarray:
        """
        Total-return swap MTM ≈ notional · (S_t/S_0 - 1 - funding accrual).
        Receiver of equity return is long the index performance.
        """
        n_paths, n_time = spot_paths.shape
        mtm = np.zeros((n_paths, n_time))
        S0 = self.spot
        for ti in range(n_time):
            t = time_grid[ti]
            equity_ret = spot_paths[:, ti] / S0 - 1.0
            funding = funding_spread * t
            val = notional * (equity_ret - funding)
            mtm[:, ti] = val if receive_equity else -val
        return mtm

    @staticmethod
    def exposure_metrics(mtm: np.ndarray, time_grid: np.ndarray) -> Dict:
        pos = np.maximum(mtm, 0.0); neg = np.minimum(mtm, 0.0)
        ee = pos.mean(0); ene = neg.mean(0)
        pfe = np.percentile(pos, 95, axis=0)
        dt = np.diff(time_grid, prepend=0.0)
        T = max(float(time_grid[-1]), 1e-8)
        epe = float(np.dot(ee, dt) / T)
        return {'time_grid': time_grid, 'EE': ee, 'ENE': ene, 'PFE': pfe, 'EPE': epe}


class EquityHeston(EquityGBM):
    """
    Stochastic-volatility (Heston) index simulator — a drop-in for EquityGBM.

    This wires the Heston model (``src/pricing/heston.py``) into the equity
    exposure engine, closing the "Heston is unwired" gap. Two things change
    relative to GBM:

    1. **Spot dynamics.** Paths are generated with the Andersen (2008) QE
       scheme, so variance is itself a mean-reverting stochastic process
       correlated with the spot. Exposure therefore picks up genuine
       vol-of-vol and forward-smile tail risk that GBM cannot represent.

    2. **Dynamic smile.** Options are repriced along each path at the path's
       *own* simulated instantaneous vol √v(t), rather than a frozen quadratic
       smile — the smile evolves with the state, which is the whole point of a
       stochastic-vol model.

    The Heston parameters can be calibrated to the existing quadratic vol smile
    with :meth:`from_smile` — no market data beyond the smile already in use.
    """

    def __init__(self, spot: float, params: 'HestonParams', div_yield: float = 0.0):
        super().__init__(spot, vol=float(np.sqrt(max(params.v0, 1e-8))),
                         div_yield=div_yield)
        self.params = params
        self._variance: Optional[np.ndarray] = None
        self.calibration_rmse: Optional[float] = None

    @classmethod
    def from_smile(cls, spot: float, smile: EquityVolSmile, expiry: float,
                   r: float, div_yield: float = 0.0,
                   n_strikes: int = 11) -> 'EquityHeston':
        """
        Calibrate Heston (v0, κ, θ, ξ, ρ) to an existing quadratic vol smile.

        Builds a synthetic implied-vol chain by sampling the smile across
        strikes and fits the Heston parameters to it, so the static smile is
        turned into consistent stochastic-vol dynamics. No external data.
        """
        import pandas as pd
        from src.pricing.heston import calibrate_heston
        fwd = spot * np.exp((r - div_yield) * max(expiry, 1e-6))
        strikes = np.linspace(0.80, 1.20, n_strikes) * spot
        vols = [smile.vol(float(k), fwd) for k in strikes]
        chain = pd.DataFrame({'strike': strikes, 'implied_vol': vols})
        params, rmse = calibrate_heston(chain, spot, max(expiry, 1e-6), r, div_yield)
        obj = cls(spot, params, div_yield)
        obj.calibration_rmse = rmse
        return obj

    def simulate(self, time_grid: np.ndarray, n_paths: int,
                 ois_curve: OISCurve,
                 equity_normals: Optional[np.ndarray] = None,
                 seed: int = 42) -> np.ndarray:
        """
        Simulate spot paths under Heston QE and cache the variance paths.

        The QE scheme uses a flat risk-neutral rate; we anchor it to the curve
        zero rate over the horizon (per-step curve drift is a GBM-only
        feature). ``equity_normals`` is accepted for interface compatibility
        but the Heston scheme draws its own correlated (spot, vol) shocks.
        """
        from src.pricing.heston import simulate_heston_qe
        tg = np.asarray(time_grid, dtype=float)
        T = float(tg[-1])
        r = ois_curve.zero_rate(T) if T > 1e-6 else 0.0
        sim = simulate_heston_qe(self.spot, r, self.q, tg, self.params,
                                 n_paths=n_paths, seed=seed)
        self._variance = sim['variance']
        return sim['spot']

    def option_mtm_paths(self, spot_paths: np.ndarray, time_grid: np.ndarray,
                         ois_curve: OISCurve, strike: float, maturity: float,
                         units: float, call: bool = True,
                         smile: Optional[EquityVolSmile] = None) -> np.ndarray:
        """
        Reprice the option along each path at its own stochastic vol √v(t).

        Falls back to the GBM/smile repricing if no variance has been cached
        (i.e. :meth:`simulate` has not been called on this instance).
        """
        if self._variance is None:
            return super().option_mtm_paths(spot_paths, time_grid, ois_curve,
                                            strike, maturity, units, call, smile)
        n_paths, n_time = spot_paths.shape
        mtm = np.zeros((n_paths, n_time))
        for ti in range(n_time):
            t = time_grid[ti]
            tau = max(maturity - t, 0.0)
            r_t = ois_curve.zero_rate(maturity) if maturity > 1e-6 else 0.0
            S_t = spot_paths[:, ti]
            if tau <= 0:
                payoff = np.maximum(S_t - strike, 0.0) if call \
                    else np.maximum(strike - S_t, 0.0)
                mtm[:, ti] = units * payoff
            else:
                vol_t = np.sqrt(np.maximum(self._variance[:, ti], 1e-8))
                mtm[:, ti] = units * bsm_price(S_t, strike, tau, r_t, self.q,
                                               vol_t, call)
        return mtm
