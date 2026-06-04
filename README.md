# INR OTC Derivatives Risk & XVA Analytics Platform

An end-to-end quantitative finance platform for pricing, CCR risk management,
XVA computation, and regulatory capital on Indian OTC interest rate derivatives.
Built with free data sources only (RBI DBIE, FIMMDA, FBIL — all publicly available).

---

## What This Platform Does

This platform mirrors the actual workflow of a Rates / CCR / XVA desk at an Indian
or global bank: trade-level pricing → Monte Carlo exposure simulation → collateral
modelling → CVA/FVA/KVA/MVA calculation → SA-CCR regulatory capital → EOD batch
reporting → database persistence.

---

## Features

### Core Pricing & Curves
- **INR OIS Curve Bootstrapping** — Log-linear DF interpolation from FBIL MIBOR market rates
- **G-Sec Yield Curve** — Cubic spline on RBI auction yields; OIS-G-Sec spread calculation
- **Multi-Curve Framework** — OIS discounting + separate MIBOR projection curve (dual-curve)
- **INR IRS/OIS Swap Pricer** — MTM, par rate, DV01, PV01, Key Rate DV01, Gamma
- **SABR Vol Surface** — Hagan (2002) normal SABR for INR swaptions; full expiry × tenor grid

### Exposure Simulation
- **Hull-White 1F Monte Carlo** — Exact simulation, 10,000 paths, EE/EPE/PFE/ENE/EEPE
- **HW1F Calibration** — OLS regression on RBI DBIE MIBOR history; both `a` and `σ` are data-driven
- **Persistent Exposure Cube** — PyArrow/Parquet storage of path × time × trade NPVs; correct portfolio netting
- **CSA Collateral Engine** — Threshold, MTA, MPOR-aware; uncollateralised vs. CCP comparison

### XVA Engines
- **CVA / DVA** — Bootstrapped hazard rates from synthetic Indian CDS spreads; bilateral CVA
- **FVA (v1)** — EE/ENE profile-based FCA + FBA with Indian bank funding spreads
- **FVA (v2 — Pathwise)** — Path-by-path funding cost; eliminates Jensen's inequality bias
- **MVA** — ISDA SIMM v2.7 IM calculation; DIM profile; discounted MVA
- **KVA** — SA-CCR-based term structure; RBI Basel III risk weights and capital ratio
- **Wrong-Way Risk** — Regime-switching Cholesky model; WWR multiplier term structure

### Regulatory Capital
- **SA-CCR (Basel III)** — RC, supervisory duration, effective notional, netting set PFE
- **RWA & Capital Charge** — Per RBI Basel III guidelines; counterparty type risk weights

### Analytics & Reporting
- **PnL Attribution** — Carry, Roll-Down, Delta, Gamma, New Fixing, Unexplained decomposition
- **XVA Attribution** — Day-over-day CVA changes; spread, exposure, time-decay components
- **Stress Testing** — RBI rate shock (±100/200bps) × credit spread widening scenarios
- **Model Validation Suite** — 8 quantitative MRM tests: MC convergence, bootstrap repricing,
  antithetic VR, CVA analytical benchmark, no-arbitrage forwards, SA-CCR formulas, HW1F fit

### Infrastructure
- **EOD Batch Engine** — Full portfolio CVA/FVA/KVA/MVA/EAD per counterparty in one run
- **SQLite Persistence** — XVAResult, CurveSnapshot, MarketDataSnapshot via SQLAlchemy
- **FastAPI REST Layer** — `/price/swap`, `/risk/exposure`, `/risk/cva`, `/curves/ois`, `/curves/cds/{name}`
- **Streamlit Dashboard** — 14 pages across v1–v4 feature tiers

---

## Data Sources (All Free)

| Source | Data | URL |
|---|---|---|
| RBI DBIE | Overnight MIBOR history, G-Sec yields, policy rates | https://dbie.rbi.org.in |
| FIMMDA | OIS curve market rates, IRS benchmark rates | https://www.fimmda.org |
| FBIL | MIBOR overnight fixing, OIS benchmarks | https://www.fbil.org.in |
| CCIL | OIS/IRS volumes, settlement data | https://www.ccilindia.com |

**Note:** INR swaption implied vols and Indian CDS curves require Bloomberg/Refinitiv
and are not freely available. Accordingly:
- SABR ATM vol is anchored to realised MIBOR vol (RBI DBIE — free)
- CDS spreads are synthetic market-convention proxies (documented in `counterparties.csv`)
- HW1F calibration uses realised MIBOR vol, not swaption vols

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run full test suite
pytest tests/ -v

# Run EOD batch (generates comprehensive report)
python -m src.eod.risk_engine

# Launch dashboard
streamlit run app/streamlit_app.py

# Start REST API
uvicorn api.main:app --reload --port 8000
```

---

## Project Structure
XVA Engine/
├── src/
│   ├── calibration/      # HW1F calibration from MIBOR history
│   ├── curves/           # OIS/G-Sec bootstrapping, multi-curve, CDS bootstrapper
│   ├── data_ingestion/   # Market data (FIMMDA/DBIE anchored)
│   ├── exposure/         # Persistent Parquet exposure cube
│   ├── montecarlo/       # Hull-White 1F simulation + calibrate_hw1f()
│   ├── csa/              # CSA collateral engine
│   ├── portfolio/        # Netting engine, capital optimizer
│   ├── pricing/          # Swap pricer, swaption (Bachelier), SABR
│   ├── sa_ccr/           # SA-CCR EAD, RWA, capital
│   ├── stress/           # Rate shock + credit spread stress tests
│   ├── validation/       # Model validation suite (MRM)
│   ├── wwr/              # Wrong-way risk (regime-switching)
│   └── xva/              # CVA, DVA, FVA v1/v2, KVA, MVA, SIMM,
│                         # PnL attribution, XVA attribution
├── api/                  # FastAPI REST endpoints
├── app/                  # Streamlit dashboard (14 pages)
├── db/                   # SQLAlchemy models + SQLite
├── data/                 # Portfolio, counterparties, CSA master, exposure cube
├── reports/              # EOD batch output CSV
├── tests/                # Pytest suite (12 test files, 80+ tests)
└── requirements.txt

---

## Tech Stack

Python 3.10+ · NumPy · SciPy · Pandas · PyArrow · Plotly · Streamlit · FastAPI · SQLAlchemy · Pytest

---

## Resume Line

> Built an end-to-end INR OTC derivatives risk and XVA analytics platform. Implemented FBIL MIBOR OIS curve bootstrapping, SABR vol surface, Hull-White 1F Monte Carlo (calibrated from RBI DBIE MIBOR history via OLS), CSA-aware exposure simulation with a persistent Parquet exposure cube, and full XVA suite (CVA/DVA/FVA/MVA/KVA) including ISDA SIMM v2.7 IM, pathwise FVA, regime-switching wrong-way risk, Basel SA-CCR capital, and a PnL attribution engine. Includes an MRM model validation suite, FastAPI REST layer, SQLite persistence, and a 14-page Streamlit dashboard. All data from free Indian market sources (RBI DBIE, FIMMDA, FBIL).
