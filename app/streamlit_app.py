"""
INR OTC Derivatives Risk & XVA Analytics Dashboard.

A 5-page Streamlit application for interactive portfolio analytics:
    Page 1: Portfolio Summary — Trade inputs, MTM, XVA metrics
    Page 2: Exposure Analytics — EE/PFE profiles, collateral comparison
    Page 3: Credit Analytics — Hazard rates, survival, CVA, WWR
    Page 4: Capital Analytics — SA-CCR, RWA, KVA
    Page 5: Stress Testing — RBI rate shock & credit spread scenarios
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Project imports
from src.data_ingestion.market_data import (
    get_ois_market_data, get_gsec_market_data, get_counterparty_data,
    get_sample_portfolio, get_csa_scenarios, get_stress_scenarios,
    get_policy_rates, OIS_TENOR_LABELS
)
from src.curves.ois_curve import OISCurve, GSecCurve
from src.pricing.swap_pricer import SwapPricer, price_portfolio
from src.montecarlo.hull_white import HullWhite1F, run_exposure_simulation
from src.csa.collateral import CSAEngine, compare_csa_scenarios
from src.xva.cva import CVAEngine, CreditCurve
from src.xva.fva import FVAEngine
from src.xva.kva import KVAEngine
from src.sa_ccr.regulatory import SACCRCalculator, compute_rwa, compute_capital_requirement
from src.stress.stress_testing import stress_test_swap, run_full_stress_test

# V2 Imports
from src.data_ingestion.portfolio_manager import PortfolioManager
from src.portfolio.netting_engine import NettingEngine
from src.portfolio.capital_optimizer import CapitalOptimizer



# Page Configuration
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="INR XVA Analytics Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# Custom CSS for Premium Look
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@300;400;500;700&display=swap');

    /* Global font */
    html, body, [class*="css"] {
        font-family: 'Roboto Mono', monospace;
        color: #d2c6b2;
        text-transform: uppercase;
        font-size: 0.85rem;
    }

    /* Hide Streamlit Header */
    header[data-testid="stHeader"] {
        display: none !important;
    }

    /* Zero out block-container padding and leave room for footer */
    .block-container {
        padding-top: 0rem !important;
        padding-bottom: 40px !important;
        padding-left: 0.5rem !important;
        padding-right: 0.5rem !important;
        max-width: 100% !important;
    }

    /* Main background */
    .stApp {
        background: #111111;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: #13110E;
        border-right: 1px solid #352a1a;
    }
    
    /* Remove padding inside sidebar */
    section[data-testid="stSidebar"] .block-container {
        padding-top: 0rem !important;
    }

    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        background: none;
        color: #D2B48C !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        -webkit-text-fill-color: initial;
        margin-bottom: 0px !important;
    }

    /* DataFrames */
    .stDataFrame {
        border-radius: 0px;
        border: 1px solid #352a1a;
        overflow: hidden;
    }
    
    div[data-testid="stDataFrame"] th {
        background-color: #13110E !important;
        color: #D2B48C !important;
        font-weight: bold;
        text-transform: uppercase;
    }

    /* Buttons */
    .stButton > button {
        background: #13110E;
        color: #D2B48C;
        border: 1px solid #352a1a;
        border-radius: 0px;
        font-weight: 600;
        transition: none;
        text-transform: uppercase;
    }

    /* Expanders */
    .streamlit-expanderHeader {
        background: #111111;
        border: 1px solid #352a1a;
        border-radius: 0px;
        color: #D2B48C !important;
    }

    /* Selectbox / Input */
    input, select, textarea {
        border-radius: 0px !important;
        border: 1px solid #352a1a !important;
        background-color: #111111 !important;
        color: #FFFFFF !important;
    }

    /* Ensure Plotly charts have height and look like panels */
    .stPlotlyChart {
        min-height: 400px;
        border: 1px solid #352a1a;
        padding: 4px;
        background: #111111;
    }
    
    /* Footer sticky class */
    .bbg-footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #13110E;
        border-top: 1px solid #352a1a;
        color: #888888;
        font-size: 0.75rem;
        padding: 4px 16px;
        z-index: 99999;
        display: flex;
        justify-content: space-between;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Plotly theme for consistent dark styling
# ─────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    template='plotly_dark',
    paper_bgcolor='#111111',
    plot_bgcolor='#111111',
    font=dict(family='Roboto Mono, monospace', color='#d2c6b2', size=11),
    title_font=dict(size=12, color='#D2B48C'),
    legend=dict(
        bgcolor='#13110E',
        bordercolor='#352a1a',
        borderwidth=1,
        font=dict(size=10)
    ),
    margin=dict(l=40, r=20, t=30, b=30),
    xaxis=dict(
        gridcolor='#1c1611', 
        zerolinecolor='#352a1a',
        showline=True,
        linecolor='#352a1a',
        tickfont=dict(color='#888888')
    ),
    yaxis=dict(
        gridcolor='#1c1611', 
        zerolinecolor='#352a1a',
        showline=True,
        linecolor='#352a1a',
        tickfont=dict(color='#888888')
    )
)

COLORS = {
    'primary': '#00BFFF',     # Terminal Cyan
    'secondary': '#888888',   # Muted grey
    'accent': '#D2B48C',      # Tan Gold
    'success': '#00ff00',     # Terminal Green
    'warning': '#f5a623',     # Orange
    'danger': '#ff4d4d',      # Red
    'info': '#d2c6b2',        # Text color
}

# ─────────────────────────────────────────────────────────────
# Bloomberg Terminal UI Monkey Patch
# ─────────────────────────────────────────────────────────────
from datetime import datetime
import streamlit.delta_generator

def bbg_metric(*args, **kwargs):
    # Handle both method call (self, label, value) and wrapper call (label, value)
    if len(args) > 0 and hasattr(args[0], 'markdown'):
        self = args[0]
        args = args[1:]
    else:
        import streamlit as st
        self = st
        
    label = args[0] if len(args) > 0 else kwargs.get("label", "")
    value = args[1] if len(args) > 1 else kwargs.get("value", "")
    delta = args[2] if len(args) > 2 else kwargs.get("delta", None)
    
    delta_html = ""
    if delta:
        val_str = str(delta)
        color = "#00ff00" if val_str.startswith("+") or "↑" in val_str or "green" in val_str.lower() else "#ff4d4d" if val_str.startswith("-") or "↓" in val_str or "red" in val_str.lower() else "#D2B48C"
        delta_html = f"<span style='color: {color}; font-size: 0.9rem; margin-left: 8px;'>{delta}</span>"
    
    html = f"<div style='background: #111111; border: 1px solid #352a1a; padding: 12px 16px; margin-bottom: 8px; display: flex; flex-direction: column; justify-content: center;'><div style='color: #D2B48C; font-size: 0.65rem; font-weight: bold; text-transform: uppercase; margin-bottom: 4px;'>{label}</div><div style='display: flex; align-items: baseline;'><span style='color: #FFFFFF; font-size: 1.6rem; font-weight: bold; font-family: \"Roboto Mono\", monospace;'>{value}</span>{delta_html}</div></div>"
    self.markdown(html, unsafe_allow_html=True)

import streamlit as st
st.metric = bbg_metric
streamlit.delta_generator.DeltaGenerator.metric = bbg_metric

# ─────────────────────────────────────────────────────────────
# Render Custom Top Header Bar
# ─────────────────────────────────────────────────────────────
header_html = """
<div style='background: #13110E; border-bottom: 1px solid #352a1a; display: flex; align-items: center; padding: 8px 16px; margin-bottom: 16px; margin-top: 0px;'>
    <div style='color: #D2B48C; font-weight: 800; font-size: 1.1rem; letter-spacing: 1px; margin-right: 24px;'>TERMINAL.V1</div>
    <div style='flex-grow: 1; display: flex; align-items: center;'>
        <div style='background: #1A1815; border: 1px solid #352a1a; padding: 4px 12px; color: #888888; font-size: 0.8rem; width: 300px; display: flex; align-items: center;'>
            <span style='margin-right: 8px;'>🔍</span> CMD >
        </div>
    </div>
    <div style='display: flex; gap: 16px; color: #D2B48C; font-size: 1.1rem;'>
        <span>🔔</span>
        <span>⚙️</span>
        <span>⏻</span>
    </div>
</div>
"""
st.markdown(header_html, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Render Sticky Footer
# ─────────────────────────────────────────────────────────────
footer_html = f"""
<div class="bbg-footer">
    <div><span style='color: #00ff00; font-weight: bold;'>CONNECTION: SECURE</span> &nbsp;&nbsp;|&nbsp;&nbsp; NODE: TERMINAL_ALPHA_V1</div>
    <div>{datetime.now().strftime('%H:%M:%S UTC')} &nbsp;&nbsp;|&nbsp;&nbsp; © 2026 CAPITAL_ANALYTICS_GROUP</div>
</div>
"""
st.markdown(footer_html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Cached data loading and computation
# ─────────────────────────────────────────────────────────────
@st.cache_data
def run_v2_engines():
    """Run the V2 engines to compute portfolio metrics."""
    ois_data = get_ois_market_data()
    ois_curve = OISCurve(ois_data['tenor_years'].values, ois_data['ois_rate'].values)
    cptys_df = PortfolioManager.load_counterparties()
    portfolio_df = PortfolioManager.load_portfolio()
    trades = portfolio_df.to_dict('records')
    
    hw_model = HullWhite1F(ois_curve, a=0.10, sigma=0.01)
    time_grid, rate_paths = hw_model.simulate_rates(n_paths=2000, n_steps=60, horizon=10.0)
    
    netting = NettingEngine(time_grid, rate_paths, hw_model)
    trade_mtm_paths = netting.calculate_trade_mtm_paths(trades)
    
    current_mtms = {tid: paths[:, 0].mean() for tid, paths in trade_mtm_paths.items()}
    csa_mtm = netting.aggregate_by_csa(trades, trade_paths=trade_mtm_paths)
    csa_exposures = netting.apply_collateral(csa_mtm)
    portfolio_exposure = netting.aggregate_portfolio(csa_exposures)
    
    cap_opt = CapitalOptimizer(ois_curve, cptys_df)
    ranked_df = cap_opt.rank_portfolio(trades, current_mtms)
    
    return {
        'ois_curve': ois_curve, 'cptys_df': cptys_df, 'portfolio_df': portfolio_df,
        'trades': trades, 'time_grid': time_grid, 'current_mtms': current_mtms,
        'csa_exposures': csa_exposures, 'portfolio_exposure': portfolio_exposure,
        'ranked_df': ranked_df
    }

@st.cache_data
def load_market_data():
    """Load all market data."""
    ois_data = get_ois_market_data()
    gsec_data = get_gsec_market_data()
    counterparties = get_counterparty_data()
    portfolio = get_sample_portfolio()
    policy_rates = get_policy_rates()
    return ois_data, gsec_data, counterparties, portfolio, policy_rates


@st.cache_data
def build_curves():
    """Build OIS and G-Sec curves."""
    ois_data = get_ois_market_data()
    gsec_data = get_gsec_market_data()

    ois_curve = OISCurve(
        tenors=ois_data['tenor_years'].values,
        rates=ois_data['ois_rate'].values
    )
    gsec_curve = GSecCurve(
        tenors=gsec_data['tenor_years'].values,
        yields=gsec_data['yield_rate'].values
    )
    return ois_curve, gsec_curve


@st.cache_data
def run_simulation(_curve, notional, fixed_rate, maturity, direction,
                   n_paths, a, sigma):
    """Run Monte Carlo simulation (cached)."""
    return run_exposure_simulation(
        _curve, notional=notional, fixed_rate=fixed_rate,
        maturity=maturity, direction=direction,
        n_paths=n_paths, a=a, sigma=sigma
    )


# ─────────────────────────────────────────────────────────────
# Sidebar Navigation
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<div style='color: #D2B48C; font-size: 0.8rem; font-weight: bold; margin-bottom: 0px;'>ANALYTICS</div>", unsafe_allow_html=True)
    st.markdown("<div style='color: #00ff00; font-size: 0.65rem; font-weight: bold; margin-bottom: 16px;'>SYSTEM_ACTIVE</div>", unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        [
         "⊞ PORTFOLIO OVERVIEW",
         "⇆ TRADE MANAGEMENT",
         "🏦 CAPITAL OPTIMIZATION",
         "📄 SINGLE TRADE SUMMARY", 
         "📈 EXPOSURE ANALYTICS",
         "🛡️ CREDIT ANALYTICS", 
         "💰 CAPITAL ANALYTICS",
         "⚡ STRESS TESTING",
         "📁 V3 DATABASE STATUS",
         "📊 V3 XVA ATTRIBUTION",
         "📈 V3 EXOTICS & SWAPTIONS",
         "🌐 V4 MULTI-CURVE & SABR",
         "🧊 V4 EXPOSURE CUBE & MVA",
         "💸 V4 PNL ATTRIBUTION",
         "✅ V4 MODEL VALIDATION"
        ],
        label_visibility="collapsed"
    )


    st.markdown("### ⚙️ Single Trade Parameters (V1)")

    counterparties = get_counterparty_data()
    cpty_selected = st.selectbox(
        "Counterparty",
        counterparties['counterparty'].tolist(),
        index=0
    )

    notional = st.number_input("Notional (₹ Cr)", value=500.0,
                                min_value=10.0, max_value=10000.0, step=50.0)
    fixed_rate = st.number_input("Fixed Rate (%)", value=7.00,
                                  min_value=1.0, max_value=15.0, step=0.05) / 100
    maturity = st.slider("Maturity (Years)", min_value=1, max_value=10,
                          value=5)
    direction = st.selectbox("Direction",
                              ["Receive Fixed", "Pay Fixed"])

    st.markdown("---")
    st.markdown("### 🎛️ Model Parameters")
    n_paths = st.select_slider("MC Paths",
                                options=[1000, 2000, 5000, 10000],
                                value=5000)
    mean_rev = st.slider("Mean Reversion (a)", 0.01, 0.30, 0.10, 0.01)
    vol = st.slider("Volatility (σ)", 0.005, 0.025, 0.010, 0.001)

    st.markdown("---")
    st.caption("INR XVA Analytics Platform v1.0")
    st.caption("June 2026")


# ─────────────────────────────────────────────────────────────
# Build curves and run simulation
# ─────────────────────────────────────────────────────────────
ois_curve, gsec_curve = build_curves()

sim_result = run_simulation(
    ois_curve, notional, fixed_rate, float(maturity), direction,
    n_paths, mean_rev, vol
)

time_grid = sim_result['time_grid']
rate_paths = sim_result['rate_paths']
mtm_paths = sim_result['mtm_paths']
metrics = sim_result['metrics']

# Get counterparty data for selected counterparty
cpty_row = counterparties[counterparties['counterparty'] == cpty_selected].iloc[0]
credit_curve = CreditCurve(cpty_row['cds_spread_bps'], cpty_row['recovery_rate'])

# Swap pricer
swap = SwapPricer(notional, fixed_rate, float(maturity), direction)

# ─────────────────────────────────────────────────────────────
# PAGE 1: Single Trade Summary
# ─────────────────────────────────────────────────────────────
if page == "📄 SINGLE TRADE SUMMARY":
    st.markdown("# Single Trade Summary")
    st.markdown(f"**Counterparty:** {cpty_selected} · **Direction:** {direction} · "
                f"**Notional:** ₹{notional:.0f} Cr · **Maturity:** {maturity}Y · "
                f"**Fixed Rate:** {fixed_rate*100:.2f}%")

    # Compute XVA components
    mtm_value = swap.mtm(ois_curve)
    par_rate_val = swap.par_rate(ois_curve)
    dv01_val = swap.dv01(ois_curve)

    cva_engine = CVAEngine(ois_curve)
    cva_val = cva_engine.compute_cva(metrics['EE'], time_grid, credit_curve)

    own_curve = CreditCurve(40.0)
    dva_val = cva_engine.compute_dva(metrics['ENE'], time_grid, own_curve)

    fva_engine = FVAEngine(ois_curve, cpty_row['funding_spread_bps'])
    fva_result = fva_engine.compute_fva(metrics['EE'], metrics['ENE'], time_grid)

    kva_engine = KVAEngine(ois_curve)
    kva_result = kva_engine.compute_kva_from_exposure(
        metrics['EE'], time_grid, cpty_row['risk_weight']
    )

    xva_adjusted = mtm_value - cva_val + dva_val - abs(fva_result['FVA']) - kva_result['KVA']

    # ── Metric Cards ──
    st.markdown("### Key Metrics")
    cols = st.columns(4)
    cols[0].metric("Portfolio MTM", f"₹{mtm_value:.4f} Cr",
                    delta=f"Par: {par_rate_val*100:.2f}%")
    cols[1].metric("DV01", f"₹{dv01_val:.4f} Cr",
                    delta=f"PV01: {swap.pv01(ois_curve):.4f}")
    cols[2].metric("EPE", f"₹{metrics['EPE']:.4f} Cr")
    cols[3].metric("EEPE", f"₹{metrics['EEPE']:.4f} Cr")

    st.markdown("---")
    st.markdown("### XVA Components")
    cols2 = st.columns(5)
    cols2[0].metric("CVA", f"₹{cva_val:.4f} Cr")
    cols2[1].metric("DVA", f"₹{dva_val:.4f} Cr")
    cols2[2].metric("FVA", f"₹{fva_result['FVA']:.4f} Cr")
    cols2[3].metric("KVA", f"₹{kva_result['KVA']:.4f} Cr")
    cols2[4].metric("XVA-Adjusted MTM", f"₹{xva_adjusted:.4f} Cr",
                     delta=f"{(xva_adjusted - mtm_value):.4f} Cr adjustment")

    st.markdown("---")

    # ── Curve Charts ──
    st.markdown("### Yield Curves")
    col_curve1, col_curve2 = st.columns(2)

    with col_curve1:
        # OIS zero curve + forward curve
        plot_tenors = np.linspace(0.1, 10.0, 100)
        zero_rates = ois_curve.zero_rate_array(plot_tenors) * 100
        fwd_rates = np.array([ois_curve.instantaneous_forward(t) for t in plot_tenors]) * 100

        fig_curves = go.Figure()
        fig_curves.add_trace(go.Scatter(
            x=plot_tenors, y=zero_rates,
            name='OIS Zero Rate', mode='lines',
            line=dict(color=COLORS['primary'], width=2.5)
        ))
        fig_curves.add_trace(go.Scatter(
            x=plot_tenors, y=fwd_rates,
            name='Forward Rate', mode='lines',
            line=dict(color=COLORS['accent'], width=2, dash='dash')
        ))
        fig_curves.update_layout(
            title='INR OIS Zero & Forward Curves',
            xaxis_title='Tenor (Years)',
            yaxis_title='Rate (%)',
            **PLOTLY_LAYOUT
        )
        st.plotly_chart(fig_curves, use_container_width=True)

    with col_curve2:
        # Discount factor curve
        df_values = ois_curve.df_array(plot_tenors)

        fig_df = go.Figure()
        fig_df.add_trace(go.Scatter(
            x=plot_tenors, y=df_values,
            name='Discount Factor', mode='lines',
            line=dict(color=COLORS['success'], width=2.5),
            fill='tozeroy',
            fillcolor='rgba(52, 211, 153, 0.1)'
        ))
        fig_df.update_layout(
            title='OIS Discount Factor Curve',
            xaxis_title='Tenor (Years)',
            yaxis_title='Discount Factor',
            **PLOTLY_LAYOUT
        )
        st.plotly_chart(fig_df, use_container_width=True)

    # ── OIS vs G-Sec Spread ──
    st.markdown("### OIS vs G-Sec Sovereign Basis")
    spread_tenors = np.linspace(0.5, 10.0, 50)
    spreads = gsec_curve.spread_over_ois(ois_curve, spread_tenors) * 10000  # in bps

    fig_spread = go.Figure()
    fig_spread.add_trace(go.Bar(
        x=spread_tenors, y=spreads,
        name='G-Sec vs OIS Spread',
        marker=dict(
            color=spreads,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='bps')
        )
    ))
    fig_spread.update_layout(
        title='G-Sec vs OIS Zero Rate Spread (Sovereign Basis)',
        xaxis_title='Tenor (Years)',
        yaxis_title='Spread (bps)',
        **PLOTLY_LAYOUT
    )
    st.plotly_chart(fig_spread, use_container_width=True)

    # ── Cash Flow Schedule ──
    with st.expander("📋 Cash Flow Schedule"):
        cf_df = swap.cash_flow_schedule(ois_curve)
        st.dataframe(cf_df.style.format({
            'payment_date_years': '{:.2f}',
            'accrual_fraction': '{:.4f}',
            'fixed_cf_cr': '{:.4f}',
            'float_cf_cr': '{:.4f}',
            'net_cf_cr': '{:.4f}',
            'discount_factor': '{:.6f}',
            'pv_net_cf_cr': '{:.4f}',
        }), use_container_width=True)

    # ── Key Rate DV01 ──
    with st.expander("📊 Key Rate DV01 Ladder"):
        kr_dv01 = swap.key_rate_dv01(ois_curve)
        kr_df = pd.DataFrame({
            'Tenor': list(kr_dv01.keys()),
            'KR-DV01 (₹ Cr)': list(kr_dv01.values())
        })

        fig_kr = go.Figure()
        fig_kr.add_trace(go.Bar(
            x=kr_df['Tenor'], y=kr_df['KR-DV01 (₹ Cr)'],
            marker=dict(
                color=kr_df['KR-DV01 (₹ Cr)'],
                colorscale='RdYlGn',
                showscale=False
            )
        ))
        fig_kr.update_layout(
            title='Key Rate DV01 Ladder',
            xaxis_title='Tenor',
            yaxis_title='KR-DV01 (₹ Cr)',
            **PLOTLY_LAYOUT
        )
        st.plotly_chart(fig_kr, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# PAGE 2: Exposure Analytics
# ─────────────────────────────────────────────────────────────
elif page == "📈 EXPOSURE ANALYTICS":
    st.markdown("# Exposure Analytics")
    st.markdown(f"**{n_paths:,} Monte Carlo paths** · **{maturity}Y horizon** · "
                f"Hull-White 1F (a={mean_rev}, σ={vol})")

    # ── EE/PFE Profile ──
    st.markdown("### Expected Exposure & PFE Profile")

    fig_exposure = go.Figure()

    # PFE band
    fig_exposure.add_trace(go.Scatter(
        x=time_grid, y=metrics['PFE'],
        name='PFE 95%', mode='lines',
        line=dict(color=COLORS['danger'], width=2),
        fill='tonexty' if len(fig_exposure.data) > 0 else None,
    ))

    # EE
    fig_exposure.add_trace(go.Scatter(
        x=time_grid, y=metrics['EE'],
        name='Expected Exposure (EE)', mode='lines',
        line=dict(color=COLORS['primary'], width=2.5),
    ))

    # Fill between EE and PFE
    fig_exposure.add_trace(go.Scatter(
        x=np.concatenate([time_grid, time_grid[::-1]]),
        y=np.concatenate([metrics['PFE'], metrics['EE'][::-1]]),
        fill='toself',
        fillcolor='rgba(248, 113, 113, 0.1)',
        line=dict(color='rgba(0,0,0,0)'),
        showlegend=False,
        name='PFE band'
    ))

    # ENE
    fig_exposure.add_trace(go.Scatter(
        x=time_grid, y=metrics['ENE'],
        name='Expected Negative Exposure', mode='lines',
        line=dict(color=COLORS['warning'], width=1.5, dash='dot'),
    ))

    fig_exposure.update_layout(
        title='Exposure Profile Over Time',
        xaxis_title='Time (Years)',
        yaxis_title='Exposure (₹ Cr)',
        **PLOTLY_LAYOUT,
        hovermode='x unified'
    )
    st.plotly_chart(fig_exposure, use_container_width=True)

    # ── Rate Paths Fan Chart ──
    col_fan, col_hist = st.columns(2)

    with col_fan:
        st.markdown("### Simulated Rate Paths")
        n_show = min(200, n_paths)
        fig_fan = go.Figure()

        for i in range(n_show):
            fig_fan.add_trace(go.Scatter(
                x=time_grid, y=rate_paths[i] * 100,
                mode='lines',
                line=dict(color=COLORS['primary'], width=0.3),
                opacity=0.15,
                showlegend=False,
            ))

        # Mean path
        mean_rate = np.mean(rate_paths, axis=0) * 100
        fig_fan.add_trace(go.Scatter(
            x=time_grid, y=mean_rate,
            name='Mean Rate', mode='lines',
            line=dict(color=COLORS['warning'], width=2.5),
        ))

        fig_fan.update_layout(
            title=f'Simulated Short Rate Paths ({n_show} shown)',
            xaxis_title='Time (Years)',
            yaxis_title='Short Rate (%)',
            **PLOTLY_LAYOUT
        )
        st.plotly_chart(fig_fan, use_container_width=True)

    with col_hist:
        st.markdown("### Exposure Distribution")
        horizon_step = st.slider("Select Horizon (step index)",
                                  1, len(time_grid)-1,
                                  len(time_grid)//2,
                                  key='hist_horizon')

        mtm_at_t = mtm_paths[:, horizon_step]

        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=mtm_at_t,
            nbinsx=50,
            name=f'MTM at t={time_grid[horizon_step]:.1f}Y',
            marker=dict(
                color=COLORS['primary'],
                line=dict(color=COLORS['secondary'], width=0.5)
            ),
            opacity=0.8
        ))

        # Add EE and PFE vertical lines
        ee_val = metrics['EE'][horizon_step]
        pfe_val = metrics['PFE'][horizon_step]

        fig_hist.add_vline(x=ee_val, line_color=COLORS['success'],
                           line_dash='dash', annotation_text=f'EE={ee_val:.2f}')
        fig_hist.add_vline(x=pfe_val, line_color=COLORS['danger'],
                           line_dash='dash', annotation_text=f'PFE95={pfe_val:.2f}')
        fig_hist.add_vline(x=0, line_color='white', line_width=1)

        fig_hist.update_layout(
            title=f'MTM Distribution at t={time_grid[horizon_step]:.1f}Y',
            xaxis_title='MTM (₹ Cr)',
            yaxis_title='Frequency',
            **PLOTLY_LAYOUT
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # ── CSA Comparison ──
    st.markdown("### Collateralised vs Uncollateralised Exposure")

    csa_scenarios = get_csa_scenarios()
    csa_results = compare_csa_scenarios(mtm_paths, time_grid, csa_scenarios)

    fig_csa = go.Figure()
    csa_colors = [COLORS['danger'], COLORS['warning'],
                  COLORS['primary'], COLORS['success']]

    for idx, (name, res) in enumerate(csa_results.items()):
        fig_csa.add_trace(go.Scatter(
            x=time_grid, y=res['EE'],
            name=f'{name} (EPE={res["EPE"]:.3f})',
            mode='lines',
            line=dict(color=csa_colors[idx % len(csa_colors)], width=2)
        ))

    fig_csa.update_layout(
        title='Expected Exposure Under Different CSA Scenarios',
        xaxis_title='Time (Years)',
        yaxis_title='Expected Exposure (₹ Cr)',
        **PLOTLY_LAYOUT,
        hovermode='x unified'
    )
    st.plotly_chart(fig_csa, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# PAGE 3: Credit Analytics
# ─────────────────────────────────────────────────────────────
elif page == "🛡️ CREDIT ANALYTICS":
    st.markdown("# Credit Analytics")

    # ── Counterparty Credit Table ──
    st.markdown("### Counterparty Credit Summary")

    cpty_data = get_counterparty_data()

    # Add derived fields
    cpty_display = cpty_data.copy()
    cpty_display['hazard_rate_pct'] = (cpty_display['cds_spread_bps'] / 10000 /
                                        (1 - cpty_display['recovery_rate'])) * 100
    cpty_display['survival_5y'] = np.exp(
        -(cpty_display['cds_spread_bps'] / 10000 /
          (1 - cpty_display['recovery_rate'])) * 5.0
    ) * 100

    st.dataframe(
        cpty_display[['counterparty', 'rating', 'cds_spread_bps',
                       'recovery_rate', 'hazard_rate_pct', 'survival_5y']].style.format({
            'cds_spread_bps': '{:.0f}',
            'recovery_rate': '{:.0%}',
            'hazard_rate_pct': '{:.2f}%',
            'survival_5y': '{:.1f}%',
        }),
        use_container_width=True
    )

    # ── Survival Probability Curves ──
    col_surv, col_hz = st.columns(2)

    with col_surv:
        st.markdown("### Survival Probability Curves")
        t_plot = np.linspace(0, 10, 100)

        fig_surv = go.Figure()
        for i, row in cpty_data.iterrows():
            cc = CreditCurve(row['cds_spread_bps'], row['recovery_rate'])
            sp = cc.survival_probability_array(t_plot) * 100
            fig_surv.add_trace(go.Scatter(
                x=t_plot, y=sp,
                name=f"{row['counterparty']} ({row['cds_spread_bps']}bps)",
                mode='lines',
                line=dict(width=2)
            ))

        fig_surv.update_layout(
            title='Counterparty Survival Probabilities',
            xaxis_title='Time (Years)',
            yaxis_title='Survival Probability (%)',
            **PLOTLY_LAYOUT,
            hovermode='x unified'
        )
        st.plotly_chart(fig_surv, use_container_width=True)

    with col_hz:
        st.markdown("### Hazard Rate Comparison")
        fig_hz = go.Figure()
        fig_hz.add_trace(go.Bar(
            x=cpty_data['counterparty'],
            y=cpty_display['hazard_rate_pct'],
            marker=dict(
                color=cpty_data['cds_spread_bps'],
                colorscale='Reds',
                showscale=True,
                colorbar=dict(title='CDS (bps)')
            ),
            text=[f'{h:.2f}%' for h in cpty_display['hazard_rate_pct']],
            textposition='outside'
        ))
        fig_hz.update_layout(
            title='Implied Hazard Rates by Counterparty',
            xaxis_title='Counterparty',
            yaxis_title='Hazard Rate (%)',
            **PLOTLY_LAYOUT
        )
        st.plotly_chart(fig_hz, use_container_width=True)

    # ── CVA Sensitivity ──
    st.markdown("### CVA Sensitivity to CDS Spread")

    cva_engine = CVAEngine(ois_curve)
    spread_range = np.arange(10, 800, 10)
    cva_values = []
    for s in spread_range:
        cc = CreditCurve(s, cpty_row['recovery_rate'])
        cva_values.append(cva_engine.compute_cva(metrics['EE'], time_grid, cc))

    fig_cva_sens = go.Figure()
    fig_cva_sens.add_trace(go.Scatter(
        x=spread_range, y=cva_values,
        name='CVA', mode='lines',
        line=dict(color=COLORS['danger'], width=2.5),
        fill='tozeroy',
        fillcolor='rgba(248, 113, 113, 0.1)'
    ))

    # Mark current spread
    current_cva = cva_engine.compute_cva(metrics['EE'], time_grid, credit_curve)
    fig_cva_sens.add_trace(go.Scatter(
        x=[cpty_row['cds_spread_bps']], y=[current_cva],
        name=f'{cpty_selected} ({cpty_row["cds_spread_bps"]}bps)',
        mode='markers',
        marker=dict(color=COLORS['warning'], size=12, symbol='diamond')
    ))

    fig_cva_sens.update_layout(
        title=f'CVA vs CDS Spread — {cpty_selected}',
        xaxis_title='CDS Spread (bps)',
        yaxis_title='CVA (₹ Cr)',
        **PLOTLY_LAYOUT
    )
    st.plotly_chart(fig_cva_sens, use_container_width=True)

    # ── WWR vs Standard CVA ──
    st.markdown("### Wrong-Way Risk: Standard CVA vs WWR-CVA")

    wwr_correlations = [-0.30, 0.0, 0.30, 0.50, 0.70]
    wwr_cva_values = []

    for rho in wwr_correlations:
        # Approximate WWR multiplier
        if rho == 0:
            mult = 1.0
        elif rho > 0:
            mult = 1.0 + rho * 0.8  # Simplified multiplier
        else:
            mult = 1.0 + rho * 0.5

        wwr_cva_values.append(current_cva * mult)

    fig_wwr = go.Figure()
    colors_wwr = ['#34d399', '#818cf8', '#fbbf24', '#fb923c', '#f87171']
    fig_wwr.add_trace(go.Bar(
        x=[f'ρ={r:.2f}' for r in wwr_correlations],
        y=wwr_cva_values,
        marker=dict(color=colors_wwr),
        text=[f'₹{v:.4f}' for v in wwr_cva_values],
        textposition='outside',
    ))

    fig_wwr.add_hline(y=current_cva, line_dash='dash',
                       line_color='white', opacity=0.5,
                       annotation_text='Independent CVA')

    fig_wwr.update_layout(
        title='CVA Under Different Rate-Credit Correlations',
        xaxis_title='Correlation (ρ)',
        yaxis_title='CVA (₹ Cr)',
        **PLOTLY_LAYOUT
    )
    st.plotly_chart(fig_wwr, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# PAGE 4: Capital Analytics
# ─────────────────────────────────────────────────────────────
elif page == "💰 CAPITAL ANALYTICS":
    st.markdown("# Capital Analytics")

    # ── SA-CCR Computation ──
    st.markdown("### SA-CCR — Exposure at Default")

    calculator = SACCRCalculator()
    trade_addon = calculator.compute_trade_addon(
        notional=notional, maturity=float(maturity),
        direction=direction
    )

    mtm_val = swap.mtm(ois_curve)
    rc = max(mtm_val, 0.0)
    ead = 1.4 * (rc + trade_addon['trade_addon'])
    rw = cpty_row['risk_weight']
    rwa = ead * rw
    capital_req = rwa * 0.105

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Replacement Cost", f"₹{rc:.4f} Cr")
    col2.metric("PFE Add-On", f"₹{trade_addon['trade_addon']:.4f} Cr")
    col3.metric("EAD (α=1.4)", f"₹{ead:.4f} Cr")
    col4.metric("Capital Requirement", f"₹{capital_req:.4f} Cr")

    st.markdown("---")

    # ── SA-CCR Breakdown ──
    col_saccr1, col_saccr2 = st.columns(2)

    with col_saccr1:
        st.markdown("### SA-CCR Component Breakdown")
        fig_saccr = go.Figure(go.Waterfall(
            x=['MTM', 'RC', 'Adj. Notional', 'Sup. Duration',
               'PFE Add-On', 'α × (RC+PFE)', 'RWA', 'Capital'],
            y=[mtm_val, rc, trade_addon['adjusted_notional'],
               trade_addon['supervisory_duration'],
               trade_addon['trade_addon'], ead, rwa, capital_req],
            text=[f'₹{v:.3f}' for v in [mtm_val, rc,
                  trade_addon['adjusted_notional'],
                  trade_addon['supervisory_duration'],
                  trade_addon['trade_addon'], ead, rwa, capital_req]],
            textposition='outside',
            connector=dict(line=dict(color='rgba(99,102,241,0.3)')),
            increasing=dict(marker=dict(color=COLORS['primary'])),
            decreasing=dict(marker=dict(color=COLORS['danger'])),
            totals=dict(marker=dict(color=COLORS['success']))
        ))
        fig_saccr.update_layout(
            title='SA-CCR Waterfall',
            yaxis_title='₹ Crores',
            **PLOTLY_LAYOUT
        )
        st.plotly_chart(fig_saccr, use_container_width=True)

    with col_saccr2:
        st.markdown("### RWA by Counterparty Type")
        rw_data = get_counterparty_data()
        fig_rw = go.Figure()
        fig_rw.add_trace(go.Bar(
            x=rw_data['counterparty'],
            y=rw_data['risk_weight'] * 100,
            marker=dict(
                color=rw_data['risk_weight'],
                colorscale='YlOrRd',
                showscale=True,
                colorbar=dict(title='RW')
            ),
            text=[f'{w:.0f}%' for w in rw_data['risk_weight'] * 100],
            textposition='outside',
        ))
        fig_rw.update_layout(
            title='RBI Basel III Risk Weights',
            xaxis_title='Counterparty',
            yaxis_title='Risk Weight (%)',
            **PLOTLY_LAYOUT
        )
        st.plotly_chart(fig_rw, use_container_width=True)

    # ── KVA Profile ──
    st.markdown("### KVA — Capital Valuation Adjustment")

    kva_engine = KVAEngine(ois_curve)
    kva_result = kva_engine.compute_kva_from_exposure(
        metrics['EE'], time_grid, cpty_row['risk_weight']
    )

    col_kva1, col_kva2, col_kva3 = st.columns(3)
    col_kva1.metric("KVA", f"₹{kva_result['KVA']:.4f} Cr")
    col_kva2.metric("Peak Capital", f"₹{kva_result['peak_capital_cr']:.4f} Cr")
    col_kva3.metric("Avg Capital", f"₹{kva_result['avg_capital_cr']:.4f} Cr")

    fig_kva = go.Figure()
    fig_kva.add_trace(go.Scatter(
        x=time_grid, y=kva_result['capital_profile'],
        name='Capital Requirement', mode='lines',
        line=dict(color=COLORS['accent'], width=2.5),
        fill='tozeroy',
        fillcolor='rgba(192, 132, 252, 0.15)'
    ))
    fig_kva.add_trace(go.Scatter(
        x=time_grid, y=kva_result['ead_profile'],
        name='EAD Profile', mode='lines',
        line=dict(color=COLORS['primary'], width=2, dash='dash'),
    ))
    fig_kva.update_layout(
        title=f'Capital & EAD Profile Over Time (RW={rw*100:.0f}%)',
        xaxis_title='Time (Years)',
        yaxis_title='₹ Crores',
        **PLOTLY_LAYOUT,
        hovermode='x unified'
    )
    st.plotly_chart(fig_kva, use_container_width=True)

    # ── SA-CCR Parameters Table ──
    with st.expander("📋 SA-CCR Trade Parameters"):
        saccr_df = pd.DataFrame({
            'Parameter': ['Delta (δ)', 'Supervisory Duration',
                          'Adjusted Notional', 'Effective Notional',
                          'Maturity Factor', 'Maturity Bucket',
                          'Supervisory Factor', 'Trade Add-On'],
            'Value': [
                f"{trade_addon['delta']:+.0f}",
                f"{trade_addon['supervisory_duration']:.4f}",
                f"₹{trade_addon['adjusted_notional']:.2f} Cr",
                f"₹{trade_addon['effective_notional']:.2f} Cr",
                f"{trade_addon['maturity_factor']:.4f}",
                trade_addon['maturity_bucket'].upper(),
                f"{trade_addon['supervisory_factor']:.4%}",
                f"₹{trade_addon['trade_addon']:.4f} Cr",
            ]
        })
        st.dataframe(saccr_df, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────
# PAGE 5: Stress Testing
# ─────────────────────────────────────────────────────────────
elif page == "⚡ STRESS TESTING":
    st.markdown("# Stress Testing")

    # ── Interactive Sliders ──
    st.markdown("### RBI Rate Shock Scenario Builder")

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        rate_shock = st.slider("Rate Shock (bps)",
                                min_value=-200, max_value=300,
                                value=0, step=25,
                                help="Parallel shift to OIS curve")
    with col_s2:
        credit_shock = st.slider("Credit Spread Shock (bps)",
                                  min_value=-50, max_value=500,
                                  value=0, step=25,
                                  help="Parallel shift to all CDS spreads")

    # Apply shocks
    shocked_curve = ois_curve.shift(rate_shock)
    shocked_mtm = swap.mtm(shocked_curve)
    base_mtm = swap.mtm(ois_curve)

    shocked_cds = max(cpty_row['cds_spread_bps'] + credit_shock, 1.0)
    shocked_credit = CreditCurve(shocked_cds, cpty_row['recovery_rate'])
    cva_engine = CVAEngine(ois_curve)
    shocked_cva = cva_engine.compute_cva(metrics['EE'], time_grid, shocked_credit)
    base_cva = cva_engine.compute_cva(metrics['EE'], time_grid, credit_curve)

    st.markdown("### Scenario Impact")
    cols = st.columns(4)
    cols[0].metric("MTM",
                    f"₹{shocked_mtm:.4f} Cr",
                    delta=f"{shocked_mtm - base_mtm:.4f} Cr")
    cols[1].metric("CVA",
                    f"₹{shocked_cva:.4f} Cr",
                    delta=f"{shocked_cva - base_cva:.4f} Cr")
    cols[2].metric("CDS Spread",
                    f"{shocked_cds:.0f} bps",
                    delta=f"{credit_shock:+.0f} bps")
    cols[3].metric("Survival (5Y)",
                    f"{shocked_credit.survival_probability(5)*100:.1f}%",
                    delta=f"{(shocked_credit.survival_probability(5) - credit_curve.survival_probability(5))*100:.2f}%")

    st.markdown("---")

    # ── Predefined Stress Scenarios ──
    st.markdown("### Predefined Stress Scenario Matrix")
    stress_scenarios = get_stress_scenarios()

    stress_results = []
    for _, scenario in stress_scenarios.iterrows():
        sc_curve = ois_curve.shift(scenario['rate_shock_bps'])
        sc_mtm = swap.mtm(sc_curve)
        sc_cds = max(cpty_row['cds_spread_bps'] + scenario['credit_spread_shock_bps'], 1.0)
        sc_credit = CreditCurve(sc_cds, cpty_row['recovery_rate'])
        sc_cva = cva_engine.compute_cva(metrics['EE'], time_grid, sc_credit)

        stress_results.append({
            'Scenario': scenario['scenario'],
            'Description': scenario['description'],
            'Rate Shock': f"{scenario['rate_shock_bps']:+d} bps",
            'Credit Shock': f"{scenario['credit_spread_shock_bps']:+d} bps",
            'MTM (₹ Cr)': f"{sc_mtm:.4f}",
            'MTM Change': f"{sc_mtm - base_mtm:.4f}",
            'CVA (₹ Cr)': f"{sc_cva:.4f}",
            'CVA Change': f"{sc_cva - base_cva:.4f}",
        })

    stress_df = pd.DataFrame(stress_results)
    st.dataframe(stress_df, use_container_width=True, hide_index=True)

    # ── Stress Test Charts ──
    col_st1, col_st2 = st.columns(2)

    with col_st1:
        st.markdown("### MTM Sensitivity to Rate Shocks")
        rate_range = np.arange(-200, 350, 25)
        mtm_sensitivity = []
        for r in rate_range:
            sc = ois_curve.shift(r)
            mtm_sensitivity.append(swap.mtm(sc))

        fig_mtm_stress = go.Figure()
        fig_mtm_stress.add_trace(go.Scatter(
            x=rate_range, y=mtm_sensitivity,
            mode='lines+markers',
            line=dict(color=COLORS['primary'], width=2.5),
            marker=dict(size=4),
            name='MTM'
        ))
        fig_mtm_stress.add_vline(x=0, line_dash='dash', line_color='white',
                                  opacity=0.3)
        fig_mtm_stress.add_hline(y=0, line_dash='dash', line_color='white',
                                  opacity=0.3)
        fig_mtm_stress.add_vline(x=rate_shock, line_dash='dash', line_color=COLORS['danger'],
                                  annotation_text=f'Your Shock ({rate_shock} bps)')
        fig_mtm_stress.update_layout(
            title='MTM vs Parallel Rate Shock',
            xaxis_title='Rate Shock (bps)',
            yaxis_title='MTM (₹ Cr)',
            **PLOTLY_LAYOUT
        )
        st.plotly_chart(fig_mtm_stress, use_container_width=True)

    with col_st2:
        st.markdown("### CVA Sensitivity to Credit Spread Shock")
        credit_range = np.arange(0, 800, 25)
        cva_sensitivity = []
        for c in credit_range:
            shocked_cc = CreditCurve(max(cpty_row['cds_spread_bps'] + c, 1), cpty_row['recovery_rate'])
            cva_sensitivity.append(cva_engine.compute_cva(
                metrics['EE'], time_grid, shocked_cc
            ))

        fig_cva_stress = go.Figure()
        fig_cva_stress.add_trace(go.Scatter(
            x=credit_range + cpty_row['cds_spread_bps'],
            y=cva_sensitivity,
            mode='lines+markers',
            line=dict(color=COLORS['danger'], width=2.5),
            marker=dict(size=4),
            name='CVA'
        ))
        fig_cva_stress.add_vline(x=cpty_row['cds_spread_bps'],
                                  line_dash='dash', line_color='white', opacity=0.4,
                                  annotation_text='Base')
        fig_cva_stress.add_vline(x=shocked_cds,
                                  line_dash='dash', line_color=COLORS['danger'],
                                  annotation_text=f'Your Shock (+{credit_shock} bps)')
        fig_cva_stress.update_layout(
            title=f'CVA vs CDS Spread — {cpty_selected}',
            xaxis_title='CDS Spread (bps)',
            yaxis_title='CVA (₹ Cr)',
            **PLOTLY_LAYOUT
        )
        st.plotly_chart(fig_cva_stress, use_container_width=True)

    # ── Shocked Curve Comparison ──
    st.markdown("### Shocked vs Base Yield Curve")
    plot_tenors = np.linspace(0.1, 10.0, 100)

    fig_shocked = go.Figure()
    fig_shocked.add_trace(go.Scatter(
        x=plot_tenors,
        y=ois_curve.zero_rate_array(plot_tenors) * 100,
        name='Base OIS Curve', mode='lines',
        line=dict(color=COLORS['primary'], width=2.5),
    ))

    fig_shocked.add_trace(go.Scatter(
        x=plot_tenors,
        y=shocked_curve.zero_rate_array(plot_tenors) * 100,
        name=f'Shocked Curve ({rate_shock:+} bps)', mode='lines',
        line=dict(color=COLORS['danger'], width=2.5, dash='dash'),
    ))

    fig_shocked.update_layout(
        title='OIS Zero Curve Under Rate Shock Scenarios',
        xaxis_title='Tenor (Years)',
        yaxis_title='Zero Rate (%)',
        **PLOTLY_LAYOUT,
        hovermode='x unified'
    )
    st.plotly_chart(fig_shocked, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# V2 PAGES
# ─────────────────────────────────────────────────────────────
elif page == "⊞ PORTFOLIO OVERVIEW":
    res = run_v2_engines()
    st.title("Portfolio Overview (V2 Netting Engine)")
    
    if not res['portfolio_exposure']:
        st.info("Portfolio is empty. Go to Trade Management to add trades.")
    else:
        total_mtm = sum(res['current_mtms'].values())
        total_ee = res['portfolio_exposure']['EE']
        total_pfe = res['portfolio_exposure']['PFE']
        
        cols = st.columns(3)
        cols[0].metric("Total Portfolio MTM", f"₹ {total_mtm:.2f} Cr")
        cols[1].metric("Max Expected Exposure", f"₹ {np.max(total_ee):.2f} Cr")
        cols[2].metric("Max Potential Future Exposure", f"₹ {np.max(total_pfe):.2f} Cr")
        
        st.markdown("### Aggregated Portfolio Exposure Profile")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=res['time_grid'], y=total_ee, name='Expected Exposure', line=dict(color='#818cf8', width=2)))
        fig.add_trace(go.Scatter(x=res['time_grid'], y=total_pfe, name='PFE 95%', line=dict(color='#f87171', width=2)))
        fig.update_layout(title="Netted Portfolio Exposure (₹ Cr)", xaxis_title="Years", yaxis_title="Exposure", **PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

elif page == "⇆ TRADE MANAGEMENT":
    res = run_v2_engines()
    st.title("Trade Management")
    
    st.markdown("### Current Portfolio")
    st.dataframe(res['portfolio_df'], use_container_width=True)
    
    st.markdown("### Add New Trade")
    with st.form("add_trade_form"):
        col1, col2, col3 = st.columns(3)
        trade_type = col1.selectbox("Type", ["IRS", "OIS"])
        cpty = col2.selectbox("Counterparty", res['cptys_df']['Counterparty'].tolist())
        csa = col3.selectbox("CSA_ID", ["CSA_HDFC_01", "CSA_SBI_01", "CSA_KOTAK_01", "CSA_RELIANCE_01", "CSA_NBFC_01"])
        
        col4, col5, col6, col7 = st.columns(4)
        notional = col4.number_input("Notional (₹ Cr)", 10.0, 10000.0, 500.0)
        start_date = col5.date_input("Start Date")
        maturity = col6.number_input("Maturity (Y)", 1.0, 30.0, 5.0)
        fixed_rate = col7.number_input("Fixed Rate (%)", 1.0, 15.0, 7.0)
        
        direction = st.selectbox("Direction", ["Receive Fixed", "Pay Fixed"])
        
        submitted = st.form_submit_button("Add Trade")
        if submitted:
            new_trade = {
                'TradeType': trade_type,
                'Counterparty': cpty,
                'Notional': notional,
                'StartDate': start_date.strftime("%Y-%m-%d"),
                'Maturity': maturity,
                'FixedRate': fixed_rate,
                'Direction': direction,
                'CSA_ID': csa
            }
            PortfolioManager.add_trade(new_trade)
            run_v2_engines.clear()
            st.success("Trade Added Successfully! Refresh to see updates.")

elif page == "🏦 CAPITAL OPTIMIZATION":
    res = run_v2_engines()
    st.title("Capital Optimization & Ranking")
    
    st.markdown("### Trade Capital Efficiency (RoC)")
    df = res['ranked_df']
    if not df.empty:
        st.dataframe(df.style.format({
            'MTM': '₹ {:.2f}', 'EAD': '₹ {:.2f}', 'RWA': '₹ {:.2f}', 
            'Capital': '₹ {:.2f}', 'KVA': '₹ {:.2f}', 'Revenue': '₹ {:.2f}', 
            'RoC': '{:.2%}'
        }), use_container_width=True)
    else:
        st.info("No trades to optimize.")

# ─────────────────────────────────────────────────────────────
# PAGE: V3 Modules
# ─────────────────────────────────────────────────────────────
elif page == "📁 V3 DATABASE STATUS":
    st.markdown("## Database Layer")
    st.markdown("This database layer tracks the End-Of-Day (EOD) risk snapshots, manages trade portfolios, and stores counterparty reference data. It is currently powered by SQLite for local deployment.")
    # Show real counts from the actual CSV files
    from src.data_ingestion.portfolio_manager import PortfolioManager
    port_df = PortfolioManager.load_portfolio()
    cpty_df = PortfolioManager.load_counterparties()
    csa_df = PortfolioManager.load_csas()

    col1, col2, col3 = st.columns(3)
    col1.metric("Trades", len(port_df))
    col2.metric("Counterparties", len(cpty_df))
    col3.metric("CSA Agreements", len(csa_df))

    st.markdown("### Trade Book")
    st.dataframe(port_df, use_container_width=True)
    st.markdown("### Counterparty Master")
    st.dataframe(cpty_df, use_container_width=True)


elif page == "📊 V3 XVA ATTRIBUTION":
    st.markdown("## Daily XVA Attribution")
    st.markdown("Explains day-over-day changes in CVA using a first-order "
                "Taylor expansion decomposition.")

    from src.xva.attribution import XVAAttribution
    from src.xva.cva import CVAEngine, CreditCurve
    from src.montecarlo.hull_white import run_exposure_simulation

    ois_data, _, _, _, _ = load_market_data()
    ois_curve_attr = OISCurve(ois_data['tenor_years'].values, ois_data['ois_rate'].values)

    st.markdown("### Attribution Inputs")
    col1, col2 = st.columns(2)
    cva_yesterday = col1.number_input("CVA Yesterday (₹ Cr)", value=4.20, step=0.1)
    cva_today = col2.number_input("CVA Today (₹ Cr)", value=4.65, step=0.1)
    spread_move = col1.number_input("CDS Spread Move (bps)", value=8.0, step=1.0)
    exposure_chg = col2.number_input("Exposure Change (%)", value=2.5, step=0.5) / 100.0

    attribution = XVAAttribution.explain_cva_change(
        cva_yesterday=cva_yesterday,
        cva_today=cva_today,
        spread_move_bps=spread_move,
        exposure_change_pct=exposure_chg,
    )

    st.markdown("### Attribution Breakdown")
    cols = st.columns(5)
    labels = ['Total Change', 'Spread Move', 'Exposure Move', 'Time Decay', 'Unexplained']
    for col, label in zip(cols, labels):
        val = attribution[label]
        col.metric(label, f"₹{val:+.3f} Cr")

    import plotly.graph_objects as pgo
    fig_attr = pgo.Figure(pgo.Waterfall(
        name="CVA Attribution",
        orientation="v",
        measure=["relative", "relative", "relative", "relative", "total"],
        x=["Spread Move", "Exposure Move", "Time Decay", "Unexplained", "Total Change"],
        y=[attribution["Spread Move"], attribution["Exposure Move"],
           attribution["Time Decay"], attribution["Unexplained"],
           attribution["Total Change"]],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
    ))
    fig_attr.update_layout(title="CVA Waterfall Attribution", template="plotly_dark",
                           height=400)
    st.plotly_chart(fig_attr, use_container_width=True)

elif page == "📈 V3 EXOTICS & SWAPTIONS":
    st.markdown("## European Swaption Pricer")
    st.markdown("Priced using the **Bachelier (Normal) model** — the market standard "
                "for INR swaptions, valid for near-zero and negative rate environments.")

    from src.pricing.swaption import EuropeanSwaption

    ois_data, _, _, _, _ = load_market_data()
    ois_curve_sw = OISCurve(ois_data['tenor_years'].values, ois_data['ois_rate'].values)

    col1, col2, col3 = st.columns(3)
    sw_notional = col1.number_input("Notional (₹ Cr)", value=500.0, step=50.0)
    sw_expiry = col2.slider("Option Expiry (Years)", 1, 5, 2)
    sw_tenor = col3.slider("Swap Tenor (Years)", 1, 10, 5)
    sw_strike = col1.number_input("Strike Rate (%)", value=7.0, step=0.1) / 100.0
    sw_vol = col2.number_input("Normal Vol (bps)", value=55.0, step=5.0) / 10000.0
    sw_type = col3.selectbox("Option Type", ["Payer", "Receiver"])

    # Forward swap rate from OIS curve
    pricer_underlying = SwapPricer(
        notional=sw_notional, fixed_rate=sw_strike,
        maturity=float(sw_expiry + sw_tenor), direction="Receive Fixed"
    )
    fwd_rate = pricer_underlying.par_rate(ois_curve_sw)
    annuity = pricer_underlying.annuity(ois_curve_sw)

    sw = EuropeanSwaption(
        notional=sw_notional,
        strike=sw_strike,
        maturity=float(sw_expiry),
        swap_tenor=float(sw_tenor),
    )

    # For receiver swaption: flip the sign of (F - K) per put-call parity
    if sw_type == "Receiver":
        from scipy.stats import norm
        import numpy as np
        d = (sw.strike - fwd_rate) / (sw_vol * np.sqrt(sw.maturity))
        pv = annuity * ((sw.strike - fwd_rate) * norm.cdf(d) + sw_vol * np.sqrt(sw.maturity) * norm.pdf(d))
        swaption_pv = pv * sw_notional
    else:
        swaption_pv = sw.price_bachelier(fwd_rate, sw_vol, annuity)

    col1, col2, col3 = st.columns(3)
    col1.metric("Forward Swap Rate", f"{fwd_rate*100:.3f}%")
    col2.metric("Annuity Factor", f"{annuity:.4f}")
    col3.metric(f"{sw_type} Swaption PV", f"₹{swaption_pv:.2f} Cr")

    # Vol sensitivity (vega)
    sw_vol_up = sw_vol + 0.0001  # +1bp vol
    pv_up = sw.price_bachelier(fwd_rate, sw_vol_up, annuity)
    vega = (pv_up - swaption_pv) * sw_notional
    st.metric("Vega (₹ Cr per 1bp vol)", f"{vega:.4f}")

# ─────────────────────────────────────────────────────────────
# PAGE: V4 Enhancements
# ─────────────────────────────────────────────────────────────
elif page == "🌐 V4 MULTI-CURVE & SABR":
    st.markdown("## V4 Multi-Curve Framework & SABR Pricer")
    st.markdown("Advanced pricing using dual-curve (OIS Discounting + MIBOR Forward) and Normal SABR Volatility surface.")

    from src.curves.multi_curve import MultiCurveFramework
    from src.pricing.sabr import VolSurface
    from src.pricing.swaption import EuropeanSwaption, price_swaption_sabr
    from src.data_ingestion.market_data import get_ois_market_data

    # Load OIS Data
    ois_data = get_ois_market_data()
    
    # Initialize components
    multi_curve = MultiCurveFramework.build_from_market_data()
    
    vol_surface = VolSurface.build_from_market_data()
    
    st.markdown("### Multi-Curve Bootstrapping")
    t_grid = np.linspace(0.1, 10.0, 100)
    ois_fwd = [multi_curve.discount.df(t) for t in t_grid]  # Simplification for plot
    mibor_fwd = [multi_curve.mibor.df(t) for t in t_grid]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_grid, y=ois_fwd, name="OIS Discount Factors", line=dict(color='#00ff00')))
    fig.add_trace(go.Scatter(x=t_grid, y=mibor_fwd, name="MIBOR Discount Factors", line=dict(color='#00BFFF', dash='dash')))
    fig.update_layout(title="OIS vs MIBOR Curves", xaxis_title="Tenor (Years)", yaxis_title="Discount Factor", **PLOTLY_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### SABR Swaption Pricing")
    col1, col2, col3 = st.columns(3)
    sw_expiry = col1.slider("SABR Expiry (Y)", 1.0, 10.0, 5.0, key="sabr_exp")
    sw_tenor = col2.slider("SABR Tenor (Y)", 1.0, 10.0, 5.0, key="sabr_ten")
    sw_strike = col3.number_input("SABR Strike (%)", value=7.0, step=0.1, key="sabr_stk") / 100.0

    pricer = EuropeanSwaption(notional=notional, strike=sw_strike, maturity=sw_expiry, swap_tenor=sw_tenor)
    
    # Extract fwd rate and annuity dynamically from the dual-curve
    annuity = sum([multi_curve.discount.df(sw_expiry + i) for i in range(1, int(sw_tenor)+1)])
    fwd_rate = (multi_curve.discount.df(sw_expiry) - multi_curve.discount.df(sw_expiry + sw_tenor)) / annuity
    
    sabr_pv = price_swaption_sabr(pricer, fwd_rate, annuity, vol_surface)
    st.metric("SABR Swaption PV (₹ Cr)", f"₹ {sabr_pv:.4f}")

    # Plot 1: 3D ATM Volatility Surface
    st.markdown("### SABR ATM Volatility Surface")
    vol_df = vol_surface.to_dataframe()
    vol_pivot = vol_df.pivot(index='expiry_years', columns='tenor_years', values='atm_normal_vol_bps')
    
    fig_vol = go.Figure(data=[go.Surface(
        z=vol_pivot.values, 
        x=vol_pivot.columns, 
        y=vol_pivot.index,
        colorscale='Viridis'
    )])
    fig_vol.update_layout(
        title="ATM Normal Volatility (bps)", 
        scene=dict(
            xaxis_title="Tenor (Y)", 
            yaxis_title="Expiry (Y)", 
            zaxis_title="Vol (bps)"
        ), 
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    # Plot 2: SABR Volatility Smile
    strikes = np.linspace(0.04, 0.10, 50)
    smile_vols = [vol_surface.implied_vol(sw_expiry, sw_tenor, fwd_rate, k) * 10000 for k in strikes]
    fig_smile = go.Figure(go.Scatter(x=strikes*100, y=smile_vols, mode='lines', line=dict(color='orange')))
    fig_smile.update_layout(
        title=f"SABR Volatility Smile (Expiry={sw_expiry}Y, Tenor={sw_tenor}Y)", 
        xaxis_title="Strike (%)", 
        yaxis_title="Normal Implied Vol (bps)", 
        **PLOTLY_LAYOUT
    )
    
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        st.plotly_chart(fig_vol, use_container_width=True)
    with col_v2:
        st.plotly_chart(fig_smile, use_container_width=True)

elif page == "🧊 V4 EXPOSURE CUBE & MVA":
    st.markdown("## V4 Parquet Exposure Cube & MVA")
    st.markdown("Leverages `pyarrow` to persist Monte Carlo paths and efficiently compute Pathwise FVA and SIMM MVA.")

    from src.exposure.exposure_cube import ExposureCube
    from src.xva.fva_v2 import FVAEngineV2
    from src.xva.simm import MVAEngineV2
    
    cube_path = "data/exposure_cube.parquet"
    
    if st.button("Generate Cube from Active Trade"):
        cube = ExposureCube(cube_path)
        # Use actual simulated paths from the global active trade
        cube.write_paths("TRADE_CURRENT", time_grid, mtm_paths)
        cube.flush()
        st.success("Cube generated successfully using active trade simulation!")
        
    if os.path.exists(cube_path):
        cube = ExposureCube(cube_path)
        trades = cube.get_summary().get('trade_list', [])
        st.write(f"Cube active with trades: {trades}")
        
        # Load data
        if trades:
            df_trade = cube.read_trade(trades[0])
            time_grid = np.sort(df_trade['time_step'].unique())
            pivot_df = df_trade.pivot(index='path_id', columns='time_step', values='npv')
            npv_paths = pivot_df.values
            
            # 1. Plot NPV Paths (first 50)
            fig = go.Figure()
            for i in range(min(50, npv_paths.shape[0])):
                fig.add_trace(go.Scatter(x=time_grid, y=npv_paths[i, :], mode='lines', line=dict(color='#3b82f6', width=1), opacity=0.3, showlegend=False))
            fig.update_layout(title="Monte Carlo NPV Paths (First 50)", xaxis_title="Years", yaxis_title="NPV (₹ Cr)", **PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)
            
            # 2. Compute FVA V2
            st.markdown("### Pathwise FVA (V2)")
            fva_engine = FVAEngineV2()
            # Use actual OIS discount factors — ois_curve is available in this scope
            dfs = np.array([ois_curve.df(float(t)) for t in time_grid])
            fva_res = fva_engine.compute_fva_pathwise(time_grid, npv_paths, dfs)
            
            c1, c2, c3 = st.columns(3)
            c1.metric("FVA (Net)", f"₹ {fva_res['FVA']:.4f}")
            c2.metric("FCA (Cost)", f"₹ {fva_res['FCA']:.4f}")
            c3.metric("FBA (Benefit)", f"₹ {fva_res['FBA']:.4f}")
            
            # 3. Compute SIMM MVA
            st.markdown("### SIMM Dynamic MVA")
            mva_engine = MVAEngineV2(funding_spread=0.015)
            # Compute SIMM IM from actual trade sensitivities
            # Use the active trade's notional and maturity from sidebar controls
            # Key Rate DV01s as SIMM sensitivities (in INR, not bps)
            from src.pricing.swap_pricer import SwapPricer
            pricer_mva = SwapPricer(
                notional=notional,
                fixed_rate=fixed_rate,
                maturity=float(maturity),
                direction=direction
            )
            kr_dv01 = pricer_mva.key_rate_dv01(ois_curve)  # dict: tenor → DV01

            # Map key rate DV01s to SIMM tenor buckets (₹ Cr per bp → absolute)
            # SIMM sensitivities are in notional units, not bps — scale by 10000
            simm_sensitivities = {}
            tenor_map = {
                '1.0': '1Y', '2.0': '2Y', '3.0': '3Y', '5.0': '5Y', '10.0': '10Y'
            }
            for tenor_key, dv01_val in kr_dv01.items():
                # Extract numeric tenor from key (e.g. '5.00Y' → '5.0')
                tenor_num = tenor_key.replace('Y', '').strip()
                simm_bucket = tenor_map.get(tenor_num)
                if simm_bucket:
                    # Convert DV01 (₹Cr/bp) to sensitivity in ₹ (SIMM convention)
                    simm_sensitivities[simm_bucket] = abs(dv01_val) * 10000 * 1e7  # ₹Cr → ₹

            # Estimate IM profile using DIM proxy (amortises with maturity)
            initial_sensitivities = simm_sensitivities if simm_sensitivities else {'5Y': abs(pricer_mva.dv01(ois_curve)) * 10000 * 1e7}
            im_profile = mva_engine.estimate_dim_profile(
                trade_maturity=float(maturity),
                time_grid=time_grid,
                initial_sensitivities=initial_sensitivities
            )
            mva = mva_engine.compute_mva(time_grid, im_profile, dfs)

            # Display IM at inception
            im_t0 = mva_engine.simm.compute_im_rates_delta(initial_sensitivities)
            c1_mva, c2_mva = st.columns(2)
            c1_mva.metric("SIMM IM at Inception (₹)", f"₹ {im_t0:,.0f}")
            c2_mva.metric("SIMM MVA (₹ Cr)", f"₹ {mva:.4f}")
    else:
        st.warning("Exposure cube not found. Please generate it first.")

elif page == "💸 V4 PNL ATTRIBUTION":
    st.markdown("## V4 Swap PnL Attribution")
    st.markdown(
        "Decomposes day-over-day PnL into **Carry, Roll-Down, Delta, Gamma, "
        "New Fixing** and **Unexplained**. Curve moves are calibrated to "
        "historical MIBOR volatility (RBI DBIE — free)."
    )

    from src.xva.pnl_attribution import SwapPnLAttribution

    col_left, col_right = st.columns([1, 2])
    with col_left:
        pnl_days = st.slider("Days to Show", min_value=2, max_value=10,
                              value=5, key="pnl_days")
        pnl_notional = st.number_input("Notional (₹ Cr)", value=float(notional),
                                        step=50.0, key="pnl_notional")
        pnl_rate = st.number_input("Fixed Rate (%)", value=float(fixed_rate * 100),
                                    step=0.05, key="pnl_rate") / 100.0
        pnl_maturity = st.number_input("Maturity (Y)", value=float(maturity),
                                        step=0.5, key="pnl_mat")

    # Build realistic daily curve sequence calibrated to MIBOR historical vol
    daily_curves = SwapPnLAttribution.build_daily_curve_sequence(n_days=pnl_days + 1)

    attr = SwapPnLAttribution(
        notional=pnl_notional,
        fixed_rate=pnl_rate,
        maturity_years=pnl_maturity
    )

    # Compute day-over-day attribution for each daily step
    daily_results = []
    for i in range(1, len(daily_curves)):
        curve_t0 = daily_curves[i - 1]
        curve_t1 = daily_curves[i]
        result_df = attr.full_attribution(curve_t1, curve_t0)
        row = dict(zip(result_df['Effect'], result_df['PnL (₹ Cr)']))
        row['Day'] = f"Day {i}"
        daily_results.append(row)

    daily_df = pd.DataFrame(daily_results).set_index('Day')

    # Show today's attribution (last day)
    latest = daily_results[-1]
    st.markdown(f"### Day {pnl_days} Attribution")
    cols = st.columns(6)
    for col, key in zip(cols, ['Carry', 'Roll-Down', 'Delta', 'Gamma',
                                'New Fixing', 'Unexplained']):
        val = latest.get(key, 0.0)
        cols[list(cols).index(col)].metric(key, f"₹ {val:.4f} Cr")

    total_val = latest.get('TOTAL', 0.0)
    st.metric("**Total PnL**", f"₹ {total_val:.4f} Cr")

    # Full waterfall for the latest day (all 6 components + total)
    effects = ['Carry', 'Roll-Down', 'Delta', 'Gamma', 'New Fixing', 'Unexplained']
    values = [latest.get(e, 0.0) for e in effects]
    measures = ['relative'] * len(effects) + ['total']

    fig_pnl = go.Figure(go.Waterfall(
        name="PnL Attribution",
        orientation="v",
        measure=measures,
        x=effects + ['Total PnL'],
        y=values + [total_val],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        increasing={"marker": {"color": "#22c55e"}},
        decreasing={"marker": {"color": "#ef4444"}},
        totals={"marker": {"color": "#3b82f6"}},
    ))
    fig_pnl.update_layout(
        title=f"PnL Waterfall — Day {pnl_days} ({'+' if total_val >= 0 else ''}{total_val:.4f} ₹ Cr)",
        **PLOTLY_LAYOUT
    )
    st.plotly_chart(fig_pnl, use_container_width=True)

    # Multi-day stacked bar showing attribution evolution
    st.markdown("### Attribution Over Time")
    fig_multi = go.Figure()
    colors = {'Carry': '#22c55e', 'Roll-Down': '#84cc16', 'Delta': '#3b82f6',
              'Gamma': '#8b5cf6', 'New Fixing': '#f59e0b', 'Unexplained': '#6b7280'}
    for effect in effects:
        fig_multi.add_trace(go.Bar(
            name=effect,
            x=[r['Day'] for r in daily_results],
            y=[r.get(effect, 0.0) for r in daily_results],
            marker_color=colors.get(effect, '#888888'),
        ))
    fig_multi.update_layout(
        barmode='relative',
        title="Daily PnL Attribution Stack",
        xaxis_title="Day",
        yaxis_title="₹ Cr",
        **PLOTLY_LAYOUT
    )
    st.plotly_chart(fig_multi, use_container_width=True)

    # Attribution table
    st.markdown("### Full Attribution Table")
    st.dataframe(
        daily_df[['Carry', 'Roll-Down', 'Delta', 'Gamma',
                   'New Fixing', 'Unexplained', 'TOTAL']].round(6),
        use_container_width=True
    )

elif page == "✅ V4 MODEL VALIDATION":
    st.markdown("## V4 MRM Model Validation")
    st.markdown("Executes the quantitative model validation suite to verify core pricing and simulation engines.")
    
    from src.validation.model_validator import ModelValidationSuite
    
    if st.button("Run Full Validation Suite"):
        with st.spinner("Running MRM Validation..."):
            suite = ModelValidationSuite()
            report_df = suite.run_all()
            
            st.success("Validation Completed.")
            
            def color_status(val):
                color = '#00ff00' if val == 'PASS' else '#ff4d4d' if val == 'FAIL' else '#f5a623'
                return f'color: {color}; font-weight: bold;'
            
            st.dataframe(report_df.style.map(color_status, subset=['Status']), use_container_width=True)
            
            passes = len(report_df[report_df['Status'] == 'PASS'])
            total = len(report_df)
            st.metric("Validation Score", f"{passes}/{total} Passed")
