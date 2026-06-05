"""
Market Data Ingestion for INR OTC Derivatives Platform.

Supports 3-tier fallback architecture:
1. Live fetch (from FIMMDA or RBI DBIE).
2. Local cache (data/raw/fallback_cache.json).
3. Original hardcoded synthetic values.
"""

import os
import json
import logging
import requests
import io
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

# ---------------------------------------------------------------------------
# Fallback Constants (Tier 3)
# ---------------------------------------------------------------------------

OIS_TENORS_YEARS = np.array([
    1/365, 7/365, 14/365, 1/12, 2/12, 3/12, 6/12, 9/12, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0
])

OIS_TENOR_LABELS = [
    'O/N', '1W', '2W', '1M', '2M', '3M', '6M', '9M', '1Y', '2Y', '3Y', '4Y', '5Y', '7Y', '10Y'
]

OIS_RATES = np.array([
    0.0625, 0.0628, 0.0630, 0.0635, 0.0640, 0.0648, 0.0660, 0.0668, 0.0675, 0.0695, 0.0710, 0.0720, 0.0728, 0.0740, 0.0750,
])

GSEC_TENORS_YEARS = np.array([91/365, 182/365, 364/365, 2.0, 5.0, 10.0, 30.0])
GSEC_TENOR_LABELS = ['91D T-Bill', '182D T-Bill', '364D T-Bill', '2Y G-Sec', '5Y G-Sec', '10Y G-Sec', '30Y G-Sec']
GSEC_YIELDS = np.array([0.0640, 0.0655, 0.0670, 0.0700, 0.0735, 0.0760, 0.0790])

POLICY_RATES = {
    'repo_rate':         0.0550,
    'sdf_rate':          0.0525,
    'msf_rate':          0.0575,
    'reverse_repo_rate': 0.0335,
    'crr':               0.0400,
    '_last_updated':     '2025-08-06',
    '_next_mpc':         '2025-10-06',
}

_RATING_SPREAD_LADDER = {
    'AAA':  50,   'AA+':  65,  'AA':   85,
    'AA-': 100,   'A+':  120,  'A':   150,
    'A-':  180,   'BBB+':220,  'BBB': 280,
    'BBB-':340,   'BB+': 420,  'BB':  550,
    'BB-': 700,   'B':   900,
}

_FUNDING_SPREAD_LADDER = {
    'AAA':  25,  'AA+':  35,  'AA':   40,
    'AA-':  50,  'A+':   80,  'A':   150,
    'BBB': 250,  'BB':   400,
}

# ---------------------------------------------------------------------------
# Cache Module
# ---------------------------------------------------------------------------

CACHE_PATH = Path('data/raw/fallback_cache.json')

def _load_cache() -> dict:
    try:
        if CACHE_PATH.exists():
            return json.loads(CACHE_PATH.read_text())
    except Exception as e:
        logging.warning(f'[Cache] Load failed: {e}')
    return {}

def _save_cache(key: str, data) -> None:
    try:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        cache = _load_cache()
        cache[key] = data
        cache['_last_updated'] = datetime.now().isoformat()
        CACHE_PATH.write_text(json.dumps(cache, default=str, indent=2))
    except Exception as e:
        logging.warning(f'[Cache] Save failed: {e}')

# ---------------------------------------------------------------------------
# Core Fetchers
# ---------------------------------------------------------------------------

def _fetch_fimmda_ois() -> pd.DataFrame:
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Referer': 'https://www.fimmda.org/',
        'Accept': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,*/*',
    }
    
    for days_back in range(0, 4):
        date = datetime.today() - timedelta(days=days_back)
        date_str = date.strftime('%d%m%Y')
        url = f'https://www.fimmda.org/uploads/RateFiles/{date_str}_FIMMDA.xlsx'
        
        try:
            resp = requests.get(url, headers=HEADERS, timeout=12)
            if resp.status_code != 200:
                continue
            
            xls = pd.ExcelFile(io.BytesIO(resp.content))
            
            ois_sheet = None
            for sheet in xls.sheet_names:
                if 'OIS' in sheet.upper() or 'SWAP' in sheet.upper():
                    ois_sheet = sheet
                    break
            if ois_sheet is None:
                ois_sheet = xls.sheet_names[0]
            
            df = xls.parse(ois_sheet, header=0)
            df.columns = [str(c).strip().upper() for c in df.columns]
            tenor_col = next((c for c in df.columns if 'TENOR' in c), None)
            rate_col  = next((c for c in df.columns if 'RATE' in c or 'YIELD' in c), None)
            
            if tenor_col is None or rate_col is None:
                continue
            
            df = df[[tenor_col, rate_col]].dropna()
            df.columns = ['tenor_label', 'rate_pct']
            df['rate_pct'] = pd.to_numeric(df['rate_pct'], errors='coerce')
            df = df.dropna()
            
            if df['rate_pct'].max() > 1.0:
                df['rate_pct'] = df['rate_pct'] / 100.0
            
            return df
        
        except Exception:
            continue
    
    raise RuntimeError('FIMMDA fetch failed for last 4 days')


def _build_full_ois_curve(fimmda_df: pd.DataFrame) -> pd.DataFrame:
    TENOR_MAP = {
        'ON': 1/365, 'O/N': 1/365, 'OVERNIGHT': 1/365,
        '1W': 7/365, '1 WEEK': 7/365,
        '2W': 14/365, '2 WEEK': 14/365,
        '1M': 1/12,  '2M': 2/12,  '3M': 3/12,
        '6M': 6/12,  '9M': 9/12,
        '1Y': 1.0,   '2Y': 2.0,   '3Y': 3.0,
        '4Y': 4.0,   '5Y': 5.0,   '7Y': 7.0,   '10Y': 10.0,
    }
    
    available_years = []
    available_rates = []
    for _, row in fimmda_df.iterrows():
        key = str(row['tenor_label']).strip().upper()
        if key in TENOR_MAP:
            available_years.append(TENOR_MAP[key])
            available_rates.append(float(row['rate_pct']))
    
    if len(available_years) < 3:
        raise ValueError('Not enough FIMMDA tenors to build curve')
    
    pairs = sorted(zip(available_years, available_rates))
    avail_t = [p[0] for p in pairs]
    avail_r = [p[1] for p in pairs]
    
    TARGET_YEARS  = [1/365, 7/365, 14/365, 1/12, 2/12, 3/12,
                     6/12, 9/12, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]
    TARGET_LABELS = ['O/N','1W','2W','1M','2M','3M',
                     '6M','9M','1Y','2Y','3Y','4Y','5Y','7Y','10Y']
    
    repo = POLICY_RATES['repo_rate']
    short_anchor = {1/365: repo + 0.0025, 7/365: repo + 0.003, 14/365: repo + 0.003}
    
    final_rates = []
    for t, label in zip(TARGET_YEARS, TARGET_LABELS):
        if t in short_anchor and t < min(avail_t):
            final_rates.append(short_anchor[t])
        else:
            final_rates.append(float(np.interp(t, avail_t, avail_r)))
    
    return pd.DataFrame({
        'tenor_label': TARGET_LABELS,
        'tenor_years': TARGET_YEARS,
        'ois_rate': final_rates,
    })


def _fetch_rbi_dbie_series(series_id: str, years_back: int = 3) -> pd.Series:
    end_year   = datetime.today().year
    start_year = end_year - years_back
    
    url = (
        f'https://dbie.rbi.org.in/DBIE/dbie.rbi'
        f'?site=statistics'
        f'&seriesid={series_id}'
        f'&startYear={start_year}'
        f'&endYear={end_year}'
        f'&format=csv'
    )
    
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Referer': 'https://dbie.rbi.org.in/',
        'Accept': 'text/csv,application/csv,text/plain,*/*',
    }
    
    resp = requests.get(url, headers=HEADERS, timeout=12)
    resp.raise_for_status()
    
    lines = resp.text.strip().split('\n')
    
    data_start = 0
    for i, line in enumerate(lines):
        first_cell = line.split(',')[0].strip().strip('"')
        if len(first_cell) >= 4 and (first_cell[:4].isdigit() or '-' in first_cell or '/' in first_cell):
            try:
                pd.to_datetime(first_cell)
                data_start = i
                break
            except:
                continue
    
    df = pd.read_csv(
        io.StringIO('\n'.join(lines[data_start:])),
        header=None,
        names=['date', 'value'],
        usecols=[0, 1],
    )
    df['date']  = pd.to_datetime(df['date'], errors='coerce', dayfirst=True)
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df = df.dropna().set_index('date').sort_index()
    
    s = df['value']
    if s.max() > 1.0:
        s = s / 100.0
    
    s = s[s.between(0.01, 0.25)]
    return s


def _synthetic_mibor_fallback(n_days: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    mean_rate, kappa, sigma, dt = 0.0625, 0.10, 0.0080, 1/252
    rates = np.zeros(n_days)
    rates[0] = 0.0650
    for i in range(1, n_days):
        dr = kappa * (mean_rate - rates[i-1]) * dt + sigma * np.sqrt(dt) * rng.standard_normal()
        rates[i] = max(rates[i-1] + dr, 0.01)
    dates = pd.bdate_range(end=datetime.today(), periods=n_days)
    return pd.DataFrame({'date': dates, 'mibor_rate': rates})


def _pad_mibor_with_simulation(series: pd.Series, n_days: int, seed: int) -> pd.Series:
    missing_days = n_days - len(series)
    if missing_days <= 0:
        return series
    
    synth = _synthetic_mibor_fallback(missing_days + 1, seed)
    synth_rates = synth['mibor_rate'].values.copy()
    
    anchor = series.iloc[0]
    shift = anchor - synth_rates[-1]
    synth_rates += shift
    
    dates = pd.bdate_range(end=series.index[0] - pd.tseries.offsets.BDay(1), periods=missing_days)
    pad_series = pd.Series(synth_rates[:-1], index=dates)
    
    return pd.concat([pad_series, series])


def _fetch_fimmda_bond_zspread() -> dict:
    """
    Fetch FIMMDA bond Z-spreads from publicly available FIMMDA data.

    Attempts to download the FIMMDA daily Excel valuation sheet and parse
    a corporate bond spread column. Falls back to published FIMMDA
    sector-average Z-spreads from quarterly market reports (free PDF).

    Z-Spread definition:
        Constant spread z such that:
        Bond Price = Σ C_i × DF(t_i) × exp(-z × t_i)

    Primary:  FIMMDA daily bond valuation sheet (free Excel at fimmda.org)
    Fallback: Published FIMMDA sector-average spreads (quarterly reports)

    Returns:
        Dict mapping issuer/sector → Z-spread in basis points.
    """
    import requests
    import io
    from datetime import datetime, timedelta

    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
        'Referer':    'https://www.fimmda.org/',
    }

    # Try FIMMDA daily Excel for up to 5 business days back
    for days_back in range(0, 5):
        try:
            date     = datetime.today() - timedelta(days=days_back)
            date_str = date.strftime('%d%m%Y')
            url      = f'https://www.fimmda.org/uploads/RateFiles/{date_str}_FIMMDA.xlsx'
            resp     = requests.get(url, headers=HEADERS, timeout=12)
            if resp.status_code != 200:
                continue

            xls = pd.ExcelFile(io.BytesIO(resp.content))
            spread_sheet = next(
                (s for s in xls.sheet_names
                 if any(kw in s.upper() for kw in ['SPREAD','BOND','CORP','ZSPREAD'])),
                None
            )
            if spread_sheet is None:
                continue

            df = xls.parse(spread_sheet, header=0)
            df.columns = [str(c).strip().upper() for c in df.columns]
            issuer_col = next((c for c in df.columns if any(
                k in c for k in ['ISSUER','NAME','SCRIP'])), None)
            spread_col = next((c for c in df.columns if any(
                k in c for k in ['SPREAD','Z-SPREAD','ZSPREAD'])), None)

            if issuer_col and spread_col:
                result = {}
                for _, row in df.iterrows():
                    try:
                        spread = float(row[spread_col])
                        if 0 < spread < 1000:
                            result[str(row[issuer_col]).strip()] = spread
                    except (ValueError, TypeError):
                        continue
                if result:
                    return result
        except Exception:
            continue

    # ── Fallback: FIMMDA published sector-average Z-spreads ─────────────────
    # Source: FIMMDA Quarterly Report "Corporate Bond Market in India" (free PDF)
    # Values calibrated to 2025-2026 INR market conditions.
    return {
        'PSU_AAA':           25.0,
        'PSU_AA':            45.0,
        'Private_Bank_AAA':  35.0,
        'Private_Bank_AA':   60.0,
        'NBFC_AAA':          55.0,
        'NBFC_AA':           95.0,
        'NBFC_A':           160.0,
        'Corporate_AAA':     45.0,
        'Corporate_AA':      80.0,
        'Corporate_A':      140.0,
        'Corporate_BBB':    250.0,
        'Infra_AAA':         30.0,
        'Infra_AA':          55.0,
    }


# ---------------------------------------------------------------------------
# Exported Public API
# ---------------------------------------------------------------------------

def get_policy_rates() -> Dict[str, float]:
    return {k: v for k, v in POLICY_RATES.items() if not k.startswith('_')}


def get_ois_market_data() -> pd.DataFrame:
    try:
        fimmda_raw = _fetch_fimmda_ois()
        df = _build_full_ois_curve(fimmda_raw)
        _save_cache('ois_data', df.to_dict('records'))
        logging.info('[MarketData] OIS curve loaded from FIMMDA')
        return df
    except Exception as e:
        logging.warning(f'[MarketData] FIMMDA OIS fetch failed: {e}. Trying cache.')
        cached = _load_cache().get('ois_data')
        if cached:
            return pd.DataFrame(cached)
        logging.warning('[MarketData] Cache miss. Using synthetic OIS fallback.')
        return pd.DataFrame({
            'tenor_label': OIS_TENOR_LABELS,
            'tenor_years': OIS_TENORS_YEARS,
            'ois_rate':    OIS_RATES,
        })


def get_gsec_market_data() -> pd.DataFrame:
    try:
        rows = {}
        for sid, label, t in [
            ('TBILL91_YLD',  '91D T-Bill',  91/365),
            ('TBILL182_YLD', '182D T-Bill', 182/365),
            ('TBILL364_YLD', '364D T-Bill', 364/365),
            ('GSEC10_YLD',   '10Y G-Sec',   10.0),
        ]:
            try:
                s = _fetch_rbi_dbie_series(sid, years_back=1)
                rows[t] = {'tenor_label': label, 'yield_rate': float(s.iloc[-1])}
            except Exception:
                pass
        
        if len(rows) < 2:
            raise RuntimeError('Fewer than 2 G-Sec tenors fetched from DBIE')
        
        avail_t = sorted(rows.keys())
        avail_r = [rows[t]['yield_rate'] for t in avail_t]
        
        TARGET = [
            (91/365,  '91D T-Bill'),
            (182/365, '182D T-Bill'),
            (364/365, '364D T-Bill'),
            (2.0,     '2Y G-Sec'),
            (5.0,     '5Y G-Sec'),
            (10.0,    '10Y G-Sec'),
            (30.0,    '30Y G-Sec'),
        ]
        
        final_yields = []
        for t, label in TARGET:
            if t <= max(avail_t):
                y = float(np.interp(t, avail_t, avail_r))
            else:
                y = avail_r[-1] + (avail_r[-1] - np.interp(5.0, avail_t, avail_r)) * 0.5
            final_yields.append(y)
        
        df = pd.DataFrame({
            'tenor_label': [t[1] for t in TARGET],
            'tenor_years': [t[0] for t in TARGET],
            'yield_rate':  final_yields,
        })
        _save_cache('gsec_data', df.to_dict('records'))
        logging.info('[MarketData] G-Sec curve loaded from RBI DBIE')
        return df
    
    except Exception as e:
        logging.warning(f'[MarketData] G-Sec DBIE fetch failed: {e}. Trying cache.')
        cached = _load_cache().get('gsec_data')
        if cached:
            return pd.DataFrame(cached)
        logging.warning('[MarketData] Cache miss. Using synthetic G-Sec fallback.')
        return pd.DataFrame({
            'tenor_label': GSEC_TENOR_LABELS,
            'tenor_years': GSEC_TENORS_YEARS,
            'yield_rate':  GSEC_YIELDS,
        })


def get_historical_mibor(n_days: int = 504, seed: int = 42) -> pd.DataFrame:
    try:
        series = None
        for sid in ['II_PR_MIBOR_D', 'MIBOR_ON', 'PR_MIBOR_ON']:
            try:
                series = _fetch_rbi_dbie_series(sid, years_back=3)
                if len(series) > 100:
                    break
            except Exception:
                continue
        
        if series is None or len(series) < 100:
            raise RuntimeError('All MIBOR series IDs failed or returned < 100 rows')
        
        series = series.resample('B').last().ffill()
        
        if len(series) >= n_days:
            series = series.iloc[-n_days:]
        else:
            logging.warning(f'[MarketData] DBIE returned {len(series)} rows, need {n_days}. Padding start.')
            series = _pad_mibor_with_simulation(series, n_days, seed)
        
        assert series.between(0.01, 0.20).all(), 'MIBOR values out of expected range'
        
        result = pd.DataFrame({'date': series.index, 'mibor_rate': series.values})
        _save_cache('historical_mibor', result.assign(date=result['date'].astype(str)).to_dict('records'))
        logging.info(f'[MarketData] MIBOR history loaded from RBI DBIE: {len(result)} rows')
        return result
    
    except Exception as e:
        logging.warning(f'[MarketData] MIBOR live fetch failed: {e}. Trying cache.')
        
        cached = _load_cache().get('historical_mibor')
        if cached:
            df = pd.DataFrame(cached)
            df['date'] = pd.to_datetime(df['date'])
            return df.tail(n_days).reset_index(drop=True)
        
        logging.warning('[MarketData] Cache miss. Falling back to synthetic MIBOR.')
        return _synthetic_mibor_fallback(n_days, seed)


def get_counterparty_data() -> pd.DataFrame:
    COUNTERPARTY_MASTER = [
        {'counterparty': 'SBI',                      'entity_type': 'PSU Bank',      'rating': 'AAA',  'risk_weight': 0.20},
        {'counterparty': 'HDFC Bank',                'entity_type': 'Private Bank',  'rating': 'AA+',  'risk_weight': 0.20},
        {'counterparty': 'ICICI Bank',               'entity_type': 'Private Bank',  'rating': 'AA',   'risk_weight': 0.20},
        {'counterparty': 'Kotak Bank',               'entity_type': 'Private Bank',  'rating': 'AA',   'risk_weight': 0.20},
        {'counterparty': 'Corporate A (Large NBFC)', 'entity_type': 'Large NBFC',    'rating': 'A',    'risk_weight': 0.75},
        {'counterparty': 'Corporate B (Mid NBFC)',   'entity_type': 'Mid NBFC',      'rating': 'BBB',  'risk_weight': 1.00},
        {'counterparty': 'Corporate C (Stressed)',   'entity_type': 'Stressed Entity','rating': 'BB',  'risk_weight': 1.50},
    ]
    
    try:
        live_spreads = _fetch_fimmda_bond_zspread()
        
        rows = []
        for cp in COUNTERPARTY_MASTER:
            name = cp['counterparty']
            cds_bps = live_spreads.get(name, _RATING_SPREAD_LADDER.get(cp['rating'], 150))
            fund_bps = cds_bps * 0.6
            rows.append({**cp,
                         'cds_spread_bps': cds_bps,
                         'recovery_rate': 0.40,
                         'funding_spread_bps': fund_bps})
        
        df = pd.DataFrame(rows)
        _save_cache('counterparty_data', df.to_dict('records'))
        return df
    
    except Exception as e:
        logging.warning(f'[MarketData] FIMMDA bond fetch failed: {e}. Using rating ladder.')
        
        rows = []
        for cp in COUNTERPARTY_MASTER:
            cds_bps  = _RATING_SPREAD_LADDER.get(cp['rating'], 150)
            fund_bps = _FUNDING_SPREAD_LADDER.get(cp['rating'], int(cds_bps * 0.6))
            rows.append({**cp,
                         'cds_spread_bps': float(cds_bps),
                         'recovery_rate': 0.40,
                         'funding_spread_bps': float(fund_bps)})
        return pd.DataFrame(rows)


def get_csa_scenarios() -> Dict[str, Dict]:
    return {
        'Uncollateralised': {
            'threshold_cr': float('inf'),
            'mta_cr': 0.0,
            'mpor_days': 0,
            'margin_frequency': 'none',
            'independent_amount_cr': 0.0,
        },
        'Partially Collateralised': {
            'threshold_cr': 50.0,
            'mta_cr': 5.0,
            'mpor_days': 10,
            'margin_frequency': 'weekly',
            'independent_amount_cr': 0.0,
        },
        'Fully Collateralised': {
            'threshold_cr': 0.0,
            'mta_cr': 1.0,
            'mpor_days': 10,
            'margin_frequency': 'daily',
            'independent_amount_cr': 0.0,
        },
        'CCP-Cleared': {
            'threshold_cr': 0.0,
            'mta_cr': 0.0,
            'mpor_days': 5,
            'margin_frequency': 'daily',
            'independent_amount_cr': 0.0,
        },
    }


def get_stress_scenarios() -> pd.DataFrame:
    return pd.DataFrame({
        'scenario': [
            'Base', 'RBI Tightening', 'Aggressive Tightening',
            'Rate Easing', 'NBFC Stress', 'Systemic Crisis'
        ],
        'rate_shock_bps': [0, 100, 200, -100, 50, 150],
        'credit_spread_shock_bps': [0, 30, 80, -20, 300, 500],
        'description': [
            'Current market',
            'Normalisation',
            'Inflation crisis',
            'Growth shock',
            'Sectoral credit event',
            '2008-type event'
        ],
    })

def get_historical_stress_scenarios() -> pd.DataFrame:
    """
    Return historically-calibrated stress scenarios for INR rates market.

    Each scenario captures an observed market dislocation:
      - COVID_2020: RBI cut repo 115bps in Mar-May 2020; credit spreads
        widened sharply for NBFCs; OIS rates fell ~120bps peak-to-trough.
      - RBI_2022: Rapid tightening cycle; repo hiked 250bps May-Dec 2022;
        OIS 5Y rose ~200bps; NBFC spreads widened ~150bps.
      - TAPER_2013: Taper tantrum; 10Y G-Sec spiked ~180bps in May-Sep 2013;
        INR depreciated; OIS rates rose ~160bps; credit spreads +200bps.
      - IL_FS_2018: IL&FS default; NBFC credit spreads spiked 300-500bps;
        OIS rates relatively stable (+30bps); severe liquidity squeeze.

    Shocks represent peak observed moves from scenario start date.
    All values calibrated to publicly available RBI/FBIL/CCIL data.

    Returns:
        DataFrame with columns matching get_stress_scenarios() output so
        run_full_stress_test() can consume both interchangeably.
    """
    return pd.DataFrame({
        'scenario': [
            'COVID_2020', 'RBI_2022', 'TAPER_2013', 'IL_FS_2018'
        ],
        'rate_shock_bps': [
            -120,   # OIS rates fell ~120bps (RBI emergency cuts)
            +200,   # OIS 5Y rose ~200bps (tightening cycle)
            +160,   # OIS rates rose ~160bps (taper tantrum)
            +30,    # OIS relatively stable during IL&FS
        ],
        'credit_spread_shock_bps': [
            +250,   # NBFC spreads widened ~250bps
            +150,   # NBFC spreads widened ~150bps
            +200,   # Broad credit widening ~200bps
            +400,   # NBFC/HFC spreads spiked 300-500bps
        ],
        'description': [
            'COVID-19: RBI emergency cut 115bp; NBFC stress; credit widening',
            'RBI tightening 2022: repo +250bp; OIS 5Y +200bp',
            'Taper tantrum 2013: G-Sec +180bp; INR depreciation; credit +200bp',
            'IL&FS 2018: NBFC default; credit spreads +300-500bp; liquidity crisis',
        ],
        'reference_start': [
            '2020-03-01', '2022-05-01', '2013-05-01', '2018-09-01'
        ],
        'reference_end': [
            '2020-05-31', '2022-12-31', '2013-09-30', '2018-12-31'
        ],
    })

def get_sample_portfolio() -> pd.DataFrame:
    """
    Return a sample portfolio of INR IRS/OIS swaps for analytics.
    """
    today = datetime(2026, 6, 3)

    trades = {
        'trade_id': [
            'IRS-001', 'IRS-002', 'IRS-003', 'OIS-004',
            'IRS-005', 'OIS-006', 'IRS-007', 'OIS-008'
        ],
        'counterparty': [
            'SBI', 'HDFC Bank', 'ICICI Bank', 'Kotak Bank',
            'Corporate A (Large NBFC)', 'Corporate B (Mid NBFC)',
            'SBI', 'Corporate C (Stressed)'
        ],
        'product': [
            'INR IRS', 'INR IRS', 'INR IRS', 'INR OIS',
            'INR IRS', 'INR OIS', 'INR IRS', 'INR OIS'
        ],
        'direction': [
            'Receive Fixed', 'Pay Fixed', 'Receive Fixed', 'Pay Fixed',
            'Receive Fixed', 'Receive Fixed', 'Pay Fixed', 'Receive Fixed'
        ],
        'notional_cr': [500, 300, 400, 200, 250, 100, 600, 150],
        'fixed_rate': [0.0700, 0.0680, 0.0720, 0.0660, 0.0750, 0.0690, 0.0710, 0.0770],
        'maturity_years': [5, 3, 7, 2, 5, 3, 10, 5],
        'start_date': [today] * 8,
        'csa_type': [
            'Fully Collateralised', 'Fully Collateralised',
            'Partially Collateralised', 'CCP-Cleared',
            'Uncollateralised', 'Uncollateralised',
            'Fully Collateralised', 'Uncollateralised'
        ],
    }

    df = pd.DataFrame(trades)
    df['end_date'] = df.apply(
        lambda r: r['start_date'] + timedelta(days=int(r['maturity_years'] * 365)),
        axis=1
    )
    return df