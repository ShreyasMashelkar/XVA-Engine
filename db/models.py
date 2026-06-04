from sqlalchemy import Column, Integer, String, Float, Date, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from .database import Base


class Trade(Base):
    __tablename__ = "trades"
    id            = Column(Integer, primary_key=True, index=True)
    trade_id      = Column(String, unique=True, index=True)
    counterparty  = Column(String, index=True)
    notional_cr   = Column(Float)
    fixed_rate    = Column(Float)
    maturity_years = Column(Float)
    direction     = Column(String)


class XVAResult(Base):
    """One row per counterparty per EOD run."""
    __tablename__ = "xva_results"
    id            = Column(Integer, primary_key=True, index=True)
    run_date      = Column(String, index=True)        # YYYYMMDD
    counterparty  = Column(String, index=True)
    cds_bps       = Column(Float)
    epe_cr        = Column(Float)
    cva_cr        = Column(Float)
    dva_cr        = Column(Float)
    cs01_cr       = Column(Float, default=0.0)
    ir01_cr       = Column(Float, default=0.0)
    fva_cr        = Column(Float)
    mva_cr        = Column(Float)
    kva_cr        = Column(Float)
    ead_cr        = Column(Float)
    rwa_cr        = Column(Float)
    capital_cr    = Column(Float)
    xva_total_cr  = Column(Float)
    created_at    = Column(String, default=lambda: datetime.utcnow().isoformat())


class CurveSnapshot(Base):
    """One row per tenor node per EOD run — stores the full OIS curve."""
    __tablename__ = "curve_snapshots"
    id            = Column(Integer, primary_key=True, index=True)
    run_date      = Column(String, index=True)
    tenor_label   = Column(String)
    tenor_years   = Column(Float)
    ois_rate      = Column(Float)
    discount_factor = Column(Float)
    zero_rate     = Column(Float)


class MarketDataSnapshot(Base):
    """Key policy and market rates at EOD."""
    __tablename__ = "market_data_snapshots"
    id            = Column(Integer, primary_key=True, index=True)
    run_date      = Column(String, index=True)
    metric        = Column(String)   # e.g. 'repo_rate', 'mibor_on', 'ois_5y'
    value         = Column(Float)
