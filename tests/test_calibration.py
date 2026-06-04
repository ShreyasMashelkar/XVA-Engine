import pytest
import numpy as np
from src.curves.ois_curve import OISCurve
from src.calibration.hw_calibrator import HullWhiteCalibrator
from src.data_ingestion.market_data import get_ois_market_data


def test_hw_calibrator():
    ois_df = get_ois_market_data()
    curve = OISCurve(ois_df['tenor_years'].values, ois_df['ois_rate'].values)
    calibrator = HullWhiteCalibrator(curve)

    a, sigma = calibrator.calibrate_to_historical_mibor()
    assert 0.0 < a <= 0.5
    assert 0.0 < sigma < 0.10  # Normal vol usually < 1000 bps

    hw = calibrator.get_calibrated_model()
    assert hw.a == a
    assert hw.sigma == sigma
