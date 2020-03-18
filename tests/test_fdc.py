"""Testing fdc module."""

import os
import sys

import pytest
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from de import fdc

baseline_dir = 'images'

baseline_dir_single = os.path.join(baseline_dir, 'fdc/single_fdc')
baseline_dir_dual = os.path.join(baseline_dir, 'de/dual_fdc')

WIN = sys.platform.startswith('win')

# In some cases, the fonts on Windows can be quite different
DEFAULT_TOLERANCE = 10 if WIN else 2

@pytest.mark.mpl_image_compare(baseline_dir=baseline_dir_single,
                               tolerance=DEFAULT_TOLERANCE)
def test_single_fdc():
    date_rng = pd.date_range(start='1/1/2018', periods=11)
    arr = np.array([1.5, 1, 0.8, 0.85, 1.5, 2, 2.5, 3.5, 1.8, 1.5, 1.2])
    ts = pd.Series(data=arr, index=date_rng)
    fig = fdc.fdc(ts)
    
    return fig

@pytest.mark.mpl_image_compare(baseline_dir=baseline_dir_dual,
                               tolerance=DEFAULT_TOLERANCE)
def test_dual_fdc():
    date_rng = pd.date_range(start='1/1/2018', periods=11)
    obs = np.array([1.5, 1, 0.8, 0.85, 1.5, 2, 2.5, 3.5, 1.8, 1.5, 1.2])
    ts_obs = pd.Series(data=obs, index=date_rng)
    sim = np.array([1.4, .9, 1, 0.95, 1.4, 2.1, 2.6, 3.6, 1.9, 1.4, 1.1])
    ts_sim = pd.Series(data=sim, index=date_rng)
    fig = fdc.fdc_obs_sim(ts_obs, ts_sim)

    return fig
