"""Testing kge module."""

import os
import sys

import pytest
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from de import kge

baseline_dir = 'images'

baseline_dir_single = os.path.join(baseline_dir, 'kge/single')
baseline_dir_multi = os.path.join(baseline_dir, 'kge/multi')

WIN = sys.platform.startswith('win')

# In some cases, the fonts on Windows can be quite different
DEFAULT_TOLERANCE = 10 if WIN else 2

def test_kge_for_arrays():
    eff = kge.calc_kge(obs=np.array([1.5, 1, 0.8, 0.85, 1.5, 2]),
                       sim=np.array([1.6, 1.3, 1, 0.8, 1.2, 2.5]))
    assert eff == pytest.approx(0.683901305466148)

def test_kge_skill_for_arrays():
    eff = kge.calc_kge_skill(obs=np.array([1.5, 1, 0.8, 0.85, 1.5, 2]),
                             sim=np.array([1.6, 1.3, 1, 0.8, 1.2, 2.5]),
                             bench=np.array([1, 1.1, 1.15, 1.15, 1.1, 1]))
    assert eff == pytest.approx(0.8467044616487865)

def test_beta_for_arrays():
    beta = kge.calc_kge_beta(obs=np.array([1.5, 1, 0.8, 0.85, 1.5, 2]),
                             sim=np.array([1.6, 1.3, 1, 0.8, 1.2, 2.5]))
    assert beta == pytest.approx(1.0980392156862746)

def test_alpha_for_arrays():
    alpha = kge.calc_kge_alpha(obs=np.array([1.5, 1, 0.8, 0.85, 1.5, 2]),
                               sim=np.array([1.6, 1.3, 1, 0.8, 1.2, 2.5]))
    assert alpha == pytest.approx(1.2812057455166919)

def test_gamma_for_arrays():
    gamma = kge.calc_kge_gamma(obs=np.array([1.5, 1, 0.8, 0.85, 1.5, 2]),
                               sim=np.array([1.6, 1.3, 1, 0.8, 1.2, 2.5]))
    assert gamma == pytest.approx(1.166812375381273)

def test_temp_cor_for_arrays():
    temp_cor = kge.calc_temp_cor(obs=np.array([1.5, 1, 0.8, 0.85, 1.5, 2]),
                                 sim=np.array([1.6, 1.3, 1, 0.8, 1.2, 2.5]))
    assert temp_cor == pytest.approx(0.8940281850583509)

@pytest.mark.mpl_image_compare(baseline_dir=baseline_dir_single,
                               tolerance=DEFAULT_TOLERANCE)
def test_single_diag_polar_plot():
    obs = np.array([1.5, 1, 0.8, 0.85, 1.5, 2])
    sim = np.array([1.6, 1.3, 1, 0.8, 1.2, 2.5])
    fig = kge.diag_polar_plot(obs, sim)
    return fig

@pytest.mark.mpl_image_compare(baseline_dir=baseline_dir_multi,
                               tolerance=DEFAULT_TOLERANCE)
def test_multi_diag_polar_plot():
    beta = np.array([1.1, 1.15, 1.2, 1.1, 1.05, 1.15])
    alpha = np.array([1.15, 1.1, 1.2, 1.1, 1.1, 1.2])
    r = np.array([0.9, 0.85, 0.8, 0.9, 0.85, 0.9])
    eff_kge = np.array([0.79, 0.77, 0.65, 0.83, 0.81, 0.73])
    fig = kge.diag_polar_plot_multi(beta, alpha, r, eff_kge)
    return fig
