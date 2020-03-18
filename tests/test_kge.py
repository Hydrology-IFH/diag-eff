"""Testing kge module."""

import os
import sys

import pytest
import numpy as np
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
    assert eff == 0.6839013054661

def test_kge_skill_for_arrays():
    eff = kge.calc_kge_skill(obs=np.array([1.5, 1, 0.8, 0.85, 1.5, 2]),
                             sim=np.array([1.6, 1.3, 1, 0.8, 1.2, 2.5]),
                             bench=np.array([1, 1.1, 1.15, 1.15, 1.1, 1]))
    assert eff == 0.84670446164878

def test_beta_for_arrays():
    beta = kge.calc_kge_beta(obs=np.array([1.5, 1, 0.8, 0.85, 1.5, 2]),
                             sim=np.array([1.6, 1.3, 1, 0.8, 1.2, 2.5]))
    assert beta == 1.09803921568627

def test_alpha_for_arrays():
    alpha = kge.calc_kge_alpha(obs=np.array([1.5, 1, 0.8, 0.85, 1.5, 2]),
                               sim=np.array([1.6, 1.3, 1, 0.8, 1.2, 2.5]))
    assert alpha == 1.28120574551669

def test_gamma_for_arrays():
    gamma = kge.calc_kge_gamma(obs=np.array([1.5, 1, 0.8, 0.85, 1.5, 2]),
                               sim=np.array([1.6, 1.3, 1, 0.8, 1.2, 2.5]))
    assert gamma == 1.1668123753812

def test_temp_cor_for_arrays():
    temp_cor = kge.calc_temp_cor(obs=np.array([1.5, 1, 0.8, 0.85, 1.5, 2]),
                                 sim=np.array([1.6, 1.3, 1, 0.8, 1.2, 2.5]))
    assert temp_cor == 0.89402818505835

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
