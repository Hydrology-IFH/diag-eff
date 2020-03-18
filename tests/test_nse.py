"""Testing nse module."""

import pytest
import numpy as np
import pandas as pd
from de import nse

def test_nse_for_arrays():
    eff = nse.calc_nse(obs=np.array([1.5, 1, 0.8, 0.85, 1.5, 2]),
                       sim=np.array([1.6, 1.3, 1, 0.8, 1.2, 2.5]))
    assert eff == 0.56482525366403

def test_nse_dec_for_arrays():
    eff = nse.calc_nse_dec(obs=np.array([1.5, 1, 0.8, 0.85, 1.5, 2]),
                           sim=np.array([1.6, 1.3, 1, 0.8, 1.2, 2.5]))
    assert eff == 0.92510767049232

def test_nse_beta_for_arrays():
    beta = nse.calc_nse_beta(obs=np.array([1.5, 1, 0.8, 0.85, 1.5, 2]),
                             sim=np.array([1.6, 1.3, 1, 0.8, 1.2, 2.5]))
    assert beta == 0.29078287207506

def test_nse_alpha_for_arrays():
    alpha = nse.calc_nse_alpha(obs=np.array([1.5, 1, 0.8, 0.85, 1.5, 2]),
                               sim=np.array([1.6, 1.3, 1, 0.8, 1.2, 2.5]))
    assert alpha == 1.28120574551669

def test_nse_r_for_arrays():
    lin_cor = nse.calc_nse_r(obs=np.array([1.5, 1, 0.8, 0.85, 1.5, 2]),
                             sim=np.array([1.6, 1.3, 1, 0.8, 1.2, 2.5]))
    assert lin_cor == 0.89402818505835

def test_nse_for_equal_arrays():
    sig = nse.calc_nse(obs=np.array([1, 2, 3, 4, 5]),
                       sim=np.array([1, 2, 3, 4, 5]))
    assert sig == 1

def test_nse_simulation_equals_obs_mean():
    sig = nse.calc_nse(obs=np.array([1, 2, 3, 4, 5]),
                       sim=np.array([3, 3, 3, 3, 3]))
    assert sig == 0
