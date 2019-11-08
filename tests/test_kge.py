"""Testing kge module."""

import pytest
import numpy as np
import pandas as pd
from de import kge

def test_kge_for_equal_arrays():
    sig = kge.calc_kge(obs=np.array([1, 2, 3, 4, 5]), sim=np.array([1, 2, 3, 4, 5]))
    assert sig > 0.99

def test_kge_simulation_equals_obs_mean():
    sig = kge.calc_kge(obs=np.array([1, 2, 3, 4, 5]), sim=np.array([3, 3, 3, 3, 3]))
    assert sig < -0.4

def test_temp_cor_for_equal_arrays():
    temp_cor = kge.calc_temp_cor(obs=np.array([1, 2, 3, 4, 5]), sim=np.array([1, 2, 3, 4, 5]))
    assert temp_cor > 0.99

def test_temp_cor_simulation_equals_obs_mean():
    temp_cor = kge.calc_temp_cor(obs=np.array([1, 2, 3, 4, 5]), sim=np.array([3, 3, 3, 3, 3]))
    assert temp_cor == 0
