"""Testing nse module."""

import pytest
import numpy as np
import pandas as pd
from de import nse

def test_nse_for_equal_arrays():
    sig = nse.calc_nse(obs=np.array([1, 2, 3, 4, 5]), sim=np.array([1, 2, 3, 4, 5]))
    assert sig == 1

def test_nse_simulation_equals_obs_mean():
    sig = nse.calc_nse(obs=np.array([1, 2, 3, 4, 5]), sim=np.array([3, 3, 3, 3, 3]))
    assert sig == 0
