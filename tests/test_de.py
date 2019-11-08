"""Testing de module."""

import pytest
import numpy as np
import pandas as pd
from de import de

def test_de_for_equal_arrays():
    sig = de.calc_de(obs=np.array([1, 2, 3, 4, 5]), sim=np.array([1, 2, 3, 4, 5]))
    assert sig > 0.99

def test_de_simulation_equals_obs_mean():
    sig = de.calc_de(obs=np.array([1, 2, 3, 4, 5]), sim=np.array([3, 3, 3, 3, 3]))
    assert sig < -0.2

def test_brel_mean_simulation_equals_obs_mean():
    brel_mean = de.calc_brel_mean(obs=np.array([1, 2, 3, 4, 5]), sim=np.array([3, 3, 3, 3, 3]))
    assert brel_mean > 0.2

def test_brel_mean_for_equal_arrays():
    brel_mean = de.calc_brel_mean(obs=np.array([1, 2, 3]), sim=np.array([1, 2, 3]))
    assert brel_mean == 0

def test_brel_rest_simulation_equals_obs_mean():
    brel_rest = de.calc_brel_rest(obs=np.array([1, 2, 3, 4, 5]), sim=np.array([3, 3, 3, 3, 3]))
    b_area = de.calc_bias_area(brel_rest)
    assert b_area > 0.5

def test_brel_rest_for_equal_arrays():
    brel_rest = de.calc_brel_rest(obs=np.array([1, 2, 3]), sim=np.array([1, 2, 3]))
    b_area = de.calc_bias_area(brel_rest)
    assert b_area == 0

def test_bias_dir_simulation_equals_obs_mean():
    brel_rest = de.calc_brel_rest(obs=np.array([1, 2, 3, 4, 5]), sim=np.array([3, 3, 3, 3, 3]))
    b_dir = de.calc_bias_dir(brel_rest)
    assert b_dir < 0

def test_bias_dir_for_equal_arrays():
    brel_rest = de.calc_brel_rest(obs=np.array([1, 2, 3]), sim=np.array([1, 2, 3]))
    b_dir = b_dir = de.calc_bias_dir(brel_rest)
    assert b_dir == 0

def test_bias_slope_simulation_equals_obs_mean():
    brel_rest = de.calc_brel_rest(obs=np.array([1, 2, 3, 4, 5]), sim=np.array([3, 3, 3, 3, 3]))
    b_area = de.calc_bias_area(brel_rest)
    b_dir = de.calc_bias_dir(brel_rest)
    b_slope = de.calc_bias_slope(b_area, b_dir)
    assert b_slope > 0.5

def test_bias_slope_for_equal_arrays():
    brel_rest = de.calc_brel_rest(obs=np.array([1, 2, 3]), sim=np.array([1, 2, 3]))
    b_area = de.calc_bias_area(brel_rest)
    b_dir = b_dir = de.calc_bias_dir(brel_rest)
    b_slope = de.calc_bias_slope(b_area, b_dir)
    assert b_slope == 0

def test_arctan():
    diag = np.arctan2(0, 0)
    assert diag == 0

def test_fdc_sort():
    arr = np.array([1, 2, 3, 4, 5])
    arr_sort = np.sort(arr)
    assert arr_sort[0] < arr_sort[-1]
