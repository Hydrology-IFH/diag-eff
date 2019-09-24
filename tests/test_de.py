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

def test_nse_for_equal_arrays():
    sig = de.calc_nse(obs=np.array([1, 2, 3, 4, 5]), sim=np.array([1, 2, 3, 4, 5]))
    assert sig == 1

def test_nse_simulation_equals_obs_mean():
    sig = de.calc_nse(obs=np.array([1, 2, 3, 4, 5]), sim=np.array([3, 3, 3, 3, 3]))
    assert sig == 0

def test_kge_for_equal_arrays():
    sig = de.calc_kge(obs=np.array([1, 2, 3, 4, 5]), sim=np.array([1, 2, 3, 4, 5]))
    assert sig > 0.99

def test_kge_simulation_equals_obs_mean():
    sig = de.calc_kge(obs=np.array([1, 2, 3, 4, 5]), sim=np.array([3, 3, 3, 3, 3]))
    assert sig < -0.4

def test_temp_cor_for_equal_arrays():
    temp_cor = de.calc_temp_cor(obs=np.array([1, 2, 3, 4, 5]), sim=np.array([1, 2, 3, 4, 5]))
    assert temp_cor > 0.99

def test_temp_cor_simulation_equals_obs_mean():
    temp_cor = de.calc_temp_cor(obs=np.array([1, 2, 3, 4, 5]), sim=np.array([3, 3, 3, 3, 3]))
    assert temp_cor == 0

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

def test_highunder_lowover():
    ts_idx = pd.date_range(start='1/1/2019', periods=5)
    arr = np.array([1, 2, 3, 4, 5])
    ts = pd.DataFrame(index=ts_idx, data=arr)
    ts_uo = de.highunder_lowover(ts)
    assert ts_uo.iloc[0, 0] > 1
    assert ts_uo.iloc[4, 0] < 5

def test_highover_lowunder():
    ts_idx = pd.date_range(start='1/1/2019', periods=5)
    arr = np.array([1, 2, 3, 4, 5])
    ts = pd.DataFrame(index=ts_idx, data=arr)
    ts_ou = de.highover_lowunder(ts)
    assert ts_ou.iloc[0, 0] < 1
    assert ts_ou.iloc[4, 0] > 5

def test_pos_shift_ts():
    arr = np.array([1, 2, 3, 4, 5])
    arr_pos = de.pos_shift_ts(arr)
    assert arr_pos[0] > 1
    assert arr_pos[-1] > 5

def test_neg_shift_ts():
    arr = np.array([1, 2, 3, 4, 5])
    arr_pos = de.neg_shift_ts(arr)
    assert arr_pos[0] < 1
    assert arr_pos[-1] < 5
