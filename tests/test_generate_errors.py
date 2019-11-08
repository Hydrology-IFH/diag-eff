"""Testing generate_errors module."""

import pytest
import numpy as np
import pandas as pd
from de import generate_errors

def test_highunder_lowover():
    ts_idx = pd.date_range(start='1/1/2019', periods=5)
    arr = np.array([1, 2, 3, 4, 5])
    ts = pd.DataFrame(index=ts_idx, data=arr)
    ts_uo = generate_errors.highunder_lowover(ts)
    assert ts_uo.iloc[0, 0] > 1
    assert ts_uo.iloc[4, 0] < 5

def test_highover_lowunder():
    ts_idx = pd.date_range(start='1/1/2019', periods=5)
    arr = np.array([1, 2, 3, 4, 5])
    ts = pd.DataFrame(index=ts_idx, data=arr)
    ts_ou = generate_errors.highover_lowunder(ts)
    assert ts_ou.iloc[0, 0] < 1
    assert ts_ou.iloc[4, 0] > 5

def test_pos_shift_ts():
    arr = np.array([1, 2, 3, 4, 5])
    arr_pos = generate_errors.pos_shift_ts(arr)
    assert arr_pos[0] > 1
    assert arr_pos[-1] > 5

def test_neg_shift_ts():
    arr = np.array([1, 2, 3, 4, 5])
    arr_pos = generate_errors.neg_shift_ts(arr)
    assert arr_pos[0] < 1
    assert arr_pos[-1] < 5
