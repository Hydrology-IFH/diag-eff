"""Testing generate_errors module."""

import pytest
import numpy as np
import pandas as pd
from de import generate_errors


def test_negative_dynamic():
    ts_idx = pd.date_range(start="1/1/2019", periods=5)
    arr = np.array([1, 2, 3, 4, 5])
    ts = pd.DataFrame(index=ts_idx, data=arr)
    ts_nd = generate_errors.negative_dynamic(ts)
    assert ts_nd.iloc[0, 0] > 1
    assert ts_nd.iloc[4, 0] < 5


def test_positive_dynamic():
    ts_idx = pd.date_range(start="1/1/2019", periods=5)
    arr = np.array([1, 2, 3, 4, 5])
    ts = pd.DataFrame(index=ts_idx, data=arr)
    ts_pd = generate_errors.positive_dynamic(ts)
    assert ts_pd.iloc[0, 0] < 1
    assert ts_pd.iloc[4, 0] > 5


def test_positive_constant():
    arr = np.array([1, 2, 3, 4, 5])
    arr_pos = generate_errors.constant(arr)
    assert arr_pos[0] > 1
    assert arr_pos[-1] > 5


def test_negative_constant():
    arr = np.array([1, 2, 3, 4, 5])
    arr_pos = generate_errors.constant(arr, offset=0.5)
    assert arr_pos[0] < 1
    assert arr_pos[-1] < 5


def test_timing_shift():
    arr = np.array([1, 2, 3, 4, 5])
    date_rng = pd.date_range(start="1/1/2018", periods=5)
    ts = pd.DataFrame(index=date_rng, columns=["Q"], data=arr)
    ts_tim = generate_errors.timing(ts, shuffle=False)
    assert ts_tim["Q"].values[0] == 3
    assert ts_tim["Q"].values[-1] == 2


def test_timing_shuffling():
    arr = np.array([1, 2, 3, 4, 5])
    date_rng = pd.date_range(start="1/1/2018", periods=5)
    ts = pd.DataFrame(index=date_rng, columns=["Q"], data=arr)
    ts_tim = generate_errors.timing(ts)
    arr = np.array([1, 2, 3, 4, 5])
    diff = np.all(arr == ts_tim["Q"].values)
    assert diff == False
