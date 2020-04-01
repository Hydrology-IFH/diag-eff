"""Testing de module."""

import os
import sys

import pytest
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from de import de

baseline_dir = "images"

baseline_dir_single = os.path.join(baseline_dir, "de/single")
baseline_dir_multi = os.path.join(baseline_dir, "de/multi")

WIN = sys.platform.startswith("win")

# In some cases, the fonts on Windows can be quite different
DEFAULT_TOLERANCE = 10 if WIN else 2


def test_de_for_arrays():
    eff = de.calc_de(
        obs=np.array([1.5, 1, 0.8, 0.85, 1.5, 2]),
        sim=np.array([1.6, 1.3, 1, 0.8, 1.2, 2.5]),
    )
    assert eff == pytest.approx(0.8202204384691575, rel=1e-4)


def test_lin_cor_for_arrays():
    r = de.calc_temp_cor(
        obs=np.array([1.5, 1, 0.8, 0.85, 1.5, 2]),
        sim=np.array([1.6, 1.3, 1, 0.8, 1.2, 2.5]),
    )
    assert r == pytest.approx(0.8940281850583509, rel=1e-4)


def test_nonlin_cor_for_arrays():
    r = de.calc_temp_cor(
        obs=np.array([1.5, 1, 0.8, 0.85, 1.5, 2]),
        sim=np.array([1.6, 1.3, 1, 0.8, 1.2, 2.5]),
        r="spearman",
    )
    assert r == pytest.approx(0.8406680016960504, rel=1e-4)


def test_bias_area_for_arrays():
    brel_res = de.calc_brel_res(
        obs=np.array([1.5, 1, 0.8, 0.85, 1.5, 2]),
        sim=np.array([1.6, 1.3, 1, 0.8, 1.2, 2.5]),
    )
    b_area = de.calc_bias_area(brel_res)
    assert b_area == pytest.approx(0.1112908496732026, rel=1e-4)


def test_bias_dir_for_arrays():
    brel_res = de.calc_brel_res(
        obs=np.array([1.5, 1, 0.8, 0.85, 1.5, 2]),
        sim=np.array([1.6, 1.3, 1, 0.8, 1.2, 2.5]),
    )
    b_dir = b_dir = de.calc_bias_dir(brel_res)
    assert b_dir == pytest.approx(-0.014705882352941155, rel=1e-4)


def test_bias_slope_for_arrays():
    brel_res = de.calc_brel_res(
        obs=np.array([1.5, 1, 0.8, 0.85, 1.5, 2]),
        sim=np.array([1.6, 1.3, 1, 0.8, 1.2, 2.5]),
    )
    b_area = de.calc_bias_area(brel_res)
    b_dir = de.calc_bias_dir(brel_res)
    b_slope = de.calc_bias_slope(b_area, b_dir)
    assert b_slope == pytest.approx(0.1112908496732026, rel=1e-4)


def test_brel_mean_for_arrays():
    brel_mean = de.calc_brel_mean(
        obs=np.array([1.5, 1, 0.8, 0.85, 1.5, 2]),
        sim=np.array([1.6, 1.3, 1, 0.8, 1.2, 2.5]),
    )
    assert brel_mean == pytest.approx(0.09330065359477124, rel=1e-4)


def test_brel_mean_simulation_equals_obs_mean():
    brel_mean = de.calc_brel_mean(
        obs=np.array([1, 2, 3, 4, 5]), sim=np.array([3, 3, 3, 3, 3])
    )
    assert brel_mean > 0.2


def test_brel_mean_for_equal_arrays():
    brel_mean = de.calc_brel_mean(obs=np.array([1, 2, 3]), sim=np.array([1, 2, 3]))
    assert brel_mean == 0


def test_brel_rest_simulation_equals_obs_mean():
    brel_res = de.calc_brel_res(
        obs=np.array([1, 2, 3, 4, 5]), sim=np.array([3, 3, 3, 3, 3])
    )
    b_area = de.calc_bias_area(brel_res)
    assert b_area == pytest.approx(0.5116666666666666, rel=1e-4)


def test_brel_rest_for_equal_arrays():
    brel_res = de.calc_brel_res(obs=np.array([1, 2, 3]), sim=np.array([1, 2, 3]))
    b_area = de.calc_bias_area(brel_res)
    assert b_area == 0


def test_bias_dir_simulation_equals_obs_mean():
    brel_res = de.calc_brel_res(
        obs=np.array([1, 2, 3, 4, 5]), sim=np.array([3, 3, 3, 3, 3])
    )
    b_dir = de.calc_bias_dir(brel_res)
    assert b_dir < 0


def test_bias_dir_for_equal_arrays():
    brel_res = de.calc_brel_res(obs=np.array([1, 2, 3]), sim=np.array([1, 2, 3]))
    b_dir = b_dir = de.calc_bias_dir(brel_res)
    assert b_dir == 0


def test_bias_slope_simulation_equals_obs_mean():
    brel_res = de.calc_brel_res(
        obs=np.array([1, 2, 3, 4, 5]), sim=np.array([3, 3, 3, 3, 3])
    )
    b_area = de.calc_bias_area(brel_res)
    b_dir = de.calc_bias_dir(brel_res)
    b_slope = de.calc_bias_slope(b_area, b_dir)
    assert b_slope > 0.5


def test_bias_slope_for_equal_arrays():
    brel_res = de.calc_brel_res(obs=np.array([1, 2, 3]), sim=np.array([1, 2, 3]))
    b_area = de.calc_bias_area(brel_res)
    b_dir = b_dir = de.calc_bias_dir(brel_res)
    b_slope = de.calc_bias_slope(b_area, b_dir)
    assert b_slope == 0


def test_arctan():
    phi = np.arctan2(0, 0)
    assert phi == 0


def test_fdc_sort():
    arr = np.array([1, 2, 3, 4, 5])
    arr_sort = np.sort(arr)
    assert arr_sort[0] < arr_sort[-1]


@pytest.mark.mpl_image_compare(
    baseline_dir=baseline_dir_single, tolerance=DEFAULT_TOLERANCE
)
def test_single_diag_polar_plot():
    obs = np.array([1.5, 1, 0.8, 0.85, 1.5, 2])
    sim = np.array([1.6, 1.3, 1, 0.8, 1.2, 2.5])
    fig = de.diag_polar_plot(obs, sim)
    return fig


@pytest.mark.mpl_image_compare(
    baseline_dir=baseline_dir_multi, tolerance=DEFAULT_TOLERANCE
)
def test_multi_diag_polar_plot():
    brel_mean = np.array([0.1, 0.15, 0.2, 0.1, 0.05, 0.15])
    b_area = np.array([0.15, 0.1, 0.2, 0.1, 0.1, 0.2])
    temp_cor = np.array([0.9, 0.85, 0.8, 0.9, 0.85, 0.9])
    eff_de = np.array([0.79, 0.76, 0.65, 0.82, 0.81, 0.73])
    b_dir = np.array([0.08, 0.05, 0.1, 0.05, 0.05, 0.1])
    phi = np.array([0.58, 0.98, 0.78, 0.78, 0.46, 0.64])
    fig = de.diag_polar_plot_multi(brel_mean, b_area, temp_cor, eff_de, b_dir, phi)
    return fig
