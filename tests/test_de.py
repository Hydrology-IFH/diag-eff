# -*- coding: utf-8 -*-

import pytest
import numpy as np

import de


def test_de_for_equal_arrays():
    sig = de.calc_de(obs=[1, 2, 3], sim=[1, 2, 3])
    assert sig == 1

def test_de_simulation_equals_obs_mean():
    sig = de.calc_de(obs=[1, 2, 3], sim=[2, 2, 2])
    assert sig < -1

def test_nse_for_equal_arrays():
    sig = de.calc_nse(obs=[1, 2, 3], sim=[1, 2, 3])
    assert sig == 1

def test_nse_simulation_equals_obs_mean():
    sig = de.calc_nse(obs=[1, 2, 3], sim=[2, 2, 2])
    assert sig == 0

def test_kge_for_equal_arrays():
    sig = de.calc_kge(obs=[1, 2, 3], sim=[1, 2, 3])
    assert sig == 1

def test_kge_simulation_equals_obs_mean():
    sig = de.calc_kge(obs=[1, 2, 3], sim=[2, 2, 2])
    assert sig < 0
