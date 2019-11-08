#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
de.nse
~~~~~~~~~~~
Nash-Sutcliffe efficiency measure.
:2019 by Robin Schwemmle.
:license: GNU GPLv3, see LICENSE for more details.
"""

import numpy as np
# RunTimeWarning will not be displayed (division by zeros or NaN values)
np.seterr(divide='ignore', invalid='ignore')

__title__ = 'de'
__version__ = '0.1'
#__build__ = 0x001201
__author__ = 'Robin Schwemmle'
__license__ = 'GNU GPLv3'
#__docformat__ = 'markdown'

#TODO: decompose NSE

def calc_nse(obs, sim):
    """Calculate Nash-Sutcliffe-Efficiency (NSE).

    Parameters
    ----------
    obs : (N,)array_like
        Observed time series as 1-D array

    sim : (N,)array_like
        Simulated time series as 1-D array

    Returns
    ----------
    sig : float
        Nash-Sutcliffe-Efficiency measure

    Examples
    --------
    Provide arrays with equal length

    >>> from de import de
    >>> import numpy as np
    >>> obs = np.array([1.5, 1, 0.8, 0.85, 1.5, 2])
    >>> sim = np.array([1.6, 1.3, 1, 0.8, 1.2, 2.5])
    >>> de.calc_nse(obs, sim)
    0.5648252536640361

    Notes
    ----------
    .. math::

        NSE = 1 - \frac{\sum_{t=1}^{t=T} (Q_{sim}(t) - Q_{obs}(t))^2}{\sum_{t=1}^{t=T} (Q_{obs}(t) - \overline{Q_{obs}})^2}


    References
    ----------
    Nash, J. E., and Sutcliffe, J. V.: River flow forecasting through conceptual
    models part I - A discussion of principles, Journal of Hydrology, 10,
    282-290, 10.1016/0022-1694(70)90255-6, 1970.
    """
    if len(obs) != len(sim):
        raise AssertionError("Arrays are not of equal length!")
    sim_obs_diff = np.sum((sim - obs)**2)
    obs_mean = np.mean(obs)
    obs_diff_mean = np.sum((obs - obs_mean)**2)
    sig = 1 - (sim_obs_diff/obs_diff_mean)

    return sig
