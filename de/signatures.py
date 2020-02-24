#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
de.signatures
~~~~~~~~~~~
Alternative signatures which can be used for a tailored diagnosis (e.g. low flow
behaviour)
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

def bias_rr(qobs, qsim):
    """Bias in runoff ratio.

    Parameters
    ----------
    qobs : (N,)array_like
        observed runoff time series as 1-D array

    qsim : (N,)array_like
        simulated runoff time series as 1-D array

    Returns
    ----------
    sig : float
        bias in runoff ratio (-)

    References
    ----------
    Yilmaz, K. K., Gupta, H. V., and Wagener, T.: A process-based diagnostic
    approach to model evaluation: Application to the NWS distributed hydrologic
    model, Water Resources Research, 44, 10.1029/2007wr006716, 2008.
    """
    if len(qobs) != len(qsim):
        raise AssertionError("Arrays are not of equal length!")

    sim_obs_diff = np.subtract(qsim, qobs)
    sum_diff = np.sum(sim_obs_diff)
    sum_obs = np.sum(qobs)
    sig = (sum_diff/sum_obs)

    return sig

def bias_hf(qobs, qsim):
    """Bias in high-flow segment of flow duration curve.

    Parameters
    ----------
    qobs : (N,)array_like
        observed runoff time series as 1-D array

    qsim : (N,)array_like
        simulated runoff time series as 1-D array

    Returns
    ----------
    sig : float
        bias in high-flow segment of flow duration curve (-)

    References
    ----------
    Yilmaz, K. K., Gupta, H. V., and Wagener, T.: A process-based diagnostic
    approach to model evaluation: Application to the NWS distributed hydrologic
    model, Water Resources Research, 44, 10.1029/2007wr006716, 2008.
    """
    if len(qobs) != len(qsim):
        raise AssertionError("Arrays are not of equal length!")

    qobs = np.sort(qobs)[::-1]
    qsim = np.sort(qsim)[::-1]

    hf_ind = int(.02 * len(qobs))
    qobs_hf = qobs[0:hf_ind]
    qsim_hf = qsim[0:hf_ind]

    diff_hf = np.subtract(qsim_hf, qobs_hf)
    sum_diff_hf = np.sum(diff_hf)
    sum_obs_hf = np.sum(qobs_hf)
    sig = (sum_diff_hf/sum_obs_hf)

    return sig

def bias_mf(qobs, qsim):
    """Bias in mid-flow segment of flow duration curve.

    Parameters
    ----------
    qobs : (N,)array_like
        observed runoff time series as 1-D array

    qsim : (N,)array_like
        simulated runoff time series as 1-D array

    Returns
    ----------
    sig : float
        bias in mid-flow segment of flow duration curve (-)

    References
    ----------
    Yilmaz, K. K., Gupta, H. V., and Wagener, T.: A process-based diagnostic
    approach to model evaluation: Application to the NWS distributed hydrologic
    model, Water Resources Research, 44, 10.1029/2007wr006716, 2008.
    """
    if len(qobs) != len(qsim):
        raise AssertionError("Arrays are not of equal length!")

    qobs = np.sort(qobs)[::-1]
    qsim = np.sort(qsim)[::-1]

    qobs_mf = np.log(np.median(qobs))
    qsim_mf = np.log(np.median(qsim))
    sig = ((qsim_mf - qobs_mf)/qobs_mf)

    return sig

def bias_lf(qobs, qsim):
    """Bias in low-flow segment of flow duration curve.

    Parameters
    ----------
    qobs : (N,)array_like
        observed runoff time series as 1-D array

    qsim : (N,)array_like
        simulated runoff time series as 1-D array

    Returns
    ----------
    sig : float
        bias in low-flow segment of flow duration curve (-)

    References
    ----------
    Yilmaz, K. K., Gupta, H. V., and Wagener, T.: A process-based diagnostic
    approach to model evaluation: Application to the NWS distributed hydrologic
    model, Water Resources Research, 44, 10.1029/2007wr006716, 2008.
    """
    if len(qobs) != len(qsim):
        raise AssertionError("Arrays are not of equal length!")

    qobs = np.sort(qobs)[::-1]
    qsim = np.sort(qsim)[::-1]

    l = int(.7 * len(qobs))
    L = len(qobs)
    qobs_lf = qobs[l:L]
    qsim_lf = qsim[l:L]

    diff_obs = np.log(qobs_lf) - np.log(qobs[L])
    diff_sim = np.log(qsim_lf) - np.log(qsim[L])

    sum_diff_obs = np.sum(diff_obs)
    sum_diff_sim = np.sum(diff_sim)
    sig = -1 * ((sum_diff_sim - sum_diff_obs)/sum_diff_obs)

    return sig

def bias_t(qobs, qsim, prec):
    """Bias in lag time.

    Parameters
    ----------
    qobs : (N,)array_like
        observed runoff time series as 1-D array

    qsim : (N,)array_like
        simulated runoff time series as 1-D array

    prec : (N,)array_like
        precipitation time series as 1-D array

    Returns
    ----------
    sig : float
        bias in lag time (-)

    References
    ----------
    Yilmaz, K. K., Gupta, H. V., and Wagener, T.: A process-based diagnostic
    approach to model evaluation: Application to the NWS distributed hydrologic
    model, Water Resources Research, 44, 10.1029/2007wr006716, 2008.
    """
    pass
