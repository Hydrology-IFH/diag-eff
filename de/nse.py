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
import scipy as sp

__title__ = 'de'
__version__ = '0.1'
#__build__ = 0x001201
__author__ = 'Robin Schwemmle'
__license__ = 'GNU GPLv3'
#__docformat__ = 'markdown'

#TODO: visualization of decomposed NSE

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

def calc_nse_dec(obs, sim):
    """Calculate the decomposed Nash-Sutcliffe-Efficiency (NSE).

    Parameters
    ----------
    obs : (N,)array_like
        Observed time series as 1-D array

    sim : (N,)array_like
        Simulated time series as 1-D array

    Returns
    ----------
    sig : float
        decomposed Nash-Sutcliffe-Efficiency measure

    Examples
    --------
    Provide arrays with equal length

    >>> from de import de
    >>> import numpy as np
    >>> obs = np.array([1.5, 1, 0.8, 0.85, 1.5, 2])
    >>> sim = np.array([1.6, 1.3, 1, 0.8, 1.2, 2.5])
    >>> de.calc_nse_dec(obs, sim)

    Notes
    ----------
    .. math::

        NSE = 2 \times \alpha \times r - \alpha^2 - \beta_n^2


    References
    ----------
    Gupta, H. V., Kling, H., Yilmaz, K. K., and Martinez, G. F.: Decomposition
    of the mean squared error and NSE performance criteria: Implications for
    improving hydrological modelling, Journal of Hydrology, 377, 80-91,
    10.1016/j.jhydrol.2009.08.003, 2009.
    """
    if len(obs) != len(sim):
        raise AssertionError("Arrays are not of equal length!")
    nse_alpha = calc_nse_alpha(obs, sim)
    nse_beta = calc_nse_beta(obs, sim)
    nse_r = calc_nse_r(obs, sim)
    sig = 2 * nse_alpha * nse_r - nse_alpha**1 - nse_beta**2

    return sig

def calc_nse_beta(obs, sim):
    """Calculate the beta term of decomposed Nash-Sutcliffe-Efficiency (NSE).

    Parameters
    ----------
    obs : (N,)array_like
        Observed time series as 1-D array

    sim : (N,)array_like
        Simulated time series as 1-D array

    Returns
    ----------
    nse_beta : float
        beta term of decomposed Nash-Sutcliffe-Efficiency (NSE)

    Examples
    --------
    Provide arrays with equal length

    >>> from de import de
    >>> import numpy as np
    >>> obs = np.array([1.5, 1, 0.8, 0.85, 1.5, 2])
    >>> sim = np.array([1.6, 1.3, 1, 0.8, 1.2, 2.5])
    >>> de.calc_nse_beta(obs, sim)

    Notes
    ----------
    .. math::

        \beta_{n} = \frac{\mu_{sim} - \mu_{obs}}{\sigma_{obs}}


    References
    ----------
    Gupta, H. V., Kling, H., Yilmaz, K. K., and Martinez, G. F.: Decomposition
    of the mean squared error and NSE performance criteria: Implications for
    improving hydrological modelling, Journal of Hydrology, 377, 80-91,
    10.1016/j.jhydrol.2009.08.003, 2009.
    """
    if len(obs) != len(sim):
        raise AssertionError("Arrays are not of equal length!")
    sim_mean = np.mean(sim)
    obs_mean = np.mean(obs)
    obs_std = np.std(obs)
    nse_beta = (sim_mean - obs_mean)/obs_std

    return nse_beta

def calc_nse_alpha(obs, sim):
    """Calculate the alpha term of decomposed Nash-Sutcliffe-Efficiency (NSE).

    Parameters
    ----------
    obs : (N,)array_like
        Observed time series as 1-D array

    sim : (N,)array_like
        Simulated time series as 1-D array

    Returns
    ----------
    nse_alpha : float
        alpha term of decomposed Nash-Sutcliffe-Efficiency

    Examples
    --------
    Provide arrays with equal length

    >>> from de import de
    >>> import numpy as np
    >>> obs = np.array([1.5, 1, 0.8, 0.85, 1.5, 2])
    >>> sim = np.array([1.6, 1.3, 1, 0.8, 1.2, 2.5])
    >>> de.calc_nse_alpha(obs, sim)

    Notes
    ----------
    .. math::

        \alpha = \frac{\sigma_{sim}}{\sigma_{obs}}

    References
    ----------
    Gupta, H. V., Kling, H., Yilmaz, K. K., and Martinez, G. F.: Decomposition
    of the mean squared error and NSE performance criteria: Implications for
    improving hydrological modelling, Journal of Hydrology, 377, 80-91,
    10.1016/j.jhydrol.2009.08.003, 2009.
    """
    if len(obs) != len(sim):
        raise AssertionError("Arrays are not of equal length!")

    # calculate alpha term
    obs_std = np.std(obs)
    sim_std = np.std(sim)
    nse_alpha = sim_std/obs_std

    return nse_alpha

def calc_nse_r(obs, sim):
    """Calculate linear correlation between observed and simulated
    time series.

    Parameters
    ----------
    obs : (N,)array_like
        Observed time series as 1-D array

    sim : (N,)array_like
        Simulated time series

    Returns
    ----------
    lin_cor : float
        Linear correlation between observed and simulated time series

    Examples
    --------
    Provide arrays with equal length

    >>> from de import de
    >>> import numpy as np
    >>> obs = np.array([1.5, 1, 0.8, 0.85, 1.5, 2])
    >>> sim = np.array([1.6, 1.3, 1, 0.8, 1.2, 2.5])
    >>> de.calc_nse_r(obs, sim)
    0.8940281850583509

    References
    ----------
    Gupta, H. V., Kling, H., Yilmaz, K. K., and Martinez, G. F.: Decomposition
    of the mean squared error and NSE performance criteria: Implications for
    improving hydrological modelling, Journal of Hydrology, 377, 80-91,
    10.1016/j.jhydrol.2009.08.003, 2009.
    """
    if len(obs) != len(sim):
        raise AssertionError("Arrays are not of equal length!")

    r = sp.stats.pearsonr(obs, sim)
    temp_cor = r[0]

    if np.isnan(temp_cor):
        lin_cor = 0

    return lin_cor
