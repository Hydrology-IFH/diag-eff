# -*- coding: utf-8 -*-

"""
de.generate_errors
~~~~~~~~~~~
Mimicking three different error types:
- dynamic errors
- constant erros
- timing errors
:2019 by Robin Schwemmle.
:license: GNU GPLv3, see LICENSE for more details.
"""

import numpy as np
# RunTimeWarning will not be displayed (division by zeros or NaN values)
np.seterr(divide='ignore', invalid='ignore')
import pandas as pd

__title__ = 'de'
__version__ = '0.1'
#__build__ = 0x001201
__author__ = 'Robin Schwemmle'
__license__ = 'GNU GPLv3'
#__docformat__ = 'markdown'

def negative_dynamic(ts, prop=0.5):
    """Generate negative dynamic error (i.e Underestimate high flows -
    Overestimate low flows)

    High to medium flows are decreased by linear
    increasing factors. Medium to low flows are increased by linear
    increasing factors.

    Parameters
    ----------
    ts : dataframe
        Observed time series

    prop : float, optional
        Factor by which time series is tilted.

    Returns
    ----------
    ts_smoothed : dataframe
        Smoothed time series
    """
    obs_sim = pd.DataFrame(index=ts.index, columns=['Qobs', 'Qsim'])
    obs_sim.iloc[:, 0] = ts.iloc[:, 0]
    # sort values by descending order
    obs_sort = obs_sim.sort_values(by='Qobs', ascending=False)
    nn = len(obs_sim.index)
    # factors to decrease/increase runoff
    downup = np.linspace(1.0-prop, 1.0+prop, nn)
    # tilting the fdc at median
    obs_sort.iloc[:, 1] = np.multiply(obs_sort.iloc[:, 0].values, downup)
    # sort by index
    obs_sim = obs_sort.sort_index()
    ts_smoothed = obs_sim.iloc[:, 1].copy().to_frame()

    return ts_smoothed

def positive_dynamic(ts, prop=0.5):
    """Generate positive dynamic errors (i.e. Overestimate high flows -
    Underestimate low flows)

    High to medium flows are increased by linear
    decreasing factors. Medium to low flows are decreased by linear
    decreasing factors.

    Parameters
    ----------
    ts : dataframe
        Dataframe with time series

    prop : float, optional
        Factor by which time series is tilted.

    Returns
    ----------
    ts_disagg : dataframe
        disaggregated time series
    """
    obs_sim = pd.DataFrame(index=ts.index, columns=['Qobs', 'Qsim'])
    obs_sim.iloc[:, 0] = ts.iloc[:, 0]
    # sort values by descending order
    obs_sort = obs_sim.sort_values(by='Qobs', ascending=False)
    nn = len(obs_sim.index)
    # factors to decrease/increase runoff
    updown = np.linspace(1.0+prop, 1.0-prop, nn)
    # tilting the fdc at median
    obs_sort.iloc[:, 1] = np.multiply(obs_sort.iloc[:, 0].values, updown)
    # sort by index
    obs_sim = obs_sort.sort_index()
    ts_disagg = obs_sim.iloc[:, 1].copy().to_frame()

    return ts_disagg

def constant(ts, offset=1.5):
    """Generate constant errors.

    Constant overestimation/underestimation.

    Mimicking constant errors by multiplying/adding with constant offset.

    Parameters
    ----------
    ts : (N,)array_like
        Observed time series

    offset : float, optional
        Offset multiplied to time series. If greater than 1 positive constant offset
        and if less than 1 negative constant offset.

    Returns
    ----------
    shift_ts : array_like
        Time series with positve offset
    """
    shift_ts = ts * offset

    return shift_ts

def timing(ts, tshift=3, random=True):
    """Mimicking timing errors.

    Parameters
    ----------
    ts : dataframe
        dataframe with time series

    tshift : int, optional
        days by which time series is shifted. Both positive and negative
        time shift are possible. The default is 3 days.

    random : boolean, optional
        If True, time series is shuffled. The feault is shuffling.

    Returns
    ----------
    ts_shift : dataframe
        disaggregated time series
    """
    if not random:
        ts_shift = ts.shift(periods=tshift, fill_value=0)
        if tshift > 0:
            ts_shift.iloc[:tshift, 0] = ts.iloc[:, 0].values[-tshift:]

        elif tshift < 0:
            ts_shift.iloc[tshift:, 0] = ts.iloc[:, 0].values[:-tshift]

    if random:
        ts_shift = ts
        y = ts_shift.iloc[:, 0].values
        np.random.shuffle(y)
        ts_shift.iloc[:, 0] = y

    return ts_shift

def linear_bm(ts):
    """Generate a linear flow duration as a benchmark.

    Parameters
    ----------
    ts : dataframe
        Observed time series

    Returns
    ----------
    ts_lin: dataframe
        Smoothed time series
    """
    obs_sim = pd.DataFrame(index=ts.index, columns=['Qobs', 'Qsim'])
    obs_sim.iloc[:, 0] = ts.iloc[:, 0]
    # sort values by descending order
    obs_sort = obs_sim.sort_values(by='Qobs', ascending=False)
    nn = len(obs_sim.index)
    qmax = np.max(obs_sim.iloc[:, 0].values)
    qmin = np.min(obs_sim.iloc[:, 0].values)
    # linearly interpolated array
    lin_arr = np.linspace(qmin, qmax, nn)
    obs_sort.iloc[:, 1] = lin_arr
    # sort by index
    obs_sim = obs_sort.sort_index()
    ts_lin = obs_sim.iloc[:, 1].copy().to_frame()

    return ts_lin
