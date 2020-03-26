# -*- coding: utf-8 -*-

"""
de.misc
~~~~~~~~~~~

:2019 by Robin Schwemmle.
:license: GNU GPLv3, see LICENSE for more details.
"""

import datetime as dt
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
import scipy as sp
import seaborn as sns


def sort_obs(ts):
    """
    Sort time series by observed values.

    Parameters
    ----------
    ts : dataframe
        Dataframe with two time series (e.g. observed and simulated)

    Returns
    ----------
    obs_sort : dataframe
        dataframe with two time series sorted by the observed values
        in ascending order
    """
    df_ts = pd.DataFrame(data=ts)
    obs_sort = df_ts.sort_values(by=['Qobs'], ascending=False)

    return obs_sort

def fdc_obs_sort(Q):
    """
    Plotting the flow duration curves of observed and simulated runoff.
    Descending order of ebserved time series is applied to simulated time
    series.

    Parameters
    ----------
    Q : dataframe
        Containing time series of Qobs and Qsim.
    """
    df_Q = pd.DataFrame(data=Q)
    df_Q_sort = sort_obs(df_Q)

    # calculate exceedence probability
    ranks_obs = sp.stats.rankdata(df_Q_sort['Qobs'], method='ordinal')
    ranks_obs = ranks_obs[::-1]
    prob_obs = [(ranks_obs[i]/(len(df_Q_sort['Qobs'])+1)) for i in range(len(df_Q_sort['Qobs']))]

    fig, ax = plt.subplots()
    ax.plot(prob_obs, df_Q_sort['Qsim'], color='red', label='Simulated')
    ax.plot(prob_obs, df_Q_sort['Qobs'], color='blue', label='Observed')
    ax.set(ylabel=_q_lab,
           xlabel='Exceedence probabilty [-]', yscale='log')
    ax.legend(loc=1)

def calc_de_sort(obs, sim):
    """
    Calculate Diagnostic-Efficiency (DE).

    Simulated values are sorted by order of observed values.

    Parameters
    ----------
    obs : (N,)array_like
        Observed time series as 1-D array

    sim : (N,)array_like
        Simulated time series

    Returns
    ----------
    sig : float
        diagnostic efficiency measure
    """
    # mean relative bias
    brel_mean = de.calc_brel_mean(obs, sim)
    # remaining relative bias
    brel_rest = de.calc_brel_rest(obs, sim)
    # area of relative remaing bias
    b_area = de.calc_bias_area(brel_rest)
    # diagnostic efficiency
    sig = 1 - np.sqrt((brel_mean)**2 + (b_area)**2)

    return sig

def smooth_ts(ts, win=5):
    """
    Underestimate high flows - Overestimate low flows.

    Time series is smoothed by rolling average. Maxima decrease and minima
    decrease.

    Parameters
    ----------
    ts : dataframe
        Time series

    win : int, optional
        Size of window used to apply rolling mean. The default is 5 days.

    Returns
    ----------
    smoothed_ts : series
        Smoothed time series
    """
    smoothed_ts = ts.rolling(window=win).mean()
    smoothed_ts.fillna(method='bfill', inplace=True)

    return smoothed_ts

def _datacheck_peakdetect(x_axis, y_axis):
    """
    Check input data for peak detection.

    Parameters
    ----------
    x_axis : str
        path to file with meta informations on the catchments

    y_axis : str, optional
        delimeter to use. The default is ','.

    Returns
    ----------
    x_axis : dataframe
        imported time series
    """

    if x_axis is None:
        x_axis = range(len(y_axis))

    if len(y_axis) != len(x_axis):
        raise ValueError("Input vectors y_axis and x_axis must have same length")

    #needs to be a numpy array
    y_axis = np.array(y_axis)
    x_axis = np.array(x_axis)

    return x_axis, y_axis

def peakdetect(y_axis, x_axis = None, lookahead=200, delta=0):
    """
    Converted from/based on a MATLAB scripts at:
    http://billauer.co.il/peakdet.html
    https://gist.github.com/sixtenbe/1178136

    Function for detecting local maxima and minima in a signal.
    Discovers peaks by searching for values which are surrounded by lower
    or larger values for maxima and minima respectively

    Parameters
    ----------
    y_axis : array_like
        contains the signal over which to find peaks

    x_axis : array_like, optional
        values correspond to the y_axis list and is used
        in the return to specify the position of the peaks. If omitted an
        index of the y_axis is used.
        (default: None)

    lookahead : int, optional
        distance to look ahead from a peak candidate to determine if
        it is the actual peak
        '(samples / period) / f' where '4 >= f >= 1.25' might be a good value

    delta : int, optional
        this specifies a minimum difference between a peak and
        the following points, before a peak may be considered a peak. Useful
        to hinder the function from picking up false peaks towards to end of
        the signal. It is recommended that delta should be set to
        delta >= RMSnoise * 5.
        When omitted delta function causes a 20% decrease in speed.
        When used correctly it can double the speed of the function.

    Returns
    ----------
    max_peaks : list
        containing the positive peaks. Each cell of the list contains a tuple
        of: (position, peak_value)
        to get the average peak value do: np.mean(max_peaks, 0)[1] on the
        results to unpack one of the lists into x, y coordinates do:
        x, y = zip(*max_peaks)

    min_peaks : list
        containing the negative peaks. Each cell of the list contains a tuple
        of: (position, peak_value)
        to get the average peak value do: np.mean(max_peaks, 0)[1] on the
        results to unpack one of the lists into x, y coordinates do:
        x, y = zip(*max_peaks)
    """
    max_peaks = []
    min_peaks = []
    dump = []  # Used to pop the first hit which is false

    # check input data
    x_axis, y_axis = _datacheck_peakdetect(x_axis, y_axis)
    # store data length for later use
    length = len(y_axis)

    # perform some checks
    if lookahead < 1:
        raise ValueError("Lookahead must be '1' or above in value")
    if not (np.isscalar(delta) and delta >= 0):
        raise ValueError("delta must be a positive number")

    # maxima and minima candidates are temporarily stored in
    # mx and mn respectively
    mn, mx = np.Inf, -np.Inf

    # Only detect peak if there is 'lookahead' amount of points after it
    for index, (x, y) in enumerate(zip(x_axis[:-lookahead], y_axis[:-lookahead])):
        if y > mx:
            mx = y
            mxpos = x
        if y < mn:
            mn = y
            mnpos = x

        #### look for max ####
        if y < mx-delta and mx != np.Inf:
            # Maxima peak candidate found
            # look ahead in signal to ensure that this is a peak and not jitter
            if y_axis[index:index+lookahead].max() < mx:
                max_peaks.append([mxpos, mx])
                dump.append(True)
                #set algorithm to only find minima now
                mx = np.Inf
                mn = np.Inf
                if index+lookahead >= length:
                    # end is within lookahead no more peaks can be found
                    break
                continue
            #else:  #slows shit down this does
            #    mx = ahead
            #    mxpos = x_axis[np.where(y_axis[index:index+lookahead]==mx)]

        #### look for min ####
        if y > mn+delta and mn != -np.Inf:
            # Minima peak candidate found
            # look ahead in signal to ensure that this is a peak and not jitter
            if y_axis[index:index+lookahead].min() > mn:
                min_peaks.append([mnpos, mn])
                dump.append(False)
                # set algorithm to only find maxima now
                mn = -np.Inf
                mx = -np.Inf
                if index+lookahead >= length:
                    #end is within lookahead no more peaks can be found
                    break
            #else:  #slows shit down this does
            #    mn = ahead
            #    mnpos = x_axis[np.where(y_axis[index:index+lookahead]==mn)]


    #Remove the false hit on the first value of the y_axis
    try:
        if dump[0]:
            max_peaks.pop(0)
        else:
            min_peaks.pop(0)
        del dump
    except IndexError:
        # no peaks were found, should the function return empty lists?
        pass

    return max_peaks, min_peaks

def disaggregate_obs(ts, max_peaks_ind, min_peaks_ind):
    """
    Overestimate high flows - Underestimate low flows.

    Increase max values and decrease min values.


    Parameters
    ----------
    ts : dataframe
        Dataframe with time series

    max_peaks_ind : list
        Index where max. peaks occur

    min_peaks_ind : list
        Index where min. peaks occur

    Returns
    ----------
    ts : dataframe
        Disaggregated time series
    """
    idx_max = max_peaks_ind - dt.timedelta(days=1)
    idxu_max = max_peaks_ind.union(idx_max)
    idx1_max = max_peaks_ind - dt.timedelta(days=2)

    idx_min = min_peaks_ind + dt.timedelta(days=1)
    idxu_min = min_peaks_ind.union(idx_min)
    idx1_min = min_peaks_ind + dt.timedelta(days=2)

    ts.loc[idx1_max, 'Qobs'] = ts.loc[idx1_max, 'Qobs'] * 1.1
    ts.loc[idx_max, 'Qobs'] = ts.loc[idx_max, 'Qobs'] * 1.2
    ts.loc[max_peaks_ind, 'Qobs'] = ts.loc[max_peaks_ind, 'Qobs'] * 1.3

    ts.loc[idx1_min, 'Qobs'] = ts.loc[idx1_min, 'Qobs'] * 0.5
    ts.loc[idx_min, 'Qobs'] = ts.loc[idx_min, 'Qobs'] * 0.5
    ts.loc[min_peaks_ind, 'Qobs'] = ts.loc[min_peaks_ind, 'Qobs'] * 0.5

    return ts

def plot_peaks(ts, max_peak_ts, min_peak_ts):
    """
    Plot time series.

    Parameters
    ----------
    ts : dataframe
        dataframe with time series

    max_peak_ts : dataframe
        dataframe with max. peaks of time series

    min_peak_ts : dataframe
        dataframe with min. peaks of time series
    """
    fig, ax = plt.subplots()
    ax.plot(ts.index, ts.iloc[:, 0].values, color='blue')
    ax.plot(max_peak_ts.index, max_peak_ts.iloc[:, 0].values, 'r.')
    ax.plot(min_peak_ts.index, min_peak_ts.iloc[:, 0].values, 'g.')
    ax.set(ylabel=r'[$m^{3}$ $s^{-1}$]',
           xlabel='Time [Days]')

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
