#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
de.util
~~~~~~~~~~~
Diagnosing model performance using an efficiency measure based on flow
duration curve and temoral correlation. The efficiency measure can be
visualized in 2D-Plot which facilitates decomposing potential error origins
(model errors vs. input data erros)
:2019 by Robin Schwemmle.
:license: GNU GPLv3, see LICENSE for more details.
"""

import datetime as dt
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.transforms as mtransforms
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import scipy as sp
import seaborn as sns
# controlling figure aesthetics
sns.set_style('ticks', {'xtick.major.size': 8, 'ytick.major.size': 8})
import datetime as dt
from de import de

__title__ = 'de'
__version__ = '0.1'
#__build__ = 0x001201
__author__ = 'Robin Schwemmle'
__license__ = 'GNU GPLv3'
#__docformat__ = 'markdown'

_mmd = r'[mm $d^{-1}$]'
_m3s = r'[$m^{3}$ $s^{-1}$]'
_q_lab = _mmd
_sim_lab = 'Manipulated'

def import_ts(path, sep=','):
    """
    Import .csv-file with streamflow time series (m3/s).

    Parameters
    ----------
    path : str
        Path to .csv-file which contains time series

    sep : str, optional
        Delimeter to use. The default is ‘,’.

    Returns
    ----------
    df_ts : dataframe
        Imported time series
    """
    df_ts = pd.read_csv(path, sep=sep, na_values=-9999, parse_dates=True,
                        index_col=0, dayfirst=True)
    # drop nan values
    df_ts = df_ts.dropna()

    return df_ts

def import_camels_ts(path, sep=r"\s+", catch_area=None):
    """
    Import .txt-file with streamflow time series from CAMELS dataset (cubic feet
    per second).

    Parameters
    ----------
    path : str
        Path to .csv-file which contains time series

    sep : str, optional
        Delimeter to use. The default is r"\s+".

    catch_area : float, optional
        catchment area in km2 to convert runoff to mm/day

    Returns
    ----------
    df_ts : dataframe
        Imported time series in m3/s
    """
    if catch_area is None:
        raise ValueError("Value for catchment area is missing")
    df_ts = pd.read_csv(path, sep=sep, na_values=-999, header=None,
                        parse_dates=[[1, 2, 3]], index_col=0)
    df_ts.drop(df_ts.columns[[0, 2]], axis=1, inplace=True)
    df_ts.columns = ['Qobs']
    df_ts = df_ts.dropna()

    # convert to m3/s
    df_ts['Qobs'] = df_ts['Qobs'].values/35.3
    # convert to mm/day
    df_ts['Qobs'] = (df_ts['Qobs'].values*(24*60*60)*1000) / (catch_area*1000*1000)

    return df_ts

def import_camels_obs_sim(path, sep=r"\s+"):
    """
    Import .txt-file containing observed and simulated streamflow time series
    from CAMELS dataset (cubic feet per second).

    Parameters
    ----------
    path : str
        Path to .csv-file which contains time series

    sep : str, optional
        Delimeter to use. The default is ‘,’.

    Returns
    ----------
    obs_sim : dataframe
        observed and simulated time series in m3/s
    """
    obs_sim = pd.read_csv(path, sep=sep, na_values=-999, header=0, parse_dates=[[0, 1, 2]], index_col=0)
    obs_sim.drop(['HR', 'SWE', 'PRCP', 'RAIM', 'TAIR', 'PET', 'ET'], axis=1, inplace=True)
    obs_sim.columns = ['Qsim', 'Qobs']
    obs_sim = obs_sim[['Qobs', 'Qsim']]
    obs_sim = obs_sim.dropna()

    return obs_sim

def plot_ts(ts):
    """Plot time series.

    Parameters
    ----------
    ts : dataframe
        Dataframe which contains time series
    """
    fig, ax = plt.subplots()
    ax.plot(ts.index, ts.iloc[:, 0].values, color='blue')
    ax.set(ylabel=_q_lab,
           xlabel='Time [Years]')
    ax.set_ylim(0, )
    ax.set_xlim(ts.index[0], ts.index[-1])
    years_5 = mdates.YearLocator(5)
    years = mdates.YearLocator()
    yearsFmt = mdates.DateFormatter('%Y')
    ax.xaxis.set_major_locator(years_5)
    ax.xaxis.set_major_formatter(yearsFmt)
    ax.xaxis.set_minor_locator(years)

def plot_obs_sim(obs, sim):
    """Plot observed and simulated time series.

    Parameters
    ----------
    obs : series
        observed time series

    sim : series
        simulated time series
    """
    fig, ax = plt.subplots()
    ax.plot(obs.index, obs, lw=2, color='blue')  # observed time series
    # simulated time series
    ax.plot(sim.index, sim, lw=1, ls='-.', color='red', alpha=.8)
    ax.set(ylabel=_q_lab, xlabel='Time')
    ax.set_ylim(0, )
    ax.set_xlim(obs.index[0], obs.index[-1])

def fdc_obs_sim(obs, sim):
    """Plotting the flow duration curves of two hydrologic time series (e.g.
    observed streamflow and simulated streamflow).

    Parameters
    ----------
    obs : series
        observed time series
    sim : series
        simulated time series
    """
    obs_sim = pd.DataFrame(index=obs.index, columns=['obs', 'sim'])
    obs_sim.loc[:, 'obs'] = obs.values
    obs_sim.loc[:, 'sim'] = sim.values
    obs = obs_sim.sort_values(by=['obs'], ascending=True)
    sim = obs_sim.sort_values(by=['sim'], ascending=True)

    # calculate exceedence probability
    ranks_obs = sp.stats.rankdata(obs['obs'], method='ordinal')
    ranks_obs = ranks_obs[::-1]
    prob_obs = [(ranks_obs[i]/(len(obs['obs'])+1)) for i in range(len(obs['obs']))]

    ranks_sim = sp.stats.rankdata(sim['sim'], method='ordinal')
    ranks_sim = ranks_sim[::-1]
    prob_sim = [(ranks_sim[i]/(len(sim['sim'])+1)) for i in range(len(sim['sim']))]

    fig, ax = plt.subplots()
    ax.plot(prob_obs, obs['obs'], lw=2, color='blue', alpha=.5, label='Observed')
    ax.plot(prob_sim, sim['sim'], lw=2, ls='-.', color='red', label='Manipulated')
    ax.set(yscale='log', ylabel=_q_lab, xlabel='Exceedence probabilty [-]')
    ax.set_ylim(0, )
    ax.set_xlim(0, 1)

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

def fdc_obs_sim_ax(obs, sim, ax, fig_num):
    """Plotting the flow duration curves of two hydrologic time series (e.g.
    observed streamflow and simulated streamflow).

    Parameters
    ----------
    obs : series
        observed time series
    sim : series
        simulated time series
    ax : axes
        Axes object to draw the plot onto
    fig_num : string
        string object for figure caption
    """
    obs_sim = pd.DataFrame(index=obs.index, columns=['obs', 'sim'])
    obs_sim.loc[:, 'obs'] = obs.values
    obs_sim.loc[:, 'sim'] = sim.values
    obs = obs_sim.sort_values(by=['obs'], ascending=True)
    sim = obs_sim.sort_values(by=['sim'], ascending=True)

    # calculate exceedence probability
    ranks_obs = sp.stats.rankdata(obs['obs'], method='ordinal')
    ranks_obs = ranks_obs[::-1]
    prob_obs = [(ranks_obs[i]/(len(obs['obs'])+1)) for i in range(len(obs['obs']))]

    ranks_sim = sp.stats.rankdata(sim['sim'], method='ordinal')
    ranks_sim = ranks_sim[::-1]
    prob_sim = [(ranks_sim[i]/(len(sim['sim'])+1)) for i in range(len(sim['sim']))]

    ax.plot(prob_obs, obs['obs'], lw=2, color='blue', alpha=.5, label='Observed')
    ax.plot(prob_sim, sim['sim'], lw=2, ls='-.', color='red', label='Manipulated')
    ax.text(.96, .95, fig_num, transform=ax.transAxes, ha='right', va='top')
    ax.set(yscale='log')
    ax.set_ylim(0, )
    ax.set_xlim(0, 1)

def plot_obs_sim_ax(obs, sim, ax, fig_num):
    """Plot observed and simulated time series.

    Parameters
    ----------
    obs : series
        observed time series
    sim : series
        simulated time series
    ax : axes
        Axes object to draw the plot onto
    fig_num : string
        string object for figure caption
    """
    # observed time series
    ax.plot(obs.index, obs, lw=2, color='blue', label='Observed')
    ax.plot(sim.index, sim, lw=1, ls='-.', color='red', alpha=.8,
            label='Manipulated')  # simulated time series
    ax.set_ylim(0, )
    ax.set_xlim(obs.index[0], obs.index[-1])
    ax.text(.96, .95, fig_num, transform=ax.transAxes, ha='right', va='top')
    # format the ticks
    years_20 = mdates.YearLocator(20)
    years_5 = mdates.YearLocator(5)
    yearsFmt = mdates.DateFormatter('%Y')
    ax.xaxis.set_major_locator(years_20)
    ax.xaxis.set_major_formatter(yearsFmt)
    ax.xaxis.set_minor_locator(years_5)

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

def vis2d_de_multi_fc(brel_mean, b_area, temp_cor, sig_de, b_dir, diag, fc,
                      l=0.05, ax_lim=-.6):
    """Multiple polar plot of Diagnostic-Efficiency (DE)

    Parameters
    ----------
    brel_mean : (N,)array_like
        relative mean bias as 1-D array

    b_area : (N,)array_like
        bias area as 1-D array

    temp_cor : (N,)array_like
        temporal correlation as 1-D array

    sig_de : (N,)array_like
        diagnostic efficiency as 1-D array

    b_dir : (N,)array_like
        direction of bias as 1-D array

    diag : (N,)array_like
        angle as 1-D array

    fc : list
        figure captions

    l : float, optional
        Threshold for which diagnosis can be made. The default is 0.05.

    Notes
    ----------
    .. math::

        \varphi = arctan2(\overline{B_{rel}}, B_{slope})
    """
    sig_min = np.min(sig_de)

    ll_brel_mean = brel_mean.tolist()
    ll_b_dir = b_dir.tolist()
    ll_b_area = b_area.tolist()
    ll_sig = sig_de.tolist()
    ll_diag = diag.tolist()
    ll_temp_cor = temp_cor.tolist()

    # convert temporal correlation to color
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1.0)

    delta = 0.01  # for spacing

    # determine axis limits
    yy = np.arange(ax_lim, 1.01, delta)[::-1]
    c_levels = np.arange(ax_lim, 1, .2)

    len_yy = 360
    # len_yy1 = 90

    # arrays to plot contour lines of DE
    xx = np.radians(np.linspace(0, 360, len_yy))
    theta, r = np.meshgrid(xx, yy)
    #
    # # arrays to plot contours of positive constant offset
    # xx1 = np.radians(np.linspace(45, 135, len_yy1))
    # theta1, r1 = np.meshgrid(xx1, yy)
    #
    # # arrays to plot contours of negative constant offset
    # xx2 = np.radians(np.linspace(225, 315, len_yy1))
    # theta2, r2 = np.meshgrid(xx2, yy)
    #
    # # arrays to plot contours of model errors
    # xx3 = np.radians(np.linspace(135, 225, len_yy1))
    # theta3, r3 = np.meshgrid(xx3, yy)
    #
    # # arrays to plot contours of model errors
    # len_yy2 = int(len_yy1/2)
    # if len_yy != len_yy2 + len_yy2:
    #     xx0 = np.radians(np.linspace(0, 45, len_yy2+1))
    # else:
    #     xx0 = np.radians(np.linspace(0, 45, len_yy2))
    #
    # xx360 = np.radians(np.linspace(315, 360, len_yy2))
    # xx4 = np.concatenate((xx360, xx0), axis=None)
    # theta4, r4 = np.meshgrid(xx4, yy)
    #
    # diagnostic polar plot
    fig, ax = plt.subplots(figsize=(6, 6),
                           subplot_kw=dict(projection='polar'),
                           constrained_layout=True)
    # dummie plot for colorbar of temporal correlation
    cs = np.arange(0, 1.1, 0.1)
    dummie_cax = ax.scatter(cs, cs, c=cs, cmap='plasma_r')
    # Clear axis
    ax.cla()
    # # contours positive constant offset
    # cpio = ax.contourf(theta1, r1, r1, cmap='Purples_r', alpha=.2, edgecolors=None, levels=c_levels, zorder=0)
    # # contours negative constant offset
    # cpiu = ax.contourf(theta2, r2, r2, cmap='Purples_r', alpha=.2, edgecolors=None, levels=c_levels, zorder=0)
    # # contours model errors
    # cpmou = ax.contourf(theta3, r3, r3, cmap='Greys_r', alpha=.2, edgecolors=None, levels=c_levels, zorder=0)
    # cpmuo = ax.contourf(theta4, r4, r4, cmap='Greys_r', alpha=.2, edgecolors=None, levels=c_levels, zorder=0)
    # plot regions
    ax.plot((1, np.deg2rad(45)), (1, np.min(yy)), color='lightgray', linewidth=1.5, ls='--', zorder=0)
    ax.plot((1, np.deg2rad(135)), (1, np.min(yy)), color='lightgray', linewidth=1.5, ls='--', zorder=0)
    ax.plot((1, np.deg2rad(225)), (1, np.min(yy)), color='lightgray', linewidth=1.5, ls='--', zorder=0)
    ax.plot((1, np.deg2rad(315)), (1, np.min(yy)), color='lightgray', linewidth=1.5, ls='--', zorder=0)
    # contours of DE
    cp = ax.contour(theta, r, r, colors='darkgray', levels=c_levels, zorder=1)
    cl = ax.clabel(cp, inline=False, fontsize=10, fmt='%1.1f',
                   colors='dimgrey')
    # threshold efficiency for FBM
    sig_l = 1 - np.sqrt((l)**2 + (l)**2 + (l)**2)
    # loop over each data point
    for (bm, bd, ba, r, sig, ang, txt) in zip(ll_brel_mean, ll_b_dir, ll_b_area, ll_temp_cor, ll_sig, ll_diag, fc):
        # slope of bias
        b_slope = de.calc_bias_slope(ba, bd)
        # convert temporal correlation to color
        rgba_color = cm.plasma_r(norm(r))
        # relation of b_dir which explains the error
        if abs(ba) > 0:
            exp_err = (abs(bd) * 2)/abs(ba)
        elif abs(ba) == 0:
            exp_err = 0

        # diagnose the error
        if abs(bm) <= l and exp_err > l and sig <= sig_l:
            c = ax.scatter(ang, sig, s=75, color=rgba_color, zorder=3)
            d = ax.scatter(ang, sig, color='black', marker='.', zorder=4)
            ax.annotate(txt, xy=(ang, sig), color='black', fontsize=13,
                        xytext=(-8, 0), textcoords="offset points",
                        ha='center', va='center')
        elif abs(bm) > l and exp_err <= l and sig <= sig_l:
            c = ax.scatter(ang, sig, s=75, color=rgba_color, zorder=3)
            d = ax.scatter(ang, sig, color='black', marker='.', zorder=4)
            ax.annotate(txt, xy=(ang, sig), color='black', fontsize=13,
                        xytext=(-8, 0), textcoords="offset points",
                        ha='center', va='center')
        elif abs(bm) > l and exp_err > l and sig <= sig_l:
            c = ax.scatter(ang, sig, s=75, color=rgba_color, zorder=3)
            d = ax.scatter(ang, sig, color='black', marker='.', zorder=4)
            ax.annotate(txt, xy=(ang, sig), color='black', fontsize=13,
                        xytext=(-8, 0), textcoords="offset points",
                        ha='center', va='center')
        # FBM
        elif abs(bm) <= l and exp_err <= l and sig <= sig_l:
            ax.annotate("", xytext=(0, 1), xy=(0, sig),
                        arrowprops=dict(edgecolor=rgba_color, facecolor='black', lw=3), zorder=2)
            ax.annotate("", xytext=(0, 1), xy=(np.pi, sig),
                        arrowprops=dict(edgecolor=rgba_color, facecolor='black', lw=3), zorder=2)
            ax.annotate(txt, xy=(ang, sig), color='black', fontsize=13,
                        xytext=(-8, 0), textcoords="offset points",
                        ha='center', va='center')
        # FGM
        elif abs(bm) <= l and exp_err <= l and sig > sig_l:
            c = ax.scatter(ang, sig, s=75, color=rgba_color, zorder=3)
            d = ax.scatter(ang, sig, color='black', marker='.', zorder=4)
            ax.annotate(txt, xy=(ang, sig), color='black', fontsize=13,
                        xytext=(-6, 0), textcoords="offset points",
                        ha='center', va='center')
    ax.set_rticks([])  # turn default ticks off
    ax.set_rmin(1)
    ax.set_rmax(ax_lim)
    ax.tick_params(labelleft=False, labelright=False, labeltop=False,
                  labelbottom=True, grid_alpha=.01)  # turn labels and grid off
    ax.set_xticklabels(['', '', '', '', '', '', '', ''])
    ax.text(-.14, 0.5, 'High flow overestimation -',
            va='center', ha='center', rotation=90, rotation_mode='anchor',
            transform=ax.transAxes)
    ax.text(-.09, 0.5, 'Low flow underestimation',
            va='center', ha='center', rotation=90, rotation_mode='anchor',
            transform=ax.transAxes)
    ax.text(-.04, 0.5, r'$B_{slope}$ < 0',
            va='center', ha='center', rotation=90, rotation_mode='anchor',
            transform=ax.transAxes)
    ax.text(1.14, 0.5, 'High flow underestimation -',
            va='center', ha='center', rotation=90, rotation_mode='anchor',
            transform=ax.transAxes)
    ax.text(1.09, 0.5, 'Low flow overestimation',
            va='center', ha='center', rotation=90, rotation_mode='anchor',
            transform=ax.transAxes)
    ax.text(1.04, 0.5, r'$B_{slope}$ > 0',
            va='center', ha='center', rotation=90, rotation_mode='anchor',
            transform=ax.transAxes)
    ax.text(.5, -.09, 'Constant negative offset', va='center', ha='center',
            rotation=0, rotation_mode='anchor', transform=ax.transAxes)
    ax.text(.5, -.04, r'$\overline{B_{rel}}$ < 0', va='center', ha='center',
            rotation=0, rotation_mode='anchor', transform=ax.transAxes)
    ax.text(.5, 1.09, 'Constant positive offset',
            va='center', ha='center', rotation=0, rotation_mode='anchor',
            transform=ax.transAxes)
    ax.text(.5, 1.04, r'$\overline{B_{rel}}$ > 0',
            va='center', ha='center', rotation=0, rotation_mode='anchor',
            transform=ax.transAxes)
    # add colorbar for temporal correlation
    cbar = fig.colorbar(dummie_cax, ax=ax, orientation='horizontal',
                        ticks=[1, 0.5, 0], shrink=0.8)
    cbar.set_label(r'r [-]', fontsize=12, labelpad=8)
    cbar.set_ticklabels(['1', '0.5', '<0'])
    cbar.ax.tick_params(direction='in', labelsize=10)

    return fig

def vis2d_deb_multi_fc(brel_mean, b_area, temp_cor, sig_de, sig_de_bench, b_dir, diag, fc,
                      l=0.05):
    """Multiple polar plot of benchmarked Diagnostic-Efficiency (DEB)

    Parameters
    ----------
    brel_mean : (N,)array_like
        relative mean bias as 1-D array

    b_area : (N,)array_like
        bias area as 1-D array

    temp_cor : (N,)array_like
        temporal correlation as 1-D array

    sig_de : (N,)array_like
        diagnostic efficiency as 1-D array

    sig_de_bench : (N,)array_like
        benchmark of diagnostic efficiency as 1-D array

    b_dir : (N,)array_like
        direction of bias as 1-D array

    diag : (N,)array_like
        angle as 1-D array

    fc : list
        figure captions

    l : float, optional
        Threshold for which diagnosis can be made. The default is 0.05.

    Notes
    ----------
    .. math::

        \varphi = arctan2(\overline{B_{rel}}, B_{slope})
    """
    sig_min = np.min(sig_de)

    ll_brel_mean = brel_mean.tolist()
    ll_b_dir = b_dir.tolist()
    ll_b_area = b_area.tolist()
    ll_sig = sig_de.tolist()
    ll_bench = sig_de_bench.tolist()
    ll_diag = diag.tolist()
    ll_temp_cor = temp_cor.tolist()

    # convert temporal correlation to color
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1.0)

    delta = 0.01  # for spacing

    # determine axis limits
    if sig_min > 0:
        ax_lim = sig_min - .1
        ax_lim = np.around(ax_lim, decimals=1)
        yy = np.arange(ax_lim, 1.01, delta)[::-1]
        c_levels = np.arange(ax_lim+.1, 1.1, .1)
    elif sig_min >= 0:
        ax_lim = 0.2
        yy = np.arange(-ax_lim, 1.01, delta)[::-1]
        c_levels = np.arange(0, 1, .2)
    elif sig_min < 0 and sig_min >= -1:
        ax_lim = 1.2
        yy = np.arange(-ax_lim, 1.01, delta)[::-1]
        c_levels = np.arange(-1, 1, .2)
    elif sig_min >= -2 and sig_min < -1:
        ax_lim = 2.2
        yy = np.arange(-ax_lim, 1.01, delta)[::-1]
        c_levels = np.arange(-2, 1, .2)
    elif sig_min < -2:
        raise AssertionError("Some values of 'DE' are too low for visualization!", sig_min)

    len_yy = 360
    # len_yy1 = 90

    # arrays to plot contour lines of DE
    xx = np.radians(np.linspace(0, 360, len_yy))
    theta, r = np.meshgrid(xx, yy)
    #
    # # arrays to plot contours of positive constant offset
    # xx1 = np.radians(np.linspace(45, 135, len_yy1))
    # theta1, r1 = np.meshgrid(xx1, yy)
    #
    # # arrays to plot contours of negative constant offset
    # xx2 = np.radians(np.linspace(225, 315, len_yy1))
    # theta2, r2 = np.meshgrid(xx2, yy)
    #
    # # arrays to plot contours of model errors
    # xx3 = np.radians(np.linspace(135, 225, len_yy1))
    # theta3, r3 = np.meshgrid(xx3, yy)
    #
    # # arrays to plot contours of model errors
    # len_yy2 = int(len_yy1/2)
    # if len_yy != len_yy2 + len_yy2:
    #     xx0 = np.radians(np.linspace(0, 45, len_yy2+1))
    # else:
    #     xx0 = np.radians(np.linspace(0, 45, len_yy2))
    #
    # xx360 = np.radians(np.linspace(315, 360, len_yy2))
    # xx4 = np.concatenate((xx360, xx0), axis=None)
    # theta4, r4 = np.meshgrid(xx4, yy)

    # diagnostic polar plot
    fig, ax = plt.subplots(figsize=(6, 6),
                           subplot_kw=dict(projection='polar'),
                           constrained_layout=True)
    # dummie plot for colorbar of temporal correlation
    cs = np.arange(0, 1.1, 0.1)
    dummie_cax = ax.scatter(cs, cs, c=cs, cmap='plasma_r')
    # Clear axis
    ax.cla()
    # # contours positive constant offset
    # cpio = ax.contourf(theta1, r1, r1, cmap='Purples_r', alpha=.2, edgecolors=None, levels=c_levels, zorder=0)
    # # contours negative constant offset
    # cpiu = ax.contourf(theta2, r2, r2, cmap='Purples_r', alpha=.2, edgecolors=None, levels=c_levels, zorder=0)
    # # contours model errors
    # cpmou = ax.contourf(theta3, r3, r3, cmap='Greys_r', alpha=.2, edgecolors=None, levels=c_levels, zorder=0)
    # cpmuo = ax.contourf(theta4, r4, r4, cmap='Greys_r', alpha=.2, edgecolors=None, levels=c_levels, zorder=0)
    # plot regions
    ax.plot((1, np.deg2rad(45)), (1, np.min(yy)), color='lightgray', linewidth=1.5, ls='--', zorder=0)
    ax.plot((1, np.deg2rad(135)), (1, np.min(yy)), color='lightgray', linewidth=1.5, ls='--', zorder=0)
    ax.plot((1, np.deg2rad(225)), (1, np.min(yy)), color='lightgray', linewidth=1.5, ls='--', zorder=0)
    ax.plot((1, np.deg2rad(315)), (1, np.min(yy)), color='lightgray', linewidth=1.5, ls='--', zorder=0)
    # contours of DE
    cp = ax.contour(theta, r, r, colors='darkgray', levels=c_levels, zorder=1)
    cl = ax.clabel(cp, inline=False, fontsize=10, fmt='%1.1f',
                   colors='dimgrey')
    # threshold efficiency for FBM
    sig_l = 1 - np.sqrt((l)**2 + (l)**2 + (l)**2)
    # loop over each data point
    for (bm, bd, ba, r, sig, sig_bench, ang, txt) in zip(ll_brel_mean, ll_b_dir, ll_b_area, ll_temp_cor, ll_sig, ll_bench, ll_diag, fc):
        # normalizing the threshold efficiency
        sig_lim_norm = (sig_l - sig_bench)/(1 - sig_bench)
        # slope of bias
        b_slope = de.calc_bias_slope(ba, bd)
        # convert temporal correlation to color
        rgba_color = cm.plasma_r(norm(r))
        # relation of b_dir which explains the error
        if abs(ba) > 0:
            exp_err = (abs(bd) * 2)/abs(ba)
        elif abs(ba) == 0:
            exp_err = 0

        # diagnose the error
        if abs(bm) <= l and exp_err > l and sig <= sig_lim_norm:
            c = ax.scatter(ang, sig, s=75, color=rgba_color, zorder=3)
            d = ax.scatter(ang, sig, color='black', marker='.', zorder=4)
            ax.annotate(txt, xy=(ang, sig), color='black', fontsize=13,
                        xytext=(-8, 0), textcoords="offset points",
                        ha='center', va='center')
        elif abs(bm) > l and exp_err <= l and sig <= sig_lim_norm:
            c = ax.scatter(ang, sig, s=75, color=rgba_color, zorder=3)
            d = ax.scatter(ang, sig, color='black', marker='.', zorder=4)
            ax.annotate(txt, xy=(ang, sig), color='black', fontsize=13,
                        xytext=(-8, 0), textcoords="offset points",
                        ha='center', va='center')
        elif abs(bm) > l and exp_err > l and sig <= sig_lim_norm:
            c = ax.scatter(ang, sig, s=75, color=rgba_color, zorder=3)
            d = ax.scatter(ang, sig, color='black', marker='.', zorder=4)
            ax.annotate(txt, xy=(ang, sig), color='black', fontsize=13,
                        xytext=(-8, 0), textcoords="offset points",
                        ha='center', va='center')
        # FBM
        elif abs(bm) <= l and exp_err <= l and sig <= sig_lim_norm:
            ax.annotate("", xytext=(0, 1), xy=(0, sig),
                        arrowprops=dict(edgecolor=rgba_color, facecolor='black', lw=3), zorder=2)
            ax.annotate("", xytext=(0, 1), xy=(np.pi, sig),
                        arrowprops=dict(edgecolor=rgba_color, facecolor='black', lw=3), zorder=2)
            ax.annotate(txt, xy=(ang, sig), color='black', fontsize=13,
                        xytext=(-8, 0), textcoords="offset points",
                        ha='center', va='center')
        # FGM
        elif abs(bm) <= l and exp_err <= l and sig > sig_lim_norm:
            c = ax.scatter(ang, sig, s=75, color=rgba_color, zorder=3)
            d = ax.scatter(ang, sig, color='black', marker='.', zorder=4)
            ax.annotate(txt, xy=(ang, sig), color='black', fontsize=13,
                        xytext=(-6, 0), textcoords="offset points",
                        ha='center', va='center')
    ax.set_rticks([])  # turn default ticks off
    ax.set_rmin(1)
    if sig_min > 0:
        ax.set_rmax(ax_lim)
    elif sig_min <= 0:
        ax.set_rmax(-ax_lim)
    ax.tick_params(labelleft=False, labelright=False, labeltop=False,
                  labelbottom=True, grid_alpha=.01)  # turn labels and grid off
    ax.set_xticklabels(['', '', '', '', '', '', '', ''])
    ax.text(-.14, 0.5, 'High flow overestimation -',
            va='center', ha='center', rotation=90, rotation_mode='anchor',
            transform=ax.transAxes)
    ax.text(-.09, 0.5, 'Low flow underestimation',
            va='center', ha='center', rotation=90, rotation_mode='anchor',
            transform=ax.transAxes)
    ax.text(-.04, 0.5, r'$B_{slope}$ < 0',
            va='center', ha='center', rotation=90, rotation_mode='anchor',
            transform=ax.transAxes)
    ax.text(1.14, 0.5, 'High flow underestimation -',
            va='center', ha='center', rotation=90, rotation_mode='anchor',
            transform=ax.transAxes)
    ax.text(1.09, 0.5, 'Low flow overestimation',
            va='center', ha='center', rotation=90, rotation_mode='anchor',
            transform=ax.transAxes)
    ax.text(1.04, 0.5, r'$B_{slope}$ > 0',
            va='center', ha='center', rotation=90, rotation_mode='anchor',
            transform=ax.transAxes)
    ax.text(.5, -.09, 'Constant negative offset', va='center', ha='center',
            rotation=0, rotation_mode='anchor', transform=ax.transAxes)
    ax.text(.5, -.04, r'$\overline{B_{rel}}$ < 0', va='center', ha='center',
            rotation=0, rotation_mode='anchor', transform=ax.transAxes)
    ax.text(.5, 1.09, 'Constant positive offset',
            va='center', ha='center', rotation=0, rotation_mode='anchor',
            transform=ax.transAxes)
    ax.text(.5, 1.04, r'$\overline{B_{rel}}$ > 0',
            va='center', ha='center', rotation=0, rotation_mode='anchor',
            transform=ax.transAxes)
    # add colorbar for temporal correlation
    cbar = fig.colorbar(dummie_cax, ax=ax, orientation='horizontal',
                        ticks=[1, 0.5, 0], shrink=0.8)
    cbar.set_label(r'r [-]', fontsize=12, labelpad=8)
    cbar.set_ticklabels(['1', '0.5', '<0'])
    cbar.ax.tick_params(direction='in', labelsize=10)

    return fig

def vis2d_kge_multi_fc(kge_beta, alpha_or_gamma, kge_r, sig_kge, fc, ax_lim=-.6):
    """Multiple polar plot of Kling-Gupta Efficiency (KGE)

    Parameters
    ----------
    kge_beta: (N,)array_like
        KGE beta as 1-D array

    alpha_or_gamma : (N,)array_like
        KGE alpha or KGE gamma as 1-D array

    kge_r : (N,)array_like
        KGE r as 1-D array

    sig_kge : (N,)array_like
        KGE as 1-D array

    fc : list
        figure captions
    """
    sig_min = np.min(sig_kge)

    ll_kge_beta = kge_beta.tolist()
    ll_ag = alpha_or_gamma.tolist()
    ll_kge_r = kge_r.tolist()
    ll_sig = sig_kge.tolist()

    # convert temporal correlation to color
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1.0)

    delta = 0.01  # for spacing

    # determine axis limits
    yy = np.arange(ax_lim, 1.01, delta)[::-1]
    c_levels = np.arange(ax_lim, 1, .2)

    len_yy = 360
    # len_yy1 = 90

    # arrays to plot contour lines of DE
    xx = np.radians(np.linspace(0, 360, len_yy))
    theta, r = np.meshgrid(xx, yy)

    # # arrays to plot contours of positive constant offset
    # xx1 = np.radians(np.linspace(45, 135, len_yy1))
    # theta1, r1 = np.meshgrid(xx1, yy)
    #
    # # arrays to plot contours of negative constant offset
    # xx2 = np.radians(np.linspace(225, 315, len_yy1))
    # theta2, r2 = np.meshgrid(xx2, yy)
    #
    # # arrays to plot contours of model errors
    # xx3 = np.radians(np.linspace(135, 225, len_yy1))
    # theta3, r3 = np.meshgrid(xx3, yy)
    #
    # # arrays to plot contours of model errors
    # len_yy2 = int(len_yy1/2)
    # if len_yy != len_yy2 + len_yy2:
    #     xx0 = np.radians(np.linspace(0, 45, len_yy2+1))
    # else:
    #     xx0 = np.radians(np.linspace(0, 45, len_yy2))
    #
    # xx360 = np.radians(np.linspace(315, 360, len_yy2))
    # xx4 = np.concatenate((xx360, xx0), axis=None)
    # theta4, r4 = np.meshgrid(xx4, yy)

    # diagnostic polar plot
    fig, ax = plt.subplots(figsize=(6, 6),
                           subplot_kw=dict(projection='polar'),
                           constrained_layout=True)
    # dummie plot for colorbar of temporal correlation
    cs = np.arange(0, 1.1, 0.1)
    dummie_cax = ax.scatter(cs, cs, c=cs, cmap='plasma_r')
    # Clear axis
    ax.cla()
    # # contours positive constant offset
    # cpio = ax.contourf(theta1, r1, r1, cmap='Purples_r', alpha=.2, edgecolors=None, levels=c_levels, zorder=0)
    # # contours negative constant offset
    # cpiu = ax.contourf(theta2, r2, r2, cmap='Purples_r', alpha=.2, edgecolors=None, levels=c_levels, zorder=0)
    # # contours model errors
    # cpmou = ax.contourf(theta3, r3, r3, cmap='Greys_r', alpha=.2, edgecolors=None, levels=c_levels, zorder=0)
    # cpmuo = ax.contourf(theta4, r4, r4, cmap='Greys_r', alpha=.2, edgecolors=None, levels=c_levels, zorder=0)
    # plot regions
    ax.plot((1, np.deg2rad(45)), (1, np.min(yy)), color='lightgray', linewidth=1.5, ls='--', zorder=0)
    ax.plot((1, np.deg2rad(135)), (1, np.min(yy)), color='lightgray', linewidth=1.5, ls='--', zorder=0)
    ax.plot((1, np.deg2rad(225)), (1, np.min(yy)), color='lightgray', linewidth=1.5, ls='--', zorder=0)
    ax.plot((1, np.deg2rad(315)), (1, np.min(yy)), color='lightgray', linewidth=1.5, ls='--', zorder=0)
    # contours of KGE
    cp = ax.contour(theta, r, r, colors='darkgray', levels=c_levels, zorder=1)
    cl = ax.clabel(cp, inline=False, fontsize=10, fmt='%1.1f',
                   colors='dimgrey')
    # loop over each data point
    for (b, ag, r, sig, txt) in zip(ll_kge_beta, ll_ag, ll_kge_r, ll_sig, fc):
        ang = np.arctan2(b - 1, ag - 1)
        # convert temporal correlation to color
        rgba_color = cm.plasma_r(norm(r))
        c = ax.scatter(ang, sig, s=75, color=rgba_color, zorder=2)
        d = ax.scatter(ang, sig, color='black', marker='.', zorder=4)
        ax.annotate(txt, xy=(ang, sig), color='black', fontsize=13,
                    xytext=(8, 0), textcoords="offset points",
                    ha='center', va='center')
    ax.tick_params(labelleft=False, labelright=False, labeltop=False,
                  labelbottom=True, grid_alpha=.01)  # turn labels and grid off
    ax.text(-.09, 0.5, 'Variability underestimation', va='center', ha='center',
            rotation=90, rotation_mode='anchor', transform=ax.transAxes)
    ax.text(-.04, 0.5, r'($\alpha$ - 1) < 0', va='center', ha='center',
            rotation=90, rotation_mode='anchor', transform=ax.transAxes)
    ax.text(1.09, 0.5, 'Variability overestimation',
            va='center', ha='center', rotation=90, rotation_mode='anchor',
            transform=ax.transAxes)
    ax.text(1.04, 0.5, r'($\alpha$ - 1) > 0',
            va='center', ha='center', rotation=90, rotation_mode='anchor',
            transform=ax.transAxes)
    ax.text(.5, -.09, 'Mean underestimation', va='center', ha='center',
            rotation=0, rotation_mode='anchor', transform=ax.transAxes)
    ax.text(.5, -.04, r'($\beta$ - 1) < 0', va='center', ha='center',
            rotation=0, rotation_mode='anchor', transform=ax.transAxes)
    ax.text(.5, 1.09, 'Mean overestimation',
            va='center', ha='center', rotation=0, rotation_mode='anchor',
            transform=ax.transAxes)
    ax.text(.5, 1.04, r'($\beta$ - 1) > 0',
            va='center', ha='center', rotation=0, rotation_mode='anchor',
            transform=ax.transAxes)
    ax.set_xticklabels(['', '', '', '', '', '', '', ''])
    ax.set_rticks([])  # turn default ticks off
    ax.set_rmin(1)
    ax.set_rmax(ax_lim)
    # add colorbar for temporal correlation
    cbar = fig.colorbar(dummie_cax, ax=ax, orientation='horizontal',
                        ticks=[1, 0.5, 0], shrink=0.8)
    cbar.set_label(r'r [-]', fontsize=12, labelpad=8)
    cbar.set_ticklabels(['1', '0.5', '<0'])
    cbar.ax.tick_params(direction='in', labelsize=10)

    return fig
