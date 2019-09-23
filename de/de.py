#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
de.de
~~~~~~~~~~~
Diagnosing model performance using an efficiency measure based on flow
duration curve and temoral correlation. The efficiency measure can be
visualized in 2D-Plot which facilitates decomposing potential error origins
(model errors vs. input data erros)
:2019 by Robin Schwemmle.
:license: GNU GPLv3, see LICENSE for more details.
"""

import datetime as dt
import numpy as np
# RunTimeWarning will not be displayed (division by zeros or NaN values)
np.seterr(divide='ignore', invalid='ignore')
import matplotlib
import matplotlib.dates as mdates
import matplotlib.transforms as mtransforms
from matplotlib import cm
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import scipy as sp
import scipy.integrate as integrate
import seaborn as sns
# controlling figure aesthetics
sns.set_style('ticks', {'xtick.major.size': 8, 'ytick.major.size': 8})

__title__ = 'de'
__version__ = '0.1'
#__build__ = 0x001201
__author__ = 'Robin Schwemmle'
__license__ = 'GNU GPLv3'
#__docformat__ = 'markdown'

#TODO: consistent datatype
#TODO: match Qsim to Qobs
#TODO: zero values in Qobs calculating rel. bias
#TODO: colormap r
#TODO: DE vs KGE, B_bal vs beta, B_slope vs gamma

_mmd = r'[mm $d^{-1}$]'
_m3s = r'[$m^{3}$ $s^{-1}$]'
_q_lab = _m3s
_sim_lab = 'Manipulated'

def plot_ts(ts):
    """
    Plot time series.

    Parameters
    ----------
    ts : dataframe
        Dataframe with time series
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
    """
    Plot observed and simulated time series.

    Parameters
    ----------
    obs : series
        observed time series

    sim : series
        simulated time series
    """
    fig, ax = plt.subplots()
    ax.plot(obs.index, obs, lw=2, color='blue')  # observed time series
    ax.plot(sim.index, sim, lw=1, ls='-.', color='red', alpha=.8)  # simulated time series
    ax.set(ylabel=_q_lab, xlabel='Time')
    ax.set_ylim(0, )
    ax.set_xlim(obs.index[0], obs.index[-1])

def plot_obs_sim_ax(obs, sim, ax, fig_num):
    """
    Plot observed and simulated time series.

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
    ax.plot(obs.index, obs, lw=2, color='blue')  # observed time series
    ax.plot(sim.index, sim, lw=1, ls='-.', color='red', alpha=.8)  # simulated time series
    ax.set_ylim(0, )
    ax.set_xlim(obs.index[0], obs.index[-1])
    ax.text(.88, .93, fig_num, transform=ax.transAxes)
    # format the ticks
    years_20 = mdates.YearLocator(20)
    years_5 = mdates.YearLocator(5)
    yearsFmt = mdates.DateFormatter('%Y')
    ax.xaxis.set_major_locator(years_20)
    ax.xaxis.set_major_formatter(yearsFmt)
    ax.xaxis.set_minor_locator(years_5)

def fdc(ts):
    """
    Generate a flow duration curve for a single hydrologic time series.

    Parameters
    ----------
    ts : dataframe
        Containing a hydrologic time series
    """
    data = ts.dropna()
    data = np.sort(data.values)  # sort values by ascending order
    ranks = sp.stats.rankdata(data, method='ordinal')  # rank data
    ranks = ranks[::-1]
    # calculate exceedence probability
    prob = [(ranks[i]/(len(data)+1)) for i in range(len(data))]

    fig, ax = plt.subplots()
    ax.plot(prob, data, color='blue')
    ax.set(ylabel=_q_lab,
           xlabel='Exceedence probabilty [-]', yscale='log')

def fdc_obs_sim(obs, sim):
    """
    Plotting the flow duration curves of two hydrologic time series (e.g.
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
    ax.plot(prob_obs, obs['obs'], color='blue', lw=2, label='Observed')
    ax.plot(prob_sim, sim['sim'], color='red', lw=1, ls='-.', label=_sim_lab,
            alpha=.8)
    ax.set(ylabel=_q_lab,
           xlabel='Exceedence probabilty [-]', yscale='log')
    ax.legend(loc=1)
    ax.set_ylim(0, )
    ax.set_xlim(0, 1)

def fdc_obs_sim_ax(obs, sim, ax, fig_num):
    """
    Plotting the flow duration curves of two hydrologic time series (e.g.
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

    ax.plot(prob_obs, obs['obs'], lw=2, color='blue')
    ax.plot(prob_sim, sim['sim'], lw=1, ls='-.', color='red')
    ax.text(.88, .93, fig_num, transform=ax.transAxes)
    ax.set(yscale='log')
    ax.set_ylim(0, )
    ax.set_xlim(0, 1)

def calc_brel_mean(obs, sim, sort=True):
    """
    Calculate arithmetic mean of relative bias.

    Parameters
    ----------
    obs : (N,)array_like
        observed time series as 1-D array

    sim : (N,)array_like
        simulated time series

    sort : boolean, optional
        If True, time series are sorted by ascending order. If False, time series
        are not sorted. The default is to sort.

    Returns
    ----------
    brel_mean : float
        average relative bias
    """
    if sort:
        obs = np.sort(obs)[::-1]
        sim = np.sort(sim)[::-1]
    sim_obs_diff = np.subtract(sim, obs)
    brel = np.divide(sim_obs_diff, obs)
    brel[sim_obs_diff==0] = 0
    brel = brel[np.isfinite(brel)]
    brel_mean = np.mean(brel)

    return brel_mean

def calc_brel_rest(obs, sim, sort=True):
    """
    Subtract arithmetic mean of relative bias from relative bias.

    Parameters
    ----------
    obs : (N,)array_like
        observed time series as 1-D array

    sim : (N,)array_like
        simulated time series

    sort : boolean, optional
        If True, time series are sorted by ascending order. If False, time
        series are not sorted. The default is to sort.

    Returns
    ----------
    brel_rest : array_like
        remaining relative bias
    """
    if sort:
        obs = np.sort(obs)[::-1]
        sim = np.sort(sim)[::-1]
    sim_obs_diff = np.subtract(sim, obs)
    brel = np.divide(sim_obs_diff, obs)
    brel[sim_obs_diff==0] = 0
    brel = brel[np.isfinite(brel)]
    brel_mean = np.mean(brel)
    brel_rest = brel - brel_mean

    return brel_rest

def integrand(y, x):
    """
    Function to intergrate bias.

    f(x)

    Parameters
    ----------
    y : array_like
        time series

    x : float

    Returns
    ----------
    y[i] : float

    """
    i = int(x * len(y))  # convert to index
    return y[i]

def calc_b_area(brel_rest):
    """
    Calculate absolute bias area for high flow and low flow.

    Parameters
    ----------
    brel_rest : (N,)array_like
        remaining relative bias as 1-D array

    Returns
    ----------
    b_area[0] : float
        bias area
    """
    brel_rest_abs = abs(brel_rest)
    # area of bias
    b_area = integrate.quad(lambda x: integrand(brel_rest_abs, x), 0.001, .999,
                            limit=10000)

    return b_area[0]

def calc_bias_dir(brel_rest):
    """
    Calculate absolute bias area for high flow and low flow.

    Parameters
    ----------
    brel_rest : (N,)array_like
        remaining relative bias as 1-D array

    Returns
    ----------
    b_dir : float
        direction of bias
    """
    # integral of relative bias < 50 %
    hf_area = integrate.quad(lambda x: integrand(brel_rest, x), 0.001, .5,
                             limit=10000)

    # direction of bias
    b_dir = hf_area[0]

    return b_dir

def calc_bias_slope(b_area, b_dir):
    """
    Calculate slope of bias balance.

    Parameters
    ----------
    b_area : float
        absolute area of remaining bias

    b_dir : float
        direction of bias

    Returns
    ----------
    b_slope : float
        slope of bias
    """
    if b_dir > 0:
        b_slope = b_area * (-1)

    elif b_dir < 0:
        b_slope = b_area

    elif b_dir == 0:
        b_slope = 0

    return b_slope

def calc_temp_cor(obs, sim, r='pearson'):
    """
    Calculate temporal correlation between observed and simulated
    time series.

    Parameters
    ----------
    obs : (N,)array_like
        Observed time series as 1-D array

    sim : (N,)array_like
        Simulated time series

    r : str, optional
        Either Ppearman correlation coefficient ('spearman') or Pearson
        correlation coefficient ('pearson') can be used to describe the temporal
        correlation. The default is to calculate the Pearson correlation.

    Returns
    ----------
    temp_cor : float
        Rank correlation between observed and simulated time series
    """
    if r == 'spearman':
        r = sp.stats.spearmanr(obs, sim)
        temp_cor = r[0]

        if np.isnan(temp_cor):
            temp_cor = 0

    elif r == 'pearson':
        r = sp.stats.pearsonr(obs, sim)
        temp_cor = r[0]

        if np.isnan(temp_cor):
            temp_cor = 0

    return temp_cor

def calc_de(obs, sim, sort=True):
    """
    Calculate Diagnostic-Efficiency (DE).

    Parameters
    ----------
    obs : (N,)array_like
        Observed time series as 1-D array

    sim : (N,)array_like
        Simulated time series

    sort : boolean, optional
        If True, time series are sorted by ascending order. If False, time
        series are not sorted. The default is to sort.

    Returns
    ----------
    sig : float
        Diagnostic efficiency measure
    """
    # mean relative bias
    brel_mean = calc_brel_mean(obs, sim, sort=sort)
    # remaining relative bias
    brel_rest = calc_brel_rest(obs, sim, sort=sort)
    # area of relative remaing bias
    b_area = calc_b_area(brel_rest)
    # temporal correlation
    temp_cor = calc_temp_cor(obs, sim)
    # diagnostic efficiency
    sig = 1 - np.sqrt((brel_mean)**2 + (b_area)**2 + (temp_cor - 1)**2)

    return sig

def calc_kge_alpha(obs, sim):
    """
    Calculate the alpha term of Kling-Gupta-Efficiency (KGE).

    Parameters
    ----------
    obs : (N,)array_like
        Observed time series as 1-D array

    sim : (N,)array_like
        Simulated time series

    Returns
    ----------
    kge_alpha : float
        alpha value

    References
    ----------
    Gupta, H. V., Kling, H., Yilmaz, K. K., and Martinez, G. F.: Decomposition
    of the mean squared error and NSE performance criteria: Implications for
    improving hydrological modelling, Journal of Hydrology, 377, 80-91,
    10.1016/j.jhydrol.2009.08.003, 2009.

    Kling, H., Fuchs, M., and Paulin, M.: Runoff conditions in the upper
    Danube basin under an ensemble of climate change scenarios, Journal of
    Hydrology, 424-425, 264-277, 10.1016/j.jhydrol.2012.01.011, 2012.

    Pool, S., Vis, M., and Seibert, J.: Evaluating model performance: towards a
    non-parametric variant of the Kling-Gupta efficiency, Hydrological Sciences
    Journal, 63, 1941-1953, 10.1080/02626667.2018.1552002, 2018.
    """
    # calculate alpha term
    obs_mean = np.mean(obs)
    sim_mean = np.mean(sim)
    kge_alpha = sim_mean/obs_mean

    return kge_alpha

def calc_kge_beta(obs, sim):
    """
    Calculate the beta term of the Kling-Gupta-Efficiency (KGE).

    Parameters
    ----------
    obs : (N,)array_like
        Observed time series as 1-D array

    sim : (N,)array_like
        Simulated time series

    Returns
    ----------
    kge_beta : float
        beta value

    References
    ----------
    Gupta, H. V., Kling, H., Yilmaz, K. K., and Martinez, G. F.: Decomposition
    of the mean squared error and NSE performance criteria: Implications for
    improving hydrological modelling, Journal of Hydrology, 377, 80-91,
    10.1016/j.jhydrol.2009.08.003, 2009.

    Kling, H., Fuchs, M., and Paulin, M.: Runoff conditions in the upper
    Danube basin under an ensemble of climate change scenarios, Journal of
    Hydrology, 424-425, 264-277, 10.1016/j.jhydrol.2012.01.011, 2012.

    Pool, S., Vis, M., and Seibert, J.: Evaluating model performance: towards a
    non-parametric variant of the Kling-Gupta efficiency, Hydrological Sciences
    Journal, 63, 1941-1953, 10.1080/02626667.2018.1552002, 2018.
    """
    obs_std = np.std(obs)
    sim_std = np.std(sim)
    kge_beta = sim_std/obs_std

    return kge_beta

def calc_kge_gamma(obs, sim):
    """
    Calculate Kling-Gupta-Efficiency (KGE).

    Parameters
    ----------
    obs : (N,)array_like
        Observed time series as 1-D array

    sim : (N,)array_like
        Simulated time series

    Returns
    ----------
    kge_gamma : float
        gamma value

    References
    ----------
    Gupta, H. V., Kling, H., Yilmaz, K. K., and Martinez, G. F.: Decomposition
    of the mean squared error and NSE performance criteria: Implications for
    improving hydrological modelling, Journal of Hydrology, 377, 80-91,
    10.1016/j.jhydrol.2009.08.003, 2009.

    Kling, H., Fuchs, M., and Paulin, M.: Runoff conditions in the upper
    Danube basin under an ensemble of climate change scenarios, Journal of
    Hydrology, 424-425, 264-277, 10.1016/j.jhydrol.2012.01.011, 2012.

    Pool, S., Vis, M., and Seibert, J.: Evaluating model performance: towards a
    non-parametric variant of the Kling-Gupta efficiency, Hydrological Sciences
    Journal, 63, 1941-1953, 10.1080/02626667.2018.1552002, 2018.
    """
    obs_mean = np.mean(obs)
    sim_mean = np.mean(sim)
    obs_std = np.std(obs)
    sim_std = np.std(sim)
    obs_cv = obs_std/obs_mean
    sim_cv = sim_std/sim_mean
    kge_gamma = sim_cv/obs_cv

    return kge_gamma

def calc_kge(obs, sim, r='pearson', var='std'):
    """
    Calculate Kling-Gupta-Efficiency (KGE).

    Parameters
    ----------
    obs : (N,)array_like
        Observed time series as 1-D array

    sim : (N,)array_like
        Simulated time series

    r : str, optional
        Either Ppearman correlation coefficient ('spearman'; Pool et al. 2018)
        or Pearson correlation coefficient ('pearson'; Gupta et al. 2009) can
        be used to describe the temporal correlation. The default is to
        calculate the Pearson correlation.

    var : str, optional
        Either coefficient of variation ('cv'; Kling et al. 2012) or standard
        deviation ('std'; Gupta et al. 2009, Pool et al. 2018) to describe the
        gamma term. The default is to calculate the standard deviation.

    Returns
    ----------
    sig : float
        Kling-Gupta-Efficiency measure

    References
    ----------
    Gupta, H. V., Kling, H., Yilmaz, K. K., and Martinez, G. F.: Decomposition
    of the mean squared error and NSE performance criteria: Implications for
    improving hydrological modelling, Journal of Hydrology, 377, 80-91,
    10.1016/j.jhydrol.2009.08.003, 2009.

    Kling, H., Fuchs, M., and Paulin, M.: Runoff conditions in the upper
    Danube basin under an ensemble of climate change scenarios, Journal of
    Hydrology, 424-425, 264-277, 10.1016/j.jhydrol.2012.01.011, 2012.

    Pool, S., Vis, M., and Seibert, J.: Evaluating model performance: towards a
    non-parametric variant of the Kling-Gupta efficiency, Hydrological Sciences
    Journal, 63, 1941-1953, 10.1080/02626667.2018.1552002, 2018.
    """
    # calculate alpha term
    obs_mean = np.mean(obs)
    sim_mean = np.mean(sim)
    kge_alpha = sim_mean/obs_mean

    # calculate KGE with gamma term
    if var == 'cv':
        kge_gamma = calc_kge_gamma(obs, sim)
        temp_cor = calc_temp_cor(obs, sim, r=r)

        sig = 1 - np.sqrt((kge_alpha - 1)**2 + (kge_gamma - 1)**2  + (temp_cor - 1)**2)

    # calculate KGE with beta term
    elif var == 'std':
        kge_beta = calc_kge_beta(obs, sim)
        temp_cor = calc_temp_cor(obs, sim, r=r)

        sig = 1 - np.sqrt((kge_alpha - 1)**2 + (kge_beta - 1)**2  + (temp_cor - 1)**2)

    return sig

def calc_nse(obs, sim):
    """
    Calculate Nash-Sutcliffe-Efficiency (NSE).

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

    References
    ----------
    Nash, J. E., and Sutcliffe, J. V.: River flow forecasting through conceptual
    models part I - A discussion of principles, Journal of Hydrology, 10,
    282-290, 10.1016/0022-1694(70)90255-6, 1970.
    """
    sim_obs_diff = np.sum((sim - obs)**2)
    obs_mean = np.mean(obs)
    obs_diff_mean = np.sum((obs - obs_mean)**2)
    sig = 1 - (sim_obs_diff/obs_diff_mean)

    return sig

def vis2d_de(obs, sim, sort=True, lim=0.05, extended=False):
    """
    Polar plot of Diagnostic-Efficiency (DE)

    Parameters
    ----------
    obs : (N,)array_like
        Observed time series as 1-D array

    sim : (N,)array_like
        Simulated time series as 1-D array

    sort : boolean, optional
        If True, time series are sorted by ascending order. If False, time series
        are not sorted. The default is to sort.

    lim : float, optional
        Threshold for which diagnosis can be made. The default is 0.05.

    extended : boolean, optional
        If True, extended diagnostic plot is displayed. In addtion, the duration
        curve of B_rest is plotted besides the polar plot. The default is,
        that only the diagnostic polar plot is displayed.
    """
    # mean relative bias
    brel_mean = calc_brel_mean(obs, sim, sort=sort)

    str_brel_mean = 'Brel_mean: {}'.format(np.round(brel_mean, decimals=3))
    print(str_brel_mean)

    # remaining relative bias
    brel_rest = calc_brel_rest(obs, sim, sort=sort)
    # area of relative remaing bias
    b_area = calc_b_area(brel_rest)
    # temporal correlation
    temp_cor = calc_temp_cor(obs, sim)
    # diagnostic efficiency
    sig = 1 - np.sqrt((brel_mean)**2 + (b_area)**2 + (temp_cor - 1)**2)
    sig = np.round(sig, decimals=2)  # round to 2 decimals

    # direction of bias
    b_dir = calc_bias_dir(brel_rest)

    str_b_dir = 'B_dir: {}'.format(np.round(b_dir, decimals=3))
    print(str_b_dir)

    # slope of bias
    b_slope = calc_bias_slope(b_area, b_dir)

    str_b_slope = 'B_slope: {}'.format(np.round(b_slope, decimals=3))
    print(str_b_slope)

    # convert to radians
    # (y, x) Trigonometric inverse tangent
    diag = np.arctan2(brel_mean, b_slope)

    str_diag = 'diag: {}'.format(np.round(diag, decimals=3))
    print(str_diag)

    # convert temporal correlation to color
    norm = matplotlib.colors.Normalize(vmin=-1.0, vmax=1.0)
    rgba_color = cm.YlGnBu(norm(temp_cor))

    delta = 0.01  # for spacing

    # determine axis limits
    if sig > 0:
        yy = np.arange(0, 1, delta)[::-1]
        ax_lim = 0
    elif sig <= 0 and sig > -1:
        yy = np.arange(-1, 1, delta)[::-1]
        ax_lim = 1
    elif sig <= -1:
        yy = np.arange(-2, 1, delta)[::-1]
        ax_lim = 2
    elif sig <= -2:
        yy = np.arange(-3, 1, delta)[::-1]
        ax_lim = 3

    len_yy = len(yy)

    # arrays to plot contour lines of DE
    xx = np.radians(np.linspace(0, 360, len_yy))
    theta, r = np.meshgrid(xx, yy)

    # arrays to plot contours of P overestimation
    xx1 = np.radians(np.linspace(45, 135, len_yy))
    theta1, r1 = np.meshgrid(xx1, yy)

    # arrays to plot contours of P underestimation
    xx2 = np.radians(np.linspace(225, 315, len_yy))
    theta2, r2 = np.meshgrid(xx2, yy)

    # arrays to plot contours of model errors
    xx3 = np.radians(np.linspace(135, 225, len_yy))
    theta3, r3 = np.meshgrid(xx3, yy)

    # arrays to plot contours of model errors
    len_yy2 = int(len_yy/2)
    if len_yy != len_yy2 + len_yy2:
        xx0 = np.radians(np.linspace(0, 45, len_yy2+1))
    else:
        xx0 = np.radians(np.linspace(0, 45, len_yy2))

    xx360 = np.radians(np.linspace(315, 360, len_yy2))
    xx4 = np.concatenate((xx360, xx0), axis=None)
    theta4, r4 = np.meshgrid(xx4, yy)

    # diagnostic polar plot
    if not extended:
        fig, ax = plt.subplots(figsize=(6, 6),
                               subplot_kw=dict(projection='polar'),
                               constrained_layout=True)
        # dummie plot for colorbar of temporal correlation
        cs = np.arange(-1, 1.1, 0.1)
        dummie_cax = ax.scatter(cs, cs, c=cs, cmap='YlGnBu')
        # Clear axis
        ax.cla()
        # contours P overestimation
        cpio = ax.contourf(theta1, r1, r1, cmap='Purples_r', alpha=.3)
        # contours P underestimation
        cpiu = ax.contourf(theta2, r2, r2, cmap='Purples_r', alpha=.3)
        # contours model errors
        cpmou = ax.contourf(theta3, r3, r3, cmap='Greys_r', alpha=.3)
        cpmuo = ax.contourf(theta4, r4, r4, cmap='Greys_r', alpha=.3)
        # contours of DE
        cp = ax.contour(theta, r, r, colors='black', alpha=.7)
        cl = ax.clabel(cp, inline=True, fontsize=10, fmt='%1.1f', inline_spacing=6)
        # threshold efficiency for FBM
        sig_lim = 1 - np.sqrt((lim)**2 + (lim)**2 + (lim)**2)
        # relation of b_dir which explains the error
        if abs(b_area) > 0:
            exp_err = (abs(b_dir) * 2)/abs(b_area)
        elif abs(b_area) == 0:
            exp_err = 0
        # diagnose the error
        if abs(brel_mean) <= lim and exp_err > lim and sig <= sig_lim:
            c = ax.scatter(diag, sig, color=rgba_color)
        elif abs(brel_mean) > lim and exp_err <= lim and sig <= sig_lim:
            c = ax.scatter(diag, sig, color=rgba_color)
        elif abs(brel_mean) > lim and exp_err > lim and sig <= sig_lim:
            c = ax.scatter(diag, sig, color=rgba_color)
        # FBM
        elif abs(brel_mean) <= lim and exp_err <= lim and sig <= sig_lim:
            c = ax.arrow(0, 1, 0, -abs(1-sig), color=rgba_color, lw=5, zorder=1,
                         width=.05, length_includes_head=True,
                         transform=mtransforms.Affine2D().translate(0, 0) + ax.transData)
            c1 = ax.arrow(0, 1, 0, -abs(1-sig), color=rgba_color, lw=5,
                          zorder=1, width=.05, length_includes_head=True,
                          transform=mtransforms.Affine2D().translate(np.pi, 0) + ax.transData)
        # FGM
        elif abs(brel_mean) <= lim and exp_err <= lim and sig > sig_lim:
            c = ax.scatter(diag, sig, color=rgba_color)
        ax.set_rticks([])  # turn default ticks off
        ax.set_rmin(1)
        ax.set_rmax(-ax_lim)
        ax.tick_params(labelleft=False, labelright=False, labeltop=False,
                      labelbottom=True, grid_alpha=.01)  # turn labels and grid off
        ax.set_xticklabels(['', '', 'P overestimation', '', '', '', 'P underestimation', ''])
        ax.text(-.05, 0.5, 'High flow overestimation - \n Low flow underestimation',
                va='center', ha='center', rotation=90, rotation_mode='anchor',
                transform=ax.transAxes)
        ax.text(1.05, 0.5, 'High flow underestimation - \n Low flow overestimation',
                va='center', ha='center', rotation=90, rotation_mode='anchor',
                transform=ax.transAxes)
        # add colorbar for temporal correlation
        cbar = fig.colorbar(dummie_cax, ax=ax, orientation='vertical',
                            ticks=[1, 0.5, 0, -0.5, -1], shrink=0.8)
        cbar.set_label(r'r [-]', fontsize=12, labelpad=8)
        cbar.set_ticklabels(['1', '0.5', '0', '-0.5', '-1'])
        cbar.ax.tick_params(direction='in', labelsize=10)

    elif extended:
            fig = plt.figure(figsize=(12, 6), constrained_layout=True)
            gs = fig.add_gridspec(1, 2)
            ax = fig.add_subplot(gs[0, 0], projection='polar')
            ax1 = fig.add_axes([.64, .3, .33, .33], frameon=True)
            # dummie plot for colorbar of temporal correlation
            cs = np.arange(-1, 1.1, 0.1)
            dummie_cax = ax.scatter(cs, cs, c=cs, cmap='YlGnBu')
            # Clear axis
            ax.cla()
            # contours P overestimation
            cpio = ax.contourf(theta1, r1, r1, cmap='Purples_r', alpha=.3)
            # contours P underestimation
            cpiu = ax.contourf(theta2, r2, r2, cmap='Purples_r', alpha=.3)
            # contours model errors
            cpmou = ax.contourf(theta3, r3, r3, cmap='Greys_r', alpha=.3)
            cpmuo = ax.contourf(theta4, r4, r4, cmap='Greys_r', alpha=.3)
            # contours of DE
            cp = ax.contour(theta, r, r, colors='black', alpha=.7)
            cl = ax.clabel(cp, inline=True, fontsize=10, fmt='%1.1f',
                           inline_spacing=6)
            # threshold efficiency for FBM
            sig_lim = 1 - np.sqrt((lim)**2 + (lim)**2 + (lim)**2)
            # relation of b_dir which explains the error
            if abs(b_area) > 0:
                exp_err = (abs(b_dir) * 2)/abs(b_area)
            elif abs(b_area) == 0:
                exp_err = 0
            # diagnose the error
            if abs(brel_mean) <= lim and exp_err > lim and sig <= sig_lim:
                c = ax.scatter(diag, sig, color=rgba_color)
            elif abs(brel_mean) > lim and exp_err <= lim and sig <= sig_lim:
                c = ax.scatter(diag, sig, color=rgba_color)
            elif abs(brel_mean) > lim and exp_err > lim and sig <= sig_lim:
                c = ax.scatter(diag, sig, color=rgba_color)
            # FBM
            elif abs(brel_mean) <= lim and exp_err <= lim and sig <= sig_lim:
                c = ax.arrow(0, 1, 0, -abs(1-sig), color=rgba_color, lw=5, zorder=1,
                             width=.05, length_includes_head=True,
                             transform=mtransforms.Affine2D().translate(0, 0) + ax.transData)
                c1 = ax.arrow(0, 1, 0, -abs(1-sig), color=rgba_color, lw=5,
                              zorder=1, width=.05, length_includes_head=True,
                              transform=mtransforms.Affine2D().translate(np.pi, 0) + ax.transData)
            # FGM
            elif abs(brel_mean) <= lim and exp_err <= lim and sig > sig_lim:
                c = ax.scatter(diag, sig, color=rgba_color)
            ax.set_rticks([])  # turn default ticks off
            ax.set_rmin(1)
            ax.set_rmax(-ax_lim)
            ax.tick_params(labelleft=False, labelright=False, labeltop=False,
                          labelbottom=True, grid_alpha=.01)  # turn labels and grid off
            ax.set_xticklabels(['', '', 'P overestimation', '', '', '', 'P underestimation', ''])
            ax.text(-.05, 0.5, 'High flow overestimation - \n Low flow underestimation',
                    va='center', ha='center', rotation=90,
                    rotation_mode='anchor', transform=ax.transAxes)
            ax.text(1.05, 0.5, 'High flow underestimation - \n Low flow overestimation',
                    va='center', ha='center', rotation=90,
                    rotation_mode='anchor', transform=ax.transAxes)
            # add colorbar for temporal correlation
            cbar = fig.colorbar(dummie_cax, ax=ax, orientation='vertical',
                                ticks=[1, 0.5, 0, -0.5, -1], shrink=0.8)
            cbar.set_label(r'r [-]', fontsize=12, labelpad=8)
            cbar.set_ticklabels(['1', '0.5', '0', '-0.5', '-1'])
            cbar.ax.tick_params(direction='in', labelsize=10)

            # plot B_rest
            # calculate exceedence probability
            prob = np.linspace(0, 1, len(brel_rest))
            ax1.axhline(y=0, color='slategrey')
            ax1.axvline(x=0.5, color='slategrey')
            ax1.plot(prob, brel_rest, color='black')
            ax1.fill_between(prob, brel_rest, where=0 < brel_rest, facecolor='purple')
            ax1.fill_between(prob, brel_rest, where=0 > brel_rest, facecolor='red')
            ax1.set(ylabel=r'$B_{rest}$ [-]',
                    xlabel='Exceedence probabilty [-]')

def vis2d_de_multi(brel_mean, b_area, temp_cor, sig_de, b_dir, diag,
                   lim=0.05, extended=False):
    """
    Multiple polar plot of Diagnostic-Efficiency (DE)

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

    lim : float, optional
        Threshold for which diagnosis can be made. The default is 0.05.

    extended : boolean, optional
        If True, density plot is displayed. In addtion, the density plot
        is displayed besides the polar plot. The default is,
        that only the diagnostic polar plot is displayed.
    """
    sig_min = np.min(sig_de)

    ll_brel_mean = brel_mean.tolist()
    ll_b_dir = b_dir.tolist()
    ll_b_area = b_area.tolist()
    ll_sig = sig_de.tolist()
    ll_diag = diag.tolist()
    ll_temp_cor = temp_cor.tolist()

    # convert temporal correlation to color
    norm = matplotlib.colors.Normalize(vmin=-1.0, vmax=1.0)

    delta = 0.01  # for spacing

    # determine axis limits
    if sig_min > 0:
        yy = np.arange(0, 1, delta)[::-1]
        ax_lim = 0
    elif sig_min <= 0 and sig_min > -1:
        yy = np.arange(-1, 1, delta)[::-1]
        ax_lim = 1
    elif sig_min <= -1:
        yy = np.arange(-2, 1, delta)[::-1]
        ax_lim = 2
    elif sig_min <= -2:
        yy = np.arange(-3, 1, delta)[::-1]
        ax_lim = 3

    len_yy = len(yy)

    # arrays to plot contour lines of DE
    xx = np.radians(np.linspace(0, 360, len_yy))
    theta, r = np.meshgrid(xx, yy)

    # arrays to plot contours of P overestimation
    xx1 = np.radians(np.linspace(45, 135, len_yy))
    theta1, r1 = np.meshgrid(xx1, yy)

    # arrays to plot contours of P underestimation
    xx2 = np.radians(np.linspace(225, 315, len_yy))
    theta2, r2 = np.meshgrid(xx2, yy)

    # arrays to plot contours of model errors
    xx3 = np.radians(np.linspace(135, 225, len_yy))
    theta3, r3 = np.meshgrid(xx3, yy)

    # arrays to plot contours of model errors
    len_yy2 = int(len_yy/2)
    if len_yy != len_yy2 + len_yy2:
        xx0 = np.radians(np.linspace(0, 45, len_yy2+1))
    else:
        xx0 = np.radians(np.linspace(0, 45, len_yy2))

    xx360 = np.radians(np.linspace(315, 360, len_yy2))
    xx4 = np.concatenate((xx360, xx0), axis=None)
    theta4, r4 = np.meshgrid(xx4, yy)

    # diagnostic polar plot
    if not extended:
        fig, ax = plt.subplots(figsize=(6, 6),
                               subplot_kw=dict(projection='polar'),
                               constrained_layout=True)
        # dummie plot for colorbar of temporal correlation
        cs = np.arange(-1, 1.1, 0.1)
        dummie_cax = ax.scatter(cs, cs, c=cs, cmap='YlGnBu')
        # Clear axis
        ax.cla()
        # contours P overestimation
        cpio = ax.contourf(theta1, r1, r1, cmap='Purples_r', alpha=.3)
        # contours P underestimation
        cpiu = ax.contourf(theta2, r2, r2, cmap='Purples_r', alpha=.3)
        # contours model errors
        cpmou = ax.contourf(theta3, r3, r3, cmap='Greys_r', alpha=.3)
        cpmuo = ax.contourf(theta4, r4, r4, cmap='Greys_r', alpha=.3)
        # contours of DE
        cp = ax.contour(theta, r, r, colors='black', alpha=.7)
        cl = ax.clabel(cp, inline=True, fontsize=10, fmt='%1.1f',
                       inline_spacing=6)
        # threshold efficiency for FBM
        sig_lim = 1 - np.sqrt((lim)**2 + (lim)**2 + (lim)**2)
        # loop over each data point
        for (bm, bd, ba, r, sig, ang) in zip(ll_brel_mean, ll_b_dir, ll_b_area, ll_temp_cor, ll_sig, ll_diag):
            # slope of bias
            b_slope = calc_bias_slope(ba, bd)
            # convert temporal correlation to color
            rgba_color = cm.YlGnBu(norm(r))
            # relation of b_dir which explains the error
            if abs(ba) > 0:
                exp_err = (abs(bd) * 2)/abs(ba)
            elif abs(ba) == 0:
                exp_err = 0
            # diagnose the error
            if abs(bm) <= lim and exp_err > lim and sig <= sig_lim:
                c = ax.scatter(ang, sig, color=rgba_color, zorder=2)
            elif abs(bm) > lim and exp_err <= lim and sig <= sig_lim:
                c = ax.scatter(ang, sig, color=rgba_color, zorder=2)
            elif abs(bm) > lim and exp_err > lim and sig <= sig_lim:
                c = ax.scatter(ang, sig, color=rgba_color, zorder=2)
            # FBM
            elif abs(bm) <= lim and exp_err <= lim and sig <= sig_lim:
                c = ax.arrow(0, 1, 0, -abs(1-sig), color=rgba_color, lw=5, zorder=1,
                             width=.05, length_includes_head=True,
                             transform=mtransforms.Affine2D().translate(0, 0) + ax.transData)
                c1 = ax.arrow(0, 1, 0, -abs(1-sig), color=rgba_color, lw=5,
                              zorder=1, width=.05, length_includes_head=True,
                              transform=mtransforms.Affine2D().translate(np.pi, 0) + ax.transData)
            # FGM
            elif abs(bm) <= lim and exp_err <= lim and sig > sig_lim:
                c = ax.scatter(ang, sig, color=rgba_color, zorder=2)
        ax.set_rticks([])  # turn default ticks off
        ax.set_rmin(1)
        ax.set_rmax(-ax_lim)
        ax.tick_params(labelleft=False, labelright=False, labeltop=False,
                      labelbottom=True, grid_alpha=.01)  # turn labels and grid off
        ax.set_xticklabels(['', '', 'P overestimation', '', '', r'0$^\circ$/360$^\circ$', 'P underestimation', ''])
        ax.text(-.05, 0.5, 'High flow overestimation - \n Low flow underestimation',
                va='center', ha='center', rotation=90, rotation_mode='anchor',
                transform=ax.transAxes)
        ax.text(1.05, 0.5, 'High flow underestimation - \n Low flow overestimation',
                va='center', ha='center', rotation=90, rotation_mode='anchor',
                transform=ax.transAxes)
        # add colorbar for temporal correlation
        cbar = fig.colorbar(dummie_cax, ax=ax, orientation='vertical',
                            ticks=[1, 0.5, 0, -0.5, -1], shrink=0.8)
        cbar.set_label(r'r [-]', fontsize=12, labelpad=8)
        cbar.set_ticklabels(['1', '0.5', '0', '-0.5', '-1'])
        cbar.ax.tick_params(direction='in', labelsize=10)

    elif extended:
            fig = plt.figure(figsize=(12, 6), constrained_layout=True)
            gs = fig.add_gridspec(1, 2)
            ax = fig.add_subplot(gs[0, 0], projection='polar')
            ax1 = fig.add_axes([.64, .3, .33, .33], frameon=True)
            # dummie plot for colorbar of temporal correlation
            cs = np.arange(-1, 1.1, 0.1)
            dummie_cax = ax.scatter(cs, cs, c=cs, cmap='YlGnBu')
            # Clear axis
            ax.cla()
            # contours P overestimation
            cpio = ax.contourf(theta1, r1, r1, cmap='Purples_r', alpha=.3)
            # contours P underestimation
            cpiu = ax.contourf(theta2, r2, r2, cmap='Purples_r', alpha=.3)
            # contours model errors
            cpmou = ax.contourf(theta3, r3, r3, cmap='Greys_r', alpha=.3)
            cpmuo = ax.contourf(theta4, r4, r4, cmap='Greys_r', alpha=.3)
            # contours of DE
            cp = ax.contour(theta, r, r, colors='black', alpha=.7)
            cl = ax.clabel(cp, inline=True, fontsize=10, fmt='%1.1f', inline_spacing=6)
            # threshold efficiency for FBM
            sig_lim = 1 - np.sqrt((lim)**2 + (lim)**2 + (lim)**2)
            # loop over each data point
            for (bm, bd, ba, r, sig, ang) in zip(ll_brel_mean, ll_b_dir, ll_b_area, ll_temp_cor, ll_sig, ll_diag):
                # slope of bias
                b_slope = calc_bias_slope(ba, bd)
                # convert temporal correlation to color
                rgba_color = cm.YlGnBu(norm(r))
                # relation of b_dir which explains the error
                if abs(ba) > 0:
                    exp_err = (abs(bd) * 2)/abs(ba)
                elif abs(ba) == 0:
                    exp_err = 0
                # diagnose the error
                if abs(bm) <= lim and exp_err > lim and sig <= sig_lim:
                    c = ax.scatter(ang, sig, color=rgba_color, zorder=2)
                elif abs(bm) > lim and exp_err <= lim and sig <= sig_lim:
                    c = ax.scatter(ang, sig, color=rgba_color, zorder=2)
                elif abs(bm) > lim and exp_err > lim and sig <= sig_lim:
                    c = ax.scatter(ang, sig, color=rgba_color, zorder=2)
                # FBM
                elif abs(bm) <= lim and exp_err <= lim and sig <= sig_lim:
                    c = ax.arrow(0, 1, 0, -abs(1-sig), color=rgba_color, lw=5, zorder=1,
                                 width=.05, length_includes_head=True,
                                 transform=mtransforms.Affine2D().translate(0, 0) + ax.transData)
                    c1 = ax.arrow(0, 1, 0, -abs(1-sig), color=rgba_color, lw=5,
                                  zorder=1, width=.05, length_includes_head=True,
                                  transform=mtransforms.Affine2D().translate(np.pi, 0) + ax.transData)
                # FGM
                elif abs(bm) <= lim and exp_err <= lim and sig > sig_lim:
                    c = ax.scatter(ang, sig, color=rgba_color, zorder=2)
            ax.set_rticks([])  # turn default ticks off
            ax.set_rmin(1)
            ax.set_rmax(-ax_lim)
            ax.tick_params(labelleft=False, labelright=False, labeltop=False,
                          labelbottom=True, grid_alpha=.01)  # turn labels and grid off
            ax.set_xticklabels(['', '', 'P overestimation', '', '', r'0$^\circ$/360$^\circ$', 'P underestimation', ''])
            ax.text(-.05, 0.5, 'High flow overestimation - \n Low flow underestimation',
                    va='center', ha='center', rotation=90,
                    rotation_mode='anchor', transform=ax.transAxes)
            ax.text(1.05, 0.5, 'High flow underestimation - \n Low flow overestimation',
                    va='center', ha='center', rotation=90,
                    rotation_mode='anchor', transform=ax.transAxes)
            # add colorbar for temporal correlation
            cbar = fig.colorbar(dummie_cax, ax=ax, orientation='vertical',
                                ticks=[1, 0.5, 0, -0.5, -1], shrink=0.8)
            cbar.set_label(r'r [-]', fontsize=12, labelpad=8)
            cbar.set_ticklabels(['1', '0.5', '0', '-0.5', '-1'])
            cbar.ax.tick_params(direction='in', labelsize=10)

            # convert to degrees
            diag_deg = (diag  * (180 / np.pi)) + 135
            diag_deg[diag_deg < 0] = 360 - diag_deg[diag_deg < 0]

            # 1-D density plot
            g = sns.kdeplot(diag_deg, color='k', ax=ax1)
            kde_data = g.get_lines()[0].get_data()
            kde_xx = kde_data[0]
            kde_yy = kde_data[1]
            x1 = np.where(kde_xx <= 90)[-1][-1]
            x2 = np.where(kde_xx <= 180)[-1][-1]
            x3 = np.where(kde_xx <= 270)[-1][-1]
            ax1.fill_between(kde_xx[:x1+1], kde_yy[:x1+1], facecolor='purple',
                             alpha=0.3)
            ax1.fill_between(kde_xx[x1:x2+2], kde_yy[x1:x2+2], facecolor='grey',
                             alpha=0.3)
            ax1.fill_between(kde_xx[x2+1:x3+1], kde_yy[x2+1:x3+1],
                             facecolor='purple', alpha=0.3)
            ax1.fill_between(kde_xx[x3:], kde_yy[x3:], facecolor='grey',
                             alpha=0.3)
            ax1.set_xticks([0, 90, 180, 270, 360])
            ax1.set_xlim(0, 360)
            ax1.set_ylim(0, )
            ax1.set(ylabel=r'[-]',
                    xlabel='[$^\circ$]')

            # 2-D density plot
            # g = (sns.jointplot(diag_deg, sig_de, color='k', marginal_kws={'color':'k'}).plot_joint(sns.kdeplot, zorder=0, n_levels=10))
            g = (sns.jointplot(diag_deg, sig_de, kind='kde', zorder=1,
                               n_levels=20, cmap='Greens', shade_lowest=False,
                               marginal_kws={'color':'k', 'shade':False}).plot_joint(sns.scatterplot, color='k', alpha=.5, zorder=2))
            g.set_axis_labels(r'[$^\circ$]', r'DE [-]')
            g.ax_joint.set_xticks([0, 90, 180, 270, 360])
            g.ax_joint.set_xlim(0, 360)
            g.ax_joint.set_ylim(-ax_lim, 1)
            g.ax_marg_x.set_xticks([0, 90, 180, 270, 360])
            kde_data = g.ax_marg_x.get_lines()[0].get_data()
            kde_xx = kde_data[0]
            kde_yy = kde_data[1]
            x1 = np.where(kde_xx <= 90)[-1][-1]
            x2 = np.where(kde_xx <= 180)[-1][-1]
            x3 = np.where(kde_xx <= 270)[-1][-1]
            g.ax_marg_x.fill_between(kde_xx[:x1+1], kde_yy[:x1+1],
                                     facecolor='purple', alpha=0.3)
            g.ax_marg_x.fill_between(kde_xx[x1:x2+2], kde_yy[x1:x2+2],
                                     facecolor='grey', alpha=0.3)
            g.ax_marg_x.fill_between(kde_xx[x2+1:x3+1], kde_yy[x2+1:x3+1],
                                     facecolor='purple', alpha=0.3)
            g.ax_marg_x.fill_between(kde_xx[x3:], kde_yy[x3:], facecolor='grey',
                                     alpha=0.3)
            kde_data = g.ax_marg_y.get_lines()[0].get_data()
            kde_xx = kde_data[0]
            kde_yy = kde_data[1]
            norm = matplotlib.colors.Normalize(vmin=-ax_lim, vmax=1.0)
            colors = cm.Reds_r(norm(kde_yy))
            npts = len(kde_xx)
            for i in range(npts - 1):
                g.ax_marg_y.fill_betweenx([kde_yy[i], kde_yy[i+1]],
                                          [kde_xx[i], kde_xx[i+1]],
                                          color=colors[i])
            g.fig.tight_layout()

def vis2d_kge(obs, sim, r='pearson', var='std'):
    """
    Polar plot of Kling-Gupta-Efficiency (KGE)

    Parameters
    ----------
    obs : (N,)array_like
        Observed time series as 1-D array

    sim : (N,)array_like
        Simulated time series as 1-D array
    """
    # calculate alpha term
    obs_mean = np.mean(obs)
    sim_mean = np.mean(sim)
    kge_alpha = sim_mean/obs_mean

    # calculate gamma term
    if var == 'cv':
        obs_std = np.std(obs)
        sim_std = np.std(sim)
        obs_cv = obs_std/obs_mean
        sim_cv = sim_std/sim_mean
        kge_gamma = sim_cv/obs_cv
        temp_cor = calc_temp_cor(obs, sim, r=r)

        sig = 1 - np.sqrt((kge_alpha - 1)**2 + (kge_gamma- 1)**2  + (temp_cor - 1)**2)
        sig = np.round(sig, decimals=2)

        # convert to radians
        # (y, x) Trigonometric inverse tangent
        diag = np.arctan2(kge_alpha - 1, kge_gamma - 1)

        # convert temporal correlation to color
        norm = matplotlib.colors.Normalize(vmin=-1.0, vmax=1.0)
        rgba_color = cm.YlGnBu(norm(temp_cor))

        delta = 0.01  # for spacing

        # determine axis limits
        if sig > 0:
            yy = np.arange(0, 1, delta)[::-1]
            ax_lim = 0
        elif sig <= 0 and sig > -1:
            yy = np.arange(-1, 1 - delta, delta)[::-1]
            ax_lim = 1
        elif sig <= -1:
            yy = np.arange(-2, 2 - delta, delta)[::-1]
            ax_lim = 2

        len_yy = len(yy)

        # arrays to plot contour lines of DE
        xx = np.radians(np.linspace(0, 360, len_yy))
        theta, r = np.meshgrid(xx, yy)

        # arrays to plot contours of P overestimation
        xx1 = np.radians(np.linspace(45, 135, len_yy))
        theta1, r1 = np.meshgrid(xx1, yy)

        # arrays to plot contours of P underestimation
        xx2 = np.radians(np.linspace(225, 315, len_yy))
        theta2, r2 = np.meshgrid(xx2, yy)

        # arrays to plot contours of model errors
        xx3 = np.radians(np.linspace(135, 225, len_yy))
        theta3, r3 = np.meshgrid(xx3, yy)

        # arrays to plot contours of model errors
        len_yy2 = int(len_yy/2)
        if len_yy != len_yy2 + len_yy2:
            xx0 = np.radians(np.linspace(0, 45, len_yy2+1))
        else:
            xx0 = np.radians(np.linspace(0, 45, len_yy2))

        xx360 = np.radians(np.linspace(315, 360, len_yy2))
        xx4 = np.concatenate((xx360, xx0), axis=None)
        theta4, r4 = np.meshgrid(xx4, yy)

        # diagnostic polar plot
        fig, ax = plt.subplots(figsize=(6, 6),
                               subplot_kw=dict(projection='polar'),
                               constrained_layout=True)
        # dummie plot for colorbar of temporal correlation
        cs = np.arange(-1, 1.1, 0.1)
        dummie_cax = ax.scatter(cs, cs, c=cs, cmap='YlGnBu')
        # Clear axis
        ax.cla()
        # contours P overestimation
        cpio = ax.contourf(theta1, r1, r1, cmap='Purples_r', alpha=.3)
        # contours P underestimation
        cpiu = ax.contourf(theta2, r2, r2, cmap='Purples_r', alpha=.3)
        # contours model errors
        cpmou = ax.contourf(theta3, r3, r3, cmap='Greys_r', alpha=.3)
        cpmuo = ax.contourf(theta4, r4, r4, cmap='Greys_r', alpha=.3)
        # contours of DE
        cp = ax.contour(theta, r, r, colors='black', alpha=.7)
        cl = ax.clabel(cp, inline=1, fontsize=10, fmt='%1.1f')
        # diagnose the error
        c = ax.scatter(diag, sig, color=rgba_color)
        ax.set_rticks([])  # turn defalut ticks off
        ax.set_rmin(1)
        ax.set_rmax(-ax_lim)
        ax.tick_params(labelleft=False, labelright=False, labeltop=False,
                      labelbottom=True, grid_alpha=.01)  # turn labels and grid off
        ax.text(-.05, 0.5, r'$\alpha$ - 1 [-]', va='center', ha='center',
                rotation=90, rotation_mode='anchor', transform=ax.transAxes)
        ax.set_xticklabels(['', '', '', '', '', '', r'$\gamma$ - 1 [-]', ''])
        # add colorbar for temporal correlation
        cbar = fig.colorbar(dummie_cax, ax=ax, orientation='vertical',
                            ticks=[1, 0.5, 0, -0.5, -1], shrink=0.8)
        cbar.set_label(r'r [-]', fontsize=12, labelpad=8)
        cbar.set_ticklabels(['1', '0.5', '0', '-0.5', '-1'])
        cbar.ax.tick_params(direction='in', labelsize=10)

    # calculate beta term
    elif var == 'std':
        obs_std = np.std(obs)
        sim_std = np.std(sim)
        kge_beta = sim_std/obs_std
        temp_cor = calc_temp_cor(obs, sim, r=r)

        sig = 1 - np.sqrt((kge_alpha - 1)**2 + (kge_beta- 1)**2  + (temp_cor - 1)**2)
        sig = np.round(sig, decimals=2)

        # convert to radians
        # (y, x) Trigonometric inverse tangent
        diag = np.arctan2(kge_alpha - 1, kge_beta - 1)

        # convert temporal correlation to color
        norm = matplotlib.colors.Normalize(vmin=-1.0, vmax=1.0)
        rgba_color = cm.YlGnBu(norm(temp_cor))

        delta = 0.01  # for spacing

        # determine axis limits
        if sig > 0:
            yy = np.arange(0, 1, delta)[::-1]
            ax_lim = 0
        elif sig <= 0 and sig > -1:
            yy = np.arange(-1, 1 - delta, delta)[::-1]
            ax_lim = 1
        elif sig <= -1:
            yy = np.arange(-2, 2 - delta, delta)[::-1]
            ax_lim = 2

        len_yy = len(yy)

        # arrays to plot contour lines of DE
        xx = np.radians(np.linspace(0, 360, len_yy))
        theta, r = np.meshgrid(xx, yy)

        # arrays to plot contours of P overestimation
        xx1 = np.radians(np.linspace(45, 135, len_yy))
        theta1, r1 = np.meshgrid(xx1, yy)

        # arrays to plot contours of P underestimation
        xx2 = np.radians(np.linspace(225, 315, len_yy))
        theta2, r2 = np.meshgrid(xx2, yy)

        # arrays to plot contours of model errors
        xx3 = np.radians(np.linspace(135, 225, len_yy))
        theta3, r3 = np.meshgrid(xx3, yy)

        # arrays to plot contours of model errors
        len_yy2 = int(len_yy/2)
        if len_yy != len_yy2 + len_yy2:
            xx0 = np.radians(np.linspace(0, 45, len_yy2+1))
        else:
            xx0 = np.radians(np.linspace(0, 45, len_yy2))

        xx360 = np.radians(np.linspace(315, 360, len_yy2))
        xx4 = np.concatenate((xx360, xx0), axis=None)
        theta4, r4 = np.meshgrid(xx4, yy)

        # diagnostic polar plot
        fig, ax = plt.subplots(figsize=(6, 6),
                               subplot_kw=dict(projection='polar'),
                               constrained_layout=True)
        # dummie plot for colorbar of temporal correlation
        cs = np.arange(-1, 1.1, 0.1)
        dummie_cax = ax.scatter(cs, cs, c=cs, cmap='YlGnBu')
        # Clear axis
        ax.cla()
        # contours P overestimation
        cpio = ax.contourf(theta1, r1, r1, cmap='Purples_r', alpha=.3)
        # contours P underestimation
        cpiu = ax.contourf(theta2, r2, r2, cmap='Purples_r', alpha=.3)
        # contours model errors
        cpmou = ax.contourf(theta3, r3, r3, cmap='Greys_r', alpha=.3)
        cpmuo = ax.contourf(theta4, r4, r4, cmap='Greys_r', alpha=.3)
        # contours of DE
        cp = ax.contour(theta, r, r, colors='black', alpha=.7)
        cl = ax.clabel(cp, inline=1, fontsize=10, fmt='%1.1f')
        # diagnose the error
        c = ax.scatter(diag, sig, color=rgba_color)
        ax.set_rticks([])  # turn defalut ticks off
        ax.set_rmin(1)
        ax.set_rmax(-ax_lim)
        ax.tick_params(labelleft=False, labelright=False, labeltop=False,
                      labelbottom=True, grid_alpha=.01)  # turn labels and grid off
        ax.text(-.05, 0.5, r'$\alpha$ - 1 [-]', va='center', ha='center',
        rotation=90, rotation_mode='anchor', transform=ax.transAxes)
        ax.set_xticklabels(['', '', '', '', '', '', r'$\beta$ - 1[-]', ''])
        # add colorbar for temporal correlation
        cbar = fig.colorbar(dummie_cax, ax=ax, orientation='vertical',
                            ticks=[1, 0.5, 0, -0.5, -1], shrink=0.8)
        cbar.set_label(r'r [-]', fontsize=12, labelpad=8)
        cbar.set_ticklabels(['1', '0.5', '0', '-0.5', '-1'])
        cbar.ax.tick_params(direction='in', labelsize=10)

def vis2d_kge_multi_fc(kge_alpha, beta_or_gamma, kge_r, sig_kge, fc, extended=False):
    """
    Multiple polar plot of Kling-Gupta Efficiency (KGE)

    Parameters
    ----------
    kge_alpha: (N,)array_like
        KGE alpha as 1-D array

    kge_beta : (N,)array_like
        KGE beta as 1-D array

    kge_r : (N,)array_like
        KGE r as 1-D array

    sig_kge : (N,)array_like
        KGE as 1-D array

    fc : list
        figure captions

    extended : boolean, optional
        If True, density plot is displayed. In addtion, the density plot
        is displayed besides the polar plot. The default is,
        that only the diagnostic polar plot is displayed.
    """
    sig_min = np.min(sig_kge)

    ll_kge_alpha = kge_alpha.tolist()
    ll_bg = beta_or_gamma.tolist()
    ll_kge_r = kge_r.tolist()
    ll_sig = sig_kge.tolist()

    # convert temporal correlation to color
    norm = matplotlib.colors.Normalize(vmin=-1.0, vmax=1.0)

    delta = 0.01  # for spacing

    # determine axis limits
    if sig_min > 0:
        yy = np.arange(0, 1, delta)[::-1]
        ax_lim = 0
    elif sig_min <= 0 and sig_min > -1:
        yy = np.arange(-1, 1, delta)[::-1]
        ax_lim = 1
    elif sig_min <= -1:
        yy = np.arange(-2, 1, delta)[::-1]
        ax_lim = 2
    elif sig_min <= -2:
        yy = np.arange(-3, 1, delta)[::-1]
        ax_lim = 3

    len_yy = len(yy)

    # arrays to plot contour lines of DE
    xx = np.radians(np.linspace(0, 360, len_yy))
    theta, r = np.meshgrid(xx, yy)

    # arrays to plot contours of P overestimation
    xx1 = np.radians(np.linspace(45, 135, len_yy))
    theta1, r1 = np.meshgrid(xx1, yy)

    # arrays to plot contours of P underestimation
    xx2 = np.radians(np.linspace(225, 315, len_yy))
    theta2, r2 = np.meshgrid(xx2, yy)

    # arrays to plot contours of model errors
    xx3 = np.radians(np.linspace(135, 225, len_yy))
    theta3, r3 = np.meshgrid(xx3, yy)

    # arrays to plot contours of model errors
    len_yy2 = int(len_yy/2)
    if len_yy != len_yy2 + len_yy2:
        xx0 = np.radians(np.linspace(0, 45, len_yy2+1))
    else:
        xx0 = np.radians(np.linspace(0, 45, len_yy2))

    xx360 = np.radians(np.linspace(315, 360, len_yy2))
    xx4 = np.concatenate((xx360, xx0), axis=None)
    theta4, r4 = np.meshgrid(xx4, yy)

    # diagnostic polar plot
    if not extended:
        fig, ax = plt.subplots(figsize=(6, 6),
                               subplot_kw=dict(projection='polar'),
                               constrained_layout=True)
        # dummie plot for colorbar of temporal correlation
        cs = np.arange(-1, 1.1, 0.1)
        dummie_cax = ax.scatter(cs, cs, c=cs, cmap='YlGnBu')
        # Clear axis
        ax.cla()
        # contours P overestimation
        cpio = ax.contourf(theta1, r1, r1, cmap='Purples_r', alpha=.3)
        # contours P underestimation
        cpiu = ax.contourf(theta2, r2, r2, cmap='Purples_r', alpha=.3)
        # contours model errors
        cpmou = ax.contourf(theta3, r3, r3, cmap='Greys_r', alpha=.3)
        cpmuo = ax.contourf(theta4, r4, r4, cmap='Greys_r', alpha=.3)
        # contours of DE
        cp = ax.contour(theta, r, r, colors='black', alpha=.7)
        cl = ax.clabel(cp, inline=True, fontsize=10, fmt='%1.1f', inline_spacing=6)
        # loop over each data point
        for (a, bg, r, sig, txt) in zip(ll_kge_alpha, ll_bg, ll_kge_r, ll_sig, fc):
            ang = np.arctan2(a - 1, bg - 1)
            # convert temporal correlation to color
            rgba_color = cm.YlGnBu(norm(r))
            c = ax.scatter(ang, sig, color=rgba_color)
            if ang <= np.pi/2 and ang >= 0:
                ax.annotate(txt, (ang, sig - .05), color='black', fontsize=13)
            elif ang < 0 and ang > -np.pi/2:
                ax.annotate(txt, (ang, sig - .05), color='black', fontsize=13)
            elif ang <= np.pi and ang > np.pi/2:
                ax.annotate(txt, (ang, sig + .06), color='black', fontsize=13)
            elif ang > np.pi or ang < -np.pi/2:
                ax.annotate(txt, (ang, sig  + .05), color='black', fontsize=13)

        ax.tick_params(labelleft=False, labelright=False, labeltop=False,
                      labelbottom=True, grid_alpha=.01)  # turn labels and grid off
        ax.text(-.05, 0.5, r'$\alpha$ - 1 [-]', va='center', ha='center',
                rotation=90, rotation_mode='anchor', transform=ax.transAxes)
        ax.set_xticklabels(['', '', '', '', '', '', r'$\beta$/$\gamma$ - 1 [-]', ''])
        ax.set_rticks([])  # turn default ticks off
        ax.set_rmin(1)
        ax.set_rmax(-ax_lim)
        # add colorbar for temporal correlation
        cbar = fig.colorbar(dummie_cax, ax=ax, orientation='vertical',
                            label='r [-]', ticks=[1, 0.5, 0, -0.5, -1], shrink=0.8)
        cbar.set_ticklabels(['1', '0.5', '0', '-0.5', '-1'])
        cbar.ax.tick_params(direction='in', labelsize=10)

    elif extended:
            fig = plt.figure(figsize=(12, 6), constrained_layout=True)
            gs = fig.add_gridspec(1, 2)
            ax = fig.add_subplot(gs[0, 0], projection='polar')
            ax1 = fig.add_axes([.64, .3, .33, .33], frameon=True)
            # dummie plot for colorbar of temporal correlation
            cs = np.arange(-1, 1.1, 0.1)
            dummie_cax = ax.scatter(cs, cs, c=cs, cmap='YlGnBu')
            # Clear axis
            ax.cla()
            # contours P overestimation
            cpio = ax.contourf(theta1, r1, r1, cmap='Purples_r', alpha=.3)
            # contours P underestimation
            cpiu = ax.contourf(theta2, r2, r2, cmap='Purples_r', alpha=.3)
            # contours model errors
            cpmou = ax.contourf(theta3, r3, r3, cmap='Greys_r', alpha=.3)
            cpmuo = ax.contourf(theta4, r4, r4, cmap='Greys_r', alpha=.3)
            # contours of DE
            cp = ax.contour(theta, r, r, colors='black', alpha=.7)
            cl = ax.clabel(cp, inline=True, fontsize=10, fmt='%1.1f', inline_spacing=6)
            # loop over each data point
            for (a, bg, r, sig) in zip(ll_kge_alpha, ll_bg, ll_kge_r, ll_sig):
                ang = np.arctan2(a - 1, bg - 1)
                # convert temporal correlation to color
                rgba_color = cm.YlGnBu(norm(r))
                c = ax.scatter(ang, sig, color=rgba_color)

            ax.tick_params(labelleft=False, labelright=False, labeltop=False,
                          labelbottom=True, grid_alpha=.01)  # turn labels and grid off
            ax.text(-.05, 0.5, r'$\alpha$ - 1 [-]', va='center', ha='center',
                    rotation=90, rotation_mode='anchor', transform=ax.transAxes)
            ax.set_xticklabels(['', '', '', '', '', '', r'$\beta$/$\gamma$ - 1 [-]', ''])
            ax.set_rticks([])  # turn default ticks off
            ax.set_rmin(1)
            ax.set_rmax(-ax_lim)
            # add colorbar for temporal correlation
            cbar = fig.colorbar(dummie_cax, ax=ax, orientation='vertical',
                                ticks=[1, 0.5, 0, -0.5, -1], shrink=0.8)
            cbar.set_label(r'r [-]', fontsize=12, labelpad=8)
            cbar.set_ticklabels(['1', '0.5', '0', '-0.5', '-1'])
            cbar.ax.tick_params(direction='in', labelsize=10)

            # convert to degrees
            diag = np.arctan2(kge_alpha - 1, beta_or_gamma - 1)
            diag_deg = (diag  * (180 / np.pi)) + 135
            diag_deg[diag_deg < 0] = 360 - diag_deg[diag_deg < 0]

            # 1-D density plot
            g = sns.kdeplot(diag_deg, color='k', ax=ax1)
            kde_data = g.get_lines()[0].get_data()
            kde_xx = kde_data[0]
            kde_yy = kde_data[1]
            x1 = np.where(kde_xx <= 90)[-1][-1]
            x2 = np.where(kde_xx <= 180)[-1][-1]
            x3 = np.where(kde_xx <= 270)[-1][-1]
            ax1.fill_between(kde_xx[:x1+1], kde_yy[:x1+1], facecolor='purple', alpha=0.3)
            ax1.fill_between(kde_xx[x1:x2+2], kde_yy[x1:x2+2], facecolor='grey', alpha=0.3)
            ax1.fill_between(kde_xx[x2+1:x3+1], kde_yy[x2+1:x3+1], facecolor='purple', alpha=0.3)
            ax1.fill_between(kde_xx[x3:], kde_yy[x3:], facecolor='grey', alpha=0.3)
            ax1.set_xticks([0, 90, 180, 270, 360])
            ax1.set_xlim(0, 360)
            ax1.set_ylim(0, )
            ax1.set(ylabel=r'[-]',
                    xlabel='[$^\circ$]')

            # 2-D density plot
            # g = (sns.jointplot(diag_deg, sig_de, color='k', marginal_kws={'color':'k'}).plot_joint(sns.kdeplot, zorder=0, n_levels=10))
            g = (sns.jointplot(diag_deg, sig_kge, kind='kde', zorder=1,
                               n_levels=20, cmap='Greens',
                               marginal_kws={'color':'k', 'shade':False}).plot_joint(sns.scatterplot, color='k', alpha=.5, zorder=2))
            g.set_axis_labels(r'[$^\circ$]', r'KGE [-]')
            g.ax_joint.set_xticks([0, 90, 180, 270, 360])
            g.ax_joint.set_xlim(0, 360)
            g.ax_joint.set_ylim(-ax_lim, 1)
            g.ax_marg_x.set_xticks([0, 90, 180, 270, 360])
            kde_data = g.ax_marg_x.get_lines()[0].get_data()
            kde_xx = kde_data[0]
            kde_yy = kde_data[1]
            x1 = np.where(kde_xx <= 90)[-1][-1]
            x2 = np.where(kde_xx <= 180)[-1][-1]
            x3 = np.where(kde_xx <= 270)[-1][-1]
            g.ax_marg_x.fill_between(kde_xx[:x1+1], kde_yy[:x1+1], facecolor='purple', alpha=0.3)
            g.ax_marg_x.fill_between(kde_xx[x1:x2+2], kde_yy[x1:x2+2], facecolor='grey', alpha=0.3)
            g.ax_marg_x.fill_between(kde_xx[x2+1:x3+1], kde_yy[x2+1:x3+1], facecolor='purple', alpha=0.3)
            g.ax_marg_x.fill_between(kde_xx[x3:], kde_yy[x3:], facecolor='grey', alpha=0.3)
            kde_data = g.ax_marg_y.get_lines()[0].get_data()
            kde_xx = kde_data[0]
            kde_yy = kde_data[1]
            norm = matplotlib.colors.Normalize(vmin=-ax_lim, vmax=1.0)
            colors = cm.Reds_r(norm(kde_yy))
            npts = len(kde_xx)
            for i in range(npts - 1):
                g.ax_marg_y.fill_betweenx([kde_yy[i], kde_yy[i+1]], [kde_xx[i], kde_xx[i+1]], color=colors[i])
            g.fig.tight_layout()

def vis2d_kge_multi(kge_alpha, beta_or_gamma, kge_r, sig_kge, extended=False):
    """
    Multiple polar plot of Kling-Gupta Efficiency (KGE)

    Parameters
    ----------
    kge_alpha: (N,)array_like
        KGE alpha as 1-D array

    kge_beta : (N,)array_like
        KGE beta as 1-D array

    kge_r : (N,)array_like
        KGE r as 1-D array

    sig_kge : (N,)array_like
        KGE as 1-D array

    extended : boolean, optional
        If True, density plot is displayed. In addtion, the density plot
        is displayed besides the polar plot. The default is,
        that only the diagnostic polar plot is displayed.
    """
    sig_min = np.min(sig_kge)

    ll_kge_alpha = kge_alpha.tolist()
    ll_bg = beta_or_gamma.tolist()
    ll_kge_r = kge_r.tolist()
    ll_sig = sig_kge.tolist()

    # convert temporal correlation to color
    norm = matplotlib.colors.Normalize(vmin=-1.0, vmax=1.0)

    delta = 0.01  # for spacing

    # determine axis limits
    if sig_min > 0:
        yy = np.arange(0, 1, delta)[::-1]
        ax_lim = 0
    elif sig_min <= 0 and sig_min > -1:
        yy = np.arange(-1, 1, delta)[::-1]
        ax_lim = 1
    elif sig_min <= -1:
        yy = np.arange(-2, 1, delta)[::-1]
        ax_lim = 2
    elif sig_min <= -2:
        yy = np.arange(-3, 1, delta)[::-1]
        ax_lim = 3

    len_yy = len(yy)

    # arrays to plot contour lines of DE
    xx = np.radians(np.linspace(0, 360, len_yy))
    theta, r = np.meshgrid(xx, yy)

    # arrays to plot contours of P overestimation
    xx1 = np.radians(np.linspace(45, 135, len_yy))
    theta1, r1 = np.meshgrid(xx1, yy)

    # arrays to plot contours of P underestimation
    xx2 = np.radians(np.linspace(225, 315, len_yy))
    theta2, r2 = np.meshgrid(xx2, yy)

    # arrays to plot contours of model errors
    xx3 = np.radians(np.linspace(135, 225, len_yy))
    theta3, r3 = np.meshgrid(xx3, yy)

    # arrays to plot contours of model errors
    len_yy2 = int(len_yy/2)
    if len_yy != len_yy2 + len_yy2:
        xx0 = np.radians(np.linspace(0, 45, len_yy2+1))
    else:
        xx0 = np.radians(np.linspace(0, 45, len_yy2))

    xx360 = np.radians(np.linspace(315, 360, len_yy2))
    xx4 = np.concatenate((xx360, xx0), axis=None)
    theta4, r4 = np.meshgrid(xx4, yy)

    # diagnostic polar plot
    if not extended:
        fig, ax = plt.subplots(figsize=(6, 6),
                               subplot_kw=dict(projection='polar'),
                               constrained_layout=True)
        # dummie plot for colorbar of temporal correlation
        cs = np.arange(-1, 1.1, 0.1)
        dummie_cax = ax.scatter(cs, cs, c=cs, cmap='YlGnBu')
        # Clear axis
        ax.cla()
        # contours P overestimation
        cpio = ax.contourf(theta1, r1, r1, cmap='Purples_r', alpha=.3)
        # contours P underestimation
        cpiu = ax.contourf(theta2, r2, r2, cmap='Purples_r', alpha=.3)
        # contours model errors
        cpmou = ax.contourf(theta3, r3, r3, cmap='Greys_r', alpha=.3)
        cpmuo = ax.contourf(theta4, r4, r4, cmap='Greys_r', alpha=.3)
        # contours of DE
        cp = ax.contour(theta, r, r, colors='black', alpha=.7)
        cl = ax.clabel(cp, inline=True, fontsize=10, fmt='%1.1f', inline_spacing=6)
        # loop over each data point
        for (a, bg, r, sig) in zip(ll_kge_alpha, ll_bg, ll_kge_r, ll_sig):
            ang = np.arctan2(a - 1, bg - 1)
            # convert temporal correlation to color
            rgba_color = cm.YlGnBu(norm(r))
            c = ax.scatter(ang, sig, color=rgba_color)

        ax.tick_params(labelleft=False, labelright=False, labeltop=False,
                      labelbottom=True, grid_alpha=.01)  # turn labels and grid off
        ax.text(-.05, 0.5, r'$\alpha$ - 1 [-]', va='center', ha='center',
                rotation=90, rotation_mode='anchor', transform=ax.transAxes)
        ax.set_xticklabels(['', '', '', '', '', '', r'$\beta$/$\gamma$ - 1 [-]', ''])
        ax.set_rticks([])  # turn default ticks off
        ax.set_rmin(1)
        ax.set_rmax(-ax_lim)
        # add colorbar for temporal correlation
        cbar = fig.colorbar(dummie_cax, ax=ax, orientation='vertical',
                            label='r [-]', ticks=[1, 0.5, 0, -0.5, -1], shrink=0.8)
        cbar.set_ticklabels(['1', '0.5', '0', '-0.5', '-1'])
        cbar.ax.tick_params(direction='in', labelsize=10)

    elif extended:
            fig = plt.figure(figsize=(12, 6), constrained_layout=True)
            gs = fig.add_gridspec(1, 2)
            ax = fig.add_subplot(gs[0, 0], projection='polar')
            ax1 = fig.add_axes([.64, .3, .33, .33], frameon=True)
            # dummie plot for colorbar of temporal correlation
            cs = np.arange(-1, 1.1, 0.1)
            dummie_cax = ax.scatter(cs, cs, c=cs, cmap='YlGnBu')
            # Clear axis
            ax.cla()
            # contours P overestimation
            cpio = ax.contourf(theta1, r1, r1, cmap='Purples_r', alpha=.3)
            # contours P underestimation
            cpiu = ax.contourf(theta2, r2, r2, cmap='Purples_r', alpha=.3)
            # contours model errors
            cpmou = ax.contourf(theta3, r3, r3, cmap='Greys_r', alpha=.3)
            cpmuo = ax.contourf(theta4, r4, r4, cmap='Greys_r', alpha=.3)
            # contours of DE
            cp = ax.contour(theta, r, r, colors='black', alpha=.7)
            cl = ax.clabel(cp, inline=True, fontsize=10, fmt='%1.1f', inline_spacing=6)
            # loop over each data point
            for (a, bg, r, sig) in zip(ll_kge_alpha, ll_bg, ll_kge_r, ll_sig):
                ang = np.arctan2(a - 1, bg - 1)
                # convert temporal correlation to color
                rgba_color = cm.YlGnBu(norm(r))
                c = ax.scatter(ang, sig, color=rgba_color)

            ax.tick_params(labelleft=False, labelright=False, labeltop=False,
                          labelbottom=True, grid_alpha=.01)  # turn labels and grid off
            ax.text(-.05, 0.5, r'$\alpha$ - 1 [-]', va='center', ha='center',
                    rotation=90, rotation_mode='anchor', transform=ax.transAxes)
            ax.set_xticklabels(['', '', '', '', '', '', r'$\beta$/$\gamma$ - 1 [-]', ''])
            ax.set_rticks([])  # turn default ticks off
            ax.set_rmin(1)
            ax.set_rmax(-ax_lim)
            # add colorbar for temporal correlation
            cbar = fig.colorbar(dummie_cax, ax=ax, orientation='vertical',
                                ticks=[1, 0.5, 0, -0.5, -1], shrink=0.8)
            cbar.set_label(r'r [-]', fontsize=12, labelpad=8)
            cbar.set_ticklabels(['1', '0.5', '0', '-0.5', '-1'])
            cbar.ax.tick_params(direction='in', labelsize=10)

            # convert to degrees
            diag = np.arctan2(kge_alpha - 1, beta_or_gamma - 1)
            diag_deg = (diag  * (180 / np.pi)) + 135
            diag_deg[diag_deg < 0] = 360 - diag_deg[diag_deg < 0]

            # 1-D density plot
            g = sns.kdeplot(diag_deg, color='k', ax=ax1)
            kde_data = g.get_lines()[0].get_data()
            kde_xx = kde_data[0]
            kde_yy = kde_data[1]
            x1 = np.where(kde_xx <= 90)[-1][-1]
            x2 = np.where(kde_xx <= 180)[-1][-1]
            x3 = np.where(kde_xx <= 270)[-1][-1]
            ax1.fill_between(kde_xx[:x1+1], kde_yy[:x1+1], facecolor='purple', alpha=0.3)
            ax1.fill_between(kde_xx[x1:x2+2], kde_yy[x1:x2+2], facecolor='grey', alpha=0.3)
            ax1.fill_between(kde_xx[x2+1:x3+1], kde_yy[x2+1:x3+1], facecolor='purple', alpha=0.3)
            ax1.fill_between(kde_xx[x3:], kde_yy[x3:], facecolor='grey', alpha=0.3)
            ax1.set_xticks([0, 90, 180, 270, 360])
            ax1.set_xlim(0, 360)
            ax1.set_ylim(0, )
            ax1.set(ylabel=r'[-]',
                    xlabel='[$^\circ$]')

            # 2-D density plot
            # g = (sns.jointplot(diag_deg, sig_de, color='k', marginal_kws={'color':'k'}).plot_joint(sns.kdeplot, zorder=0, n_levels=10))
            g = (sns.jointplot(diag_deg, sig_kge, kind='kde', zorder=1,
                               n_levels=20, cmap='Greens',
                               marginal_kws={'color':'k', 'shade':False}).plot_joint(sns.scatterplot, color='k', alpha=.5, zorder=2))
            g.set_axis_labels(r'[$^\circ$]', r'KGE [-]')
            g.ax_joint.set_xticks([0, 90, 180, 270, 360])
            g.ax_joint.set_xlim(0, 360)
            g.ax_joint.set_ylim(-ax_lim, 1)
            g.ax_marg_x.set_xticks([0, 90, 180, 270, 360])
            kde_data = g.ax_marg_x.get_lines()[0].get_data()
            kde_xx = kde_data[0]
            kde_yy = kde_data[1]
            x1 = np.where(kde_xx <= 90)[-1][-1]
            x2 = np.where(kde_xx <= 180)[-1][-1]
            x3 = np.where(kde_xx <= 270)[-1][-1]
            g.ax_marg_x.fill_between(kde_xx[:x1+1], kde_yy[:x1+1], facecolor='purple', alpha=0.3)
            g.ax_marg_x.fill_between(kde_xx[x1:x2+2], kde_yy[x1:x2+2], facecolor='grey', alpha=0.3)
            g.ax_marg_x.fill_between(kde_xx[x2+1:x3+1], kde_yy[x2+1:x3+1], facecolor='purple', alpha=0.3)
            g.ax_marg_x.fill_between(kde_xx[x3:], kde_yy[x3:], facecolor='grey', alpha=0.3)
            kde_data = g.ax_marg_y.get_lines()[0].get_data()
            kde_xx = kde_data[0]
            kde_yy = kde_data[1]
            norm = matplotlib.colors.Normalize(vmin=-ax_lim, vmax=1.0)
            colors = cm.Reds_r(norm(kde_yy))
            npts = len(kde_xx)
            for i in range(npts - 1):
                g.ax_marg_y.fill_betweenx([kde_yy[i], kde_yy[i+1]], [kde_xx[i], kde_xx[i+1]], color=colors[i])
            g.fig.tight_layout()


def pos_shift_ts(ts, offset=1.5, multi=True):
    """
    Generate input data errors.

    Precipitation overestimation.

    Mimicking input errors by multiplying/adding with constant offset.

    Parameters
    ----------
    ts : (N,)array_like
        Observed time series

    offset : float, optional
        Offset multiplied/added to time series. If multi true, offset
        has to be greater than 1. The default is 25 % of P overestimation.

    multi : boolean, optional
        If True, offset is multiplied. If False, offset is added. The default
        is multiplication.

    Returns
    ----------
    shift_pos : array_like
        Time series with positve offset
    """
    if multi:
        shift_pos = ts * offset
    else:
        shift_pos = ts + offset

    return shift_pos

def neg_shift_ts(ts, offset=0.5, multi=True):
    """
    Generate input data errors.

    Precipitation underestimation.

    Mimicking input errors by multiplying/subtracting with constant offset.

    Parameters
    ----------
    ts : (N,)array_like
        Observed time series

    offset : float, optional
        Offset multiplied/subtracted to time series. If multi true, offset
        has to be greater than 1. The default is 25 % of P underestimation.

    multi : boolean, optional
        If True, offset is multiplied. If False, offset is subtracted. The
        default is multiplication.

    Returns
    ----------
    shift_neg : array_like
        time series with negative offset
    """
    if multi:
        shift_neg = ts * offset
    else:
        shift_neg = ts - offset
        shift_neg[shift_neg < 0] = 0

    return shift_neg

def highunder_lowover(ts, prop=0.5):
    """
    Generate model errors.

    Underestimate high flows - Overestimate low flows

    Mimicking model errors. High to medium flows are decreased by linear
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

def highover_lowunder(ts, prop=0.5):
    """
    Generate model errors.

    Overestimate high flows - Underestimate low flows.

    Increase max values and decrease min values.

    Mimicking model errors. High to medium flows are increased by linear
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

def time_shift(ts, tshift=3, random=True):
    """
    Generate timing errors.

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
