#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
diagnostic_model_efficiency.de
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
import matplotlib
from matplotlib import cm
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.patches as mpatches
import pandas as pd
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import scipy as sp
import scipy.integrate as integrate
from sklearn import linear_model
from sklearn.utils import shuffle
from sklearn.metrics import r2_score
import seaborn as sns
# controlling figure aesthetics
sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})

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
_q_lab = r'[$m^{3}$ $s^{-1}$]'

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
           xlabel='Time [Days]')

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
    ax.plot(sim.index, sim, color='red')  # simulated time series
    ax.plot(obs.index, obs, color='blue')  # observed time series
    ax.set(ylabel=_q_lab, xlabel='Time')

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
    ax.plot(prob_obs, obs['obs'], color='blue', label='Observed')
    ax.plot(prob_sim, sim['sim'], color='red', label='Simulated')
    ax.set(ylabel=_q_lab,
           xlabel='Exceedence probabilty [-]', yscale='log')
    ax.legend(loc=1)

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
        If True, time series are sorted by ascending order. If False, time series
        are not sorted. The default is to sort.

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
    b_area = integrate.quad(lambda x: integrand(brel_rest_abs, x), 0, 1)

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
    hf_area = integrate.quad(lambda x: integrand(brel_rest, x), 0, .5)

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
        If True, time series are sorted by ascending order. If False, time series
        are not sorted. The default is to sort.

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
    brel_mean = calc_brel_mean(obs, sim)
    # remaining relative bias
    brel_rest = calc_brel_rest(obs, sim)
    # area of relative remaing bias
    b_area = calc_b_area(brel_rest)
    # diagnostic efficiency
    sig = 1 - np.sqrt((brel_mean)**2 + (b_area)**2)

    return sig

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
        obs_std = np.std(obs)
        sim_std = np.std(sim)
        obs_cv = obs_std/obs_mean
        sim_cv = sim_std/sim_mean
        kge_gamma = sim_cv/obs_cv
        temp_cor = calc_temp_cor(obs, sim, r=r)

        sig = 1 - np.sqrt((kge_alpha - 1)**2 + (kge_gamma - 1)**2  + (temp_cor - 1)**2)

    # calculate KGE with beta term
    elif var == 'std':
        obs_std = np.std(obs)
        sim_std = np.std(sim)
        kge_beta = sim_std/obs_std
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

def vis2d_de(obs, sim, sort=True, nd=0., extended=False):
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

    nd : float, optional
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
    if not extended:
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='polar'))
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
        sig_lim = 1 - np.sqrt((nd)**2 + (nd)**2 + (nd)**2)
        # relation of b_dir which explains the error
        if abs(b_area) > 0:
            exp_err = (abs(b_dir) * 2)/abs(b_area)
        elif abs(b_area) == 0:
            exp_err = 0
        # diagnose the error
        if abs(brel_mean) <= nd and exp_err > nd and sig <= sig_lim:
            c = ax.scatter(diag, sig, color=rgba_color)
        elif abs(brel_mean) > nd and exp_err <= nd and sig <= sig_lim:
            c = ax.scatter(diag, sig, color=rgba_color)
        elif abs(brel_mean) > nd and exp_err> nd and sig <= sig_lim:
            c = ax.scatter(diag, sig, color=rgba_color)
        # FBM
        elif abs(brel_mean) <= nd and exp_err <= nd and sig <= sig_lim:
            c = ax.arrow(0, 0, 0, sig, color=rgba_color, lw=6)
            c1 = ax.arrow(0, 0, 3.14, sig, color=rgba_color, lw=6)
        # FGM
        elif abs(brel_mean) <= nd and exp_err <= nd and sig > sig_lim:
            c = ax.scatter(diag, sig, color=rgba_color)
        ax.set_rticks([])  # turn default ticks off
        ax.set_rmin(1)
        ax.set_rmax(-ax_lim)
        ax.tick_params(labelleft=False, labelright=False, labeltop=False,
                      labelbottom=True, grid_alpha=.01)  # turn labels and grid off
        ax.set_xticklabels(['', '', 'P overestimation', '', '', '', 'P underestimation', ''])
        ax.text(-.05, 0.5, 'High flow overestimation - \n Low flow underestimation', va='center', ha='center', rotation=90, rotation_mode='anchor', transform=ax.transAxes)
        ax.text(1.05, 0.5, 'High flow underestimation - \n Low flow overestimation', va='center', ha='center', rotation=90, rotation_mode='anchor', transform=ax.transAxes)
        # add colorbar for temporal correlation
        cax = fig.add_axes([.97, .15, .04, .7], frameon=False)
        cbar = fig.colorbar(dummie_cax, cax=cax, orientation='vertical',
                            label='r [-]', ticks=[1, 0.5, 0, -0.5, -1])
        cbar.set_ticklabels(['1', '0.5', '0', '-0.5', '-1'])
        cbar.ax.tick_params(direction='in', labelsize=10)

    elif extended:
            fig = plt.figure(figsize=(12, 6))
            gs = fig.add_gridspec(1, 2)
            ax = fig.add_subplot(gs[0, 0], projection='polar')
            ax1 = fig.add_axes([.65, .3, .35, .35], frameon=True)
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
            sig_lim = 1 - np.sqrt((nd)**2 + (nd)**2 + (nd)**2)
            # relation of b_dir which explains the error
            if abs(b_area) > 0:
                exp_err = (abs(b_dir) * 2)/abs(b_area)
            elif abs(b_area) == 0:
                exp_err = 0
            # diagnose the error
            if abs(brel_mean) <= nd and exp_err > nd and sig <= sig_lim:
                c = ax.scatter(diag, sig, color=rgba_color)
            elif abs(brel_mean) > nd and exp_err <= nd and sig <= sig_lim:
                c = ax.scatter(diag, sig, color=rgba_color)
            elif abs(brel_mean) > nd and exp_err> nd and sig <= sig_lim:
                c = ax.scatter(diag, sig, color=rgba_color)
            # FBM
            elif abs(brel_mean) <= nd and exp_err <= nd and sig <= sig_lim:
                c = ax.arrow(0, 0, 0, sig, color=rgba_color, lw=6)
                c1 = ax.arrow(0, 0, 3.14, sig, color=rgba_color, lw=6)
            # FGM
            elif abs(brel_mean) <= nd and exp_err <= nd and sig > sig_lim:
                c = ax.scatter(diag, sig, color=rgba_color)
            ax.set_rticks([])  # turn default ticks off
            ax.set_rmin(1)
            ax.set_rmax(-ax_lim)
            ax.tick_params(labelleft=False, labelright=False, labeltop=False,
                          labelbottom=True, grid_alpha=.01)  # turn labels and grid off
            ax.set_xticklabels(['', '', 'P overestimation', '', '', '', 'P underestimation', ''])
            ax.text(-.05, 0.5, 'High flow overestimation - \n Low flow underestimation', va='center', ha='center', rotation=90, rotation_mode='anchor', transform=ax.transAxes)
            ax.text(1.05, 0.5, 'High flow underestimation - \n Low flow overestimation', va='center', ha='center', rotation=90, rotation_mode='anchor', transform=ax.transAxes)
            # add colorbar for temporal correlation
            cax = fig.add_axes([.52, .15, .02, .7], frameon=False)
            cbar = fig.colorbar(dummie_cax, cax=cax, orientation='vertical',
                                label='r [-]', ticks=[1, 0.5, 0, -0.5, -1])
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
        diag = np.arctan2(kge_alpha, kge_gamma)

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
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='polar'))
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
        ax.text(-.05, 0.5, r'$\alpha$ [-]', va='center', ha='center', rotation=90, rotation_mode='anchor', transform=ax.transAxes)
        ax.set_xticklabels(['', '', '', '', '', '', r'$\gamma$ [-]', ''])
        # add colorbar for temporal correlation
        cax = fig.add_axes([.95, .15, .04, .7], frameon=False)
        cbar = fig.colorbar(dummie_cax, cax=cax, orientation='vertical',
                            label='r [-]', ticks=[1, 0.5, 0, -0.5, -1])
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
        diag = np.arctan2(kge_alpha, kge_beta)

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
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='polar'))
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
        ax.text(-.05, 0.5, r'$\alpha$ [-]', va='center', ha='center', rotation=90, rotation_mode='anchor', transform=ax.transAxes)
        ax.set_xticklabels(['', '', '', '', '', '', r'$\beta$ [-]', ''])
        # add colorbar for temporal correlation
        cax = fig.add_axes([.95, .15, .04, .7], frameon=False)
        cbar = fig.colorbar(dummie_cax, cax=cax, orientation='vertical',
                            label='r [-]', ticks=[1, 0.5, 0, -0.5, -1])
        cbar.set_ticklabels(['1', '0.5', '0', '-0.5', '-1'])
        cbar.ax.tick_params(direction='in', labelsize=10)

def pos_shift_ts(ts, offset=1.5, multi=True):
    """
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
        If True, offset is multiplied. If False, offset is subtracted. The default
        is multiplication.

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

def highunder_lowover(ts, prop=0.5):
    """
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
#    idxu1_max = idxu_max.union(idx1_max)

    idx_min = min_peaks_ind + dt.timedelta(days=1)
    idxu_min = min_peaks_ind.union(idx_min)
    idx1_min = min_peaks_ind + dt.timedelta(days=2)
#    idxu1_min = idxu_min.union(idx1_min)

    ts.loc[idx1_max, 'Qobs'] = ts.loc[idx1_max, 'Qobs'] * 1.1
    ts.loc[idx_max, 'Qobs'] = ts.loc[idx_max, 'Qobs'] * 1.2
    ts.loc[max_peaks_ind, 'Qobs'] = ts.loc[max_peaks_ind, 'Qobs'] * 1.3

    ts.loc[idx1_min, 'Qobs'] = ts.loc[idx1_min, 'Qobs'] * 0.5
    ts.loc[idx_min, 'Qobs'] = ts.loc[idx_min, 'Qobs'] * 0.5
    ts.loc[min_peaks_ind, 'Qobs'] = ts.loc[min_peaks_ind, 'Qobs'] * 0.5

#    ts.loc[idxu1_max, 'Qobs'] = ts.loc[idxu1_max, 'Qobs'] * 1.01
#    ts.loc[idxu1_min, 'Qobs'] = ts.loc[idxu1_min, 'Qobs'] * 0.99

    return ts

def highover_lowunder(ts, prop=0.5):
    """
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

def time_shift(ts, tshift=3, random=False):
    """
    Timing errors

    Parameters
    ----------
    ts : dataframe
        dataframe with time series

    tshift : int, optional
        days by which time series is shifted. Both positive and negative
        time shift are possible.

    random : boolean, default False
        whether time series is shuffled or not.

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
        ys = shuffle(y, random_state=0)
        ts_shift.iloc[:, 0] = ys

    return ts_shift

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


if __name__ == "__main__":
   path = '/Users/robinschwemmle/Desktop/PhD/diagnostic_model_efficiency/examples/data/9960682_Q_1970_2012.csv'
##    path = '/Users/robo/Desktop/PhD/de/examples/data/9960682_Q_1970_2012.csv'

   # import observed time series
   df_ts = import_ts(path, sep=';')
#    plot_ts(df_ts)

#    # peak detection
#    obs_arr = df_ts['Qobs'].values
#    ll_ind = df_ts.index.tolist()
#    max_peaks, min_peaks = peakdetect(obs_arr, x_axis=ll_ind, lookahead=7)
#    max_peaks_ind, max_peaks_val = zip(*max_peaks)
#    min_peaks_ind, min_peaks_val = zip(*min_peaks)
#    df_max_peaks = pd.DataFrame(index=max_peaks_ind, data=max_peaks_val, columns=['max_peaks'])
#    df_min_peaks = pd.DataFrame(index=min_peaks_ind, data=min_peaks_val, columns=['min_peaks'])
#    plot_peaks(df_ts, df_max_peaks, df_min_peaks)
#
#    ### increase high flows - decrease low flows ###
#    tsd = disaggregate_obs(df_ts.copy(), df_max_peaks.index, df_min_peaks.index)
#    obs_sim = pd.DataFrame(index=df_ts.index, columns=['Qobs', 'Qsim'])
#    obs_sim.loc[:, 'Qobs'] = df_ts.loc[:, 'Qobs']
#    obs_sim.loc[:, 'Qsim'] = tsd.loc[:, 'Qobs']  # disaggregated time series
#    plot_obs_sim(obs_sim['Qobs'], obs_sim['Qsim'])
#    fdc_obs_sim(obs_sim['Qobs'], obs_sim['Qsim'])
#
#    obs_arr = obs_sim['Qobs'].values
#    sim_arr = obs_sim['Qsim'].values

#    sig_de = calc_de(obs_arr, sim_arr)
#    sig_kge = calc_kge(obs_arr, sim_arr)
#    sig_nse = calc_nse(obs_arr, sim_arr)
#
#    vis2d_de(obs_arr, sim_arr)
#    vis2d_kge(obs_arr, sim_arr)

#    ### increase high flows - decrease low flows ###
#    obs_sim = pd.DataFrame(index=df_ts.index, columns=['Qobs', 'Qsim'])
#    obs_sim.loc[:, 'Qobs'] = df_ts.loc[:, 'Qobs']
#    tsd = highover_lowunder(df_ts.copy(), prop=.99)
#    obs_sim.loc[:, 'Qsim'] = tsd.iloc[:, 0]  # disaggregated time series
#    plot_obs_sim(obs_sim['Qobs'], obs_sim['Qsim'])
#    fdc_obs_sim(obs_sim['Qobs'], obs_sim['Qsim'])
#
#    obs_arr = obs_sim['Qobs'].values
#    sim_arr = obs_sim['Qsim'].values
#
#    sig_de = calc_de(obs_arr, sim_arr)
#    sig_kge = calc_kge(obs_arr, sim_arr)
#    sig_nse = calc_nse(obs_arr, sim_arr)
#
#    vis2d_de(obs_arr, sim_arr)
#    vis2d_kge(obs_arr, sim_arr)

    # obs_sim_sort = sort_obs(obs_sim)
    # obs_arr_sort = obs_sim_sort['Qobs'].values
    # sim_arr_sort = obs_sim_sort['Qsim'].values
    # sig_der_sort = calc_de_sort(obs_arr_sort, sim_arr_sort)
    # sig_de_sort = calc_de(obs_arr_sort, sim_arr_sort, sort=False)
    # fdc_obs_sort(obs_sim_sort)
    # vis2d_de(obs_arr_sort, sim_arr_sort, sort=False)

#    ### decrease high flows - increase low flows ###
#    obs_sim = pd.DataFrame(index=df_ts.index, columns=['Qobs', 'Qsim'])
#    obs_sim.loc[:, 'Qobs'] = df_ts.loc[:, 'Qobs']
#    obs_sim.loc[:, 'Qsim'] = smooth_ts(df_ts['Qobs'], win=5)  # smoothed time series
#    plot_obs_sim(obs_sim['Qobs'], obs_sim['Qsim'])
#    fdc_obs_sim(obs_sim['Qobs'], obs_sim['Qsim'])
#
#    obs_arr = obs_sim['Qobs'].values
#    sim_arr = obs_sim['Qsim'].values
#
#    sig_de = calc_de(obs_arr, sim_arr)
#    sig_kge = calc_kge(obs_arr, sim_arr)
#    sig_nse = calc_nse(obs_arr, sim_arr)
#
#    vis2d_de(obs_arr, sim_arr)
#    vis2d_kge(obs_arr, sim_arr)
#
#    ### decrease high flows - increase low flows ###
#    obs_sim = pd.DataFrame(index=df_ts.index, columns=['Qobs', 'Qsim'])
#    obs_sim.loc[:, 'Qobs'] = df_ts.loc[:, 'Qobs']
#    tss = highunder_lowover(df_ts.copy())  # smoothed time series
#    obs_sim.loc[:, 'Qsim'] = tss.iloc[:, 0]  # smoothed time series
#    plot_obs_sim(obs_sim['Qobs'], obs_sim['Qsim'])
#    fdc_obs_sim(obs_sim['Qobs'], obs_sim['Qsim'])
#
#    obs_arr = obs_sim['Qobs'].values
#    sim_arr = obs_sim['Qsim'].values
#
#    sig_de = calc_de(obs_arr, sim_arr)
#    sig_kge = calc_kge(obs_arr, sim_arr)
#    sig_nse = calc_nse(obs_arr, sim_arr)
#
#    vis2d_de(obs_arr, sim_arr)
#    vis2d_kge(obs_arr, sim_arr)
#
   ### precipitation surplus ###
   obs_sim = pd.DataFrame(index=df_ts.index, columns=['Qobs', 'Qsim'])
   obs_sim.loc[:, 'Qobs'] = df_ts.loc[:, 'Qobs']
   obs_sim.loc[:, 'Qsim'] = pos_shift_ts(df_ts['Qobs'].values)  # positive offset
   plot_obs_sim(obs_sim['Qobs'], obs_sim['Qsim'])
   fdc_obs_sim(obs_sim['Qobs'], obs_sim['Qsim'])

   obs_arr = obs_sim['Qobs'].values
   sim_arr = obs_sim['Qsim'].values

   sig_de = calc_de(obs_arr, sim_arr)
   sig_kge = calc_kge(obs_arr, sim_arr)
   sig_nse = calc_nse(obs_arr, sim_arr)

   vis2d_de(obs_arr, sim_arr)
   vis2d_kge(obs_arr, sim_arr)

#    ### precipitation shortage ###
#    obs_sim = pd.DataFrame(index=df_ts.index, columns=['Qobs', 'Qsim'])
#    obs_sim.loc[:, 'Qobs'] = df_ts.loc[:, 'Qobs']
#    obs_sim.loc[:, 'Qsim'] = neg_shift_ts(df_ts['Qobs'].values)  # negative offset
#    plot_obs_sim(obs_sim['Qobs'], obs_sim['Qsim'])
#    fdc_obs_sim(obs_sim['Qobs'], obs_sim['Qsim'])
#
#    obs_arr = obs_sim['Qobs'].values
#    sim_arr = obs_sim['Qsim'].values
#
#    sig_de = calc_de(obs_arr, sim_arr)
#    sig_kge = calc_kge(obs_arr, sim_arr)
#    sig_nse = calc_nse(obs_arr, sim_arr)
#
#    vis2d_de(obs_arr, sim_arr)
#    vis2d_kge(obs_arr, sim_arr)


#    ### averaged time series ###
#    obs_sim = pd.DataFrame(index=df_ts.index, columns=['Qobs', 'Qsim'])
#    obs_sim.loc[:, 'Qobs'] = df_ts.loc[:, 'Qobs']
#    obs_mean = np.mean(obs_sim['Qobs'].values)
#    obs_sim.loc[:, 'Qsim'] = np.repeat(obs_mean, len(obs_sim['Qobs'].values))
#    plot_obs_sim(obs_sim['Qobs'], obs_sim['Qsim'])
#    fdc_obs_sim(obs_sim['Qobs'], obs_sim['Qsim'])
#
#    obs_arr = obs_sim['Qobs'].values
#    sim_arr = obs_sim['Qsim'].values
#
#    sig_de = calc_de(obs_arr, sim_arr)
#    sig_kge = calc_kge(obs_arr, sim_arr)
#    sig_nse = calc_nse(obs_arr, sim_arr)
#
#    vis2d_de(obs_arr, sim_arr)
#    vis2d_kge(obs_arr, sim_arr)


#    ### Tier-1 ###
#    path_wrr1 = '/Users/robinschwemmle/Desktop/PhD/diagnostic_model_efficiency/examples/data/GRDC_4103631_wrr1.csv'
#    df_wrr1 = import_ts(path_wrr1, sep=';')
#    fdc_obs_sim(df_wrr1['Qobs'], df_wrr1['Qsim'])
#    plot_obs_sim(df_wrr1['Qobs'], df_wrr1['Qsim'])
#
#    obs_arr = df_wrr1['Qobs'].values
#    sim_arr = df_wrr1['Qsim'].values
#
#    sig_de = calc_de(obs_arr, sim_arr)
#    sig_kge = calc_kge(obs_arr, sim_arr)
#    sig_nse = calc_nse(obs_arr, sim_arr)
#
#    vis2d_de(obs_arr, sim_arr)
#    vis2d_kge(obs_arr, sim_arr)

   # ### Tier-2 ###
   # path_wrr2 = '/Users/robinschwemmle/Desktop/PhD/diagnostic_model_efficiency/examples/data/GRDC_4103631_wrr2.csv'
   # df_wrr2 = import_ts(path_wrr2, sep=';')
   # fdc_obs_sim(df_wrr2['Qobs'], df_wrr2['Qsim'])
   # plot_obs_sim(df_wrr2['Qobs'], df_wrr2['Qsim'])
   #
   # obs_arr = df_wrr2['Qobs'].values
   # sim_arr = df_wrr2['Qsim'].values
   #
   # sig_de = calc_de(obs_arr, sim_arr)
   # sig_kge = calc_kge(obs_arr, sim_arr)
   # sig_nse = calc_nse(obs_arr, sim_arr)
   #
   # vis2d_de(obs_arr, sim_arr)
   # vis2d_kge(obs_arr, sim_arr)
