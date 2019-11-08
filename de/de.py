#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
de.de
~~~~~~~~~~~
Diagnosing model performance using an efficiency measure based on flow
duration curve and temoral correlation. The efficiency measure can be
visualized in 2D-Plot which facilitates decomposing potential error origins
(dynamic errors vs. constant erros vs. timing errors)
:2019 by Robin Schwemmle.
:license: GNU GPLv3, see LICENSE for more details.
"""

import numpy as np
# RunTimeWarning will not be displayed (division by zeros or NaN values)
np.seterr(divide='ignore', invalid='ignore')
import matplotlib
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
sns.set_context("paper", font_scale=1.5)

__title__ = 'de'
__version__ = '0.1'
#__build__ = 0x001201
__author__ = 'Robin Schwemmle'
__license__ = 'GNU GPLv3'
#__docformat__ = 'markdown'

#TODO: consistent datatype
#TODO: match Qsim to Qobs

_mmd = r'[mm $d^{-1}$]'
_m3s = r'[$m^{3}$ $s^{-1}$]'
_q_lab = _mmd
_sim_lab = 'Manipulated'

def fdc(ts):
    """Generate a flow duration curve for a single hydrologic time series.

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
    ax.plot(prob_obs, obs['obs'], color='blue', lw=2, label='Observed')
    ax.plot(prob_sim, sim['sim'], color='red', lw=1, ls='-.', label=_sim_lab,
            alpha=.8)
    ax.set(ylabel=_q_lab,
           xlabel='Exceedence probabilty [-]', yscale='log')
    ax.legend(loc=1)
    ax.set_ylim(0, )
    ax.set_xlim(0, 1)

def calc_brel_mean(obs, sim, sort=True):
    """Calculate arithmetic mean of relative bias.

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

    Notes
    ----------
    .. math::

        \overline{B_{rel}} = \frac{1}{N}\sum_{i=1}^{N} B_{rel}(i)


    Examples
    --------
    Provide arrays with equal length

    >>> from de import de
    >>> import numpy as np
    >>> obs = np.array([1.5, 1, 0.8, 0.85, 1.5, 2])
    >>> sim = np.array([1.6, 1.3, 1, 0.8, 1.2, 2.5])
    >>> de.calc_brel_mean(obs, sim)
    0.09330065359477124
    """
    if len(obs) != len(sim):
        raise AssertionError("Arrays are not of equal length!")

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
    """Subtract arithmetic mean of relative bias from relative bias.

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

    Notes
    ----------
    .. math::

        B_{rest}(i) = B_{rel}(i) - \overline{B_{rel}}

    Examples
    --------
    Provide arrays with equal length

    >>> from de import de
    >>> import numpy as np
    >>> obs = np.array([1.5, 1, 0.8, 0.85, 1.5, 2])
    >>> sim = np.array([1.6, 1.3, 1, 0.8, 1.2, 2.5])
    >>> de.calc_brel_rest(obs, sim)
    array([ 0.15669935, -0.02663399, -0.22663399,  0.10669935,  0.08316993,
       -0.09330065])
    """
    if len(obs) != len(sim):
        raise AssertionError("Arrays are not of equal length!")

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
    """Function to intergrate bias.

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

def calc_bias_area(brel_rest):
    """Calculate absolute bias area for high flow and low flow.

    Parameters
    ----------
    brel_rest : (N,)array_like
        remaining relative bias as 1-D array

    Returns
    ----------
    b_area[0] : float
        bias area

    Notes
    ----------
    .. math::

        \vert B_{area}\vert = \int_{0}^{1}\vert B_{rest}(i)\vert di

    Examples
    --------
    Provide arrays with equal length

    >>> from de import de
    >>> import numpy as np
    >>> obs = np.array([1.5, 1, 0.8, 0.85, 1.5, 2])
    >>> sim = np.array([1.6, 1.3, 1, 0.8, 1.2, 2.5])
    >>> b_rest = de.calc_brel_rest(obs, sim)
    >>> de.calc_bias_area(b_rest)
    0.11527287549968694
    """
    brel_rest_abs = abs(brel_rest)
    # area of bias
    b_area = integrate.quad(lambda x: integrand(brel_rest_abs, x), 0.001, .999,
                            limit=10000)

    return b_area[0]

def calc_bias_dir(brel_rest):
    """Calculate absolute bias area for high flow and low flow.

    Parameters
    ----------
    brel_rest : (N,)array_like
        remaining relative bias as 1-D array

    Returns
    ----------
    b_dir : float
        direction of bias

    Notes
    ----------
    .. math::

        B_{dir} = \int_{0}^{0.5}B_{rest}(i) di

    Examples
    --------
    Provide arrays with equal length

    >>> from de import de
    >>> import numpy as np
    >>> obs = np.array([1.5, 1, 0.8, 0.85, 1.5, 2])
    >>> sim = np.array([1.6, 1.3, 1, 0.8, 1.2, 2.5])
    >>> b_rest = de.calc_brel_rest(obs, sim)
    >>> de.calc_bias_dir(b_rest)
    -0.0160625816993464
    """
    # integral of relative bias < 50 %
    hf_area = integrate.quad(lambda x: integrand(brel_rest, x), 0.001, .5,
                             limit=10000)

    # direction of bias
    b_dir = hf_area[0]

    return b_dir

def calc_bias_slope(b_area, b_dir):
    """Calculate slope of bias balance.

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

    Notes
    ----------
    .. math::

        B_{slope} =
        \begin{cases}
        \vert B_{area}\vert \times (-1) & \text{if } B_{dir} > 0 \\
        \vert B_{area}\vert       & \text{if } B_{dir} < 0 \\
        0       & \text{if } B_{dir} = 0
        \end{cases}

    Examples
    --------
    Provide arrays with equal length

    >>> from de import de
    >>> import numpy as np
    >>> obs = np.array([1.5, 1, 0.8, 0.85, 1.5, 2])
    >>> sim = np.array([1.6, 1.3, 1, 0.8, 1.2, 2.5])
    >>> b_rest = de.calc_brel_rest(obs, sim)
    >>> b_dir = de.calc_bias_dir(b_rest)
    >>> b_area = de.calc_bias_area(b_rest)
    >>> de.calc_bias_slope(b_area, b_dir)
    0.11527287549968694
    """
    if b_dir > 0:
        b_slope = b_area * (-1)

    elif b_dir < 0:
        b_slope = b_area

    elif b_dir == 0:
        b_slope = 0

    return b_slope

def calc_temp_cor(obs, sim, r='pearson'):
    """Calculate temporal correlation between observed and simulated
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

    Examples
    --------
    Provide arrays with equal length

    >>> from de import de
    >>> import numpy as np
    >>> obs = np.array([1.5, 1, 0.8, 0.85, 1.5, 2])
    >>> sim = np.array([1.6, 1.3, 1, 0.8, 1.2, 2.5])
    >>> de.calc_temp_cor(obs, sim)
    0.8940281850583509
    """
    if len(obs) != len(sim):
        raise AssertionError("Arrays are not of equal length!")

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

def calc_de_nn(obs, sim, sort=True):
    r"""
    Calculate non-normalized Diagnostic-Efficiency (DE).

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
        non-normalized Diagnostic efficiency

    Notes
    ----------
    .. math::

        DE = 1 - \sqrt{\overline{B_{rel}}^2 + \vert B_{area}\vert^2 + (r - 1)^2}

    Examples
    --------
    Provide arrays with equal length

    >>> from de import de
    >>> import numpy as np
    >>> obs = np.array([1.5, 1, 0.8, 0.85, 1.5, 2])
    >>> sim = np.array([1.6, 1.3, 1, 0.8, 1.2, 2.5])
    >>> de.calc_de_nn(obs, sim)
    0.8177285723180813
    """
    if len(obs) != len(sim):
        raise AssertionError("Arrays are not of equal length!")
    # mean relative bias
    brel_mean = calc_brel_mean(obs, sim, sort=sort)
    # remaining relative bias
    brel_rest = calc_brel_rest(obs, sim, sort=sort)
    # area of relative remaing bias
    b_area = calc_bias_area(brel_rest)
    # temporal correlation
    temp_cor = calc_temp_cor(obs, sim)
    # diagnostic efficiency
    sig_de = 1 - np.sqrt((brel_mean)**2 + (b_area)**2 + (temp_cor - 1)**2)
    sig_mfb = calc_de_mfb(obs)
    sig = (sig_de - sig_mfb)/(1 - sig_mfb)

    return sig

def calc_de_mfb(obs, sort=True):
    r"""
    Calculate mean flow benchmark of Diagnostic-Efficiency (DE).

    Parameters
    ----------
    obs : (N,)array_like
        Observed time series as 1-D array

    sort : boolean, optional
        If True, time series are sorted by ascending order. If False, time
        series are not sorted. The default is to sort.

    Returns
    ----------
    sig : float
        Diagnostic efficiency of mean flow benchmark

    Notes
    ----------
    .. math::

        DE = 1 - \sqrt{\overline{B_{rel}}^2 + \vert B_{area}\vert^2 + (r - 1)^2}

    Examples
    --------
    Provide arrays with equal length

    >>> from de import de
    >>> import numpy as np
    >>> obs = np.array([1.5, 1, 0.8, 0.85, 1.5, 2])
    >>> de.calc_de_mfb(obs)
    """
    obs_mean = np.mean(obs)
    sim = np.repeat(obs_mean, len(obs))
    # mean relative bias
    brel_mean = calc_brel_mean(obs, sim, sort=sort)
    # remaining relative bias
    brel_rest = calc_brel_rest(obs, sim, sort=sort)
    # area of relative remaing bias
    b_area = calc_bias_area(brel_rest)
    # temporal correlation
    temp_cor = calc_temp_cor(obs, sim)
    # diagnostic efficiency
    sig = 1 - np.sqrt((brel_mean)**2 + (b_area)**2 + (temp_cor - 1)**2)

    return sig

def calc_de(obs, sim, sort=True):
    r"""
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
        Diagnostic efficiency

    Notes
    ----------
    .. math::

        DE_{nn} = 1 - \sqrt{\overline{B_{rel}}^2 + \vert B_{area}\vert^2 + (r - 1)^2}
        DE = \frac{DE_{nn} - DE_{mfb}}{1 - DE_{mfb}}

    Examples
    --------
    Provide arrays with equal length

    >>> from de import de
    >>> import numpy as np
    >>> obs = np.array([1.5, 1, 0.8, 0.85, 1.5, 2])
    >>> sim = np.array([1.6, 1.3, 1, 0.8, 1.2, 2.5])
    >>> de.calc_de(obs, sim)
    0.8177285723180813
    """
    if len(obs) != len(sim):
        raise AssertionError("Arrays are not of equal length!")
    # mean relative bias
    brel_mean = calc_brel_mean(obs, sim, sort=sort)
    # remaining relative bias
    brel_rest = calc_brel_rest(obs, sim, sort=sort)
    # area of relative remaing bias
    b_area = calc_bias_area(brel_rest)
    # temporal correlation
    temp_cor = calc_temp_cor(obs, sim)
    # diagnostic efficiency
    sig_de = 1 - np.sqrt((brel_mean)**2 + (b_area)**2 + (temp_cor - 1)**2)
    sig_mfb = calc_de_mfb(obs)
    sig = (sig_de - sig_mfb)/(1 - sig_mfb)

    return sig

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

def vis2d_de_nn(obs, sim, sort=True, lim=0.05, extended=False):
    """Polar plot of non-normalized Diagnostic-Efficiency (DE)

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

    Notes
    ----------
    .. math::

        \varphi = arctan2(\overline{B_{rel}}, B_{slope})

    Examples
    --------
    Provide arrays with equal length

    >>> from de import de
    >>> import numpy as np
    >>> obs = np.array([1.5, 1, 0.8, 0.85, 1.5, 2])
    >>> sim = np.array([1.6, 1.3, 1, 0.8, 1.2, 2.5])
    >>> de.vis2d_de(obs, sim)
    """
    if len(obs) != len(sim):
        raise AssertionError("Arrays are not of equal length!")
    # mean relative bias
    brel_mean = calc_brel_mean(obs, sim, sort=sort)

    # remaining relative bias
    brel_rest = calc_brel_rest(obs, sim, sort=sort)
    # area of relative remaing bias
    b_area = calc_bias_area(brel_rest)
    # temporal correlation
    temp_cor = calc_temp_cor(obs, sim)
    # diagnostic efficiency
    sig = 1 - np.sqrt((brel_mean)**2 + (b_area)**2 + (temp_cor - 1)**2)
    sig = np.round(sig, decimals=2)  # round to 2 decimals

    # direction of bias
    b_dir = calc_bias_dir(brel_rest)

    # slope of bias
    b_slope = calc_bias_slope(b_area, b_dir)

    # convert to radians
    # (y, x) Trigonometric inverse tangent
    diag = np.arctan2(brel_mean, b_slope)

    # convert temporal correlation to color
    norm = matplotlib.colors.Normalize(vmin=-1.0, vmax=1.0)
    rgba_color = cm.YlGnBu(norm(temp_cor))

    delta = 0.01  # for spacing

    # determine axis limits
    if sig > 0:
        ax_lim = sig - .1
        ax_lim = np.around(ax_lim, decimals=1)
        yy = np.arange(ax_lim, 1.01, delta)[::-1]
        c_levels = np.arange(ax_lim+.1, 1.1, .1)
    elif sig >= 0:
        ax_lim = 0.2
        yy = np.arange(-ax_lim, 1.01, delta)[::-1]
        c_levels = np.arange(0, 1, .2)
    elif sig < 0 and sig >= -1:
        ax_lim = 1.2
        yy = np.arange(-ax_lim, 1.01, delta)[::-1]
        c_levels = np.arange(-1, 1, .2)
    elif sig >= -2 and sig < -1:
        ax_lim = 2.2
        yy = np.arange(-ax_lim, 1.01, delta)[::-1]
        c_levels = np.arange(-2, 1, .2)
    elif sig <= -2:
        raise AssertionError("Value of 'DE' is too low for visualization!", sig)

    len_yy = len(yy)

    # arrays to plot contour lines of DE
    xx = np.radians(np.linspace(0, 360, len_yy))
    theta, r = np.meshgrid(xx, yy)

    # # arrays to plot contours of positive constant offset
    # xx1 = np.radians(np.linspace(45, 135, len_yy))
    # theta1, r1 = np.meshgrid(xx1, yy)
    #
    # # arrays to plot contours of negative constant offset
    # xx2 = np.radians(np.linspace(225, 315, len_yy))
    # theta2, r2 = np.meshgrid(xx2, yy)
    #
    # # arrays to plot contours of dynamic model errors
    # xx3 = np.radians(np.linspace(135, 225, len_yy))
    # theta3, r3 = np.meshgrid(xx3, yy)
    #
    # # arrays to plot contours of dynamic model errors
    # len_yy2 = int(len_yy/2)
    # if len_yy != len_yy2 + len_yy2:
    #     xx0 = np.radians(np.linspace(0, 45, len_yy2+1))
    # else:
    #     xx0 = np.radians(np.linspace(0, 45, len_yy2))
    #
    # xx360 = np.radians(np.linspace(315, 360, len_yy2))
    # xx4 = np.concatenate((xx360, xx0), axis=None)
    # theta4, r4 = np.meshgrid(xx4, yy)

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
        # # contours positive constant offset
        # cpio = ax.contourf(theta1, r1, r1, cmap='Purples_r', alpha=.2)
        # # contours negative constant offset
        # cpiu = ax.contourf(theta2, r2, r2, cmap='Purples_r', alpha=.2)
        # # contours dynamic model errors
        # cpmou = ax.contourf(theta3, r3, r3, cmap='Greys_r', alpha=.2)
        # cpmuo = ax.contourf(theta4, r4, r4, cmap='Greys_r', alpha=.2)
        # plot regions
        ax.plot((1, np.deg2rad(45)), (1, np.min(yy)), color='lightgray', linewidth=1.5, ls='--', zorder=0)
        ax.plot((1, np.deg2rad(135)), (1, np.min(yy)), color='lightgray', linewidth=1.5, ls='--', zorder=0)
        ax.plot((1, np.deg2rad(225)), (1, np.min(yy)), color='lightgray', linewidth=1.5, ls='--', zorder=0)
        ax.plot((1, np.deg2rad(315)), (1, np.min(yy)), color='lightgray', linewidth=1.5, ls='--', zorder=0)
        # contours of DE
        cp = ax.contour(theta, r, r, colors='darkgray', levels=c_levels, zorder=1)
        cl = ax.clabel(cp, inline=True, fontsize=10, fmt='%1.1f',
                       colors='dimgrey')
        # threshold efficiency for FBM
        sig_lim = 1 - np.sqrt((lim)**2 + (lim)**2 + (lim)**2)
        # relation of b_dir which explains the error
        if abs(b_area) > 0:
            exp_err = (abs(b_dir) * 2)/abs(b_area)
        elif abs(b_area) == 0:
            exp_err = 0
        # diagnose the error
        if abs(brel_mean) <= lim and exp_err > lim and sig <= sig_lim:
            ax.annotate("", xytext=(0, 1), xy=(diag, sig),
                        arrowprops=dict(facecolor=rgba_color))
        elif abs(brel_mean) > lim and exp_err <= lim and sig <= sig_lim:
            ax.annotate("", xytext=(0, 1), xy=(diag, sig),
                        arrowprops=dict(facecolor=rgba_color))
        elif abs(brel_mean) > lim and exp_err > lim and sig <= sig_lim:
            ax.annotate("", xytext=(0, 1), xy=(diag, sig),
                        arrowprops=dict(facecolor=rgba_color))
        # FBM
        elif abs(brel_mean) <= lim and exp_err <= lim and sig <= sig_lim:
            ax.annotate("", xytext=(0, 1), xy=(0, sig),
                        arrowprops=dict(facecolor=rgba_color))
            ax.annotate("", xytext=(0, 1), xy=(np.pi, sig),
                        arrowprops=dict(facecolor=rgba_color))
        # FGM
        elif abs(brel_mean) <= lim and exp_err <= lim and sig > sig_lim:
            c = ax.scatter(diag, sig, color=rgba_color)
        ax.set_rticks([])  # turn default ticks off
        ax.set_rmin(1)
        if sig > 0:
            ax.set_rmax(ax_lim)
        elif sig <= 0:
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
                            ticks=[1, 0.5, 0, -0.5, -1], shrink=0.8)
        cbar.set_label('r [-]', labelpad=4)
        cbar.set_ticklabels(['1', '0.5', '0', '-0.5', '-1'])
        cbar.ax.tick_params(direction='in')

    elif extended:
            fig = plt.figure(figsize=(12, 6), constrained_layout=True)
            gs = fig.add_gridspec(1, 2)
            ax = fig.add_subplot(gs[0, 0], projection='polar')
            ax1 = fig.add_axes([.65, .3, .33, .33], frameon=True)
            # dummie plot for colorbar of temporal correlation
            cs = np.arange(-1, 1.1, 0.1)
            dummie_cax = ax.scatter(cs, cs, c=cs, cmap='YlGnBu')
            # Clear axis
            ax.cla()
            # # contours positive constant offset
            # cpio = ax.contourf(theta1, r1, r1, cmap='Purples_r', alpha=.2)
            # # contours negative constant offset
            # cpiu = ax.contourf(theta2, r2, r2, cmap='Purples_r', alpha=.2)
            # # contours dynamic model errors
            # cpmou = ax.contourf(theta3, r3, r3, cmap='Greys_r', alpha=.2)
            # cpmuo = ax.contourf(theta4, r4, r4, cmap='Greys_r', alpha=.2)
            # plot regions
            ax.plot((1, np.deg2rad(45)), (1, np.min(yy)), color='lightgray', linewidth=1.5, ls='--', zorder=0)
            ax.plot((1, np.deg2rad(135)), (1, np.min(yy)), color='lightgray', linewidth=1.5, ls='--', zorder=0)
            ax.plot((1, np.deg2rad(225)), (1, np.min(yy)), color='lightgray', linewidth=1.5, ls='--', zorder=0)
            ax.plot((1, np.deg2rad(315)), (1, np.min(yy)), color='lightgray', linewidth=1.5, ls='--', zorder=0)
            # contours of DE
            cp = ax.contour(theta, r, r, colors='darkgray', levels=c_levels, zorder=1)
            cl = ax.clabel(cp, inline=True, fontsize=10, fmt='%1.1f',
                           colors='dimgrey')
            # threshold efficiency for FBM
            sig_lim = 1 - np.sqrt((lim)**2 + (lim)**2 + (lim)**2)
            # relation of b_dir which explains the error
            if abs(b_area) > 0:
                exp_err = (abs(b_dir) * 2)/abs(b_area)
            elif abs(b_area) == 0:
                exp_err = 0
            # diagnose the error
            if abs(brel_mean) <= lim and exp_err > lim and sig <= sig_lim:
                ax.annotate("", xytext=(0, 1), xy=(diag, sig),
                            arrowprops=dict(facecolor=rgba_color))
            elif abs(brel_mean) > lim and exp_err <= lim and sig <= sig_lim:
                ax.annotate("", xytext=(0, 1), xy=(diag, sig),
                            arrowprops=dict(facecolor=rgba_color))
            elif abs(brel_mean) > lim and exp_err > lim and sig <= sig_lim:
                ax.annotate("", xytext=(0, 1), xy=(diag, sig),
                            arrowprops=dict(facecolor=rgba_color))
            # FBM
            elif abs(brel_mean) <= lim and exp_err <= lim and sig <= sig_lim:
                ax.annotate("", xytext=(0, 1), xy=(0, sig),
                            arrowprops=dict(facecolor=rgba_color))
                ax.annotate("", xytext=(0, 1), xy=(np.pi, sig),
                            arrowprops=dict(facecolor=rgba_color))
            # FGM
            elif abs(brel_mean) <= lim and exp_err <= lim and sig > sig_lim:
                c = ax.scatter(diag, sig, color=rgba_color)
            ax.set_rticks([])  # turn default ticks off
            ax.set_rmin(1)
            if sig > 0:
                ax.set_rmax(ax_lim)
            elif sig <= 0:
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
                                ticks=[1, 0.5, 0, -0.5, -1], shrink=0.8)
            cbar.set_label('r [-]', labelpad=4)
            cbar.set_ticklabels(['1', '0.5', '0', '-0.5', '-1'])
            cbar.ax.tick_params(direction='in')

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

def vis2d_de_nn_multi(brel_mean, b_area, temp_cor, sig_de, b_dir, diag,
                      lim=0.05, extended=False):
    """Multiple polar plot of non-normalized Diagnostic-Efficiency (DE)

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
    norm = matplotlib.colors.Normalize(vmin=-1.0, vmax=1.0)

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
    elif sig_min <= -2:
        raise ValueError("Some values of 'DE' are too low for visualization!", sig_min)

    len_yy = len(yy)

    # arrays to plot contour lines of DE
    xx = np.radians(np.linspace(0, 360, len_yy))
    theta, r = np.meshgrid(xx, yy)

    # # arrays to plot contours of positive constant offset
    # xx1 = np.radians(np.linspace(45, 135, len_yy))
    # theta1, r1 = np.meshgrid(xx1, yy)
    #
    # # arrays to plot contours of negative constant offset
    # xx2 = np.radians(np.linspace(225, 315, len_yy))
    # theta2, r2 = np.meshgrid(xx2, yy)
    #
    # # arrays to plot contours of dynamic model errors
    # xx3 = np.radians(np.linspace(135, 225, len_yy))
    # theta3, r3 = np.meshgrid(xx3, yy)
    #
    # # arrays to plot contours of dynamic model errors
    # len_yy2 = int(len_yy/2)
    # if len_yy != len_yy2 + len_yy2:
    #     xx0 = np.radians(np.linspace(0, 45, len_yy2+1))
    # else:
    #     xx0 = np.radians(np.linspace(0, 45, len_yy2))
    #
    # xx360 = np.radians(np.linspace(315, 360, len_yy2))
    # xx4 = np.concatenate((xx360, xx0), axis=None)
    # theta4, r4 = np.meshgrid(xx4, yy)

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
        # # contours positive constant offset
        # cpio = ax.contourf(theta1, r1, r1, cmap='Purples_r', alpha=.2)
        # # contours negative constant offset
        # cpiu = ax.contourf(theta2, r2, r2, cmap='Purples_r', alpha=.2)
        # # contours dynamic model errors
        # cpmou = ax.contourf(theta3, r3, r3, cmap='Greys_r', alpha=.2)
        # cpmuo = ax.contourf(theta4, r4, r4, cmap='Greys_r', alpha=.2)
        # plot regions
        ax.plot((1, np.deg2rad(45)), (1, np.min(yy)), color='lightgray', linewidth=1.5, ls='--', zorder=0)
        ax.plot((1, np.deg2rad(135)), (1, np.min(yy)), color='lightgray', linewidth=1.5, ls='--', zorder=0)
        ax.plot((1, np.deg2rad(225)), (1, np.min(yy)), color='lightgray', linewidth=1.5, ls='--', zorder=0)
        ax.plot((1, np.deg2rad(315)), (1, np.min(yy)), color='lightgray', linewidth=1.5, ls='--', zorder=0)
        # contours of DE
        cp = ax.contour(theta, r, r, colors='darkgray', levels=c_levels, zorder=1)
        cl = ax.clabel(cp, inline=True, fontsize=10, fmt='%1.1f',
                       colors='dimgrey')
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
                ax.annotate("", xytext=(0, 1), xy=(0, sig),
                            arrowprops=dict(facecolor=rgba_color))
                ax.annotate("", xytext=(0, 1), xy=(np.pi, sig),
                            arrowprops=dict(facecolor=rgba_color))
            # FGM
            elif abs(bm) <= lim and exp_err <= lim and sig > sig_lim:
                c = ax.scatter(ang, sig, color=rgba_color, zorder=2)
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
                            ticks=[1, 0.5, 0, -0.5, -1], shrink=0.8)
        cbar.set_label('r [-]', labelpad=4)
        cbar.set_ticklabels(['1', '0.5', '0', '-0.5', '-1'])
        cbar.ax.tick_params(direction='in')

    elif extended:
            fig = plt.figure(figsize=(12, 6), constrained_layout=True)
            gs = fig.add_gridspec(1, 2)
            ax = fig.add_subplot(gs[0, 0], projection='polar')
            ax1 = fig.add_axes([.66, .3, .32, .32], frameon=True)
            # dummie plot for colorbar of temporal correlation
            cs = np.arange(-1, 1.1, 0.1)
            dummie_cax = ax.scatter(cs, cs, c=cs, cmap='YlGnBu')
            # Clear axis
            ax.cla()
            # # contours positive constant offset
            # cpio = ax.contourf(theta1, r1, r1, cmap='Purples_r', alpha=.2)
            # # contours negative constant offset
            # cpiu = ax.contourf(theta2, r2, r2, cmap='Purples_r', alpha=.2)
            # # contours dynamic model errors
            # cpmou = ax.contourf(theta3, r3, r3, cmap='Greys_r', alpha=.2)
            # cpmuo = ax.contourf(theta4, r4, r4, cmap='Greys_r', alpha=.2)
            # plot regions
            ax.plot((1, np.deg2rad(45)), (1, np.min(yy)), color='lightgray', linewidth=1.5, ls='--', zorder=0)
            ax.plot((1, np.deg2rad(135)), (1, np.min(yy)), color='lightgray', linewidth=1.5, ls='--', zorder=0)
            ax.plot((1, np.deg2rad(225)), (1, np.min(yy)), color='lightgray', linewidth=1.5, ls='--', zorder=0)
            ax.plot((1, np.deg2rad(315)), (1, np.min(yy)), color='lightgray', linewidth=1.5, ls='--', zorder=0)
            # contours of DE
            cp = ax.contour(theta, r, r, colors='darkgray', levels=c_levels, zorder=1)
            cl = ax.clabel(cp, inline=True, fontsize=10, fmt='%1.1f',
                           colors='dimgrey')
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
                    ax.annotate("", xytext=(0, 1), xy=(0, sig),
                                arrowprops=dict(facecolor=rgba_color))
                    ax.annotate("", xytext=(0, 1), xy=(np.pi, sig),
                                arrowprops=dict(facecolor=rgba_color))
                # FGM
                elif abs(bm) <= lim and exp_err <= lim and sig > sig_lim:
                    c = ax.scatter(ang, sig, color=rgba_color, zorder=2)
            ax.set_rticks([])  # turn default ticks off
            ax.set_rmin(1)
            if sig_min > 0:
                ax.set_rmax(ax_lim)
            elif sig_min <= 0:
                ax.set_rmax(-ax_lim)
            ax.tick_params(labelleft=False, labelright=False, labeltop=False,
                          labelbottom=True, grid_alpha=.01)  # turn labels and grid off
            ax.set_xticklabels(['', '', '', '', '', r'0$^{\circ}$ (360$^{\circ}$)', '', ''])
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
                                ticks=[1, 0.5, 0, -0.5, -1], shrink=0.8)
            cbar.set_label('r [-]', labelpad=4)
            cbar.set_ticklabels(['1', '0.5', '0', '-0.5', '-1'])
            cbar.ax.tick_params(direction='in')

            # convert to degrees
            diag_deg = (diag  * (180 / np.pi)) + 135
            diag_deg[diag_deg < 0] = 360 - diag_deg[diag_deg < 0]

            # 1-D density plot
            g = sns.kdeplot(diag_deg, color='k', ax=ax1)
            # kde_data = g.get_lines()[0].get_data()
            # kde_xx = kde_data[0]
            # kde_yy = kde_data[1]
            # x1 = np.where(kde_xx <= 90)[-1][-1]
            # x2 = np.where(kde_xx <= 180)[-1][-1]
            # x3 = np.where(kde_xx <= 270)[-1][-1]
            # ax1.fill_between(kde_xx[:x1+1], kde_yy[:x1+1], facecolor='purple',
            #                  alpha=0.2)
            # ax1.fill_between(kde_xx[x1:x2+2], kde_yy[x1:x2+2], facecolor='grey',
            #                  alpha=0.2)
            # ax1.fill_between(kde_xx[x2+1:x3+1], kde_yy[x2+1:x3+1],
            #                  facecolor='purple', alpha=0.2)
            # ax1.fill_between(kde_xx[x3:], kde_yy[x3:], facecolor='grey',
            #                  alpha=0.2)
            ax1.set_xticks([0, 90, 180, 270, 360])
            ax1.set_xlim(0, 360)
            ax1.set_ylim(0, )
            ax1.set(ylabel='Density',
                    xlabel='[$^\circ$]')

            # 2-D density plot
            # g = (sns.jointplot(diag_deg, sig_de, color='k', marginal_kws={'color':'k'}).plot_joint(sns.kdeplot, zorder=0, n_levels=10))
            g = (sns.jointplot(diag_deg, sig_de, kind='kde', zorder=1,
                               n_levels=20, cmap='Greens', shade_lowest=False,
                               marginal_kws={'color':'k', 'shade':False}).plot_joint(sns.scatterplot, color='k', alpha=.5, zorder=2))
            g.set_axis_labels(r'[$^\circ$]', r'$DE_{nn}$ [-]')
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
                                     facecolor='purple', alpha=0.2)
            g.ax_marg_x.fill_between(kde_xx[x1:x2+2], kde_yy[x1:x2+2],
                                     facecolor='grey', alpha=0.2)
            g.ax_marg_x.fill_between(kde_xx[x2+1:x3+1], kde_yy[x2+1:x3+1],
                                     facecolor='purple', alpha=0.2)
            g.ax_marg_x.fill_between(kde_xx[x3:], kde_yy[x3:], facecolor='grey',
                                     alpha=0.2)
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

def vis2d_de(obs, sim, sort=True, lim=0.05, extended=False):
    """Polar plot of Diagnostic-Efficiency (DE)

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

    Notes
    ----------
    .. math::

        \varphi = arctan2(\overline{B_{rel}}, B_{slope})

    Examples
    --------
    Provide arrays with equal length

    >>> from de import de
    >>> import numpy as np
    >>> obs = np.array([1.5, 1, 0.8, 0.85, 1.5, 2])
    >>> sim = np.array([1.6, 1.3, 1, 0.8, 1.2, 2.5])
    >>> de.vis2d_de(obs, sim)
    """
    if len(obs) != len(sim):
        raise AssertionError("Arrays are not of equal length!")
    # mean relative bias
    brel_mean = calc_brel_mean(obs, sim, sort=sort)

    # remaining relative bias
    brel_rest = calc_brel_rest(obs, sim, sort=sort)
    # area of relative remaing bias
    b_area = calc_bias_area(brel_rest)
    # temporal correlation
    temp_cor = calc_temp_cor(obs, sim)
    # diagnostic efficiency
    sig = calc_de(obs, sim)
    sig = np.round(sig, decimals=2)  # round to 2 decimals

    # direction of bias
    b_dir = calc_bias_dir(brel_rest)

    # slope of bias
    b_slope = calc_bias_slope(b_area, b_dir)

    # convert to radians
    # (y, x) Trigonometric inverse tangent
    diag = np.arctan2(brel_mean, b_slope)

    # convert temporal correlation to color
    norm = matplotlib.colors.Normalize(vmin=-1.0, vmax=1.0)
    rgba_color = cm.YlGnBu(norm(temp_cor))

    delta = 0.01  # for spacing

    # determine axis limits
    if sig > 0:
        ax_lim = sig - .1
        ax_lim = np.around(ax_lim, decimals=1)
        yy = np.arange(ax_lim, 1.01, delta)[::-1]
        c_levels = np.arange(ax_lim+.1, 1.1, .1)
    elif sig >= 0:
        ax_lim = 0.2
        yy = np.arange(-ax_lim, 1.01, delta)[::-1]
        c_levels = np.arange(0, 1, .2)
    elif sig < 0 and sig >= -1:
        ax_lim = 1.2
        yy = np.arange(-ax_lim, 1.01, delta)[::-1]
        c_levels = np.arange(-1, 1, .2)
    elif sig >= -2 and sig < -1:
        ax_lim = 2.2
        yy = np.arange(-ax_lim, 1.01, delta)[::-1]
        c_levels = np.arange(-2, 1, .2)
    elif sig <= -2:
        raise AssertionError("Value of 'DE' is too low for visualization!", sig)

    len_yy = 360

    # arrays to plot contour lines of DE
    xx = np.radians(np.linspace(0, 360, len_yy))
    theta, r = np.meshgrid(xx, yy)

    # # arrays to plot contours of positive constant offset
    # xx1 = np.radians(np.linspace(45, 135, len_yy))
    # theta1, r1 = np.meshgrid(xx1, yy)
    #
    # # arrays to plot contours of negative constant offset
    # xx2 = np.radians(np.linspace(225, 315, len_yy))
    # theta2, r2 = np.meshgrid(xx2, yy)
    #
    # # arrays to plot contours of dynamic model errors
    # xx3 = np.radians(np.linspace(135, 225, len_yy))
    # theta3, r3 = np.meshgrid(xx3, yy)
    #
    # # arrays to plot contours of dynamic model errors
    # len_yy2 = int(len_yy/2)
    # if len_yy != len_yy2 + len_yy2:
    #     xx0 = np.radians(np.linspace(0, 45, len_yy2+1))
    # else:
    #     xx0 = np.radians(np.linspace(0, 45, len_yy2))
    #
    # xx360 = np.radians(np.linspace(315, 360, len_yy2))
    # xx4 = np.concatenate((xx360, xx0), axis=None)
    # theta4, r4 = np.meshgrid(xx4, yy)

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
        # # contours positive constant offset
        # cpio = ax.contourf(theta1, r1, r1, cmap='Purples_r', alpha=.2)
        # # contours negative constant offset
        # cpiu = ax.contourf(theta2, r2, r2, cmap='Purples_r', alpha=.2)
        # # contours dynamic model errors
        # cpmou = ax.contourf(theta3, r3, r3, cmap='Greys_r', alpha=.2)
        # cpmuo = ax.contourf(theta4, r4, r4, cmap='Greys_r', alpha=.2)
        # plot regions
        ax.plot((1, np.deg2rad(45)), (1, np.min(yy)), color='lightgray', linewidth=1.5, ls='--', zorder=0)
        ax.plot((1, np.deg2rad(135)), (1, np.min(yy)), color='lightgray', linewidth=1.5, ls='--', zorder=0)
        ax.plot((1, np.deg2rad(225)), (1, np.min(yy)), color='lightgray', linewidth=1.5, ls='--', zorder=0)
        ax.plot((1, np.deg2rad(315)), (1, np.min(yy)), color='lightgray', linewidth=1.5, ls='--', zorder=0)
        # contours of DE
        cp = ax.contour(theta, r, r, colors='darkgray', levels=c_levels, zorder=1)
        cl = ax.clabel(cp, inline=True, fontsize=10, fmt='%1.1f',
                       colors='dimgrey')
        # threshold efficiency for FBM
        sig_lim = 1 - np.sqrt((lim)**2 + (lim)**2 + (lim)**2)
        # mean flow becnhmark
        mfb = calc_de_mfb(obs)
        # normalize threshold with mean flow becnhmark
        sig_lim_norm = (sig_lim - mfb)/(1 - mfb)
        # relation of b_dir which explains the error
        if abs(b_area) > 0:
            exp_err = (abs(b_dir) * 2)/abs(b_area)
        elif abs(b_area) == 0:
            exp_err = 0
        # diagnose the error
        if abs(brel_mean) <= lim and exp_err > lim and sig <= sig_lim_norm:
            ax.annotate("", xytext=(0, 1), xy=(diag, sig),
                        arrowprops=dict(facecolor=rgba_color))
        elif abs(brel_mean) > lim and exp_err <= lim and sig <= sig_lim_norm:
            ax.annotate("", xytext=(0, 1), xy=(diag, sig),
                        arrowprops=dict(facecolor=rgba_color))
        elif abs(brel_mean) > lim and exp_err > lim and sig <= sig_lim_norm:
            ax.annotate("", xytext=(0, 1), xy=(diag, sig),
                        arrowprops=dict(facecolor=rgba_color))
        # FBM
        elif abs(brel_mean) <= lim and exp_err <= lim and sig <= sig_lim_norm:
            ax.annotate("", xytext=(0, 1), xy=(0, sig),
                        arrowprops=dict(facecolor=rgba_color))
            ax.annotate("", xytext=(0, 1), xy=(np.pi, sig),
                        arrowprops=dict(facecolor=rgba_color))
        # FGM
        elif abs(brel_mean) <= lim and exp_err <= lim and sig > sig_lim_norm:
            c = ax.scatter(diag, sig, color=rgba_color)
        ax.set_rticks([])  # turn default ticks off
        ax.set_rmin(1)
        if sig > 0:
            ax.set_rmax(ax_lim)
        elif sig <= 0:
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
                            ticks=[1, 0.5, 0, -0.5, -1], shrink=0.8)
        cbar.set_label('r [-]', labelpad=4)
        cbar.set_ticklabels(['1', '0.5', '0', '-0.5', '-1'])
        cbar.ax.tick_params(direction='in')

    elif extended:
            fig = plt.figure(figsize=(12, 6), constrained_layout=True)
            gs = fig.add_gridspec(1, 2)
            ax = fig.add_subplot(gs[0, 0], projection='polar')
            ax1 = fig.add_axes([.65, .3, .33, .33], frameon=True)
            # dummie plot for colorbar of temporal correlation
            cs = np.arange(-1, 1.1, 0.1)
            dummie_cax = ax.scatter(cs, cs, c=cs, cmap='YlGnBu')
            # Clear axis
            ax.cla()
            # # contours positive constant offset
            # cpio = ax.contourf(theta1, r1, r1, cmap='Purples_r', alpha=.2)
            # # contours negative constant offset
            # cpiu = ax.contourf(theta2, r2, r2, cmap='Purples_r', alpha=.2)
            # # contours dynamic model errors
            # cpmou = ax.contourf(theta3, r3, r3, cmap='Greys_r', alpha=.2)
            # cpmuo = ax.contourf(theta4, r4, r4, cmap='Greys_r', alpha=.2)
            # plot regions
            ax.plot((1, np.deg2rad(45)), (1, np.min(yy)), color='lightgray', linewidth=1.5, ls='--', zorder=0)
            ax.plot((1, np.deg2rad(135)), (1, np.min(yy)), color='lightgray', linewidth=1.5, ls='--', zorder=0)
            ax.plot((1, np.deg2rad(225)), (1, np.min(yy)), color='lightgray', linewidth=1.5, ls='--', zorder=0)
            ax.plot((1, np.deg2rad(315)), (1, np.min(yy)), color='lightgray', linewidth=1.5, ls='--', zorder=0)
            # contours of DE
            cp = ax.contour(theta, r, r, colors='darkgray', levels=c_levels, zorder=1)
            cl = ax.clabel(cp, inline=True, fontsize=10, fmt='%1.1f',
                           colors='dimgrey')
            # threshold efficiency for FBM
            sig_lim = 1 - np.sqrt((lim)**2 + (lim)**2 + (lim)**2)
            # mean flow benchmark
            mfb = calc_de_mfb(obs)
            # normalize threshold with mean flow becnhmark
            sig_lim_norm = (sig_lim - mfb)/(1 - mfb)
            # relation of b_dir which explains the error
            if abs(b_area) > 0:
                exp_err = (abs(b_dir) * 2)/abs(b_area)
            elif abs(b_area) == 0:
                exp_err = 0
            # diagnose the error
            if abs(brel_mean) <= lim and exp_err > lim and sig <= sig_lim_norm:
                ax.annotate("", xytext=(0, 1), xy=(diag, sig),
                            arrowprops=dict(facecolor=rgba_color))
            elif abs(brel_mean) > lim and exp_err <= lim and sig <= sig_lim_norm:
                ax.annotate("", xytext=(0, 1), xy=(diag, sig),
                            arrowprops=dict(facecolor=rgba_color))
            elif abs(brel_mean) > lim and exp_err > lim and sig <= sig_lim_norm:
                ax.annotate("", xytext=(0, 1), xy=(diag, sig),
                            arrowprops=dict(facecolor=rgba_color))
            # FBM
            elif abs(brel_mean) <= lim and exp_err <= lim and sig <= sig_lim_norm:
                ax.annotate("", xytext=(0, 1), xy=(0, sig),
                            arrowprops=dict(facecolor=rgba_color))
                ax.annotate("", xytext=(0, 1), xy=(np.pi, sig),
                            arrowprops=dict(facecolor=rgba_color))
            # FGM
            elif abs(brel_mean) <= lim and exp_err <= lim and sig > sig_lim_norm:
                c = ax.scatter(diag, sig, color=rgba_color)
            ax.set_rticks([])  # turn default ticks off
            ax.set_rmin(1)
            if sig > 0:
                ax.set_rmax(ax_lim)
            elif sig <= 0:
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
                                ticks=[1, 0.5, 0, -0.5, -1], shrink=0.8)
            cbar.set_label('r [-]', labelpad=4)
            cbar.set_ticklabels(['1', '0.5', '0', '-0.5', '-1'])
            cbar.ax.tick_params(direction='in')

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

def vis2d_de_multi(brel_mean, b_area, temp_cor, sig_de, de_mfb, b_dir, diag,
                   lim=0.05, extended=False):
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

    de_mfb : (N,)array_like
        mean flow benchmark of diagnostic efficiency as 1-D array

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
    ll_mfb = de_mfb.tolist()
    ll_diag = diag.tolist()
    ll_temp_cor = temp_cor.tolist()

    # convert temporal correlation to color
    norm = matplotlib.colors.Normalize(vmin=-1.0, vmax=1.0)

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
    elif sig_min <= -2:
        raise ValueError("Some values of 'DE' are too low for visualization!", sig_min)

    len_yy = len(yy)

    # arrays to plot contour lines of DE
    xx = np.radians(np.linspace(0, 360, len_yy))
    theta, r = np.meshgrid(xx, yy)

    # # arrays to plot contours of positive constant offset
    # xx1 = np.radians(np.linspace(45, 135, len_yy))
    # theta1, r1 = np.meshgrid(xx1, yy)
    #
    # # arrays to plot contours of negative constant offset
    # xx2 = np.radians(np.linspace(225, 315, len_yy))
    # theta2, r2 = np.meshgrid(xx2, yy)
    #
    # # arrays to plot contours of dynamic model errors
    # xx3 = np.radians(np.linspace(135, 225, len_yy))
    # theta3, r3 = np.meshgrid(xx3, yy)
    #
    # # arrays to plot contours of dynamic model errors
    # len_yy2 = int(len_yy/2)
    # if len_yy != len_yy2 + len_yy2:
    #     xx0 = np.radians(np.linspace(0, 45, len_yy2+1))
    # else:
    #     xx0 = np.radians(np.linspace(0, 45, len_yy2))
    #
    # xx360 = np.radians(np.linspace(315, 360, len_yy2))
    # xx4 = np.concatenate((xx360, xx0), axis=None)
    # theta4, r4 = np.meshgrid(xx4, yy)

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
        # # contours positive constant offset
        # cpio = ax.contourf(theta1, r1, r1, cmap='Purples_r', alpha=.2)
        # # contours negative constant offset
        # cpiu = ax.contourf(theta2, r2, r2, cmap='Purples_r', alpha=.2)
        # # contours dynamic model errors
        # cpmou = ax.contourf(theta3, r3, r3, cmap='Greys_r', alpha=.2)
        # cpmuo = ax.contourf(theta4, r4, r4, cmap='Greys_r', alpha=.2)
        # plot regions
        ax.plot((1, np.deg2rad(45)), (1, np.min(yy)), color='lightgray', linewidth=1.5, ls='--', zorder=0)
        ax.plot((1, np.deg2rad(135)), (1, np.min(yy)), color='lightgray', linewidth=1.5, ls='--', zorder=0)
        ax.plot((1, np.deg2rad(225)), (1, np.min(yy)), color='lightgray', linewidth=1.5, ls='--', zorder=0)
        ax.plot((1, np.deg2rad(315)), (1, np.min(yy)), color='lightgray', linewidth=1.5, ls='--', zorder=0)
        # contours of DE
        cp = ax.contour(theta, r, r, colors='darkgray', levels=c_levels, zorder=1)
        cl = ax.clabel(cp, inline=True, fontsize=10, fmt='%1.1f',
                       colors='dimgrey')
        # threshold efficiency for FBM
        sig_lim = 1 - np.sqrt((lim)**2 + (lim)**2 + (lim)**2)
        # loop over each data point
        for (bm, bd, ba, r, sig, mfb, ang) in zip(ll_brel_mean, ll_b_dir, ll_b_area, ll_temp_cor, ll_sig, ll_mfb, ll_diag):
            # normalize threshold with mean flow becnhmark
            sig_lim_norm = (sig_lim - mfb)/(1 - mfb)
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
            if abs(bm) <= lim and exp_err > lim and sig <= sig_lim_norm:
                c = ax.scatter(ang, sig, color=rgba_color, zorder=2)
            elif abs(bm) > lim and exp_err <= lim and sig <= sig_lim_norm:
                c = ax.scatter(ang, sig, color=rgba_color, zorder=2)
            elif abs(bm) > lim and exp_err > lim and sig <= sig_lim_norm:
                c = ax.scatter(ang, sig, color=rgba_color, zorder=2)
            # FBM
            elif abs(bm) <= lim and exp_err <= lim and sig <= sig_lim_norm:
                ax.annotate("", xytext=(0, 1), xy=(0, sig),
                            arrowprops=dict(facecolor=rgba_color))
                ax.annotate("", xytext=(0, 1), xy=(np.pi, sig),
                            arrowprops=dict(facecolor=rgba_color))
            # FGM
            elif abs(bm) <= lim and exp_err <= lim and sig > sig_lim_norm:
                c = ax.scatter(ang, sig, color=rgba_color, zorder=2)
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
                            ticks=[1, 0.5, 0, -0.5, -1], shrink=0.8)
        cbar.set_label('r [-]', labelpad=4)
        cbar.set_ticklabels(['1', '0.5', '0', '-0.5', '-1'])
        cbar.ax.tick_params(direction='in')

    elif extended:
            fig = plt.figure(figsize=(12, 6), constrained_layout=True)
            gs = fig.add_gridspec(1, 2)
            ax = fig.add_subplot(gs[0, 0], projection='polar')
            ax1 = fig.add_axes([.66, .3, .32, .32], frameon=True)
            # dummie plot for colorbar of temporal correlation
            cs = np.arange(-1, 1.1, 0.1)
            dummie_cax = ax.scatter(cs, cs, c=cs, cmap='YlGnBu')
            # Clear axis
            ax.cla()
            # # contours positive constant offset
            # cpio = ax.contourf(theta1, r1, r1, cmap='Purples_r', alpha=.2)
            # # contours negative constant offset
            # cpiu = ax.contourf(theta2, r2, r2, cmap='Purples_r', alpha=.2)
            # # contours dynamic model errors
            # cpmou = ax.contourf(theta3, r3, r3, cmap='Greys_r', alpha=.2)
            # cpmuo = ax.contourf(theta4, r4, r4, cmap='Greys_r', alpha=.2)
            # plot regions
            ax.plot((1, np.deg2rad(45)), (1, np.min(yy)), color='lightgray', linewidth=1.5, ls='--', zorder=0)
            ax.plot((1, np.deg2rad(135)), (1, np.min(yy)), color='lightgray', linewidth=1.5, ls='--', zorder=0)
            ax.plot((1, np.deg2rad(225)), (1, np.min(yy)), color='lightgray', linewidth=1.5, ls='--', zorder=0)
            ax.plot((1, np.deg2rad(315)), (1, np.min(yy)), color='lightgray', linewidth=1.5, ls='--', zorder=0)
            # contours of DE
            cp = ax.contour(theta, r, r, colors='darkgray', levels=c_levels, zorder=1)
            cl = ax.clabel(cp, inline=True, fontsize=10, fmt='%1.1f',
                           colors='dimgrey')
            # threshold efficiency for FBM
            sig_lim = 1 - np.sqrt((lim)**2 + (lim)**2 + (lim)**2)
            # loop over each data point
            for (bm, bd, ba, r, sig, mfb, ang) in zip(ll_brel_mean, ll_b_dir, ll_b_area, ll_temp_cor, ll_sig, ll_mfb, ll_diag):
                sig_lim_norm = (sig_lim - mfb)/(1 - mfb)
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
                if abs(bm) <= lim and exp_err > lim and sig <= sig_lim_norm:
                    c = ax.scatter(ang, sig, color=rgba_color, zorder=2)
                elif abs(bm) > lim and exp_err <= lim and sig <= sig_lim_norm:
                    c = ax.scatter(ang, sig, color=rgba_color, zorder=2)
                elif abs(bm) > lim and exp_err > lim and sig <= sig_lim_norm:
                    c = ax.scatter(ang, sig, color=rgba_color, zorder=2)
                # FBM
                elif abs(bm) <= lim and exp_err <= lim and sig <= sig_lim_norm:
                    ax.annotate("", xytext=(0, 1), xy=(0, sig),
                                arrowprops=dict(facecolor=rgba_color))
                    ax.annotate("", xytext=(0, 1), xy=(np.pi, sig),
                                arrowprops=dict(facecolor=rgba_color))
                # FGM
                elif abs(bm) <= lim and exp_err <= lim and sig > sig_lim_norm:
                    c = ax.scatter(ang, sig, color=rgba_color, zorder=2)
            ax.set_rticks([])  # turn default ticks off
            ax.set_rmin(1)
            if sig_min > 0:
                ax.set_rmax(ax_lim)
            elif sig_min <= 0:
                ax.set_rmax(-ax_lim)
            ax.tick_params(labelleft=False, labelright=False, labeltop=False,
                          labelbottom=True, grid_alpha=.01)  # turn labels and grid off
            ax.set_xticklabels(['', '', '', '', '', r'0$^{\circ}$ (360$^{\circ}$)', '', ''])
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
                                ticks=[1, 0.5, 0, -0.5, -1], shrink=0.8)
            cbar.set_label('r [-]', labelpad=4)
            cbar.set_ticklabels(['1', '0.5', '0', '-0.5', '-1'])
            cbar.ax.tick_params(direction='in')

            # convert to degrees
            diag_deg = (diag  * (180 / np.pi)) + 135
            diag_deg[diag_deg < 0] = 360 - diag_deg[diag_deg < 0]

            # 1-D density plot
            g = sns.kdeplot(diag_deg, color='k', ax=ax1)
            # kde_data = g.get_lines()[0].get_data()
            # kde_xx = kde_data[0]
            # kde_yy = kde_data[1]
            # x1 = np.where(kde_xx <= 90)[-1][-1]
            # x2 = np.where(kde_xx <= 180)[-1][-1]
            # x3 = np.where(kde_xx <= 270)[-1][-1]
            # ax1.fill_between(kde_xx[:x1+1], kde_yy[:x1+1], facecolor='purple',
            #                  alpha=0.2)
            # ax1.fill_between(kde_xx[x1:x2+2], kde_yy[x1:x2+2], facecolor='grey',
            #                  alpha=0.2)
            # ax1.fill_between(kde_xx[x2+1:x3+1], kde_yy[x2+1:x3+1],
            #                  facecolor='purple', alpha=0.2)
            # ax1.fill_between(kde_xx[x3:], kde_yy[x3:], facecolor='grey',
            #                  alpha=0.2)
            ax1.set_xticks([0, 90, 180, 270, 360])
            ax1.set_xlim(0, 360)
            ax1.set_ylim(0, )
            ax1.set(ylabel='Density',
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
            # kde_data = g.ax_marg_x.get_lines()[0].get_data()
            # kde_xx = kde_data[0]
            # kde_yy = kde_data[1]
            # x1 = np.where(kde_xx <= 90)[-1][-1]
            # x2 = np.where(kde_xx <= 180)[-1][-1]
            # x3 = np.where(kde_xx <= 270)[-1][-1]
            # g.ax_marg_x.fill_between(kde_xx[:x1+1], kde_yy[:x1+1],
            #                          facecolor='purple', alpha=0.2)
            # g.ax_marg_x.fill_between(kde_xx[x1:x2+2], kde_yy[x1:x2+2],
            #                          facecolor='grey', alpha=0.2)
            # g.ax_marg_x.fill_between(kde_xx[x2+1:x3+1], kde_yy[x2+1:x3+1],
            #                          facecolor='purple', alpha=0.2)
            # g.ax_marg_x.fill_between(kde_xx[x3:], kde_yy[x3:], facecolor='grey',
            #                          alpha=0.2)
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
