#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
de.de
~~~~~~~~~~~
Diagnosing model performance using an efficiency measure based on flow
duration curve and temporal correlation. The efficiency measure can be
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
    """Calculate absolute bias area for entire flow duration curve.

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
        Either Spearman correlation coefficient ('spearman') or Pearson
        correlation coefficient ('pearson') can be used to describe the temporal
        correlation. The default is to calculate the Pearson correlation.

    Returns
    ----------
    temp_cor : float
        correlation between observed and simulated time series

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
    >>> de.calc_de(obs, sim)
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

    return sig

def calc_de_bench(obs, bench, sort=True):
    r"""
    Calculate the benchmark of Diagnostic-Efficiency (DE).

    Parameters
    ----------
    obs : (N,)array_like
        Observed time series as 1-D array

    bench : (N,)array_like
        Benchmarked time series as 1-D array

    sort : boolean, optional
        If True, time series are sorted by ascending order. If False, time
        series are not sorted. The default is to sort.

    Returns
    ----------
    sigb : float
        Diagnostic efficiency of benchmark

    Notes
    ----------
    .. math::

        DE_{bench} = 1 - \sqrt{\overline{B_{rel}}^2 + \vert B_{area}\vert^2 + (r - 1)^2}

    Examples
    --------
    Provide arrays with equal length

    >>> from de import de
    >>> import numpy as np
    >>> obs = np.array([1.5, 1, 0.8, 0.85, 1.5, 2])
    >>> bench = np.array([1, 1, 1, 1, 1, 1])
    >>> de.calc_de_bench(obs, bench)
    """
    # mean relative bias
    brel_mean = calc_brel_mean(obs, bench, sort=sort)
    # remaining relative bias
    brel_rest = calc_brel_rest(obs, bench, sort=sort)
    # area of relative remaing bias
    b_area = calc_bias_area(brel_rest)
    # temporal correlation
    temp_cor = calc_temp_cor(obs, bench)
    # diagnostic efficiency
    sig = 1 - np.sqrt((brel_mean)**2 + (b_area)**2 + (temp_cor - 1)**2)

    return sig

def calc_deb(obs, sim, bench, sort=True):
    r"""
    Calculate benchmarked Diagnostic-Efficiency (DEB).

    Parameters
    ----------
    obs : (N,)array_like
        Observed time series as 1-D array

    sim : (N,)array_like
        Simulated time series

    bench : (N,)array_like
        Benchmarked time series as 1-D array

    sort : boolean, optional
        If True, time series are sorted by ascending order. If False, time
        series are not sorted. The default is to sort.

    Returns
    ----------
    sig : float
        Benchmarked diagnostic efficiency

    Notes
    ----------
    .. math::

        DE = 1 - \sqrt{\overline{B_{rel}}^2 + \vert B_{area}\vert^2 + (r - 1)^2}
        DE_{bench} = 1 - \sqrt{\overline{B_{rel}}^2 + \vert B_{area}\vert^2 + (r - 1)^2}
        DEB = \frac{DE - DE_{bench}}{1 - DE_{bench}}

    Examples
    --------
    Provide arrays with equal length

    >>> from de import de
    >>> import numpy as np
    >>> obs = np.array([1.5, 1, 0.8, 0.85, 1.5, 2])
    >>> sim = np.array([1.6, 1.3, 1, 0.8, 1.2, 2.5])
    >>> bench = np.array([1, 1, 1, 1, 1, 1])
    >>> de.calc_deb(obs, sim, bench)

    """
    if len(obs) != len(sim):
        raise AssertionError("Arrays are not of equal length!")
    if len(obs) != len(bench):
        raise AssertionError("Arrays are not of equal length!")
    # diagnostic efficiency
    sig_de = calc_de(obs, sim)
    sig_bench = calc_de_bench(obs, bench)
    sig = (sig_de - sig_bench)/(1 - sig_bench)

    return sig

def diag_polar_plot(obs, sim, sort=True, l=0.05, extended=False):
    """Diagnostic polar plot of Diagnostic efficiency (DE) for a single value

    Parameters
    ----------
    obs : (N,)array_like
        Observed time series as 1-D array

    sim : (N,)array_like
        Simulated time series as 1-D array

    sort : boolean, optional
        If True, time series are sorted by ascending order. If False, time series
        are not sorted. The default is to sort.

    l : float, optional
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
    >>> de.diag_polar_plot(obs, sim)
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
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1.0)
    rgba_color = cm.plasma_r(norm(temp_cor))

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
        raise AssertionError("Value of 'DE' is out of bounds for visualization!", sig)

    len_yy = len(yy)

    # arrays to plot contour lines of DE
    xx = np.radians(np.linspace(0, 360, len_yy))
    theta, r = np.meshgrid(xx, yy)

    # diagnostic polar plot
    if not extended:
        fig, ax = plt.subplots(figsize=(6, 6),
                               subplot_kw=dict(projection='polar'),
                               constrained_layout=True)
        # dummie plot for colorbar of temporal correlation
        cs = np.arange(0, 1.1, 0.1)
        dummie_cax = ax.scatter(cs, cs, c=cs, cmap='plasma_r')
        # Clear axis
        ax.cla()
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
        sig_l = 1 - np.sqrt((l)**2 + (l)**2 + (l)**2)
        # relation of b_dir which explains the error
        if abs(b_area) > 0:
            exp_err = (abs(b_dir) * 2)/abs(b_area)
        elif abs(b_area) == 0:
            exp_err = 0
        # diagnose the error
        if abs(brel_mean) <= l and exp_err > l and sig <= sig_l:
            ax.annotate("", xytext=(0, 1), xy=(diag, sig),
                        arrowprops=dict(facecolor=rgba_color))
        elif abs(brel_mean) > l and exp_err <= l and sig <= sig_l:
            ax.annotate("", xytext=(0, 1), xy=(diag, sig),
                        arrowprops=dict(facecolor=rgba_color))
        elif abs(brel_mean) > l and exp_err > l and sig <= sig_l:
            ax.annotate("", xytext=(0, 1), xy=(diag, sig),
                        arrowprops=dict(facecolor=rgba_color))
        # FBM
        elif abs(brel_mean) <= l and exp_err <= l and sig <= sig_l:
            ax.annotate("", xytext=(0, 1), xy=(0, sig),
                        arrowprops=dict(facecolor=rgba_color))
            ax.annotate("", xytext=(0, 1), xy=(np.pi, sig),
                        arrowprops=dict(facecolor=rgba_color))
        # FGM
        elif abs(brel_mean) <= l and exp_err <= l and sig > sig_l:
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
                            ticks=[1, 0.5, 0], shrink=0.8)
        cbar.set_label('r [-]', labelpad=4)
        cbar.set_ticklabels(['1', '0.5', '<0'])
        cbar.ax.tick_params(direction='in')

    elif extended:
        fig = plt.figure(figsize=(12, 6), constrained_layout=True)
        gs = fig.add_gridspec(1, 2)
        ax = fig.add_subplot(gs[0, 0], projection='polar')
        ax1 = fig.add_axes([.65, .3, .33, .33], frameon=True)
        # dummie plot for colorbar of temporal correlation
        cs = np.arange(0, 1.1, 0.1)
        dummie_cax = ax.scatter(cs, cs, c=cs, cmap='plasma_r')
        # Clear axis
        ax.cla()
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
        sig_l = 1 - np.sqrt((l)**2 + (l)**2 + (l)**2)
        # relation of b_dir which explains the error
        if abs(b_area) > 0:
            exp_err = (abs(b_dir) * 2)/abs(b_area)
        elif abs(b_area) == 0:
            exp_err = 0
        # diagnose the error
        if abs(brel_mean) <= l and exp_err > l and sig <= sig_l:
            ax.annotate("", xytext=(0, 1), xy=(diag, sig),
                        arrowprops=dict(facecolor=rgba_color))
        elif abs(brel_mean) > l and exp_err <= l and sig <= sig_l:
            ax.annotate("", xytext=(0, 1), xy=(diag, sig),
                        arrowprops=dict(facecolor=rgba_color))
        elif abs(brel_mean) > l and exp_err > l and sig <= sig_l:
            ax.annotate("", xytext=(0, 1), xy=(diag, sig),
                        arrowprops=dict(facecolor=rgba_color))
        # FBM
        elif abs(brel_mean) <= l and exp_err <= l and sig <= sig_l:
            ax.annotate("", xytext=(0, 1), xy=(0, sig),
                        arrowprops=dict(facecolor=rgba_color))
            ax.annotate("", xytext=(0, 1), xy=(np.pi, sig),
                        arrowprops=dict(facecolor=rgba_color))
        # FGM
        elif abs(brel_mean) <= l and exp_err <= l and sig > sig_l:
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
                            ticks=[1, 0.5, 0], shrink=0.8)
        cbar.set_label('r [-]', labelpad=4)
        cbar.set_ticklabels(['1', '0.5', '<0'])
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

def diag_polar_plot_multi(brel_mean, b_area, temp_cor, sig_de, b_dir, diag,
                   l=0.05, extended=False):
    """Diagnostic polar plot of Diagnostic efficiency (DE) with multiple values.

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

    l : float, optional
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
    elif sig_min <= -2:
        raise ValueError("Some values of 'DE' are too low for visualization!", sig_min)

    len_yy = len(yy)

    # arrays to plot contour lines of DE
    xx = np.radians(np.linspace(0, 360, len_yy))
    theta, r = np.meshgrid(xx, yy)

    # diagnostic polar plot
    if not extended:
        fig, ax = plt.subplots(figsize=(6, 6),
                               subplot_kw=dict(projection='polar'),
                               constrained_layout=True)
        # dummie plot for colorbar of temporal correlation
        cs = np.arange(0, 1.1, 0.1)
        dummie_cax = ax.scatter(cs, cs, c=cs, cmap='plasma_r')
        # Clear axis
        ax.cla()
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
        sig_l = 1 - np.sqrt((l)**2 + (l)**2 + (l)**2)
        # loop over each data point
        for (bm, bd, ba, r, sig, ang) in zip(ll_brel_mean, ll_b_dir, ll_b_area, ll_temp_cor, ll_sig, ll_diag):
            # slope of bias
            b_slope = calc_bias_slope(ba, bd)
            # convert temporal correlation to color
            rgba_color = cm.plasma_r(norm(r))
            # relation of b_dir which explains the error
            if abs(ba) > 0:
                exp_err = (abs(bd) * 2)/abs(ba)
            elif abs(ba) == 0:
                exp_err = 0
            # diagnose the error
            if abs(bm) <= l and exp_err > l and sig <= sig_l:
                c = ax.scatter(ang, sig, color=rgba_color, zorder=2)
            elif abs(bm) > l and exp_err <= l and sig <= sig_l:
                c = ax.scatter(ang, sig, color=rgba_color, zorder=2)
            elif abs(bm) > l and exp_err > l and sig <= sig_l:
                c = ax.scatter(ang, sig, color=rgba_color, zorder=2)
            # FBM
            elif abs(bm) <= l and exp_err <= l and sig <= sig_l:
                ax.annotate("", xytext=(0, 1), xy=(0, sig),
                            arrowprops=dict(facecolor=rgba_color))
                ax.annotate("", xytext=(0, 1), xy=(np.pi, sig),
                            arrowprops=dict(facecolor=rgba_color))
            # FGM
            elif abs(bm) <= l and exp_err <= l and sig > sig_l:
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
                            ticks=[1, 0.5, 0], shrink=0.8)
        cbar.set_label('r [-]', labelpad=4)
        cbar.set_ticklabels(['1', '0.5', '<0'])
        cbar.ax.tick_params(direction='in')

    elif extended:
        fig = plt.figure(figsize=(12, 6), constrained_layout=True)
        gs = fig.add_gridspec(1, 2)
        ax = fig.add_subplot(gs[0, 0], projection='polar')
        ax1 = fig.add_axes([.66, .3, .32, .32], frameon=True)
        # dummie plot for colorbar of temporal correlation
        cs = np.arange(0, 1.1, 0.1)
        dummie_cax = ax.scatter(cs, cs, c=cs, cmap='plasma_r')
        # Clear axis
        ax.cla()
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
        sig_l = 1 - np.sqrt((l)**2 + (l)**2 + (l)**2)
        # loop over each data point
        for (bm, bd, ba, r, sig, ang) in zip(ll_brel_mean, ll_b_dir, ll_b_area, ll_temp_cor, ll_sig, ll_diag):
            # slope of bias
            b_slope = calc_bias_slope(ba, bd)
            # convert temporal correlation to color
            rgba_color = cm.plasma_r(norm(r))
            # relation of b_dir which explains the error
            if abs(ba) > 0:
                exp_err = (abs(bd) * 2)/abs(ba)
            elif abs(ba) == 0:
                exp_err = 0
            # diagnose the error
            if abs(bm) <= l and exp_err > l and sig <= sig_l:
                c = ax.scatter(ang, sig, color=rgba_color, zorder=2)
            elif abs(bm) > l and exp_err <= l and sig <= sig_l:
                c = ax.scatter(ang, sig, color=rgba_color, zorder=2)
            elif abs(bm) > l and exp_err > l and sig <= sig_l:
                c = ax.scatter(ang, sig, color=rgba_color, zorder=2)
            # FBM
            elif abs(bm) <= l and exp_err <= l and sig <= sig_l:
                ax.annotate("", xytext=(0, 1), xy=(0, sig),
                            arrowprops=dict(facecolor=rgba_color))
                ax.annotate("", xytext=(0, 1), xy=(np.pi, sig),
                            arrowprops=dict(facecolor=rgba_color))
            # FGM
            elif abs(bm) <= l and exp_err <= l and sig > sig_l:
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
                            ticks=[1, 0.5, 0], shrink=0.8)
        cbar.set_label('r [-]', labelpad=4)
        cbar.set_ticklabels(['1', '0.5', '<0'])
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
        g.set_axis_labels(r'[$^\circ$]', 'DE [-]')
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

def bdiag_polar_plot(obs, sim, bench, sort=True, l=0.05, extended=False):
    """Diagnostic polar plot of benchmarked Diagnostic efficiency (DEB)
    with a single value

    Parameters
    ----------
    obs : (N,)array_like
        Observed time series as 1-D array

    sim : (N,)array_like
        Simulated time series as 1-D array

    bench : (N,)array_like
        Benchmarked time series as 1-D array

    sort : boolean, optional
        If True, time series are sorted by ascending order. If False, time series
        are not sorted. The default is to sort.

    l : float, optional
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
    >>> de.diag_polar_plot(obs, sim)
    """
    if len(obs) != len(sim):
        raise AssertionError("Arrays are not of equal length!")
    if len(obs) != len(bench):
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
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1.0)
    rgba_color = cm.plasma_r(norm(temp_cor))

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
        raise AssertionError("Value of 'DE' is out of bounds for visualization!", sig)

    len_yy = 360

    # arrays to plot contour lines of DE
    xx = np.radians(np.linspace(0, 360, len_yy))
    theta, r = np.meshgrid(xx, yy)

    # diagnostic polar plot
    if not extended:
        fig, ax = plt.subplots(figsize=(6, 6),
                               subplot_kw=dict(projection='polar'),
                               constrained_layout=True)
        # dummie plot for colorbar of temporal correlation
        cs = np.arange(0, 1.1, 0.1)
        dummie_cax = ax.scatter(cs, cs, c=cs, cmap='plasma_r')
        # Clear axis
        ax.cla()
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
        sig_l = 1 - np.sqrt((l)**2 + (l)**2 + (l)**2)
        # mean flow becnhmark
        sig_deb = calc_de_bench(obs, bench)
        # normalize threshold with the becnhmark
        sig_lim_norm = (sig_l - sig_deb)/(1 - sig_deb)
        # relation of b_dir which explains the error
        if abs(b_area) > 0:
            exp_err = (abs(b_dir) * 2)/abs(b_area)
        elif abs(b_area) == 0:
            exp_err = 0
        # diagnose the error
        if abs(brel_mean) <= l and exp_err > l and sig <= sig_lim_norm:
            ax.annotate("", xytext=(0, 1), xy=(diag, sig),
                        arrowprops=dict(facecolor=rgba_color))
        elif abs(brel_mean) > l and exp_err <= l and sig <= sig_lim_norm:
            ax.annotate("", xytext=(0, 1), xy=(diag, sig),
                        arrowprops=dict(facecolor=rgba_color))
        elif abs(brel_mean) > l and exp_err > l and sig <= sig_lim_norm:
            ax.annotate("", xytext=(0, 1), xy=(diag, sig),
                        arrowprops=dict(facecolor=rgba_color))
        # FBM
        elif abs(brel_mean) <= l and exp_err <= l and sig <= sig_lim_norm:
            ax.annotate("", xytext=(0, 1), xy=(0, sig),
                        arrowprops=dict(facecolor=rgba_color))
            ax.annotate("", xytext=(0, 1), xy=(np.pi, sig),
                        arrowprops=dict(facecolor=rgba_color))
        # FGM
        elif abs(brel_mean) <= l and exp_err <= l and sig > sig_lim_norm:
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
                            ticks=[1, 0.5, 0], shrink=0.8)
        cbar.set_label('r [-]', labelpad=4)
        cbar.set_ticklabels(['1', '0.5', '<0'])
        cbar.ax.tick_params(direction='in')

    elif extended:
        fig = plt.figure(figsize=(12, 6), constrained_layout=True)
        gs = fig.add_gridspec(1, 2)
        ax = fig.add_subplot(gs[0, 0], projection='polar')
        ax1 = fig.add_axes([.65, .3, .33, .33], frameon=True)
        # dummie plot for colorbar of temporal correlation
        cs = np.arange(0, 1.1, 0.1)
        dummie_cax = ax.scatter(cs, cs, c=cs, cmap='plasma_r')
        # Clear axis
        ax.cla()
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
        sig_l = 1 - np.sqrt((l)**2 + (l)**2 + (l)**2)
        # benchmark
        sig_bench = calc_de_bench(obs, bench)
        # normalize threshold with mean flow becnhmark
        sig_lim_norm = (sig_l - sig_bench)/(1 - sig_bench)
        # relation of b_dir which explains the error
        if abs(b_area) > 0:
            exp_err = (abs(b_dir) * 2)/abs(b_area)
        elif abs(b_area) == 0:
            exp_err = 0
        # diagnose the error
        if abs(brel_mean) <= l and exp_err > l and sig <= sig_lim_norm:
            ax.annotate("", xytext=(0, 1), xy=(diag, sig),
                        arrowprops=dict(facecolor=rgba_color))
        elif abs(brel_mean) > l and exp_err <= l and sig <= sig_lim_norm:
            ax.annotate("", xytext=(0, 1), xy=(diag, sig),
                        arrowprops=dict(facecolor=rgba_color))
        elif abs(brel_mean) > l and exp_err > l and sig <= sig_lim_norm:
            ax.annotate("", xytext=(0, 1), xy=(diag, sig),
                        arrowprops=dict(facecolor=rgba_color))
        # FBM
        elif abs(brel_mean) <= l and exp_err <= l and sig <= sig_lim_norm:
            ax.annotate("", xytext=(0, 1), xy=(0, sig),
                        arrowprops=dict(facecolor=rgba_color))
            ax.annotate("", xytext=(0, 1), xy=(np.pi, sig),
                        arrowprops=dict(facecolor=rgba_color))
        # FGM
        elif abs(brel_mean) <= l and exp_err <= l and sig > sig_lim_norm:
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
                            ticks=[1, 0.5, 0], shrink=0.8)
        cbar.set_label('r [-]', labelpad=4)
        cbar.set_ticklabels(['1', '0.5', '<0'])
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

def bdiag_polar_plot_multi(brel_mean, b_area, temp_cor, sig_de, sig_de_bench, b_dir, diag,
                   l=0.05, extended=False):
    """Diagnostic polar plot of benchmarked Diagnostic efficiency (DEB) with
    multiple values

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

    l : float, optional
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
    elif sig_min <= -2:
        raise ValueError("Some values of 'DE' are too low for visualization!", sig_min)

    len_yy = len(yy)

    # arrays to plot contour lines of DE
    xx = np.radians(np.linspace(0, 360, len_yy))
    theta, r = np.meshgrid(xx, yy)

    # diagnostic polar plot
    if not extended:
        fig, ax = plt.subplots(figsize=(6, 6),
                               subplot_kw=dict(projection='polar'),
                               constrained_layout=True)
        # dummie plot for colorbar of temporal correlation
        cs = np.arange(0, 1.1, 0.1)
        dummie_cax = ax.scatter(cs, cs, c=cs, cmap='plasma_r')
        # Clear axis
        ax.cla()
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
        sig_l = 1 - np.sqrt((l)**2 + (l)**2 + (l)**2)
        # loop over each data point
        for (bm, bd, ba, r, sig, sig_bench, ang) in zip(ll_brel_mean, ll_b_dir, ll_b_area, ll_temp_cor, ll_sig, ll_bench, ll_diag):
            # normalize threshold with mean flow becnhmark
            sig_lim_norm = (sig_l - sig_bench)/(1 - sig_bench)
            # slope of bias
            b_slope = calc_bias_slope(ba, bd)
            # convert temporal correlation to color
            rgba_color = cm.plasma_r(norm(r))
            # relation of b_dir which explains the error
            if abs(ba) > 0:
                exp_err = (abs(bd) * 2)/abs(ba)
            elif abs(ba) == 0:
                exp_err = 0
            # diagnose the error
            if abs(bm) <= l and exp_err > l and sig <= sig_lim_norm:
                c = ax.scatter(ang, sig, color=rgba_color, zorder=2)
            elif abs(bm) > l and exp_err <= l and sig <= sig_lim_norm:
                c = ax.scatter(ang, sig, color=rgba_color, zorder=2)
            elif abs(bm) > l and exp_err > l and sig <= sig_lim_norm:
                c = ax.scatter(ang, sig, color=rgba_color, zorder=2)
            # FBM
            elif abs(bm) <= l and exp_err <= l and sig <= sig_lim_norm:
                ax.annotate("", xytext=(0, 1), xy=(0, sig),
                            arrowprops=dict(facecolor=rgba_color))
                ax.annotate("", xytext=(0, 1), xy=(np.pi, sig),
                            arrowprops=dict(facecolor=rgba_color))
            # FGM
            elif abs(bm) <= l and exp_err <= l and sig > sig_lim_norm:
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
                            ticks=[1, 0.5, 0], shrink=0.8)
        cbar.set_label('r [-]', labelpad=4)
        cbar.set_ticklabels(['1', '0.5', '<0'])
        cbar.ax.tick_params(direction='in')

    elif extended:
            fig = plt.figure(figsize=(12, 6), constrained_layout=True)
            gs = fig.add_gridspec(1, 2)
            ax = fig.add_subplot(gs[0, 0], projection='polar')
            ax1 = fig.add_axes([.66, .3, .32, .32], frameon=True)
            # dummie plot for colorbar of temporal correlation
            cs = np.arange(0, 1.1, 0.1)
            dummie_cax = ax.scatter(cs, cs, c=cs, cmap='plasma_r')
            # Clear axis
            ax.cla()
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
            sig_l = 1 - np.sqrt((l)**2 + (l)**2 + (l)**2)
            # loop over each data point
            for (bm, bd, ba, r, sig, sig_bench, ang) in zip(ll_brel_mean, ll_b_dir, ll_b_area, ll_temp_cor, ll_sig, ll_bench, ll_diag):
                sig_lim_norm = (sig_l - sig_bench)/(1 - sig_bench)
                # slope of bias
                b_slope = calc_bias_slope(ba, bd)
                # convert temporal correlation to color
                rgba_color = cm.plasma_r(norm(r))
                # relation of b_dir which explains the error
                if abs(ba) > 0:
                    exp_err = (abs(bd) * 2)/abs(ba)
                elif abs(ba) == 0:
                    exp_err = 0
                # diagnose the error
                if abs(bm) <= l and exp_err > l and sig <= sig_lim_norm:
                    c = ax.scatter(ang, sig, color=rgba_color, zorder=2)
                elif abs(bm) > l and exp_err <= l and sig <= sig_lim_norm:
                    c = ax.scatter(ang, sig, color=rgba_color, zorder=2)
                elif abs(bm) > l and exp_err > l and sig <= sig_lim_norm:
                    c = ax.scatter(ang, sig, color=rgba_color, zorder=2)
                # FBM
                elif abs(bm) <= l and exp_err <= l and sig <= sig_lim_norm:
                    ax.annotate("", xytext=(0, 1), xy=(0, sig),
                                arrowprops=dict(facecolor=rgba_color))
                    ax.annotate("", xytext=(0, 1), xy=(np.pi, sig),
                                arrowprops=dict(facecolor=rgba_color))
                # FGM
                elif abs(bm) <= l and exp_err <= l and sig > sig_lim_norm:
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
                                ticks=[1, 0.5, 0], shrink=0.8)
            cbar.set_label('r [-]', labelpad=4)
            cbar.set_ticklabels(['1', '0.5', '<0'])
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

def gdiag_polar_plot(eff, comp1, comp2, comp3, l=0.05):
    """Generic diagnostic polar plot with a single value

    Parameters
    ----------
    eff : float
        efficiency measure

    comp1 : float
        metric component 1

    comp2 : float
        metric component 2

    comp3 : float
        metric component 3

    l : float, optional
        Threshold for which diagnosis can be made. The default is 0.05.

    Notes
    ----------
    .. math::

        \varphi = arctan2(comp1, comp2)

    Examples
    --------
    Provide arrays with equal length

    >>> from de import de
    >>> import numpy as np
    >>> obs = np.array([1.5, 1, 0.8, 0.85, 1.5, 2])
    >>> sim = np.array([1.6, 1.3, 1, 0.8, 1.2, 2.5])
    >>> sig_de = de.calc_de(obs, sim)
    >>> brel_mean = de.calc_brel_mean(obs, sim)
    >>> b_rest = de.calc_brel_rest(obs, sim)
    >>> b_dir = de.calc_bias_dir(b_rest)
    >>> b_area = de.calc_bias_area(b_rest)
    >>> b_slope = de.calc_bias_slope(b_area, b_dir)
    >>> temp_cor = de.calc_temp_cor(obs, sim)
    >>> de.diag_polar_plot(sig_de, brel_mean, b_slope, temp_cor)
    """
    # convert to radians
    # (y, x) Trigonometric inverse tangent
    diag = np.arctan2(comp1, comp2)

    # convert metric component 3 to color
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1.0)
    rgba_color = cm.plasma_r(norm(comp3))

    delta = 0.01  # for spacing

    # determine axis limits
    if eff > 0:
        ax_lim = eff - .1
        ax_lim = np.around(ax_lim, decimals=1)
        yy = np.arange(ax_lim, 1.01, delta)[::-1]
        c_levels = np.arange(ax_lim+.1, 1.1, .1)
    elif eff >= 0:
        ax_lim = 0.2
        yy = np.arange(-ax_lim, 1.01, delta)[::-1]
        c_levels = np.arange(0, 1, .2)
    elif eff < 0 and eff >= -1:
        ax_lim = 1.2
        yy = np.arange(-ax_lim, 1.01, delta)[::-1]
        c_levels = np.arange(-1, 1, .2)
    elif eff >= -2 and eff < -1:
        ax_lim = 2.2
        yy = np.arange(-ax_lim, 1.01, delta)[::-1]
        c_levels = np.arange(-2, 1, .2)
    elif eff <= -2:
        raise AssertionError("Value of eff is out of bounds for visualization!", eff)

    len_yy = len(yy)

    # arrays to plot contour lines of DE
    xx = np.radians(np.linspace(0, 360, len_yy))
    theta, r = np.meshgrid(xx, yy)

    # diagnostic polar plot
    fig, ax = plt.subplots(figsize=(6, 6),
                           subplot_kw=dict(projection='polar'),
                           constrained_layout=True)
    # dummie plot for colorbar of temporal correlation
    cs = np.arange(0, 1.1, 0.1)
    dummie_cax = ax.scatter(cs, cs, c=cs, cmap='plasma_r')
    # Clear axis
    ax.cla()
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
    sig_l = 1 - np.sqrt((l)**2 + (l)**2 + (l)**2)
    # relation of b_dir which explains the error
    if abs(comp2) > 0:
        exp_err = comp2
    elif abs(comp2) == 0:
        exp_err = 0
    # diagnose the error
    if abs(comp1) <= l and exp_err > l and eff <= sig_l:
        ax.annotate("", xytext=(0, 1), xy=(diag, eff),
                    arrowprops=dict(facecolor=rgba_color))
    elif abs(comp1) > l and exp_err <= l and eff <= sig_l:
        ax.annotate("", xytext=(0, 1), xy=(diag, eff),
                    arrowprops=dict(facecolor=rgba_color))
    elif abs(comp1) > l and exp_err > l and eff <= sig_l:
        ax.annotate("", xytext=(0, 1), xy=(diag, eff),
                    arrowprops=dict(facecolor=rgba_color))
    # FBM
    elif abs(comp1) <= l and exp_err <= l and eff <= sig_l:
        ax.annotate("", xytext=(0, 1), xy=(0, eff),
                    arrowprops=dict(facecolor=rgba_color))
        ax.annotate("", xytext=(0, 1), xy=(np.pi, eff),
                    arrowprops=dict(facecolor=rgba_color))
    # FGM
    elif abs(comp1) <= l and exp_err <= l and eff > sig_l:
        c = ax.scatter(diag, eff, color=rgba_color)
    ax.set_rticks([])  # turn default ticks off
    ax.set_rmin(1)
    if eff > 0:
        ax.set_rmax(ax_lim)
    elif eff <= 0:
        ax.set_rmax(-ax_lim)
    ax.tick_params(labelleft=False, labelright=False, labeltop=False,
                  labelbottom=True, grid_alpha=.01)  # turn labels and grid off
    ax.set_xticklabels(['', '', '', '', '', '', '', ''])
    ax.text(-.04, 0.5, r'Comp2 < 0',
            va='center', ha='center', rotation=90, rotation_mode='anchor',
            transform=ax.transAxes)
    ax.text(1.04, 0.5, r'Comp2 > 0',
            va='center', ha='center', rotation=90, rotation_mode='anchor',
            transform=ax.transAxes)
    ax.text(.5, -.04, r'Comp1 < 0', va='center', ha='center',
            rotation=0, rotation_mode='anchor', transform=ax.transAxes)
    ax.text(.5, 1.04, r'Comp1 > 0',
            va='center', ha='center', rotation=0, rotation_mode='anchor',
            transform=ax.transAxes)
    # add colorbar for temporal correlation
    cbar = fig.colorbar(dummie_cax, ax=ax, orientation='horizontal',
                        ticks=[1, 0.5, 0], shrink=0.8)
    cbar.set_label('Comp3', labelpad=4)
    cbar.set_ticklabels(['1', '0.5', '<0'])
    cbar.ax.tick_params(direction='in')

def gdiag_polar_plot_multi(eff, comp1, comp2, comp3, l=0.05, extended=True):
    """Generic diagnostic polar plot with multiple values

    Parameters
    ----------
    eff : (N,)array_like
        efficiency measure

    comp1 : (N,)array_like
        metric component 1

    comp2 : (N,)array_like
        metric component 2

    comp3 : (N,)array_like
        metric component 3

    l : float, optional
        Threshold for which diagnosis can be made. The default is 0.05.

    extended : boolean, optional
        If True, density plot is displayed. In addtion, the density plot
        is displayed besides the polar plot. The default is,
        that only the diagnostic polar plot is displayed.

    Notes
    ----------
    .. math::

        \varphi = arctan2(comp1, comp2)

    Examples
    --------
    Provide arrays with equal length

    >>> from de import de
    >>> import numpy as np
    >>> de.diag_polar_plot_multi(sig_de, brel_mean, b_slope, temp_cor)
    """
    eff_min = np.min(eff)

    ll_comp1 = comp1.tolist()
    ll_comp2 = comp2.tolist()
    ll_comp3 = comp3.tolist()
    ll_eff = eff.tolist()

    # convert temporal correlation to color
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1.0)

    delta = 0.01  # for spacing

    # determine axis limits
    if eff_min > 0:
        ax_lim = eff_min - .1
        ax_lim = np.around(ax_lim, decimals=1)
        yy = np.arange(ax_lim, 1.01, delta)[::-1]
        c_levels = np.arange(ax_lim+.1, 1.1, .1)
    elif eff_min >= 0:
        ax_lim = 0.2
        yy = np.arange(-ax_lim, 1.01, delta)[::-1]
        c_levels = np.arange(0, 1, .2)
    elif eff_min < 0 and eff_min >= -1:
        ax_lim = 1.2
        yy = np.arange(-ax_lim, 1.01, delta)[::-1]
        c_levels = np.arange(-1, 1, .2)
    elif eff_min >= -2 and eff_min < -1:
        ax_lim = 2.2
        yy = np.arange(-ax_lim, 1.01, delta)[::-1]
        c_levels = np.arange(-2, 1, .2)
    elif eff_min <= -2:
        raise ValueError("Some values of eff are out of bounds for visualization!", eff_min)

    len_yy = len(yy)

    # arrays to plot contour lines of DE
    xx = np.radians(np.linspace(0, 360, len_yy))
    theta, r = np.meshgrid(xx, yy)

    # diagnostic polar plot
    if not extended:
        fig, ax = plt.subplots(figsize=(6, 6),
                               subplot_kw=dict(projection='polar'),
                               constrained_layout=True)
        # dummie plot for colorbar of temporal correlation
        cs = np.arange(0, 1.1, 0.1)
        dummie_cax = ax.scatter(cs, cs, c=cs, cmap='plasma_r')
        # Clear axis
        ax.cla()
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
        sig_l = 1 - np.sqrt((l)**2 + (l)**2 + (l)**2)
        # loop over each data point
        for (c1, c2, c3, e) in zip(ll_comp1, ll_comp2, ll_comp3, ll_eff):
            # normalize threshold with mean flow becnhmark
            ang = np.arctan2(c1, c2)
            # convert temporal correlation to color
            rgba_color = cm.plasma_r(norm(c3))
            # relation of b_dir which explains the error
            if abs(c2) > 0:
                exp_err = c2
            elif abs(ba) == 0:
                exp_err = 0
            # diagnose the error
            if abs(c1) <= l and exp_err > l and e <= sig_l:
                c = ax.scatter(ang, e, color=rgba_color, zorder=2)
            elif abs(c1) > l and exp_err <= l and e <= sig_l:
                c = ax.scatter(ang, e, color=rgba_color, zorder=2)
            elif abs(c1) > l and exp_err > l and e <= sig_l:
                c = ax.scatter(ang, e, color=rgba_color, zorder=2)
            # FBM
            elif abs(c1) <= l and exp_err <= l and e <= sig_l:
                ax.annotate("", xytext=(0, 1), xy=(0, e),
                            arrowprops=dict(facecolor=rgba_color))
                ax.annotate("", xytext=(0, 1), xy=(np.pi, e),
                            arrowprops=dict(facecolor=rgba_color))
            # FGM
            elif abs(c1) <= l and exp_err <= l and e > sig_l:
                c = ax.scatter(ang, e, color=rgba_color, zorder=2)
        ax.set_rticks([])  # turn default ticks off
        ax.set_rmin(1)
        if eff_min > 0:
            ax.set_rmax(ax_lim)
        elif eff_min <= 0:
            ax.set_rmax(-ax_lim)
        ax.tick_params(labelleft=False, labelright=False, labeltop=False,
                      labelbottom=True, grid_alpha=.01)  # turn labels and grid off
        ax.set_xticklabels(['', '', '', '', '', '', '', ''])
        ax.text(-.04, 0.5, r'Comp2 < 0',
                va='center', ha='center', rotation=90, rotation_mode='anchor',
                transform=ax.transAxes)
        ax.text(1.04, 0.5, r'Comp2 > 0',
                va='center', ha='center', rotation=90, rotation_mode='anchor',
                transform=ax.transAxes)
        ax.text(.5, -.04, r'Comp1 < 0', va='center', ha='center',
                rotation=0, rotation_mode='anchor', transform=ax.transAxes)
        ax.text(.5, 1.04, r'Comp1 > 0',
                va='center', ha='center', rotation=0, rotation_mode='anchor',
                transform=ax.transAxes)
        # add colorbar for temporal correlation
        cbar = fig.colorbar(dummie_cax, ax=ax, orientation='horizontal',
                            ticks=[1, 0.5, 0], shrink=0.8)
        cbar.set_label('Comp3', labelpad=4)
        cbar.set_ticklabels(['1', '0.5', '<0'])
        cbar.ax.tick_params(direction='in')

    elif extended:
        fig = plt.figure(figsize=(12, 6), constrained_layout=True)
        gs = fig.add_gridspec(1, 2)
        ax = fig.add_subplot(gs[0, 0], projection='polar')
        ax1 = fig.add_axes([.66, .3, .32, .32], frameon=True)
        # dummie plot for colorbar of temporal correlation
        cs = np.arange(0, 1.1, 0.1)
        dummie_cax = ax.scatter(cs, cs, c=cs, cmap='plasma_r')
        # Clear axis
        ax.cla()
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
        sig_l = 1 - np.sqrt((l)**2 + (l)**2 + (l)**2)
        # loop over each data point
        for (c1, c2, c3, e) in zip(ll_comp1, ll_comp2, ll_comp3, ll_eff):
            # normalize threshold with mean flow becnhmark
            ang = np.arctan2(c1, c2)
            # convert temporal correlation to color
            rgba_color = cm.plasma_r(norm(c3))
            # relation of b_dir which explains the error
            if abs(c2) > 0:
                exp_err = c2
            elif abs(ba) == 0:
                exp_err = 0
            # diagnose the error
            if abs(c1) <= l and exp_err > l and e <= sig_l:
                c = ax.scatter(ang, e, color=rgba_color, zorder=2)
            elif abs(c1) > l and exp_err <= l and e <= sig_l:
                c = ax.scatter(ang, e, color=rgba_color, zorder=2)
            elif abs(c1) > l and exp_err > l and e <= sig_l:
                c = ax.scatter(ang, e, color=rgba_color, zorder=2)
            # FBM
            elif abs(c1) <= l and exp_err <= l and e <= sig_l:
                ax.annotate("", xytext=(0, 1), xy=(0, e),
                            arrowprops=dict(facecolor=rgba_color))
                ax.annotate("", xytext=(0, 1), xy=(np.pi, e),
                            arrowprops=dict(facecolor=rgba_color))
            # FGM
            elif abs(c1) <= l and exp_err <= l and e > sig_l:
                c = ax.scatter(ang, e, color=rgba_color, zorder=2)
        ax.set_rticks([])  # turn default ticks off
        ax.set_rmin(1)
        if eff_min > 0:
            ax.set_rmax(ax_lim)
        elif eff_min <= 0:
            ax.set_rmax(-ax_lim)
        ax.tick_params(labelleft=False, labelright=False, labeltop=False,
                      labelbottom=True, grid_alpha=.01)  # turn labels and grid off
        ax.set_xticklabels(['', '', '', '', '', '', '', ''])
        ax.text(-.04, 0.5, r'Comp2 < 0',
                va='center', ha='center', rotation=90, rotation_mode='anchor',
                transform=ax.transAxes)
        ax.text(1.04, 0.5, r'Comp2 > 0',
                va='center', ha='center', rotation=90, rotation_mode='anchor',
                transform=ax.transAxes)
        ax.text(.5, -.04, r'Comp1 < 0', va='center', ha='center',
                rotation=0, rotation_mode='anchor', transform=ax.transAxes)
        ax.text(.5, 1.04, r'Comp1 > 0',
                va='center', ha='center', rotation=0, rotation_mode='anchor',
                transform=ax.transAxes)
        # add colorbar for temporal correlation
        cbar = fig.colorbar(dummie_cax, ax=ax, orientation='horizontal',
                            ticks=[1, 0.5, 0], shrink=0.8)
        cbar.set_label('Comp3', labelpad=4)
        cbar.set_ticklabels(['1', '0.5', '<0'])
        cbar.ax.tick_params(direction='in')

        # convert to degrees
        diag = np.arctan2(comp1, comp2)
        diag_deg = (diag  * (180 / np.pi)) + 135
        diag_deg[diag_deg < 0] = 360 - diag_deg[diag_deg < 0]

        # 1-D density plot
        g = sns.kdeplot(diag_deg, color='k', ax=ax1)
        ax1.set_xticks([0, 90, 180, 270, 360])
        ax1.set_xlim(0, 360)
        ax1.set_ylim(0, )
        ax1.set(ylabel='Density',
                xlabel='[$^\circ$]')

        # 2-D density plot
        # g = (sns.jointplot(diag_deg, sig_de, color='k', marginal_kws={'color':'k'}).plot_joint(sns.kdeplot, zorder=0, n_levels=10))
        g = (sns.jointplot(diag_deg, eff, kind='kde', zorder=1,
                           n_levels=20, cmap='Greens', shade_lowest=False,
                           marginal_kws={'color':'k', 'shade':False}).plot_joint(sns.scatterplot, color='k', alpha=.5, zorder=2))
        g.set_axis_labels(r'[$^\circ$]', r'Eff [-]')
        g.ax_joint.set_xticks([0, 90, 180, 270, 360])
        g.ax_joint.set_xlim(0, 360)
        g.ax_joint.set_ylim(-ax_lim, 1)
        g.ax_marg_x.set_xticks([0, 90, 180, 270, 360])
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
