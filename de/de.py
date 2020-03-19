# -*- coding: utf-8 -*-

"""
de.de
~~~~~~~~~~~
Diagnosing model performance using an efficiency measure based on flow
duration curve and temporal correlation. The efficiency measure can be
visualized by diagnostic polar plots which facilitates decomposing potential
error origins (dynamic errors vs. constant erros vs. timing errors)
:2019 by Robin Schwemmle.
:license: GNU GPLv3, see LICENSE for more details.
"""

import numpy as np
# RunTimeWarning will not be displayed (division by zeros or NaN values)
np.seterr(divide='ignore', invalid='ignore')
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
import pandas as pd
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


def calc_brel_mean(obs, sim, sort=True):
    r"""
    Calculate arithmetic mean of relative bias.

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
    brel[sim_obs_diff == 0] = 0
    brel = brel[np.isfinite(brel)]
    brel_mean = np.mean(brel)

    return brel_mean

def calc_brel_rest(obs, sim, sort=True):
    r"""
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
    brel[sim_obs_diff == 0] = 0
    brel = brel[np.isfinite(brel)]
    brel_mean = np.mean(brel)
    brel_rest = brel - brel_mean

    return brel_rest

def calc_bias_area(brel_rest):
    r"""
    Calculate absolute bias area for entire flow duration curve.

    Parameters
    ----------
    brel_rest : (N,)array_like
        remaining relative bias as 1-D array

    Returns
    ----------
    b_area : float
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
    0.1112908496732026
    """
    perc = np.linspace(0, 1, len(brel_rest))
    # area of absolute bias
    b_area = integrate.simps(abs(brel_rest), perc)

    return b_area

def calc_bias_dir(brel_rest):
    r"""
    Calculate absolute bias area for high flow and low flow.

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
    -0.014705882352941155
    """
    mid_idx = int(len(brel_rest)/2)
    # integral of relative bias < 50 %
    perc = np.linspace(0, .5, mid_idx)
    # direction of bias
    b_dir = integrate.simps(brel_rest[:mid_idx], perc)

    return b_dir

def calc_bias_slope(b_area, b_dir):
    r"""
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
    0.1112908496732026
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
    eff : float
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
    0.8202204384691575
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
    eff = 1 - np.sqrt((brel_mean)**2 + (b_area)**2 + (temp_cor - 1)**2)

    return eff

def diag_polar_plot(obs, sim, sort=True, l=0.05, extended=False):
    r"""
    Diagnostic polar plot of Diagnostic efficiency (DE) for a single
    evaluation.

    Parameters
    ----------
    obs : (N,)array_like
        Observed time series as 1-D array

    sim : (N,)array_like
        Simulated time series as 1-D array

    sort : boolean, optional
        If True, time series are sorted by ascending order. If False, time
        series are not sorted. The default is to sort.

    l : float, optional
        Threshold for which diagnosis can be made. The default is 0.05.

    extended : boolean, optional
        If True, extended diagnostic plot is displayed. In addtion, the
        duration curve of B_rest is plotted besides the polar plot. The default
        is, that only the diagnostic polar plot is displayed.

    Returns
    ----------
    fig : Figure
        Returns a single figure if extended=False and two figures if
        extended=True.

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
    eff = 1 - np.sqrt((brel_mean)**2 + (b_area)**2 + (temp_cor - 1)**2)

    # direction of bias
    b_dir = calc_bias_dir(brel_rest)

    # slope of bias
    b_slope = calc_bias_slope(b_area, b_dir)

    # convert to radians
    # (y, x) Trigonometric inverse tangent
    phi = np.arctan2(brel_mean, b_slope)

    # convert temporal correlation to color
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1.0)
    rgba_color = cm.plasma_r(norm(temp_cor))

    delta = 0.01  # for spacing

    # determine axis limits
    if eff >= 0:
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
        raise AssertionError("Value of 'DE' is out of bounds for visualization!",
                             eff)

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
        ax.plot((1, np.deg2rad(45)), (1, np.min(yy)), color='lightgray',
                linewidth=1.5, ls='--', zorder=0)
        ax.plot((1, np.deg2rad(135)), (1, np.min(yy)), color='lightgray',
                linewidth=1.5, ls='--', zorder=0)
        ax.plot((1, np.deg2rad(225)), (1, np.min(yy)), color='lightgray',
                linewidth=1.5, ls='--', zorder=0)
        ax.plot((1, np.deg2rad(315)), (1, np.min(yy)), color='lightgray',
                linewidth=1.5, ls='--', zorder=0)
        # contours of DE
        cp = ax.contour(theta, r, r, colors='darkgray', levels=c_levels,
                        zorder=1)
        cl = ax.clabel(cp, inline=True, fontsize=10, fmt='%1.1f',
                       colors='dimgrey')
        # threshold efficiency for FBM
        eff_l = 1 - np.sqrt((l)**2 + (l)**2 + (l)**2)
        # relation of b_dir which explains the error
        if abs(b_area) > 0:
            exp_err = (abs(b_dir) * 2)/abs(b_area)
        elif abs(b_area) == 0:
            exp_err = 0
        # diagnose the error
        if abs(brel_mean) <= l and exp_err > l and eff <= eff_l:
            ax.annotate("", xytext=(0, 1), xy=(phi, eff),
                        arrowprops=dict(facecolor=rgba_color))
        elif abs(brel_mean) > l and exp_err <= l and eff <= eff_l:
            ax.annotate("", xytext=(0, 1), xy=(phi, eff),
                        arrowprops=dict(facecolor=rgba_color))
        elif abs(brel_mean) > l and exp_err > l and eff <= eff_l:
            ax.annotate("", xytext=(0, 1), xy=(phi, eff),
                        arrowprops=dict(facecolor=rgba_color))
        # FBM
        elif abs(brel_mean) <= l and exp_err <= l and eff <= eff_l:
            ax.annotate("", xytext=(0, 1), xy=(0, eff),
                        arrowprops=dict(facecolor=rgba_color))
            ax.annotate("", xytext=(0, 1), xy=(np.pi, eff),
                        arrowprops=dict(facecolor=rgba_color))
        # FGM
        elif abs(brel_mean) <= l and exp_err <= l and eff > eff_l:
            c = ax.scatter(phi, eff, color=rgba_color)
        ax.set_rticks([])  # turn default ticks off
        ax.set_rmin(1)
        if eff >= 0:
            ax.set_rmax(0)
        elif eff < 0:
            ax.set_rmax(-ax_lim + .2)
        # turn labels and grid off
        ax.tick_params(labelleft=False, labelright=False, labeltop=False,
                       labelbottom=True, grid_alpha=.01)
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
        ax.text(.5, -.04, r'$\overline{B_{rel}}$ < 0', va='center',
                ha='center', rotation=0, rotation_mode='anchor',
                transform=ax.transAxes)
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

        return fig

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
        ax.plot((1, np.deg2rad(45)), (1, np.min(yy)), color='lightgray',
                linewidth=1.5, ls='--', zorder=0)
        ax.plot((1, np.deg2rad(135)), (1, np.min(yy)), color='lightgray',
                linewidth=1.5, ls='--', zorder=0)
        ax.plot((1, np.deg2rad(225)), (1, np.min(yy)), color='lightgray',
                linewidth=1.5, ls='--', zorder=0)
        ax.plot((1, np.deg2rad(315)), (1, np.min(yy)), color='lightgray',
                linewidth=1.5, ls='--', zorder=0)
        # contours of DE
        cp = ax.contour(theta, r, r, colors='darkgray', levels=c_levels,
                        zorder=1)
        cl = ax.clabel(cp, inline=True, fontsize=10, fmt='%1.1f',
                       colors='dimgrey')
        # threshold efficiency for FBM
        eff_l = 1 - np.sqrt((l)**2 + (l)**2 + (l)**2)
        # relation of b_dir which explains the error
        if abs(b_area) > 0:
            exp_err = (abs(b_dir) * 2)/abs(b_area)
        elif abs(b_area) == 0:
            exp_err = 0
        # diagnose the error
        if abs(brel_mean) <= l and exp_err > l and eff <= eff_l:
            ax.annotate("", xytext=(0, 1), xy=(phi, eff),
                        arrowprops=dict(facecolor=rgba_color))
        elif abs(brel_mean) > l and exp_err <= l and eff <= eff_l:
            ax.annotate("", xytext=(0, 1), xy=(phi, eff),
                        arrowprops=dict(facecolor=rgba_color))
        elif abs(brel_mean) > l and exp_err > l and eff <= eff_l:
            ax.annotate("", xytext=(0, 1), xy=(phi, eff),
                        arrowprops=dict(facecolor=rgba_color))
        # FBM
        elif abs(brel_mean) <= l and exp_err <= l and eff <= eff_l:
            ax.annotate("", xytext=(0, 1), xy=(0, eff),
                        arrowprops=dict(facecolor=rgba_color))
            ax.annotate("", xytext=(0, 1), xy=(np.pi, eff),
                        arrowprops=dict(facecolor=rgba_color))
        # FGM
        elif abs(brel_mean) <= l and exp_err <= l and eff > eff_l:
            c = ax.scatter(phi, eff, color=rgba_color)
        ax.set_rticks([])  # turn default ticks off
        ax.set_rmin(1)
        if eff >= 0:
            ax.set_rmax(0)
        elif eff < 0:
            ax.set_rmax(-ax_lim + .2)
        # turn labels and grid off
        ax.tick_params(labelleft=False, labelright=False, labeltop=False,
                       labelbottom=True, grid_alpha=.01)
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
        ax.text(.5, -.04, r'$\overline{B_{rel}}$ < 0', va='center',
                ha='center', rotation=0, rotation_mode='anchor',
                transform=ax.transAxes)
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
        ax1.fill_between(prob, brel_rest, where=0 < brel_rest,
                         facecolor='purple')
        ax1.fill_between(prob, brel_rest, where=0 > brel_rest, facecolor='red')
        ax1.set(ylabel=r'$B_{rest}$ [-]',
                xlabel='Exceedence probabilty [-]')

        return fig

def diag_polar_plot_multi(brel_mean, b_area, temp_cor, eff_de, b_dir, phi,
                          l=0.05, extended=False):
    r"""
    Diagnostic polar plot of Diagnostic efficiency (DE) for multiple
    evaluations.

    Parameters
    ----------
    brel_mean : (N,)array_like
        relative mean bias as 1-D array

    b_area : (N,)array_like
        bias area as 1-D array

    temp_cor : (N,)array_like
        temporal correlation as 1-D array

    eff_de : (N,)array_like
        diagnostic efficiency as 1-D array

    b_dir : (N,)array_like
        direction of bias as 1-D array

    phi : (N,)array_like
        angle as 1-D array

    l : float, optional
        Deviation of metric terms used to calculate the threshold of DE for
        which diagnosis is available. The default is 0.05.

    extended : boolean, optional
        If True, density plot is displayed. In addtion, the density plot
        is displayed besides the polar plot. The default is,
        that only the diagnostic polar plot is displayed.

    Returns
    ----------
    fig : Figure
        diagnostic polar plot

    Notes
    ----------
    .. math::

        \varphi = arctan2(\overline{B_{rel}}, B_{slope})

    Examples
    --------
    Provide arrays with equal length

    >>> from de import de
    >>> import numpy as np
    >>> brel_mean = np.array([0.1, 0.15, 0.2, 0.1, 0.05, 0.15])
    >>> b_area = np.array([0.15, 0.1, 0.2, 0.1, 0.1, 0.2])
    >>> temp_cor = np.array([0.9, 0.85, 0.8, 0.9, 0.85, 0.9])
    >>> eff_de = np.array([0.79, 0.76, 0.65, 0.82, 0.81, 0.73])
    >>> b_dir = np.array([0.08, 0.05, 0.1, 0.05, 0.05, 0.1])
    >>> phi = np.array([0.58, 0.98, 0.78, 0.78, 0.46, 0.64])
    >>> de.diag_polar_plot_multi(brel_mean, b_area, temp_cor, eff_de, b_dir,
                                 phi)
    """
    eff_min = np.min(eff_de)

    ll_brel_mean = brel_mean.tolist()
    ll_b_dir = b_dir.tolist()
    ll_b_area = b_area.tolist()
    ll_eff = eff_de.tolist()
    ll_phi = phi.tolist()
    ll_temp_cor = temp_cor.tolist()

    # convert temporal correlation to color
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1.0)

    delta = 0.01  # for spacing

    # determine axis limits
    if eff_min >= 0:
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
        raise ValueError("Some values of 'DE' are too low for visualization!",
                         eff_min)

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
        ax.plot((1, np.deg2rad(45)), (1, np.min(yy)), color='lightgray',
                linewidth=1.5, ls='--', zorder=0)
        ax.plot((1, np.deg2rad(135)), (1, np.min(yy)), color='lightgray',
                linewidth=1.5, ls='--', zorder=0)
        ax.plot((1, np.deg2rad(225)), (1, np.min(yy)), color='lightgray',
                linewidth=1.5, ls='--', zorder=0)
        ax.plot((1, np.deg2rad(315)), (1, np.min(yy)), color='lightgray',
                linewidth=1.5, ls='--', zorder=0)
        # contours of DE
        cp = ax.contour(theta, r, r, colors='darkgray', levels=c_levels,
                        zorder=1)
        cl = ax.clabel(cp, inline=True, fontsize=10, fmt='%1.1f',
                       colors='dimgrey')
        # threshold efficiency for FBM
        eff_l = 1 - np.sqrt((l)**2 + (l)**2 + (l)**2)
        # loop over each data point
        zz = zip(ll_brel_mean, ll_b_dir, ll_b_area, ll_temp_cor, ll_eff,
                 ll_phi)
        for (bm, bd, ba, r, eff, ang) in zz:
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
            if abs(bm) <= l and exp_err > l and eff <= eff_l:
                c = ax.scatter(ang, eff, color=rgba_color, zorder=2)
            elif abs(bm) > l and exp_err <= l and eff <= eff_l:
                c = ax.scatter(ang, eff, color=rgba_color, zorder=2)
            elif abs(bm) > l and exp_err > l and eff <= eff_l:
                c = ax.scatter(ang, eff, color=rgba_color, zorder=2)
            # FBM
            elif abs(bm) <= l and exp_err <= l and eff <= eff_l:
                ax.annotate("", xytext=(0, 1), xy=(0, eff),
                            arrowprops=dict(facecolor=rgba_color))
                ax.annotate("", xytext=(0, 1), xy=(np.pi, eff),
                            arrowprops=dict(facecolor=rgba_color))
            # FGM
            elif abs(bm) <= l and exp_err <= l and eff > eff_l:
                c = ax.scatter(ang, eff, color=rgba_color, zorder=2)
        ax.set_rticks([])  # turn default ticks off
        ax.set_rmin(1)
        if eff_min >= 0:
            ax.set_rmax(0)
        elif eff_min <= 0:
            ax.set_rmax(-ax_lim + .2)
        # turn labels and grid off
        ax.tick_params(labelleft=False, labelright=False, labeltop=False,
                       labelbottom=True, grid_alpha=.01)
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
        ax.text(.5, -.04, r'$\overline{B_{rel}}$ < 0', va='center',
                ha='center', rotation=0, rotation_mode='anchor',
                transform=ax.transAxes)
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

        return fig

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
        ax.plot((1, np.deg2rad(45)), (1, np.min(yy)), color='lightgray',
                linewidth=1.5, ls='--', zorder=0)
        ax.plot((1, np.deg2rad(135)), (1, np.min(yy)), color='lightgray',
                linewidth=1.5, ls='--', zorder=0)
        ax.plot((1, np.deg2rad(225)), (1, np.min(yy)), color='lightgray',
                linewidth=1.5, ls='--', zorder=0)
        ax.plot((1, np.deg2rad(315)), (1, np.min(yy)), color='lightgray',
                linewidth=1.5, ls='--', zorder=0)
        # contours of DE
        cp = ax.contour(theta, r, r, colors='darkgray', levels=c_levels,
                        zorder=1)
        cl = ax.clabel(cp, inline=True, fontsize=10, fmt='%1.1f',
                       colors='dimgrey')
        # threshold efficiency for FBM
        eff_l = 1 - np.sqrt((l)**2 + (l)**2 + (l)**2)
        # loop over each data point
        zz = zip(ll_brel_mean, ll_b_dir, ll_b_area, ll_temp_cor, ll_eff,
                 ll_phi)
        for (bm, bd, ba, r, eff, ang) in zz:
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
            if abs(bm) <= l and exp_err > l and eff <= eff_l:
                c = ax.scatter(ang, eff, color=rgba_color, zorder=2)
            elif abs(bm) > l and exp_err <= l and eff <= eff_l:
                c = ax.scatter(ang, eff, color=rgba_color, zorder=2)
            elif abs(bm) > l and exp_err > l and eff <= eff_l:
                c = ax.scatter(ang, eff, color=rgba_color, zorder=2)
            # FBM
            elif abs(bm) <= l and exp_err <= l and eff <= eff_l:
                ax.annotate("", xytext=(0, 1), xy=(0, eff),
                            arrowprops=dict(facecolor=rgba_color))
                ax.annotate("", xytext=(0, 1), xy=(np.pi, eff),
                            arrowprops=dict(facecolor=rgba_color))
            # FGM
            elif abs(bm) <= l and exp_err <= l and eff > eff_l:
                c = ax.scatter(ang, eff, color=rgba_color, zorder=2)
        ax.set_rticks([])  # turn default ticks off
        ax.set_rmin(1)
        if eff_min >= 0:
            ax.set_rmax(0)
        elif eff_min <= 0:
            ax.set_rmax(-ax_lim + .2)
        # turn labels and grid off
        ax.tick_params(labelleft=False, labelright=False, labeltop=False,
                       labelbottom=True, grid_alpha=.01)
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
        ax.text(.5, -.04, r'$\overline{B_{rel}}$ < 0', va='center',
                ha='center', rotation=0, rotation_mode='anchor',
                transform=ax.transAxes)
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
        diag_deg = (phi * (180 / np.pi)) + 135
        diag_deg[diag_deg < 0] = 360 - diag_deg[diag_deg < 0]

        # 1-D density plot
        g = sns.kdeplot(diag_deg, color='k', ax=ax1)
        ax1.set_xticks([0, 90, 180, 270, 360])
        ax1.set_xlim(0, 360)
        ax1.set_ylim(0, )
        ax1.set(ylabel='Density', xlabel=r'[$^\circ$]')

        # 2-D density plot
        r_colors = cm.plasma_r(norm(temp_cor))
        g = (sns.jointplot(diag_deg, eff_de, kind='kde', zorder=1,
                           n_levels=20, cmap='Greys', shade_lowest=False,
                           marginal_kws={'color': 'k', 'shade': False})
                           .plot_joint(plt.scatter, c=r_colors, alpha=.4,
                                       zorder=2))
        g.set_axis_labels(r'[$^\circ$]', 'DE [-]')
        g.ax_joint.set_xticks([0, 90, 180, 270, 360])
        g.ax_joint.set_xlim(0, 360)
        g.ax_joint.set_ylim(-ax_lim, 1)
        g.ax_marg_x.set_xticks([0, 90, 180, 270, 360])
        kde_data = g.ax_marg_x.get_lines()[0].get_data()
        kde_xx = kde_data[0]
        kde_yy = kde_data[1]
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

        return fig, g.fig

def gdiag_polar_plot(eff, comp1, comp2, comp3, l=0.05): # pragma: no cover
    r"""
    Generic diagnostic polar plot for  single evaluation.

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
        Deviation of metric terms used to calculate the threshold of DE for
        which diagnosis is available. The default is 0.05.

    Returns
    ----------
    fig : Figure
        diagnostic polar plot

    Notes
    ----------
    .. math::

        \varphi = arctan2(comp1, comp2)
    """
    # convert to radians
    # (y, x) Trigonometric inverse tangent
    phi = np.arctan2(comp1, comp2)

    # convert metric component 3 to color
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1.0)
    rgba_color = cm.plasma_r(norm(comp3))

    delta = 0.01  # for spacing

    # determine axis limits
    if eff >= 0:
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
        raise AssertionError("Value of eff is out of bounds for visualization!",
                             eff)

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
    ax.plot((1, np.deg2rad(45)), (1, np.min(yy)), color='lightgray',
            linewidth=1.5, ls='--', zorder=0)
    ax.plot((1, np.deg2rad(135)), (1, np.min(yy)), color='lightgray',
            linewidth=1.5, ls='--', zorder=0)
    ax.plot((1, np.deg2rad(225)), (1, np.min(yy)), color='lightgray',
            linewidth=1.5, ls='--', zorder=0)
    ax.plot((1, np.deg2rad(315)), (1, np.min(yy)), color='lightgray',
            linewidth=1.5, ls='--', zorder=0)
    # contours of DE
    cp = ax.contour(theta, r, r, colors='darkgray', levels=c_levels, zorder=1)
    cl = ax.clabel(cp, inline=True, fontsize=10, fmt='%1.1f',
                   colors='dimgrey')
    # threshold efficiency for FBM
    eff_l = 1 - np.sqrt((l)**2 + (l)**2 + (l)**2)
    # relation of b_dir which explains the error
    if abs(comp2) > 0:
        exp_err = comp2
    elif abs(comp2) == 0:
        exp_err = 0
    # diagnose the error
    if abs(comp1) <= l and exp_err > l and eff <= eff_l:
        ax.annotate("", xytext=(0, 1), xy=(phi, eff),
                    arrowprops=dict(facecolor=rgba_color))
    elif abs(comp1) > l and exp_err <= l and eff <= eff_l:
        ax.annotate("", xytext=(0, 1), xy=(phi, eff),
                    arrowprops=dict(facecolor=rgba_color))
    elif abs(comp1) > l and exp_err > l and eff <= eff_l:
        ax.annotate("", xytext=(0, 1), xy=(phi, eff),
                    arrowprops=dict(facecolor=rgba_color))
    # FBM
    elif abs(comp1) <= l and exp_err <= l and eff <= eff_l:
        ax.annotate("", xytext=(0, 1), xy=(0, eff),
                    arrowprops=dict(facecolor=rgba_color))
        ax.annotate("", xytext=(0, 1), xy=(np.pi, eff),
                    arrowprops=dict(facecolor=rgba_color))
    # FGM
    elif abs(comp1) <= l and exp_err <= l and eff > eff_l:
        c = ax.scatter(phi, eff, color=rgba_color)
    ax.set_rticks([])  # turn default ticks off
    ax.set_rmin(1)
    if eff >= 0:
        ax.set_rmax(0)
    elif eff <= 0:
        ax.set_rmax(-ax_lim + .2)
    # turn labels and grid off
    ax.tick_params(labelleft=False, labelright=False, labeltop=False,
                   labelbottom=True, grid_alpha=.01)
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

    return fig

def gdiag_polar_plot_multi(eff, comp1, comp2, comp3, l=0.05, extended=True): # pragma: no cover
    r"""
    Generic diagnostic polar plot for multiple evaluations.

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
        Deviation of metric terms used to calculate the threshold of DE for
        which diagnosis is available. The default is 0.05.

    extended : boolean, optional
        If True, density plot is displayed. In addtion, the density plot
        is displayed besides the polar plot. The default is,
        that only the diagnostic polar plot is displayed.

    Returns
    ----------
    fig : Figure
        diagnostic polar plot

    Notes
    ----------
    .. math::

        \varphi = arctan2(comp1, comp2)
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
    if eff_min >= 0:
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
        raise ValueError("Some values of eff are out of bounds for visualization!",
                         eff_min)

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
        ax.plot((1, np.deg2rad(45)), (1, np.min(yy)), color='lightgray',
                linewidth=1.5, ls='--', zorder=0)
        ax.plot((1, np.deg2rad(135)), (1, np.min(yy)), color='lightgray',
                linewidth=1.5, ls='--', zorder=0)
        ax.plot((1, np.deg2rad(225)), (1, np.min(yy)), color='lightgray',
                linewidth=1.5, ls='--', zorder=0)
        ax.plot((1, np.deg2rad(315)), (1, np.min(yy)), color='lightgray',
                linewidth=1.5, ls='--', zorder=0)
        # contours of DE
        cp = ax.contour(theta, r, r, colors='darkgray', levels=c_levels,
                        zorder=1)
        cl = ax.clabel(cp, inline=True, fontsize=10, fmt='%1.1f',
                       colors='dimgrey')
        # threshold efficiency for FBM
        eff_l = 1 - np.sqrt((l)**2 + (l)**2 + (l)**2)
        # loop over each data point
        for (c1, c2, c3, eff) in zip(ll_comp1, ll_comp2, ll_comp3, ll_eff):
            ang = np.arctan2(c1, c2)
            # convert temporal correlation to color
            rgba_color = cm.plasma_r(norm(c3))
            # relation of b_dir which explains the error
            if abs(c2) > 0:
                exp_err = c2
            elif abs(c2) == 0:
                exp_err = 0
            # diagnose the error
            if abs(c1) <= l and exp_err > l and eff <= eff_l:
                c = ax.scatter(ang, eff, color=rgba_color, zorder=2)
            elif abs(c1) > l and exp_err <= l and eff <= eff_l:
                c = ax.scatter(ang, eff, color=rgba_color, zorder=2)
            elif abs(c1) > l and exp_err > l and eff <= eff_l:
                c = ax.scatter(ang, eff, color=rgba_color, zorder=2)
            # FBM
            elif abs(c1) <= l and exp_err <= l and eff <= eff_l:
                ax.annotate("", xytext=(0, 1), xy=(0, eff),
                            arrowprops=dict(facecolor=rgba_color))
                ax.annotate("", xytext=(0, 1), xy=(np.pi, eff),
                            arrowprops=dict(facecolor=rgba_color))
            # FGM
            elif abs(c1) <= l and exp_err <= l and eff > eff_l:
                c = ax.scatter(ang, eff, color=rgba_color, zorder=2)
        ax.set_rticks([])  # turn default ticks off
        ax.set_rmin(1)
        if eff_min >= 0:
            ax.set_rmax(0)
        elif eff_min <= 0:
            ax.set_rmax(-ax_lim + .2)
        # turn labels and grid off
        ax.tick_params(labelleft=False, labelright=False, labeltop=False,
                       labelbottom=True, grid_alpha=.01)
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

        return fig

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
        ax.plot((1, np.deg2rad(45)), (1, np.min(yy)), color='lightgray',
                linewidth=1.5, ls='--', zorder=0)
        ax.plot((1, np.deg2rad(135)), (1, np.min(yy)), color='lightgray',
                linewidth=1.5, ls='--', zorder=0)
        ax.plot((1, np.deg2rad(225)), (1, np.min(yy)), color='lightgray',
                linewidth=1.5, ls='--', zorder=0)
        ax.plot((1, np.deg2rad(315)), (1, np.min(yy)), color='lightgray',
                linewidth=1.5, ls='--', zorder=0)
        # contours of DE
        cp = ax.contour(theta, r, r, colors='darkgray', levels=c_levels,
                        zorder=1)
        cl = ax.clabel(cp, inline=True, fontsize=10, fmt='%1.1f',
                       colors='dimgrey')
        # threshold efficiency for FBM
        eff_l = 1 - np.sqrt((l)**2 + (l)**2 + (l)**2)
        # loop over each data point
        for (c1, c2, c3, eff) in zip(ll_comp1, ll_comp2, ll_comp3, ll_eff):
            # normalize threshold with mean flow becnhmark
            ang = np.arctan2(c1, c2)
            # convert temporal correlation to color
            rgba_color = cm.plasma_r(norm(c3))
            # relation of b_dir which explains the error
            if abs(c2) > 0:
                exp_err = c2
            elif abs(c2) == 0:
                exp_err = 0
            # diagnose the error
            if abs(c1) <= l and exp_err > l and eff <= eff_l:
                c = ax.scatter(ang, eff, color=rgba_color, zorder=2)
            elif abs(c1) > l and exp_err <= l and eff <= eff_l:
                c = ax.scatter(ang, eff, color=rgba_color, zorder=2)
            elif abs(c1) > l and exp_err > l and eff <= eff_l:
                c = ax.scatter(ang, eff, color=rgba_color, zorder=2)
            # FBM
            elif abs(c1) <= l and exp_err <= l and eff <= eff_l:
                ax.annotate("", xytext=(0, 1), xy=(0, eff),
                            arrowprops=dict(facecolor=rgba_color))
                ax.annotate("", xytext=(0, 1), xy=(np.pi, eff),
                            arrowprops=dict(facecolor=rgba_color))
            # FGM
            elif abs(c1) <= l and exp_err <= l and eff > eff_l:
                c = ax.scatter(ang, eff, color=rgba_color, zorder=2)
        ax.set_rticks([])  # turn default ticks off
        ax.set_rmin(1)
        if eff_min >= 0:
            ax.set_rmax(0)
        elif eff_min < 0:
            ax.set_rmax(-ax_lim + .2)
        # turn labels and grid off
        ax.tick_params(labelleft=False, labelright=False, labeltop=False,
                       labelbottom=True, grid_alpha=.01)
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
        phi = np.arctan2(comp1, comp2)
        diag_deg = (phi * (180 / np.pi)) + 135
        diag_deg[diag_deg < 0] = 360 - diag_deg[diag_deg < 0]

        # 1-D density plot
        g = sns.kdeplot(diag_deg, color='k', ax=ax1)
        ax1.set_xticks([0, 90, 180, 270, 360])
        ax1.set_xlim(0, 360)
        ax1.set_ylim(0, )
        ax1.set(ylabel='Density', xlabel=r'[$^\circ$]')

        # 2-D density plot
        c3_colors = cm.plasma_r(norm(comp3))
        g = (sns.jointplot(diag_deg, eff, kind='kde', zorder=1,
                           n_levels=20, cmap='Greys', shade_lowest=False,
                           marginal_kws={'color': 'k', 'shade': False})
                           .plot_joint(plt.scatter, c=c3_colors, alpha=.4,
                                       zorder=2))
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

        return fig, g.fig
