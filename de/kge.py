# -*- coding: utf-8 -*-

"""
de.kge
~~~~~~~~~~~
Kling-Gupta efficiency measure. The efficiency measure can be
visualized in 2D-Plot which facilitates decomposing the metric terms (bias
error vs. variability error vs. timing error)
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
        correlation coefficient ('pearson') can be used to describe the
        temporalcorrelation. The default is to calculate the Pearson
        correlation.

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

def calc_kge_beta(obs, sim):
    """
    Calculate the beta term of Kling-Gupta-Efficiency (KGE).

    Parameters
    ----------
    obs : (N,)array_like
        Observed time series as 1-D array

    sim : (N,)array_like
        Simulated time series as 1-D array

    Returns
    ----------
    kge_beta : float
        alpha value

    Notes
    ----------
    .. math::

        \beta = \frac{\mu_{sim}}{\mu_{obs}}

    Examples
    --------
    Provide arrays with equal length

    >>> from de import de
    >>> import numpy as np
    >>> obs = np.array([1.5, 1, 0.8, 0.85, 1.5, 2])
    >>> sim = np.array([1.6, 1.3, 1, 0.8, 1.2, 2.5])
    >>> de.calc_kge_beta(obs, sim)
    1.0980392156862746

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
    if len(obs) != len(sim):
        raise AssertionError("Arrays are not of equal length!")

    # calculate alpha term
    obs_mean = np.mean(obs)
    sim_mean = np.mean(sim)
    kge_beta = sim_mean/obs_mean

    return kge_beta

def calc_kge_alpha(obs, sim):
    """
    Calculate the alpha term of the Kling-Gupta-Efficiency (KGE).

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

    Notes
    ----------
    .. math::

        \alpha = \frac{\sigma_{sim}}{\sigma_{obs}}

    Examples
    --------
    Provide arrays with equal length

    >>> from de import de
    >>> import numpy as np
    >>> obs = np.array([1.5, 1, 0.8, 0.85, 1.5, 2])
    >>> sim = np.array([1.6, 1.3, 1, 0.8, 1.2, 2.5])
    >>> kge.calc_kge_alpha(obs, sim)
    1.2812057455166919

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
    if len(obs) != len(sim):
        raise AssertionError("Arrays are not of equal length!")

    obs_std = np.std(obs)
    sim_std = np.std(sim)
    kge_alpha = sim_std/obs_std

    return kge_alpha

def calc_kge_gamma(obs, sim):
    """
    Calculate the gamma term of Kling-Gupta-Efficiency (KGE).

    Parameters
    ----------
    obs : (N,)array_like
        Observed time series as 1-D array

    sim : (N,)array_like
        Simulated time series as 1-D array

    Returns
    ----------
    kge_gamma : float
        gamma value

    Notes
    ----------
    .. math::

        \gamma = \frac{CV_{sim}}{CV_{obs}}

    Examples
    --------
    Provide arrays with equal length

    >>> from de import de
    >>> import numpy as np
    >>> obs = np.array([1.5, 1, 0.8, 0.85, 1.5, 2])
    >>> sim = np.array([1.6, 1.3, 1, 0.8, 1.2, 2.5])
    >>> kge.calc_kge_gamma(obs, sim)
    1.166812375381273

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
    if len(obs) != len(sim):
        raise AssertionError("Arrays are not of equal length!")

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
        Simulated time series as 1-D array

    r : str, optional
        Either Spearman correlation coefficient ('spearman'; Pool et al. 2018)
        or Pearson correlation coefficient ('pearson'; Gupta et al. 2009) can
        be used to describe the temporal correlation. The default is to
        calculate the Pearson correlation.

    var : str, optional
        Either coefficient of variation ('cv'; Kling et al. 2012) or standard
        deviation ('std'; Gupta et al. 2009, Pool et al. 2018) to describe the
        gamma term. The default is to calculate the standard deviation.

    Returns
    ----------
    eff : float
        Kling-Gupta-Efficiency measure

    Examples
    --------
    Provide arrays with equal length

    >>> from de import de
    >>> import numpy as np
    >>> obs = np.array([1.5, 1, 0.8, 0.85, 1.5, 2])
    >>> sim = np.array([1.6, 1.3, 1, 0.8, 1.2, 2.5])
    >>> kge.calc_kge(obs, sim)
    0.683901305466148

    Notes
    ----------
    .. math::

        KGE = 1 - \sqrt{(\beta - 1)^2 + (\alpha - 1)^2 + (r - 1)^2}

        KGE = 1 - \sqrt{(\frac{\mu_{sim}}{\mu_{obs}} - 1)^2 + (\frac{\sigma_{sim}}{\sigma_{obs}} - 1)^2 + (r - 1)^2}

        KGE = 1 - \sqrt{(\beta - 1)^2 + (\gamma - 1)^2 + (r - 1)^2}

        KGE = 1 - \sqrt{(\frac{\mu_{sim}}{\mu_{obs}} - 1)^2 + (\frac{CV_{sim}}{CV_{obs}} - 1)^2 + (r - 1)^2}

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
    if len(obs) != len(sim):
        raise AssertionError("Arrays are not of equal length!")
    # calculate alpha term
    obs_mean = np.mean(obs)
    sim_mean = np.mean(sim)
    kge_beta = sim_mean/obs_mean

    # calculate KGE with gamma term
    if var == 'cv':
        kge_gamma = calc_kge_gamma(obs, sim)
        temp_cor = calc_temp_cor(obs, sim, r=r)

        eff = 1 - np.sqrt((kge_beta - 1)**2 + (kge_gamma - 1)**2 +
                          (temp_cor - 1)**2)

    # calculate KGE with beta term
    elif var == 'std':
        kge_alpha = calc_kge_alpha(obs, sim)
        temp_cor = calc_temp_cor(obs, sim, r=r)

        eff = 1 - np.sqrt((kge_beta - 1)**2 + (kge_alpha - 1)**2 +
                          (temp_cor - 1)**2)

    return eff

def calc_kge_skill(obs, sim, bench, r='pearson', var='std'):
    r"""
    Calculate the Kling-Gupta-Efficiency skill score ($KGE_{skill}$).

    Parameters
    ----------
    obs : (N,)array_like
        Observed time series as 1-D array

    sim : (N,)array_like
        Simulated time series as 1-D array

    bench : (N,)array_like
        Benchmarked time series as 1-D array

    r : str, optional
        Either Spearman correlation coefficient ('spearman'; Pool et al. 2018)
        or Pearson correlation coefficient ('pearson'; Gupta et al. 2009) can
        be used to describe the temporal correlation. The default calculates
        the Pearson correlation.

    var : str, optional
        Either coefficient of variation ('cv'; Kling et al. 2012) or standard
        deviation ('std'; Gupta et al. 2009, Pool et al. 2018) to describe the
        gamma term. The default calculates the standard deviation.

    Returns
    ----------
    eff : float
        Kling-Gupta-Efficiency measure normalized by mean flow benchmark

    Examples
    --------
    Provide arrays with equal length

    >>> from de import de
    >>> import numpy as np
    >>> obs = np.array([1.5, 1, 0.8, 0.85, 1.5, 2])
    >>> sim = np.array([1.6, 1.3, 1, 0.8, 1.2, 2.5])
    >>> bench = np.array([1, 1.1, 1.15, 1.15, 1.1, 1])
    >>> kge.calc_kge_skill(obs, sim, bench)
    0.7790394993537985

    Notes
    ----------
    .. math::

        KGE_{skill} = \frac{KGE - KGE{bench}}{1 - KGE{bench}}

    """
    if len(obs) != len(sim):
        raise AssertionError("Arrays are not of equal length!")
    if len(obs) != len(bench):
        raise AssertionError("Arrays are not of equal length!")

    # calculate KGE with gamma term
    if var == 'cv':
        eff_kge = calc_kge(obs, sim, var='cv')
        eff_bench = calc_kge(obs, bench)
        eff = (eff_kge - (eff_bench))/(1 - (eff_bench))

    # calculate KGE with beta term
    elif var == 'std':
        eff_kge = calc_kge(obs, sim, var='std')
        eff_bench = calc_kge(obs, bench)
        eff = (eff_kge - (eff_bench))/(1 - (eff_bench))

    return eff

def diag_polar_plot(obs, sim, r='pearson', var='std'):
    r"""
    Diagnostic polar plot of Diagnostic efficiency (KGE) for a single
    evaluation.

    Parameters
    ----------
    obs : (N,)array_like
        Observed time series as 1-D array

    sim : (N,)array_like
        Simulated time series as 1-D array

    r : str, optional
        Either Spearman correlation coefficient ('spearman') or Pearson
        correlation coefficient ('pearson') can be used to describe the
        temporal correlation. The default calculates the Pearson correlation.

    var : str, optional
        Either coefficient of variation ('cv') or standard deviation ('std')
        to describe the gamma term. The default calculates the standard
        deviation.

    Returns
    ----------
    fig : Figure
        diagnostic polar plot

    Examples
    --------
    Provide arrays with equal length

    >>> from de import de
    >>> import numpy as np
    >>> obs = np.array([1.5, 1, 0.8, 0.85, 1.5, 2])
    >>> sim = np.array([1.6, 1.3, 1, 0.8, 1.2, 2.5])
    >>> kge.diag_polar_plot_kge(obs, sim)
    """
    # calculate beta term
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
        kge_r = calc_temp_cor(obs, sim, r=r)

        eff = 1 - np.sqrt((kge_alpha - 1)**2 + (kge_gamma - 1)**2 +
                          (kge_r - 1)**2)
        eff = np.round(eff, decimals=2)

        # convert to radians
        # (y, x) Trigonometric inverse tangent
        phi = np.arctan2(kge_alpha - 1, kge_gamma - 1)

        # convert temporal correlation to color
        norm = matplotlib.colors.Normalize(vmin=0, vmax=1.0)
        rgba_color = cm.plasma_r(norm(kge_r))

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
            raise AssertionError("Value of 'KGE'  too low for visualization!",
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
        # contours of KGE
        cp = ax.contour(theta, r, r, colors='darkgray', levels=c_levels,
                        zorder=1)
        cl = ax.clabel(cp, inline=1, fontsize=10, fmt='%1.1f',
                       colors='dimgrey')
        # diagnose the error
        if eff < .9:
            ax.annotate("", xytext=(0, 1), xy=(phi, eff),
                        arrowprops=dict(facecolor=rgba_color))
        elif eff >= .9:
            ax.scatter(phi, eff, color=rgba_color)
        ax.set_rticks([])  # turn defalut ticks off
        ax.set_rmin(1)
        if eff > 0:
            ax.set_rmax(ax_lim)
        elif eff <= 0:
            ax.set_rmax(-ax_lim)
        # turn labels and grid off
        ax.tick_params(labelleft=False, labelright=False, labeltop=False,
                       labelbottom=True, grid_alpha=.01)
        ax.text(-.09, 0.5, 'Variability underestimation', va='center',
                ha='center', rotation=90, rotation_mode='anchor',
                transform=ax.transAxes)
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
        # add colorbar for temporal correlation
        cbar = fig.colorbar(dummie_cax, ax=ax, orientation='horizontal',
                            ticks=[1, 0.5, 0], shrink=0.8)
        cbar.set_label('r [-]', labelpad=4)
        cbar.set_ticklabels(['1', '0.5', '<0'])
        cbar.ax.tick_params(direction='in')

    # calculate alpha term
    elif var == 'std':
        obs_std = np.std(obs)
        sim_std = np.std(sim)
        kge_beta = sim_std/obs_std
        kge_r = calc_temp_cor(obs, sim, r=r)

        eff = 1 - np.sqrt((kge_alpha - 1)**2 + (kge_beta - 1)**2 +
                          (kge_r - 1)**2)
        eff = np.round(eff, decimals=2)

        # convert to radians
        # (y, x) Trigonometric inverse tangent
        phi = np.arctan2(kge_alpha - 1, kge_beta - 1)

        # convert temporal correlation to color
        norm = matplotlib.colors.Normalize(vmin=0, vmax=1.0)
        rgba_color = cm.plasma_r(norm(kge_r))

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
            raise AssertionError("Value of 'KGE' is too low for visualization!",
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
        # contours of KGE
        cp = ax.contour(theta, r, r, colors='darkgray', levels=c_levels,
                        zorder=1)
        cl = ax.clabel(cp, inline=1, fontsize=10, fmt='%1.1f',
                       colors='dimgrey')
        # diagnose the error
        if eff < .9:
            ax.annotate("", xytext=(0, 1), xy=(phi, eff),
                        arrowprops=dict(facecolor=rgba_color))
        elif eff >= .9:
            ax.scatter(phi, eff, color=rgba_color)
        ax.set_rticks([])  # turn defalut ticks off
        ax.set_rmin(1)
        if eff > 0:
            ax.set_rmax(ax_lim)
        elif eff <= 0:
            ax.set_rmax(-ax_lim)
        # turn labels and grid off
        ax.tick_params(labelleft=False, labelright=False, labeltop=False,
                       labelbottom=True, grid_alpha=.01)
        ax.text(-.09, 0.5, 'Variability underestimation', va='center',
                ha='center', rotation=90, rotation_mode='anchor',
                transform=ax.transAxes)
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
        # add colorbar for temporal correlation
        cbar = fig.colorbar(dummie_cax, ax=ax, orientation='horizontal',
                            ticks=[1, 0.5, 0], shrink=0.8)
        cbar.set_label('r [-]', labelpad=4)
        cbar.set_ticklabels(['1', '0.5', '<0'])
        cbar.ax.tick_params(direction='in')

        return fig

def diag_polar_plot_multi(kge_beta, alpha_or_gamma, kge_r, eff_kge,
                          var='std', extended=False):
    r"""
    Diagnostic polar plot for Kling-Gupta efficiency (KGE) for multiple
    evaluations.

    Parameters
    ----------
    kge_beta: (N,)array_like
        KGE beta as 1-D array

    alpha_or_gamma : (N,)array_like
        KGE alpha or KGE gamma as 1-D array

    kge_r : (N,)array_like
        KGE r as 1-D array

    eff_kge : (N,)array_like
        KGE as 1-D array

    var : str, optional
        Either coefficient of variation ('cv') or standard deviation ('std')
        to describe the gamma term or alpha term, respectively. The default
        is 'std'.

    extended : boolean, optional
        If True, density plot is displayed. In addtion, the density plot
        is displayed besides the polar plot. The default is,
        that only the diagnostic polar plot is displayed.

    Returns
    ----------
    fig : Figure
        Returns a single figure if extended=False and two figures if
        extended=True.

    Examples
    --------
    Provide arrays with equal length

    >>> from de import de
    >>> import numpy as np
    >>> beta = np.array([1.1, 1.15, 1.2, 1.1, 1.05, 1.15])
    >>> alpha = np.array([1.15, 1.1, 1.2, 1.1, 1.1, 1.2])
    >>> r = np.array([0.9, 0.85, 0.8, 0.9, 0.85, 0.9])
    >>> eff_kge = np.array([0.79, 0.77, 0.65, 0.83, 0.81, 0.73])
    >>> kge.diag_polar_plot_multi(beta, alpha, r, eff_kge)
    """
    eff_min = np.min(eff_kge)

    ll_kge_beta = kge_beta.tolist()
    ll_ag = alpha_or_gamma.tolist()
    ll_kge_r = kge_r.tolist()
    ll_eff = eff_kge.tolist()

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
        raise ValueError("Some values of 'KGE' are too low for visualization!",
                         eff_min)

    if var == 'std':
        var_lab = 'alpha'
    if var == 'cv':
        var_lab = 'gamma'
    else:
        var_lab = 'alpha'

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
        # contours of KGE
        cp = ax.contour(theta, r, r, colors='darkgray', levels=c_levels,
                        zorder=1)
        cl = ax.clabel(cp, inline=True, fontsize=10, fmt='%1.1f',
                       colors='dimgrey')
        # loop over each data point
        for (b, ag, r, eff) in zip(ll_kge_beta, ll_ag, ll_kge_r, ll_eff):
            ang = np.arctan2(b - 1, ag - 1)
            # convert temporal correlation to color
            rgba_color = cm.plasma_r(norm(r))
            c = ax.scatter(ang, eff, color=rgba_color)
        # turn labels and grid off
        ax.tick_params(labelleft=False, labelright=False, labeltop=False,
                       labelbottom=True, grid_alpha=.01)
        ax.text(-.09, 0.5, 'Variability underestimation', va='center',
                ha='center', rotation=90, rotation_mode='anchor',
                transform=ax.transAxes)
        ax.text(-.04, 0.5, r'($\%s$ - 1) < 0' % (var_lab), va='center',
                ha='center', rotation=90, rotation_mode='anchor',
                transform=ax.transAxes)
        ax.text(1.09, 0.5, 'Variability overestimation',
                va='center', ha='center', rotation=90, rotation_mode='anchor',
                transform=ax.transAxes)
        ax.text(1.04, 0.5, r'($\%s$ - 1) > 0' % (var_lab),
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
        if eff_min > 0:
            ax.set_rmax(ax_lim)
        elif eff_min <= 0:
            ax.set_rmax(-ax_lim)
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
        ax1 = fig.add_axes([.64, .3, .33, .33], frameon=True)
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
        # contours of KGE
        cp = ax.contour(theta, r, r, colors='darkgray', levels=c_levels,
                        zorder=1)
        cl = ax.clabel(cp, inline=True, fontsize=10, fmt='%1.1f',
                       colors='dimgrey')
        # loop over each data point
        for (b, ag, r, eff) in zip(ll_kge_beta, ll_ag, ll_kge_r, ll_eff):
            ang = np.arctan2(b - 1, ag - 1)
            # convert temporal correlation to color
            rgba_color = cm.plasma_r(norm(r))
            c = ax.scatter(ang, eff, color=rgba_color)
        # turn labels and grid off
        ax.tick_params(labelleft=False, labelright=False, labeltop=False,
                       labelbottom=True, grid_alpha=.01)
        ax.text(-.09, 0.5, 'Variability underestimation', va='center',
                ha='center', rotation=90, rotation_mode='anchor',
                transform=ax.transAxes)
        ax.text(-.04, 0.5, r'($\%s$ - 1) < 0' % (var_lab), va='center',
                ha='center', rotation=90, rotation_mode='anchor',
                transform=ax.transAxes)
        ax.text(1.09, 0.5, 'Variability overestimation',
                va='center', ha='center', rotation=90,
                rotation_mode='anchor', transform=ax.transAxes)
        ax.text(1.04, 0.5, r'($\%s$ - 1) > 0' % (var_lab),
                va='center', ha='center', rotation=90,
                rotation_mode='anchor', transform=ax.transAxes)
        ax.text(.5, -.09, 'Mean underestimation', va='center', ha='center',
                rotation=0, rotation_mode='anchor', transform=ax.transAxes)
        ax.text(.5, -.04, r'($\beta$ - 1) < 0', va='center', ha='center',
                rotation=0, rotation_mode='anchor', transform=ax.transAxes)
        ax.text(.5, 1.09, 'Mean overestimation',
                va='center', ha='center', rotation=0,
                rotation_mode='anchor', transform=ax.transAxes)
        ax.text(.5, 1.04, r'($\beta$ - 1) > 0',
                va='center', ha='center', rotation=0,
                rotation_mode='anchor', transform=ax.transAxes)
        ax.set_xticklabels(['', '', '', '', '', r'0$^{\circ}$ (360$^{\circ}$)', '', ''])
        ax.set_rticks([])  # turn default ticks off
        ax.set_rmin(1)
        if eff_min > 0:
            ax.set_rmax(ax_lim)
        elif eff_min <= 0:
            ax.set_rmax(-ax_lim)
        # add colorbar for temporal correlation
        cbar = fig.colorbar(dummie_cax, ax=ax, orientation='horizontal',
                            ticks=[1, 0.5, 0], shrink=0.8)
        cbar.set_label('r [-]', labelpad=4)
        cbar.set_ticklabels(['1', '0.5', '<0'])
        cbar.ax.tick_params(direction='in')

        # convert to degrees
        phi = np.arctan2(kge_beta - 1, alpha_or_gamma - 1)
        diag_deg = (phi * (180 / np.pi)) + 135
        diag_deg[diag_deg < 0] = 360 - diag_deg[diag_deg < 0]

        # 1-D density plot
        g = sns.kdeplot(diag_deg, color='k', ax=ax1)
        ax1.set_xticks([0, 90, 180, 270, 360])
        ax1.set_xlim(0, 360)
        ax1.set_ylim(0, )
        ax1.set(ylabel='Density', xlabel=r'[$^\circ$]')

        # 2-D density plot
        r_colors = cm.plasma_r(norm(kge_r))
        g = (sns.jointplot(diag_deg, eff_kge, kind='kde', zorder=1,
                           n_levels=20, cmap='Greys', shade_lowest=False,
                           marginal_kws={'color': 'k', 'shade': False})
                           .plot_joint(plt.scatter, c=r_colors, alpha=.4,
                                       zorder=2))
        g.set_axis_labels(r'[$^\circ$]', r'KGE [-]')
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
