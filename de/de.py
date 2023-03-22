# -*- coding: utf-8 -*-

"""
de.de
~~~~~~~~~~~
Diagnosing model errors using an efficiency measure based on flow
duration curve and temporal correlation. The efficiency measure can be
visualized by diagnostic polar plots which facilitates decomposing potential
error contributions (dynamic errors vs. constant erros vs. timing errors)
:2021 by Robin Schwemmle.
:license: GNU GPLv3, see LICENSE for more details.
"""

import numpy as np
import matplotlib
from matplotlib import cm
import matplotlib.ticker as mticker
import pandas as pd
import scipy as sp
import scipy.integrate as integrate
import seaborn as sns
matplotlib.use("agg")
import matplotlib.pyplot as plt  # noqa: E402

# RunTimeWarning will not be displayed (division by zeros or NaN values)
np.seterr(divide="ignore", invalid="ignore")

# controlling figure aesthetics
sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})


def calc_brel(obs, sim, sort=True):
    r"""
    Calculate relative bias.

    Parameters
    ----------
    obs : (N,)array_like
        observed time series as 1-D array

    sim : (N,)array_like
        simulated time series as 1-D array

    sort : boolean, optional
        If True, time series are sorted by ascending order. If False, time
        series are not sorted. The default is to sort.

    Returns
    ----------
    brel : (N,)array_like
        relative bias

    Notes
    ----------
    .. math::

        B_{rel} = \frac{Q_{sim}(i) - Q_{obs}(i)}{Q_{obs}(i)}


    Examples
    --------
    Provide arrays with equal length

    >>> from de import de
    >>> import numpy as np
    >>> obs = np.array([1.5, 1, 0.8, 0.85, 1.5, 2])
    >>> sim = np.array([1.6, 1.3, 1, 0.8, 1.2, 2.5])
    >>> brel = de.calc_brel(obs, sim)
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

    return brel


def calc_brel_mean(obs, sim, sort=True):
    r"""
    Calculate the arithmetic mean of the relative bias.

    Parameters
    ----------
    obs : (N,)array_like
        observed time series as 1-D array

    sim : (N,)array_like
        simulated time series as 1-D array

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

    # set numerical artefacts to zero
    if abs(brel_mean) < 0.001:
        brel_mean = 0

    return brel_mean


def calc_brel_res(obs, sim, sort=True):
    r"""
    Subtracting arithmetic mean of the relative bias from the relative bias.

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
    brel_res : (N,)array_like
        remaining relative bias

    Notes
    ----------
    .. math::

        B_{res}(i) = B_{rel}(i) - \overline{B_{rel}}

    Examples
    --------
    Provide arrays with equal length

    >>> from de import de
    >>> import numpy as np
    >>> obs = np.array([1.5, 1, 0.8, 0.85, 1.5, 2])
    >>> sim = np.array([1.6, 1.3, 1, 0.8, 1.2, 2.5])
    >>> de.calc_brel_res(obs, sim)
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
    brel_res = brel - brel_mean

    return brel_res


def calc_bias_area(brel_res):
    r"""
    Calculate the integrated residual bias for the entire flow duration curve.

    Parameters
    ----------
    brel_res : (N,)array_like
        remaining relative bias as 1-D array

    Returns
    ----------
    b_area : float
        bias area

    Notes
    ----------
    .. math::

        \vert B_{area}\vert = \int_{0}^{1}\vert B_{res}(i)\vert di

    Examples
    --------
    Provide arrays with equal length

    >>> from de import de
    >>> import numpy as np
    >>> obs = np.array([1.5, 1, 0.8, 0.85, 1.5, 2])
    >>> sim = np.array([1.6, 1.3, 1, 0.8, 1.2, 2.5])
    >>> b_res = de.calc_brel_res(obs, sim)
    >>> de.calc_bias_area(b_res)
    0.1112908496732026

    See Also
    --------
    de.calc_brel_res
    """
    perc = np.linspace(0, 1, len(brel_res))
    # area of absolute bias
    b_area = integrate.simps(abs(brel_res), perc)

    # set numerical artefacts to zero
    if abs(b_area) < 0.001:
        b_area = 0

    return b_area


def calc_bias_tot(brel):
    r"""
    Calculate the integrated relative bias for the entire flow duration curve.

    Parameters
    ----------
    brel : (N,)array_like
        relative bias as 1-D array

    Returns
    ----------
    b_tot : float
        bias area of the entire flow domain

    Notes
    ----------
    .. math::

        \vert B_{area}\vert = \int_{0}^{1}\vert B_{rel}(i)\vert di

    Examples
    --------
    Provide arrays with equal length

    >>> from de import de
    >>> import numpy as np
    >>> obs = np.array([1.5, 1, 0.8, 0.85, 1.5, 2])
    >>> sim = np.array([1.6, 1.3, 1, 0.8, 1.2, 2.5])
    >>> b_rel = de.calc_brel(obs, sim)
    >>> de.calc_bias_tot(b_rel)
    0.14017973856209148

    See Also
    --------
    de.calc_brel
    """
    perc = np.linspace(0, 1, len(brel))
    # area of absolute bias
    brel_abs = np.abs(brel)
    b_tot = integrate.simps(brel_abs, perc)

    # set numerical artefacts to zero
    if abs(b_tot) < 0.001:
        b_tot = 0

    return b_tot


def calc_bias_hf(brel):
    r"""
    Calculate the integrated relative bias for high flows.

    Parameters
    ----------
    brel : (N,)array_like
        relative bias as 1-D array

    Returns
    ----------
    b_hf : float
        absolute bias area of high flows (i.e. 0th percentile to 50th percentile)

    Notes
    ----------
    .. math::

        B_{hf} = \int_{0}^{0.5}B_{rel}(i) di

    See Also
    --------
    de.calc_brel

    Examples
    --------
    Provide arrays with equal length

    >>> from de import de
    >>> import numpy as np
    >>> obs = np.array([1.5, 1, 0.8, 0.85, 1.5, 2])
    >>> sim = np.array([1.6, 1.3, 1, 0.8, 1.2, 2.5])
    >>> b_rel = de.calc_brel(obs, sim)
    >>> de.calc_bias_hf(b_rel)
    0.031944444444444456
    """
    mid_idx = int(len(brel) / 2)
    # integral of relative bias < 50 %
    n = len(brel[:mid_idx])
    perc_hf = np.linspace(0, 0.5, n)
    # direction of bias from high flows
    b_hf = integrate.simps(brel[:mid_idx], perc_hf)

    # set numerical artefacts to zero
    if abs(b_hf) < 0.001:
        b_hf = 0

    return b_hf


def calc_err_hf(b_hf, b_tot):
    r"""
    Calculate the error contribution of high flows.

    Parameters
    ----------
    b_hf : float
        absolute bias area of high flows (i.e. 0th percentile to 50th
        percentile)

    b_tot : float
        bias area of the entire flow domain

    Returns
    ----------
    err_hf : float
        contribution of high flows to model error

    Notes
    ----------
    .. math::

        \epsilon_{hf} = \frac{B_{hf}}{B_{tot}}

    See Also
    --------
    de.calc_bias_hf and de.calc_bias_tot

    Examples
    --------
    Provide arrays with equal length

    >>> from de import de
    >>> import numpy as np
    >>> obs = np.array([1.5, 1, 0.8, 0.85, 1.5, 2])
    >>> sim = np.array([1.6, 1.3, 1, 0.8, 1.2, 2.5])
    >>> b_rel = de.calc_brel(obs, sim)
    >>> b_hf = de.calc_bias_hf(b_rel)
    >>> b_tot = de.calc_bias_tot(b_rel)
    >>> de.calc_err_hf(b_hf, b_tot)
    0.2278820375335122
    """
    if b_tot > 0:
        err_hf = b_hf / b_tot
    else:
        err_hf = 0

    # set nan to zero
    if err_hf == np.nan:
        err_hf = 0

    return err_hf


def calc_bias_lf(brel):
    r"""
    Calculate the integrated relative for low flows.

    Parameters
    ----------
    brel : (N,)array_like
        relative bias as 1-D array

    Returns
    ----------
    b_lf : float
        absolute bias area of low flows (i.e. 50th percentile to 100th percentile)

    Notes
    ----------
    .. math::

        B_{lf} = \int_{0.5}^{1}B_{rel}(i) di

    Examples
    --------
    Provide arrays with equal length

    >>> from de import de
    >>> import numpy as np
    >>> obs = np.array([1.5, 1, 0.8, 0.85, 1.5, 2])
    >>> sim = np.array([1.6, 1.3, 1, 0.8, 1.2, 2.5])
    >>> b_rel = de.calc_brel(obs, sim)
    >>> de.calc_bias_lf(b_rel)
    0.07549019607843138
    """
    mid_idx = int(len(brel) / 2)
    # integral of relative bias < 50 %
    n = len(brel[mid_idx:])
    perc_low = np.linspace(0.5, 1, n)
    # direction of bias from high flows
    b_lf = integrate.simps(brel[mid_idx:], perc_low)

    # set numerical artefacts to zero
    if abs(b_lf) < 0.001:
        b_lf = 0

    return b_lf


def calc_err_lf(b_lf, b_tot):
    r"""
    Calculate the error contribution of low flows.

    Parameters
    ----------
    b_lf : float
        absolute bias area of low flows (i.e. 50th percentile to 100th percentile)

    b_tot : float
        bias area of the entire flow domain

    Returns
    ----------
    err_lf : float
        contribution of low flows to dynamic error

    Notes
    ----------
    .. math::

        \epsilon_{lf} = \frac{B_{lf}}{B_{tot}}

    See Also
    --------
    de.calc_bias_hf and de.calc_bias_tot

    Examples
    --------
    Provide arrays with equal length

    >>> from de import de
    >>> import numpy as np
    >>> obs = np.array([1.5, 1, 0.8, 0.85, 1.5, 2])
    >>> sim = np.array([1.6, 1.3, 1, 0.8, 1.2, 2.5])
    >>> b_rel = de.calc_brel(obs, sim)
    >>> b_lf = de.calc_bias_lf(b_rel)
    >>> b_tot = de.calc_bias_tot(b_rel)
    >>> de.calc_err_lf(b_lf, b_tot)
    0.5385243035318803
    """
    if b_tot > 0:
        err_lf = b_lf / b_tot
    else:
        err_lf = 0

    # set nan to zero
    if err_lf == np.nan:
        err_lf = 0

    return err_lf

def calc_bias_dir(brel_res):
    r"""
    Calculate the direction of the dynamic error.

    Parameters
    ----------
    brel_res : (N,)array_like
        remaining relative bias

    Returns
    ----------
    b_dir : float
        direction of bias

    Notes
    ----------
    .. math::

        B_{dir} =
        \begin{cases}
        -1 & \text{if } (B_{res-hf} > 0 & B_{lf} < 0) | (B_{res-hf} = 0 & B_{res-lf} < 0) | (B_{res-hf} > 0 & B_{res-lf} = 0) \\
        1 & \text{if } (B_{res-hf} < 0 & B_{lf} > 0) | (B_{res-hf} = 0 & B_{res-lf} > 0) | (B_{res-hf} < 0 & B_{res-lf} = 0) \\
        0  & \text{if } (B_{res-hf} > 0 & B_{lf} > 0) | (B_{res-hf} < 0 & B_{res-lf} < 0) | (B_{res-hf} = 0 & B_{res-lf} = 0)
        \end{cases}

    Examples
    --------
    Provide arrays with equal length

    >>> from de import de
    >>> import numpy as np
    >>> obs = np.array([1.5, 1, 0.8, 0.85, 1.5, 2])
    >>> sim = np.array([1.6, 1.3, 1, 0.8, 1.2, 2.5])
    >>> brel_res = de.calc_brel_res(obs, sim)
    >>> de.calc_bias_dir(brel_res)
    1
    """
    b_res_hf = calc_bias_hf(brel_res)
    b_res_lf = calc_bias_lf(brel_res)
    if (b_res_hf > 0 and b_res_lf < 0) or (b_res_hf == 0 and b_res_lf < 0) or (b_res_hf > 0 and b_res_lf == 0):
        b_dir = -1

    elif (b_res_hf < 0 and b_res_lf > 0) or (b_res_hf == 0 and b_res_lf > 0) or (b_res_hf < 0 and b_res_lf == 0):
        b_dir = 1

    elif (b_res_hf > 0 and b_res_lf > 0) or (b_res_hf < 0 and b_res_lf < 0) or (b_res_hf == 0 and b_res_lf == 0):
        b_dir = 0

    return b_dir


def calc_bias_slope(b_area, b_dir):
    r"""
    Calculate the slope of the residual bias.

    Parameters
    ----------
    b_area : float
        absolute area of residual bias

    b_dir : float
        direction of bias

    Returns
    ----------
    b_slope : float
        slope of bias

    Notes
    ----------
    .. math::

        B_{slope} = \vert B_{area}\vert \times B_{dir}

    See Also
    --------
    de.calc_bias_area and de.calc_bias_dir

    Examples
    --------
    Provide arrays with equal length

    >>> from de import de
    >>> import numpy as np
    >>> obs = np.array([1.5, 1, 0.8, 0.85, 1.5, 2])
    >>> sim = np.array([1.6, 1.3, 1, 0.8, 1.2, 2.5])
    >>> brel_res = de.calc_brel(obs, sim)
    >>> b_dir = de.calc_bias_dir(brel_res)
    >>> b_area = de.calc_bias_area(b_res)
    >>> de.calc_bias_slope(b_area, b_dir)
    0.11
    """
    b_slope = b_area * b_dir

    return b_slope


def calc_temp_cor(obs, sim, r="pearson"):
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
    0.89
    """
    if len(obs) != len(sim):
        raise AssertionError("Arrays are not of equal length!")

    if r == "spearman":
        r = sp.stats.spearmanr(obs, sim)
        temp_cor = r[0]

        if np.isnan(temp_cor):
            temp_cor = 0

    elif r == "pearson":
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
        Diagnostic efficiency

    Notes
    ----------
    .. math::

        DE = \sqrt{\overline{B_{rel}}^2 + \vert B_{area}\vert^2 + (r - 1)^2}
    Examples
    --------
    Provide arrays with equal length

    >>> from de import de
    >>> import numpy as np
    >>> obs = np.array([1.5, 1, 0.8, 0.85, 1.5, 2])
    >>> sim = np.array([1.6, 1.3, 1, 0.8, 1.2, 2.5])
    >>> de.calc_de(obs, sim)
    0.18
    """
    if len(obs) != len(sim):
        raise AssertionError("Arrays are not of equal length!")
    # mean relative bias
    brel_mean = calc_brel_mean(obs, sim, sort=sort)
    # remaining relative bias
    brel_res = calc_brel_res(obs, sim, sort=sort)
    # area of relative remaing bias
    b_area = calc_bias_area(brel_res)
    # temporal correlation
    temp_cor = calc_temp_cor(obs, sim)
    # diagnostic efficiency
    eff = np.sqrt((brel_mean) ** 2 + (b_area) ** 2 + (temp_cor - 1) ** 2)

    return eff


def calc_phi(brel_mean, b_slope):
    """
    Calculate trigonometric inverse tangent.

    Parameters
    ----------
    brel_mean : float
        average relative bias

    b_slope : float
        slope of bias

    Returns
    ----------
    phi : float
        trigonometric inverse tangent

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
    >>> brel_mean = de.calc_brel_mean(obs, sim)
    >>> brel_res = de.calc_brel(obs, sim)
    >>> b_dir = de.calc_bias_dir(brel_res)
    >>> b_area = de.calc_bias_area(brel_res)
    >>> b_slope = de.calc_bias_slope(b_area, b_dir)
    >>> de.calc_phi(brel_mean, b_slope)
    1.5707963267948966
    """
    phi = np.arctan2(brel_mean, b_slope)
    # set numerical artefacts to zero
    if abs(phi) < 0.001:
        phi = 0

    # set numerical artefacts pi
    if phi > 3.1414:
        phi = 3.1414

    return phi


def diag_polar_plot(obs, sim, sort=True, limit=0.05, extended=False):
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

    limit : float, optional
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

    # relative bias
    brel = calc_brel(obs, sim, sort=sort)

    # mean relative bias
    brel_mean = calc_brel_mean(obs, sim, sort=sort)

    # residual relative bias
    brel_res = calc_brel_res(obs, sim, sort=sort)
    # area of relative remaing bias
    b_area = calc_bias_area(brel_res)
    # temporal correlation
    temp_cor = calc_temp_cor(obs, sim)
    # diagnostic efficiency
    eff = np.sqrt((brel_mean) ** 2 + (b_area) ** 2 + (temp_cor - 1) ** 2)

    # direction of bias
    b_dir = calc_bias_dir(brel_res)

    # slope of bias
    b_slope = calc_bias_slope(b_area, b_dir)

    # total bias
    b_tot = calc_bias_tot(brel)
    # bias of high flows
    b_hf = calc_bias_hf(brel)
    # bias of low flows
    b_lf = calc_bias_lf(brel)
    # contribution of high flows to dyn. error
    err_hf = calc_err_hf(b_hf, b_tot)
    # contribution of low flows to dyn. error
    err_lf = calc_err_lf(b_lf, b_tot)

    # convert to radians
    # (y, x) Trigonometric inverse tangent
    phi = calc_phi(brel_mean, b_slope)

    # convert temporal correlation to color
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1.0)
    cmap = cm.get_cmap('plasma_r')
    rgba_color = cmap(norm(temp_cor))

    delta = 0.01  # for spacing

    # determine axis limits
    if eff <= 1:
        ax_lim = 1.2
        yy = np.arange(0.01, ax_lim, delta)
        c_levels = np.arange(0, ax_lim, 0.2)
    elif eff > 1 and eff <= 2:
        ax_lim = 2.2
        yy = np.arange(0.01, ax_lim, delta)
        c_levels = np.arange(0, ax_lim, 0.2)
    elif eff > 2 and eff <= 3:
        ax_lim = 3.2
        yy = np.arange(0.01, ax_lim, delta)
        c_levels = np.arange(0, ax_lim, 0.2)
    elif eff >= 3:
        raise AssertionError("Value of 'DE' is out of bounds for visualization!", eff)

    len_yy = len(yy)

    # arrays to plot contour lines of DE
    xx = np.radians(np.linspace(0, 360, len_yy))
    theta, r = np.meshgrid(xx, yy)

    # diagnostic polar plot
    if not extended:
        fig, ax = plt.subplots(
            figsize=(3, 3), subplot_kw=dict(projection="polar"), constrained_layout=True
        )
        # dummie plot for colorbar of temporal correlation
        cs = np.arange(0, 1.1, 0.1)
        dummie_cax = ax.scatter(cs, cs, c=cs, cmap="plasma_r")
        # Clear axis
        ax.cla()
        # plot regions
        ax.plot(
            (0, np.deg2rad(45)),
            (0, np.max(yy)),
            color="lightgray",
            linewidth=1,
            ls="--",
            zorder=0,
        )
        ax.plot(
            (0, np.deg2rad(135)),
            (0, np.max(yy)),
            color="lightgray",
            linewidth=1,
            ls="--",
            zorder=0,
        )
        ax.plot(
            (0, np.deg2rad(225)),
            (0, np.max(yy)),
            color="lightgray",
            linewidth=1,
            ls="--",
            zorder=0,
        )
        ax.plot(
            (0, np.deg2rad(315)),
            (0, np.max(yy)),
            color="lightgray",
            linewidth=1,
            ls="--",
            zorder=0,
        )
        # contours of DE
        cp = ax.contour(theta, r, r, colors="darkgray", levels=c_levels, zorder=1, linewidths=1)
        cl = ax.clabel(cp, inline=True, fmt="%1.1f", colors="dimgrey")
        # threshold efficiency for FBM
        eff_l = np.sqrt((limit) ** 2 + (limit) ** 2 + (limit) ** 2)
        # relation of high flow errors and low flow errors which explain
        # the total FDC error
        if b_tot > 0:
            exp_err = (abs(b_hf) + abs(b_lf)) / b_tot
        elif b_tot == 0:
            exp_err = 0

        # calculate pies to display error contribution of high flows and low
        # flows
        # calculate the points of the first pie marker
        # these are just the origin (0, 0) + some (cos, sin) points on a circle
        r1 = abs(err_hf)/2
        x1 = np.cos(2 * np.pi * np.linspace(0.25 - r1/2, 0.25 + r1/2))
        y1 = np.sin(2 * np.pi * np.linspace(0.25 - r1/2, 0.25 + r1/2))
        xy1 = np.row_stack([[0, 0], np.column_stack([x1, y1])])
        s1 = np.abs(xy1).max()

        r2 = abs(err_lf)/2
        x2 = np.cos(2 * np.pi * np.linspace(0.75 - r2/2, 0.75 + r2/2))
        y2 = np.sin(2 * np.pi * np.linspace(0.75 - r2/2, 0.75 + r2/2))
        xy2 = np.row_stack([[0, 0], np.column_stack([x2, y2])])
        s2 = np.abs(xy2).max()

        # diagnose the error
        if abs(brel_mean) <= limit and exp_err > limit and eff > eff_l:
            c0 = ax.scatter(phi, eff, marker=xy1, s=s1 * 50, facecolor=rgba_color, zorder=2)
            c1 = ax.scatter(phi, eff, marker=xy2, s=s2 * 50, facecolor=rgba_color, zorder=2)
            c = ax.scatter(phi, eff, s=4, facecolor=rgba_color, zorder=2)
            c2 = ax.scatter(phi, eff, s=1, facecolor='grey', zorder=2)
        elif abs(brel_mean) > limit and exp_err <= limit and eff > eff_l:
            c0 = ax.scatter(phi, eff, marker=xy1, s=s1 * 50, facecolor=rgba_color, zorder=2)
            c1 = ax.scatter(phi, eff, marker=xy2, s=s2 * 50, facecolor=rgba_color, zorder=2)
            c = ax.scatter(phi, eff, s=4, facecolor=rgba_color, zorder=2)
            c2 = ax.scatter(phi, eff, s=1, facecolor='grey', zorder=2)
        elif abs(brel_mean) > limit and exp_err > limit and eff > eff_l:
            c0 = ax.scatter(phi, eff, marker=xy1, s=s1 * 50, facecolor=rgba_color, zorder=2)
            c1 = ax.scatter(phi, eff, marker=xy2, s=s2 * 50, facecolor=rgba_color, zorder=2)
            c = ax.scatter(phi, eff, s=4, facecolor=rgba_color, zorder=2)
            c2 = ax.scatter(phi, eff, s=1, facecolor='grey', zorder=2)
        # FBM
        elif abs(brel_mean) <= limit and exp_err <= limit and eff > eff_l:
            ax.annotate(
                "", xytext=(0, 0), xy=(0, eff), arrowprops=dict(facecolor=rgba_color),
                zorder=2
            )
            ax.annotate(
                "",
                xytext=(0, 0),
                xy=(np.pi, eff),
                arrowprops=dict(facecolor=rgba_color),
                zorder=2
            )
        # FGM
        elif abs(brel_mean) <= limit and exp_err <= limit and eff <= eff_l:
            c = ax.scatter(phi, eff, s=4, facecolor=rgba_color, zorder=2)
            c2 = ax.scatter(phi, eff, s=1, facecolor='grey', zorder=2)

        # legend for error contribution of high flows and low flows
        rl1 = 1/2
        xl1 = np.cos(2 * np.pi * np.linspace(0.25 - rl1/2, 0.25 + rl1/2))
        yl1 = np.sin(2 * np.pi * np.linspace(0.25 - rl1/2, 0.25 + rl1/2))
        xyl1 = np.row_stack([[0, 0], np.column_stack([xl1, yl1])])
        sl1 = np.abs(xyl1).max()

        rl2 = 1/2
        xl2 = np.cos(2 * np.pi * np.linspace(0.75 - rl2/2, 0.75 + rl2/2))
        yl2 = np.sin(2 * np.pi * np.linspace(0.75 - rl2/2, 0.75 + rl2/2))
        xyl2 = np.row_stack([[0, 0], np.column_stack([xl2, yl2])])
        sl2 = np.abs(xyl2).max()
        ax.scatter([], [], color='k', zorder=2, marker=xyl1, s=sl1 * 50, label=r'high values ($\epsilon_{hf}=1$)')
        ax.scatter([], [], color='k', zorder=2, marker=xyl2, s=sl2 * 50, label=r'low values ($\epsilon_{lf}=1$)')
        ax.legend(loc='upper right', title="Error contribution of", fancybox=False,
                  frameon=False, bbox_to_anchor=(1.45, 1.2), handletextpad=0.2)

        ax.set_rticks([])  # turn default ticks off
        ax.set_rmin(0)
        if eff <= 1:
            ax.set_rmax(1)
        elif eff > 1:
            ax.set_rmax(ax_lim)
        # turn labels and grid off
        ax.tick_params(
            labelleft=False,
            labelright=False,
            labeltop=False,
            labelbottom=True,
            grid_alpha=0.01,
        )
        ticks_loc = ax.get_xticks().tolist()
        ax.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
        ax.set_xticklabels(["", "", "", "", "", "", "", ""])
        ax.text(
            -0.16,
            0.5,
            "High value overestimation -",
            va="center",
            ha="center",
            rotation=90,
            rotation_mode="anchor",
            transform=ax.transAxes,
        )
        ax.text(
            -0.11,
            0.5,
            "Low value underestimation",
            va="center",
            ha="center",
            rotation=90,
            rotation_mode="anchor",
            transform=ax.transAxes,
        )
        ax.text(
            -0.06,
            0.5,
            r"$B_{slope}$ < 0",
            va="center",
            ha="center",
            rotation=90,
            rotation_mode="anchor",
            transform=ax.transAxes,
        )
        ax.text(
            1.16,
            0.5,
            "High value underestimation -",
            va="center",
            ha="center",
            rotation=90,
            rotation_mode="anchor",
            transform=ax.transAxes,
        )
        ax.text(
            1.11,
            0.5,
            "Low value overestimation",
            va="center",
            ha="center",
            rotation=90,
            rotation_mode="anchor",
            transform=ax.transAxes,
        )
        ax.text(
            1.06,
            0.5,
            r"$B_{slope}$ > 0",
            va="center",
            ha="center",
            rotation=90,
            rotation_mode="anchor",
            transform=ax.transAxes,
        )
        ax.text(
            0.5,
            -0.12,
            "Constant negative offset",
            va="center",
            ha="center",
            rotation=0,
            rotation_mode="anchor",
            transform=ax.transAxes,
        )
        ax.text(
            0.5,
            -0.06,
            r"$\overline{B_{rel}}$ < 0",
            va="center",
            ha="center",
            rotation=0,
            rotation_mode="anchor",
            transform=ax.transAxes,
        )
        ax.text(
            0.5,
            1.12,
            "Constant positive offset",
            va="center",
            ha="center",
            rotation=0,
            rotation_mode="anchor",
            transform=ax.transAxes,
        )
        ax.text(
            0.5,
            1.06,
            r"$\overline{B_{rel}}$ > 0",
            va="center",
            ha="center",
            rotation=0,
            rotation_mode="anchor",
            transform=ax.transAxes,
        )
        # add colorbar for temporal correlation
        cbar = fig.colorbar(
            dummie_cax, ax=ax, orientation="horizontal", ticks=[1, 0.5, 0], shrink=0.8
        )
        cbar.set_label("r [-]\nTiming error", labelpad=4)
        cbar.set_ticklabels(["1", "0.5", "<0"])
        cbar.ax.tick_params(direction="in")

        if (obs == 0).any():
            N_obs0 = np.sum(obs == 0)
            N_sim0 = np.sum(sim == 0)
            share0 = N_obs0 / obs.size
            a0 = np.round(1 - (N_sim0 / N_obs0), 2)
            ax.text(
                1.4,
                0.1,
                f"Share of zero values [-]: {share0}\n Agreement of zero values [-]: {a0}",
                va="center",
                ha="center",
                rotation=0,
                rotation_mode="anchor",
                transform=ax.transAxes,
            )

        return fig

    elif extended:
        fig = plt.figure(figsize=(6, 3), constrained_layout=True)
        gs = fig.add_gridspec(1, 2)
        ax = fig.add_subplot(gs[0, 0], projection="polar")
        ax1 = fig.add_axes([0.65, 0.3, 0.33, 0.33], frameon=True)
        # dummie plot for colorbar of temporal correlation
        cs = np.arange(0, 1.1, 0.1)
        dummie_cax = ax.scatter(cs, cs, c=cs, cmap="plasma_r")
        # Clear axis
        ax.cla()
        # plot regions
        ax.plot(
            (0, np.deg2rad(45)),
            (0, np.max(yy)),
            color="lightgray",
            linewidth=1,
            ls="--",
            zorder=0,
        )
        ax.plot(
            (0, np.deg2rad(135)),
            (0, np.max(yy)),
            color="lightgray",
            linewidth=1,
            ls="--",
            zorder=0,
        )
        ax.plot(
            (0, np.deg2rad(225)),
            (0, np.max(yy)),
            color="lightgray",
            linewidth=1,
            ls="--",
            zorder=0,
        )
        ax.plot(
            (0, np.deg2rad(315)),
            (0, np.max(yy)),
            color="lightgray",
            linewidth=1,
            ls="--",
            zorder=0,
        )
        # contours of DE
        cp = ax.contour(theta, r, r, colors="darkgray", levels=c_levels, zorder=1, linewidths=1)
        cl = ax.clabel(cp, inline=True, fmt="%1.1f", colors="dimgrey")
        # threshold efficiency for FBM
        eff_l = np.sqrt((limit) ** 2 + (limit) ** 2 + (limit) ** 2)
        # relation of high flow errors and low flow errors which explain
        # the total FDC error
        if b_tot > 0:
            exp_err = (abs(b_hf) + abs(b_lf)) / b_tot
        elif b_tot == 0:
            exp_err = 0

        # calculate pies to display error contribution of high flows and low
        # flows
        # calculate the points of the first pie marker
        # these are just the origin (0, 0) + some (cos, sin) points on a circle
        r1 = abs(err_hf)/2
        x1 = np.cos(2 * np.pi * np.linspace(0.25 - r1/2, 0.25 + r1/2))
        y1 = np.sin(2 * np.pi * np.linspace(0.25 - r1/2, 0.25 + r1/2))
        xy1 = np.row_stack([[0, 0], np.column_stack([x1, y1])])
        s1 = np.abs(xy1).max()

        r2 = abs(err_lf)/2
        x2 = np.cos(2 * np.pi * np.linspace(0.75 - r2/2, 0.75 + r2/2))
        y2 = np.sin(2 * np.pi * np.linspace(0.75 - r2/2, 0.75 + r2/2))
        xy2 = np.row_stack([[0, 0], np.column_stack([x2, y2])])
        s2 = np.abs(xy2).max()

        # diagnose the error
        if abs(brel_mean) <= limit and exp_err > limit and eff > eff_l:
            c0 = ax.scatter(phi, eff, marker=xy1, s=s1 * 50, facecolor=rgba_color, zorder=3)
            c1 = ax.scatter(phi, eff, marker=xy2, s=s2 * 50, facecolor=rgba_color, zorder=3)
            c = ax.scatter(phi, eff, s=4, facecolor=rgba_color, zorder=3)
            c2 = ax.scatter(phi, eff, s=1, facecolor='grey', zorder=3)
        elif abs(brel_mean) > limit and exp_err <= limit and eff > eff_l:
            c0 = ax.scatter(phi, eff, marker=xy1, s=s1 * 50, facecolor=rgba_color, zorder=3)
            c1 = ax.scatter(phi, eff, marker=xy2, s=s2 * 50, facecolor=rgba_color, zorder=3)
            c = ax.scatter(phi, eff, s=4, facecolor=rgba_color, zorder=3)
            c2 = ax.scatter(phi, eff, s=1, facecolor='grey', zorder=3)
        elif abs(brel_mean) > limit and exp_err > limit and eff > eff_l:
            c0 = ax.scatter(phi, eff, marker=xy1, s=s1 * 50, facecolor=rgba_color, zorder=3)
            c1 = ax.scatter(phi, eff, marker=xy2, s=s2 * 50, facecolor=rgba_color, zorder=3)
            c = ax.scatter(phi, eff, s=4, facecolor=rgba_color, zorder=3)
            c2 = ax.scatter(phi, eff, s=1, facecolor='grey', zorder=3)
        # FBM
        elif abs(brel_mean) <= limit and exp_err <= limit and eff > eff_l:
            ax.annotate(
                "", xytext=(0, 0), xy=(0, eff), arrowprops=dict(facecolor=rgba_color),
                zorder=2
            )
            ax.annotate(
                "",
                xytext=(0, 0),
                xy=(np.pi, eff),
                arrowprops=dict(facecolor=rgba_color),
                zorder=2
            )
        # FGM
        elif abs(brel_mean) <= limit and exp_err <= limit and eff <= eff_l:
            c = ax.scatter(phi, eff, color=rgba_color, zorder=3)
            c2 = ax.scatter(phi, eff, s=1, facecolor='grey', zorder=3)

        # legend for error contribution of high flows and low flows
        rl1 = 1/2
        xl1 = np.cos(2 * np.pi * np.linspace(0.25 - rl1/2, 0.25 + rl1/2))
        yl1 = np.sin(2 * np.pi * np.linspace(0.25 - rl1/2, 0.25 + rl1/2))
        xyl1 = np.row_stack([[0, 0], np.column_stack([xl1, yl1])])
        sl1 = np.abs(xyl1).max()

        rl2 = 1/2
        xl2 = np.cos(2 * np.pi * np.linspace(0.75 - rl2/2, 0.75 + rl2/2))
        yl2 = np.sin(2 * np.pi * np.linspace(0.75 - rl2/2, 0.75 + rl2/2))
        xyl2 = np.row_stack([[0, 0], np.column_stack([xl2, yl2])])
        sl2 = np.abs(xyl2).max()
        ax.scatter([], [], color='k', zorder=2, marker=xyl1, s=sl1 * 50, label=r'high values ($\epsilon_{hf}=1$)')
        ax.scatter([], [], color='k', zorder=2, marker=xyl2, s=sl2 * 50, label=r'low values ($\epsilon_{lf}=1$)')
        ax.legend(loc='upper right', title="Error contribution of", fancybox=False,
                  frameon=False, bbox_to_anchor=(1.3, 1.1), handletextpad=0.1)

        ax.set_rticks([])  # turn default ticks off
        ax.set_rmin(0)
        if eff <= 1:
            ax.set_rmax(1)
        elif eff > 1:
            ax.set_rmax(ax_lim)
        # turn labels and grid off
        ax.tick_params(
            labelleft=False,
            labelright=False,
            labeltop=False,
            labelbottom=True,
            grid_alpha=0.01,
        )
        ticks_loc = ax.get_xticks().tolist()
        ax.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
        ax.set_xticklabels(["", "", "", "", "", "", "", ""])
        ax.text(
            -0.16,
            0.5,
            "High value overestimation -",
            va="center",
            ha="center",
            rotation=90,
            rotation_mode="anchor",
            transform=ax.transAxes,
        )
        ax.text(
            -0.11,
            0.5,
            "Low value underestimation",
            va="center",
            ha="center",
            rotation=90,
            rotation_mode="anchor",
            transform=ax.transAxes,
        )
        ax.text(
            -0.04,
            0.5,
            r"$B_{slope}$ < 0",
            va="center",
            ha="center",
            rotation=90,
            rotation_mode="anchor",
            transform=ax.transAxes,
        )
        ax.text(
            1.17,
            0.5,
            "High value underestimation -",
            va="center",
            ha="center",
            rotation=90,
            rotation_mode="anchor",
            transform=ax.transAxes,
        )
        ax.text(
            1.12,
            0.5,
            "Low value overestimation",
            va="center",
            ha="center",
            rotation=90,
            rotation_mode="anchor",
            transform=ax.transAxes,
        )
        ax.text(
            1.06,
            0.5,
            r"$B_{slope}$ > 0",
            va="center",
            ha="center",
            rotation=90,
            rotation_mode="anchor",
            transform=ax.transAxes,
        )
        ax.text(
            0.5,
            -0.09,
            "Constant negative offset",
            va="center",
            ha="center",
            rotation=0,
            rotation_mode="anchor",
            transform=ax.transAxes,
        )
        ax.text(
            0.5,
            -0.04,
            r"$\overline{B_{rel}}$ < 0",
            va="center",
            ha="center",
            rotation=0,
            rotation_mode="anchor",
            transform=ax.transAxes,
        )
        ax.text(
            0.5,
            1.09,
            "Constant positive offset",
            va="center",
            ha="center",
            rotation=0,
            rotation_mode="anchor",
            transform=ax.transAxes,
        )
        ax.text(
            0.5,
            1.04,
            r"$\overline{B_{rel}}$ > 0",
            va="center",
            ha="center",
            rotation=0,
            rotation_mode="anchor",
            transform=ax.transAxes,
        )
        # add colorbar for temporal correlation
        cbar = fig.colorbar(
            dummie_cax, ax=ax, orientation="horizontal", ticks=[1, 0.5, 0], shrink=0.8
        )
        cbar.set_label("r [-]\nTiming error", labelpad=4)
        cbar.set_ticklabels(["1", "0.5", "<0"])
        cbar.ax.tick_params(direction="in")

        # plot residual bias
        # calculate exceedence probability
        prob = np.linspace(0, 1, len(brel_res))
        ax1.axhline(y=0, color="slategrey")
        ax1.axvline(x=0.5, color="slategrey")
        ax1.plot(prob, brel_res, color="black")
        ax1.fill_between(prob, brel_res, where=0 < brel_res, facecolor="purple")
        ax1.fill_between(prob, brel_res, where=0 > brel_res, facecolor="red")
        ax1.set(ylabel=r"$B_{res}$ [-]", xlabel="Exceedence probabilty [-]")

        return fig


def diag_polar_plot_multi(
    brel_mean, temp_cor, eff_de, b_dir, phi, b_hf, b_lf, b_tot, err_hf,
    err_lf, a0=None, share0=None, limit=0.05, extended=False):
    r"""
    Diagnostic polar plot of Diagnostic efficiency (DE) for multiple
    evaluations.

    Note that points are used rather than arrows. Displaying multiple
    arrows would deteriorate visual comprehension.

    Parameters
    ----------
    brel_mean : (N,)array_like
        relative mean bias as 1-D array

    temp_cor : (N,)array_like
        temporal correlation as 1-D array

    eff_de : (N,)array_like
        diagnostic efficiency as 1-D array

    b_dir : (N,)array_like
        direction of bias as 1-D array

    phi : (N,)array_like
        angle as 1-D array (in radians)

    b_hf : (N,)array_like
        high flow bias

    b_lf : (N,)array_like
        low flow bias

    b_tot : (N,)array_like
        absolute total relative bias

    err_hf : (N,)array_like
        contribution of high flow errors

    err_lf : (N,)array_like
        contribution of low flow errors

    a0 : (N,)array_like
        agreement of zero values

    share0 : float
        share of zero values in observations

    limit : float, optional
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
    >>> brel_mean = np.array([0.1, 0.15, 0.2])
    >>> temp_cor = np.array([0.9, 0.85, 0.8])
    >>> eff_de = np.array([0.21, 0.24, 0.35])
    >>> b_dir = np.array([1, 1, 1])
    >>> phi = np.array([0.58, 0.98, 0.78)
    >>> b_hf = np.array([0.2, 0.15, 0.2])
    >>> b_lf = np.array([0.2, 0.05, 0.3])
    >>> b_tot = np.array([0.4, 0.2, 0.5])
    >>> err_hf = np.array([0.5, 0.75, 0.4])
    >>> err_lf = np.array([0.5, 0.25, 0.6])
    >>> de.diag_polar_plot_multi(brel_mean, temp_cor, eff_de, b_dir,
                                 phi, b_hf, b_lf, b_tot, err_hf, err_lf)
    """
    eff_max = np.max(eff_de)

    ll_brel_mean = brel_mean.tolist()
    ll_temp_cor = temp_cor.tolist()
    ll_eff = eff_de.tolist()
    ll_b_dir = b_dir.tolist()
    ll_phi = phi.tolist()
    ll_b_hf = b_hf.tolist()
    ll_b_lf = b_lf.tolist()
    ll_b_tot = b_tot.tolist()
    ll_err_hf = err_hf.tolist()
    ll_err_lf = err_lf.tolist()

    # convert temporal correlation to color
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1.0)

    delta = 0.01  # for spacing

    # determine axis limits
    if eff_max <= 1:
        ax_lim = 1.2
        yy = np.arange(0.01, ax_lim, delta)
        c_levels = np.arange(0, ax_lim, 0.2)
    elif eff_max > 1 and eff_max <= 2:
        ax_lim = 2.2
        yy = np.arange(0.01, ax_lim, delta)
        c_levels = np.arange(0, ax_lim, 0.2)
    elif eff_max > 2 and eff_max <= 3:
        ax_lim = 3.2
        yy = np.arange(0.01, ax_lim, delta)
        c_levels = np.arange(0, ax_lim, 0.2)
    elif eff_max > 3:
        raise ValueError("Some values of 'DE' are too large for visualization!", eff_max)

    len_yy = len(yy)

    # arrays to plot contour lines of DE
    xx = np.radians(np.linspace(0, 360, len_yy))
    theta, r = np.meshgrid(xx, yy)

    # diagnostic polar plot
    if not extended:
        fig, ax = plt.subplots(
            figsize=(3, 3), subplot_kw=dict(projection="polar"), constrained_layout=True
        )
        # dummie plot for colorbar of temporal correlation
        cs = np.arange(0, 1.1, 0.1)
        dummie_cax = ax.scatter(cs, cs, c=cs, cmap="plasma_r")
        # Clear axis
        ax.cla()
        # plot regions
        ax.plot(
            (0, np.deg2rad(45)),
            (0, np.max(yy)),
            color="lightgray",
            linewidth=1,
            ls="--",
            zorder=0,
        )
        ax.plot(
            (0, np.deg2rad(135)),
            (0, np.max(yy)),
            color="lightgray",
            linewidth=1,
            ls="--",
            zorder=0,
        )
        ax.plot(
            (0, np.deg2rad(225)),
            (0, np.max(yy)),
            color="lightgray",
            linewidth=1,
            ls="--",
            zorder=0,
        )
        ax.plot(
            (0, np.deg2rad(315)),
            (0, np.max(yy)),
            color="lightgray",
            linewidth=1,
            ls="--",
            zorder=0,
        )
        # contours of DE
        cp = ax.contour(theta, r, r, colors="darkgray", levels=c_levels, zorder=1, linewidths=1)
        cl = ax.clabel(cp, inline=True, fmt="%1.1f", colors="dimgrey")
        # threshold efficiency for FBM
        eff_l = np.sqrt((limit) ** 2 + (limit) ** 2 + (limit) ** 2)
        # loop over each data point
        zz = zip(ll_brel_mean, ll_temp_cor, ll_eff, ll_b_dir, ll_phi, ll_b_hf,
                 ll_b_lf, ll_b_tot, ll_err_hf, ll_err_lf)
        for (bm, r, eff, bd, ang, bhf, blf, btot, errhf, errlf) in zz:
            # convert temporal correlation to color
            cmap = cm.get_cmap('plasma_r')
            rgba_color = cmap(norm(r))
            # relation of high flow errors and low flow errors which explain
            # the total FDC error
            if btot > 0:
                exp_err = (abs(bhf) + abs(blf)) / btot
            elif btot == 0:
                exp_err = 0

            # calculate pies to display error contribution of high flows and low
            # flows
            # calculate the points of the first pie marker
            # these are just the origin (0, 0) + some (cos, sin) points on a circle
            r1 = abs(errhf)/2
            x1 = np.cos(2 * np.pi * np.linspace(0.25 - r1/2, 0.25 + r1/2))
            y1 = np.sin(2 * np.pi * np.linspace(0.25 - r1/2, 0.25 + r1/2))
            xy1 = np.row_stack([[0, 0], np.column_stack([x1, y1])])
            s1 = np.abs(xy1).max()

            r2 = abs(errlf)/2
            x2 = np.cos(2 * np.pi * np.linspace(0.75 - r2/2, 0.75 + r2/2))
            y2 = np.sin(2 * np.pi * np.linspace(0.75 - r2/2, 0.75 + r2/2))
            xy2 = np.row_stack([[0, 0], np.column_stack([x2, y2])])
            s2 = np.abs(xy2).max()

            # diagnose the error
            if abs(bm) <= limit and exp_err > limit and eff > eff_l:
                c0 = ax.scatter(ang, eff, marker=xy1, s=s1 * 50, facecolor=rgba_color, zorder=3)
                c1 = ax.scatter(ang, eff, marker=xy2, s=s2 * 50, facecolor=rgba_color, zorder=3)
                c = ax.scatter(ang, eff, s=4, facecolor=rgba_color, zorder=3)
                c2 = ax.scatter(ang, eff, s=1, facecolor='grey', zorder=3)
            elif abs(bm) > limit and exp_err <= limit and eff > eff_l:
                c0 = ax.scatter(ang, eff, marker=xy1, s=s1 * 50, facecolor=rgba_color, zorder=3)
                c1 = ax.scatter(ang, eff, marker=xy2, s=s2 * 50, facecolor=rgba_color, zorder=3)
                c = ax.scatter(ang, eff, s=4, facecolor=rgba_color, zorder=3)
                c2 = ax.scatter(ang, eff, s=1, facecolor='grey', zorder=3)
            elif abs(bm) > limit and exp_err > limit and eff > eff_l:
                c0 = ax.scatter(ang, eff, marker=xy1, s=s1 * 50, facecolor=rgba_color, zorder=3)
                c1 = ax.scatter(ang, eff, marker=xy2, s=s2 * 50, facecolor=rgba_color, zorder=3)
                c = ax.scatter(ang, eff, s=4, facecolor=rgba_color, zorder=3)
                c2 = ax.scatter(ang, eff, s=1, facecolor='grey', zorder=3)
            # FBM
            elif abs(bm) <= limit and exp_err <= limit and eff > eff_l:
                ax.annotate(
                    "", xytext=(0, 0), xy=(0, eff), arrowprops=dict(facecolor=rgba_color),
                    zorder=2)
                ax.annotate(
                    "",
                    xytext=(0, 0),
                    xy=(np.pi, eff),
                    arrowprops=dict(facecolor=rgba_color),
                    zorder=2
                )
            # FGM
            elif abs(bm) <= limit and exp_err <= limit and eff <= eff_l:
                c = ax.scatter(ang, eff, color=rgba_color, zorder=3)
                c2 = ax.scatter(ang, eff, s=1, facecolor='grey', zorder=3)

        # legend for error contribution of high flows and low flows
        rl1 = 1/2
        xl1 = np.cos(2 * np.pi * np.linspace(0.25 - rl1/2, 0.25 + rl1/2))
        yl1 = np.sin(2 * np.pi * np.linspace(0.25 - rl1/2, 0.25 + rl1/2))
        xyl1 = np.row_stack([[0, 0], np.column_stack([xl1, yl1])])
        sl1 = np.abs(xyl1).max()

        rl2 = 1/2
        xl2 = np.cos(2 * np.pi * np.linspace(0.75 - rl2/2, 0.75 + rl2/2))
        yl2 = np.sin(2 * np.pi * np.linspace(0.75 - rl2/2, 0.75 + rl2/2))
        xyl2 = np.row_stack([[0, 0], np.column_stack([xl2, yl2])])
        sl2 = np.abs(xyl2).max()
        ax.scatter([], [], color='k', zorder=2, marker=xyl1, s=sl1 * 50, label=r'$\epsilon_{hf}=1$')
        ax.scatter([], [], color='k', zorder=2, marker=xyl2, s=sl2 * 50, label=r'$\epsilon_{lf}=1$')
        ax.legend(loc='upper right', fancybox=False,
                  frameon=False, bbox_to_anchor=(1.2, 1.12), handletextpad=0.2)

        ax.set_rticks([])  # turn default ticks off
        ax.set_rmin(0)
        if eff_max <= 1:
            ax.set_rmax(1)
        elif eff_max > 1:
            ax.set_rmax(ax_lim)
        # turn labels and grid off
        ax.tick_params(
            labelleft=False,
            labelright=False,
            labeltop=False,
            labelbottom=True,
            grid_alpha=0.01,
        )
        ticks_loc = ax.get_xticks().tolist()
        ax.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
        ax.set_xticklabels(["", "", "", "", "", "", "", ""])
        ax.text(
            -0.17,
            0.5,
            "High value overestimation -",
            va="center",
            ha="center",
            rotation=90,
            rotation_mode="anchor",
            transform=ax.transAxes,
        )
        ax.text(
            -0.11,
            0.5,
            "Low value underestimation",
            va="center",
            ha="center",
            rotation=90,
            rotation_mode="anchor",
            transform=ax.transAxes,
        )
        ax.text(
            -0.04,
            0.5,
            r"$B_{slope}$ < 0",
            va="center",
            ha="center",
            rotation=90,
            rotation_mode="anchor",
            transform=ax.transAxes,
        )
        ax.text(
            1.18,
            0.5,
            "High value underestimation -",
            va="center",
            ha="center",
            rotation=90,
            rotation_mode="anchor",
            transform=ax.transAxes,
        )
        ax.text(
            1.12,
            0.5,
            "Low value overestimation",
            va="center",
            ha="center",
            rotation=90,
            rotation_mode="anchor",
            transform=ax.transAxes,
        )
        ax.text(
            1.06,
            0.5,
            r"$B_{slope}$ > 0",
            va="center",
            ha="center",
            rotation=90,
            rotation_mode="anchor",
            transform=ax.transAxes,
        )
        ax.text(
            0.5,
            -0.12,
            "Constant negative offset",
            va="center",
            ha="center",
            rotation=0,
            rotation_mode="anchor",
            transform=ax.transAxes,
        )
        ax.text(
            0.5,
            -0.05,
            r"$\overline{B_{rel}}$ < 0",
            va="center",
            ha="center",
            rotation=0,
            rotation_mode="anchor",
            transform=ax.transAxes,
        )
        ax.text(
            0.5,
            1.12,
            "Constant positive offset",
            va="center",
            ha="center",
            rotation=0,
            rotation_mode="anchor",
            transform=ax.transAxes,
        )
        ax.text(
            0.5,
            1.05,
            r"$\overline{B_{rel}}$ > 0",
            va="center",
            ha="center",
            rotation=0,
            rotation_mode="anchor",
            transform=ax.transAxes,
        )
        # add colorbar for temporal correlation
        cbar = fig.colorbar(
            dummie_cax, ax=ax, orientation="horizontal", ticks=[1, 0.5, 0], shrink=0.8
        )
        cbar.set_label("r [-]\nTiming error", labelpad=4)
        cbar.set_ticklabels(["1", "0.5", "<0"])
        cbar.ax.tick_params(direction="in")

        if a0 is not None and share0 is not None:
            a0_avg = np.round(np.mean(a0), 2)
            a0_std = np.round(np.std(a0), 2)
            ax.text(
                0.8,
                0.03,
                f"$s_0$ [-]: {share0}\n$a_0$ [-]: {a0_avg} ± {a0_std}",
                va="center",
                ha="left",
                rotation=0,
                rotation_mode="anchor",
                transform=ax.transAxes,
            )

        elif a0 is not None and share0 is None:
            a0_avg = np.round(np.mean(a0), 2)
            a0_std = np.round(np.std(a0), 2)
            ax.text(
                0.8,
                0.03,
                f"$a_0$ [-]:\n {a0_avg} ± {a0_std}",
                va="center",
                ha="left",
                rotation=0,
                rotation_mode="anchor",
                transform=ax.transAxes,
            )

        return fig

    elif extended:
        fig = plt.figure(figsize=(6, 3), constrained_layout=True)
        gs = fig.add_gridspec(1, 2)
        ax = fig.add_subplot(gs[0, 0], projection="polar")
        # add_axes([xmin,ymin,dx,dy])
        ax1 = fig.add_axes([0.6, 0.15, 0.32, 0.32], frameon=True)
        ax2 = fig.add_axes([0.6, 0.6, 0.32, 0.32], frameon=True)
        # dummie plot for colorbar of temporal correlation
        cs = np.arange(0, 1.1, 0.1)
        dummie_cax = ax.scatter(cs, cs, c=cs, cmap="plasma_r")
        # Clear axis
        ax.cla()
        # plot regions
        ax.plot(
            (0, np.deg2rad(45)),
            (0, np.max(yy)),
            color="lightgray",
            linewidth=1,
            ls="--",
            zorder=0,
        )
        ax.plot(
            (0, np.deg2rad(135)),
            (0, np.max(yy)),
            color="lightgray",
            linewidth=1,
            ls="--",
            zorder=0,
        )
        ax.plot(
            (0, np.deg2rad(225)),
            (0, np.max(yy)),
            color="lightgray",
            linewidth=1,
            ls="--",
            zorder=0,
        )
        ax.plot(
            (0, np.deg2rad(315)),
            (0, np.max(yy)),
            color="lightgray",
            linewidth=1,
            ls="--",
            zorder=0,
        )
        # contours of DE
        cp = ax.contour(theta, r, r, colors="darkgray", levels=c_levels, zorder=1, linewidths=1)
        cl = ax.clabel(cp, inline=True, fmt="%1.1f", colors="dimgrey")
        # threshold efficiency for FBM
        eff_l = np.sqrt((limit) ** 2 + (limit) ** 2 + (limit) ** 2)
        # loop over each data point
        zz = zip(ll_brel_mean, ll_temp_cor, ll_eff, ll_b_dir, ll_phi, ll_b_hf,
                 ll_b_lf, ll_b_tot, ll_err_hf, ll_err_lf)
        for (bm, r, eff, bd, ang, bhf, blf, btot, errhf, errlf) in zz:
            # convert temporal correlation to color
            cmap = cm.get_cmap('plasma_r')
            rgba_color = cmap(norm(r))
            # relation of high flow errors and low flow errors which explain
            # the total FDC error
            if btot > 0:
                exp_err = (abs(bhf) + abs(blf)) / btot
            elif btot == 0:
                exp_err = 0

            # calculate pies to display error contribution of high flows and low
            # flows
            # calculate the points of the first pie marker
            # these are just the origin (0, 0) + some (cos, sin) points on a circle
            r1 = abs(errhf)/2
            x1 = np.cos(2 * np.pi * np.linspace(0.25 - r1/2, 0.25 + r1/2))
            y1 = np.sin(2 * np.pi * np.linspace(0.25 - r1/2, 0.25 + r1/2))
            xy1 = np.row_stack([[0, 0], np.column_stack([x1, y1])])
            s1 = np.abs(xy1).max()

            r2 = abs(errlf)/2
            x2 = np.cos(2 * np.pi * np.linspace(0.75 - r2/2, 0.75 + r2/2))
            y2 = np.sin(2 * np.pi * np.linspace(0.75 - r2/2, 0.75 + r2/2))
            xy2 = np.row_stack([[0, 0], np.column_stack([x2, y2])])
            s2 = np.abs(xy2).max()

            # diagnose the error
            if abs(bm) <= limit and exp_err > limit and eff > eff_l:
                c0 = ax.scatter(ang, eff, marker=xy1, s=s1 * 50, facecolor=rgba_color, zorder=3)
                c1 = ax.scatter(ang, eff, marker=xy2, s=s2 * 50, facecolor=rgba_color, zorder=3)
                c = ax.scatter(ang, eff, s=4, facecolor=rgba_color, zorder=3)
                c2 = ax.scatter(ang, eff, s=1, facecolor='grey', zorder=3)
            elif abs(bm) > limit and exp_err <= limit and eff > eff_l:
                c0 = ax.scatter(ang, eff, marker=xy1, s=s1 * 50, facecolor=rgba_color, zorder=3)
                c1 = ax.scatter(ang, eff, marker=xy2, s=s2 * 50, facecolor=rgba_color, zorder=3)
                c = ax.scatter(ang, eff, s=4, facecolor=rgba_color, zorder=3)
                c2 = ax.scatter(ang, eff, s=1, facecolor='grey', zorder=3)
            elif abs(bm) > limit and exp_err > limit and eff > eff_l:
                c0 = ax.scatter(ang, eff, marker=xy1, s=s1 * 50, facecolor=rgba_color, zorder=3)
                c1 = ax.scatter(ang, eff, marker=xy2, s=s2 * 50, facecolor=rgba_color, zorder=3)
                c = ax.scatter(ang, eff, s=4, facecolor=rgba_color, zorder=3)
                c2 = ax.scatter(ang, eff, s=1, facecolor='grey', zorder=3)
            # FBM
            elif abs(bm) <= limit and exp_err <= limit and eff > eff_l:
                ax.annotate(
                    "", xytext=(0, 0), xy=(0, eff), arrowprops=dict(facecolor=rgba_color),
                    zorder=2)
                ax.annotate(
                    "",
                    xytext=(0, 0),
                    xy=(np.pi, eff),
                    arrowprops=dict(facecolor=rgba_color),
                    zorder=2
                )
            # FGM
            elif abs(bm) <= limit and exp_err <= limit and eff <= eff_l:
                c = ax.scatter(ang, eff, color=rgba_color, zorder=3)
                c2 = ax.scatter(ang, eff, s=1, facecolor='grey', zorder=3)

        # legend for error contribution of high flows and low flows
        rl1 = 1/2
        xl1 = np.cos(2 * np.pi * np.linspace(0.25 - rl1/2, 0.25 + rl1/2))
        yl1 = np.sin(2 * np.pi * np.linspace(0.25 - rl1/2, 0.25 + rl1/2))
        xyl1 = np.row_stack([[0, 0], np.column_stack([xl1, yl1])])
        sl1 = np.abs(xyl1).max()

        rl2 = 1/2
        xl2 = np.cos(2 * np.pi * np.linspace(0.75 - rl2/2, 0.75 + rl2/2))
        yl2 = np.sin(2 * np.pi * np.linspace(0.75 - rl2/2, 0.75 + rl2/2))
        xyl2 = np.row_stack([[0, 0], np.column_stack([xl2, yl2])])
        sl2 = np.abs(xyl2).max()
        ax.scatter([], [], color='k', zorder=2, marker=xyl1, s=sl1 * 50, label=r'high values ($\epsilon_{hf}=1$)')
        ax.scatter([], [], color='k', zorder=2, marker=xyl2, s=sl2 * 50, label=r'low values ($\epsilon_{lf}=1$)')
        ax.legend(loc='upper right', title="Error contribution of", fancybox=False,
                  frameon=False, bbox_to_anchor=(1.3, 1.1), handletextpad=0.1)

        ax.set_rticks([])  # turn default ticks off
        ax.set_rmin(0)
        if eff_max <= 1:
            ax.set_rmax(1)
        elif eff_max > 1:
            ax.set_rmax(ax_lim)
        # turn labels and grid off
        ax.tick_params(
            labelleft=False,
            labelright=False,
            labeltop=False,
            labelbottom=True,
            grid_alpha=0.01,
        )
        ticks_loc = ax.get_xticks().tolist()
        ax.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
        ax.set_xticklabels(["", "", "", "", "", r"0$^{\circ}$ (360$^{\circ}$)", "", ""])
        ax.text(
            -0.16,
            0.5,
            "High value overestimation -",
            va="center",
            ha="center",
            rotation=90,
            rotation_mode="anchor",
            transform=ax.transAxes,
        )
        ax.text(
            -0.11,
            0.5,
            "Low value underestimation",
            va="center",
            ha="center",
            rotation=90,
            rotation_mode="anchor",
            transform=ax.transAxes,
        )
        ax.text(
            -0.05,
            0.5,
            r"$B_{slope}$ < 0",
            va="center",
            ha="center",
            rotation=90,
            rotation_mode="anchor",
            transform=ax.transAxes,
        )
        ax.text(
            1.16,
            0.5,
            "High value underestimation -",
            va="center",
            ha="center",
            rotation=90,
            rotation_mode="anchor",
            transform=ax.transAxes,
        )
        ax.text(
            1.11,
            0.5,
            "Low value overestimation",
            va="center",
            ha="center",
            rotation=90,
            rotation_mode="anchor",
            transform=ax.transAxes,
        )
        ax.text(
            1.05,
            0.5,
            r"$B_{slope}$ > 0",
            va="center",
            ha="center",
            rotation=90,
            rotation_mode="anchor",
            transform=ax.transAxes,
        )
        ax.text(
            0.5,
            -0.11,
            "Constant negative offset",
            va="center",
            ha="center",
            rotation=0,
            rotation_mode="anchor",
            transform=ax.transAxes,
        )
        ax.text(
            0.5,
            -0.05,
            r"$\overline{B_{rel}}$ < 0",
            va="center",
            ha="center",
            rotation=0,
            rotation_mode="anchor",
            transform=ax.transAxes,
        )
        ax.text(
            0.5,
            1.11,
            "Constant positive offset",
            va="center",
            ha="center",
            rotation=0,
            rotation_mode="anchor",
            transform=ax.transAxes,
        )
        ax.text(
            0.5,
            1.05,
            r"$\overline{B_{rel}}$ > 0",
            va="center",
            ha="center",
            rotation=0,
            rotation_mode="anchor",
            transform=ax.transAxes,
        )
        # add colorbar for temporal correlation
        cbar = fig.colorbar(
            dummie_cax, ax=ax, orientation="horizontal", ticks=[1, 0.5, 0], shrink=0.8
        )
        cbar.set_label("r [-]\nTiming error", labelpad=4)
        cbar.set_ticklabels(["1", "0.5", "<0"])
        cbar.ax.tick_params(direction="in")

        # convert to degrees
        diag_deg = (phi * (180 / np.pi)) + 135
        diag_deg[diag_deg < 0] = 360 - diag_deg[diag_deg < 0]

        # 1-D density plot
        g = sns.kdeplot(x=diag_deg, color="k", ax=ax1)
        ax1.set_xticks([0, 90, 180, 270, 360])
        ax1.set_xlim(0, 360)
        ax1.set_ylim(0,)
        ax1.set(ylabel="Density", xlabel=r"[$^\circ$]")

        # error contribution of high flows and low flows
        errc = pd.DataFrame(index=range(len(eff_de)), columns=['hf', 'lf'])
        errc.loc[:, 'hf'] = err_hf
        errc.loc[:, 'lf'] = err_lf
        sns.violinplot(data=errc, ax=ax2, inner="quartile")
        ax2.set_ylabel(r'$\epsilon$ [-]')
        ax2.set_xticklabels(['high values', 'low values'])

        # 2-D density plot
        cmap = cm.get_cmap('plasma_r')
        r_colors = cmap(norm(temp_cor))
        g = sns.jointplot(
            x=diag_deg,
            y=eff_de,
            kind="kde",
            zorder=1,
            n_levels=20,
            cmap="Greys",
            thresh=0.05,
            marginal_kws={"color": "k"},
        ).plot_joint(plt.scatter, c=r_colors, alpha=0.4, zorder=2)
        g.set_axis_labels(r"[$^\circ$]", "DE [-]")
        g.ax_joint.set_xticks([0, 90, 180, 270, 360])
        g.ax_joint.set_xlim(0, 360)
        g.ax_joint.set_ylim(0, ax_lim)
        g.ax_marg_x.set_xticks([0, 90, 180, 270, 360])
        kde_data = g.ax_marg_x.get_lines()[0].get_data()
        kde_xx = kde_data[0]
        kde_yy = kde_data[1]
        kde_data = g.ax_marg_y.get_lines()[0].get_data()
        kde_xx = kde_data[0]
        kde_yy = kde_data[1]
        norm = matplotlib.colors.Normalize(vmin=0, vmax=ax_lim)
        colors = cm.Greens_r(norm(kde_yy))
        npts = len(kde_xx)
        for i in range(npts - 1):
            g.ax_marg_y.fill_betweenx(
                [kde_yy[i], kde_yy[i + 1]], [kde_xx[i], kde_xx[i + 1]], color=colors[i]
            )
        g.fig.tight_layout()

        return fig, g.fig


def gdiag_polar_plot(eff, comp1, comp2, comp3, limit=0.05):  # pragma: no cover
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

    limit : float, optional
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
    phi = calc_phi(comp1, comp2)

    # convert metric component 3 to color
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1.0)
    cmap = cm.get_cmap('plasma_r')
    rgba_color = cmap(norm(comp3))

    delta = 0.01  # for spacing

    # determine axis limits
    if eff <= 1:
        ax_lim = 1.2
        yy = np.arange(0.01, ax_lim, delta)
        c_levels = np.arange(0, ax_lim, 0.2)
    elif eff < 1 and eff <= 2:
        ax_lim = 2.2
        yy = np.arange(0.01, ax_lim, delta)
        c_levels = np.arange(0, ax_lim, 0.2)
    elif eff < 2 and eff <= 3:
        ax_lim = 3.2
        yy = np.arange(0.01, ax_lim, delta)
        c_levels = np.arange(0, ax_lim, 0.2)
    elif eff > 3:
        raise AssertionError("Value of eff is out of bounds for visualization!", eff)

    len_yy = len(yy)

    # arrays to plot contour lines of DE
    xx = np.radians(np.linspace(0, 360, len_yy))
    theta, r = np.meshgrid(xx, yy)

    # diagnostic polar plot
    fig, ax = plt.subplots(
        figsize=(3, 3), subplot_kw=dict(projection="polar"), constrained_layout=True
    )
    # dummie plot for colorbar of temporal correlation
    cs = np.arange(0, 1.1, 0.1)
    dummie_cax = ax.scatter(cs, cs, c=cs, cmap="plasma_r")
    # Clear axis
    ax.cla()
    # plot regions
    ax.plot(
        (0, np.deg2rad(45)),
        (0, np.max(yy)),
        color="lightgray",
        linewidth=1,
        ls="--",
        zorder=0,
    )
    ax.plot(
        (0, np.deg2rad(135)),
        (0, np.max(yy)),
        color="lightgray",
        linewidth=1,
        ls="--",
        zorder=0,
    )
    ax.plot(
        (0, np.deg2rad(225)),
        (0, np.max(yy)),
        color="lightgray",
        linewidth=1,
        ls="--",
        zorder=0,
    )
    ax.plot(
        (0, np.deg2rad(315)),
        (0, np.max(yy)),
        color="lightgray",
        linewidth=1,
        ls="--",
        zorder=0,
    )
    # contours of DE
    cp = ax.contour(theta, r, r, colors="darkgray", levels=c_levels, zorder=1, linewidths=1)
    cl = ax.clabel(cp, inline=True, fmt="%1.1f", colors="dimgrey")
    # threshold efficiency for FBM
    eff_l = np.sqrt((limit) ** 2 + (limit) ** 2 + (limit) ** 2)
    # relation of b_dir which explains the error
    if abs(comp2) > 0:
        exp_err = comp2
    elif abs(comp2) == 0:
        exp_err = 0
    # diagnose the error
    if abs(comp1) <= limit and exp_err > limit and eff > eff_l:
        ax.annotate(
            "", xytext=(0, 0), xy=(phi, eff), arrowprops=dict(facecolor=rgba_color)
        )
    elif abs(comp1) > limit and exp_err <= limit and eff > eff_l:
        ax.annotate(
            "", xytext=(0, 0), xy=(phi, eff), arrowprops=dict(facecolor=rgba_color)
        )
    elif abs(comp1) > limit and exp_err > limit and eff > eff_l:
        ax.annotate(
            "", xytext=(0, 0), xy=(phi, eff), arrowprops=dict(facecolor=rgba_color)
        )
    # FBM
    elif abs(comp1) <= limit and exp_err <= limit and eff > eff_l:
        ax.annotate(
            "", xytext=(0, 0), xy=(0, eff), arrowprops=dict(facecolor=rgba_color)
        )
        ax.annotate(
            "", xytext=(0, 0), xy=(np.pi, eff), arrowprops=dict(facecolor=rgba_color)
        )
    # FGM
    elif abs(comp1) <= limit and exp_err <= limit and eff <= eff_l:
        c = ax.scatter(phi, eff, color=rgba_color)
    ax.set_rticks([])  # turn default ticks off
    ax.set_rmin(0)
    if eff <= 1:
        ax.set_rmax(1)
    elif eff > 1:
        ax.set_rmax(ax_lim)
    # turn labels and grid off
    ax.tick_params(
        labelleft=False,
        labelright=False,
        labeltop=False,
        labelbottom=True,
        grid_alpha=0.01,
    )
    ticks_loc = ax.get_xticks().tolist()
    ax.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
    ax.set_xticklabels(["", "", "", "", "", "", "", ""])
    ax.text(
        -0.04,
        0.5,
        r"Comp2 < 0",
        va="center",
        ha="center",
        rotation=90,
        rotation_mode="anchor",
        transform=ax.transAxes,
    )
    ax.text(
        1.04,
        0.5,
        r"Comp2 > 0",
        va="center",
        ha="center",
        rotation=90,
        rotation_mode="anchor",
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        -0.04,
        r"Comp1 < 0",
        va="center",
        ha="center",
        rotation=0,
        rotation_mode="anchor",
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        1.04,
        r"Comp1 > 0",
        va="center",
        ha="center",
        rotation=0,
        rotation_mode="anchor",
        transform=ax.transAxes,
    )
    # add colorbar for temporal correlation
    cbar = fig.colorbar(
        dummie_cax, ax=ax, orientation="horizontal", ticks=[1, 0.5, 0], shrink=0.8
    )
    cbar.set_label("Comp3", labelpad=4)
    cbar.set_ticklabels(["1", "0.5", "<0"])
    cbar.ax.tick_params(direction="in")

    return fig


def gdiag_polar_plot_multi(
    eff, comp1, comp2, comp3, limit=0.05, extended=True
):  # pragma: no cover
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

    limit : float, optional
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
    eff_max = np.max(eff)

    ll_comp1 = comp1.tolist()
    ll_comp2 = comp2.tolist()
    ll_comp3 = comp3.tolist()
    ll_eff = eff.tolist()

    # convert temporal correlation to color
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1.0)

    delta = 0.01  # for spacing

    # determine axis limits
    if eff_max <= 1:
        ax_lim = 1.2
        yy = np.arange(0.01, ax_lim, delta)
        c_levels = np.arange(0, ax_lim, 0.2)
    elif eff_max > 1 and eff_max <= 2:
        ax_lim = 2.2
        yy = np.arange(0.01, ax_lim, delta)
        c_levels = np.arange(0, ax_lim, 0.2)
    elif eff_max > 2 and eff_max <= 3:
        ax_lim = 3.2
        yy = np.arange(0.01, ax_lim, delta)
        c_levels = np.arange(0, ax_lim, 0.2)
    elif eff_max > 3:
        raise ValueError(
            "Some values of eff are out of bounds for visualization!", eff_max
        )

    len_yy = len(yy)

    # arrays to plot contour lines of DE
    xx = np.radians(np.linspace(0, 360, len_yy))
    theta, r = np.meshgrid(xx, yy)

    # diagnostic polar plot
    if not extended:
        fig, ax = plt.subplots(
            figsize=(3, 3), subplot_kw=dict(projection="polar"), constrained_layout=True
        )
        # dummie plot for colorbar of temporal correlation
        cs = np.arange(0, 1.1, 0.1)
        dummie_cax = ax.scatter(cs, cs, c=cs, cmap="plasma_r")
        # Clear axis
        ax.cla()
        # plot regions
        ax.plot(
            (0, np.deg2rad(45)),
            (0, np.max(yy)),
            color="lightgray",
            linewidth=1,
            ls="--",
            zorder=0,
        )
        ax.plot(
            (0, np.deg2rad(135)),
            (0, np.max(yy)),
            color="lightgray",
            linewidth=1,
            ls="--",
            zorder=0,
        )
        ax.plot(
            (0, np.deg2rad(225)),
            (0, np.max(yy)),
            color="lightgray",
            linewidth=1,
            ls="--",
            zorder=0,
        )
        ax.plot(
            (0, np.deg2rad(315)),
            (0, np.max(yy)),
            color="lightgray",
            linewidth=1,
            ls="--",
            zorder=0,
        )
        # contours of DE
        cp = ax.contour(theta, r, r, colors="darkgray", levels=c_levels, zorder=1, linewidths=1)
        cl = ax.clabel(cp, inline=True, fmt="%1.1f", colors="dimgrey")
        # threshold efficiency for FBM
        eff_l = np.sqrt((limit) ** 2 + (limit) ** 2 + (limit) ** 2)
        # loop over each data point
        for (c1, c2, c3, eff) in zip(ll_comp1, ll_comp2, ll_comp3, ll_eff):
            ang = np.arctan2(c1, c2)
            # convert temporal correlation to color
            cmap = cm.get_cmap('plasma_r')
            rgba_color = cmap(norm(c3))
            # relation of b_dir which explains the error
            if abs(c2) > 0:
                exp_err = c2
            elif abs(c2) == 0:
                exp_err = 0
            # diagnose the error
            if abs(c1) <= limit and exp_err > limit and eff > eff_l:
                c = ax.scatter(ang, eff, color=rgba_color, zorder=2)
            elif abs(c1) > limit and exp_err <= limit and eff > eff_l:
                c = ax.scatter(ang, eff, color=rgba_color, zorder=2)
            elif abs(c1) > limit and exp_err > limit and eff > eff_l:
                c = ax.scatter(ang, eff, color=rgba_color, zorder=2)
            # FBM
            elif abs(c1) <= limit and exp_err <= limit and eff > eff_l:
                ax.annotate(
                    "",
                    xytext=(0, 0),
                    xy=(0, eff),
                    arrowprops=dict(facecolor=rgba_color),
                )
                ax.annotate(
                    "",
                    xytext=(0, 0),
                    xy=(np.pi, eff),
                    arrowprops=dict(facecolor=rgba_color),
                )
            # FGM
            elif abs(c1) <= limit and exp_err <= limit and eff <= eff_l:
                c = ax.scatter(ang, eff, color=rgba_color, zorder=2)
        ax.set_rticks([])  # turn default ticks off
        ax.set_rmin(0)
        if eff_max <= 1:
            ax.set_rmax(1)
        elif eff_max > 1:
            ax.set_rmax(ax_lim)
        # turn labels and grid off
        ax.tick_params(
            labelleft=False,
            labelright=False,
            labeltop=False,
            labelbottom=True,
            grid_alpha=0.01,
        )
        ticks_loc = ax.get_xticks().tolist()
        ax.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
        ax.set_xticklabels(["", "", "", "", "", "", "", ""])
        ax.text(
            -0.04,
            0.5,
            r"Comp2 < 0",
            va="center",
            ha="center",
            rotation=90,
            rotation_mode="anchor",
            transform=ax.transAxes,
        )
        ax.text(
            1.04,
            0.5,
            r"Comp2 > 0",
            va="center",
            ha="center",
            rotation=90,
            rotation_mode="anchor",
            transform=ax.transAxes,
        )
        ax.text(
            0.5,
            -0.04,
            r"Comp1 < 0",
            va="center",
            ha="center",
            rotation=0,
            rotation_mode="anchor",
            transform=ax.transAxes,
        )
        ax.text(
            0.5,
            1.04,
            r"Comp1 > 0",
            va="center",
            ha="center",
            rotation=0,
            rotation_mode="anchor",
            transform=ax.transAxes,
        )
        # add colorbar for temporal correlation
        cbar = fig.colorbar(
            dummie_cax, ax=ax, orientation="horizontal", ticks=[1, 0.5, 0], shrink=0.8
        )
        cbar.set_label("Comp3", labelpad=4)
        cbar.set_ticklabels(["1", "0.5", "<0"])
        cbar.ax.tick_params(direction="in")

        return fig

    elif extended:
        fig = plt.figure(figsize=(6, 3), constrained_layout=True)
        gs = fig.add_gridspec(1, 2)
        ax = fig.add_subplot(gs[0, 0], projection="polar")
        ax1 = fig.add_axes([0.66, 0.3, 0.32, 0.32], frameon=True)
        # dummie plot for colorbar of temporal correlation
        cs = np.arange(0, 1.1, 0.1)
        dummie_cax = ax.scatter(cs, cs, c=cs, cmap="plasma_r")
        # Clear axis
        ax.cla()
        # plot regions
        ax.plot(
            (0, np.deg2rad(45)),
            (0, np.max(yy)),
            color="lightgray",
            linewidth=1,
            ls="--",
            zorder=0,
        )
        ax.plot(
            (0, np.deg2rad(135)),
            (0, np.max(yy)),
            color="lightgray",
            linewidth=1,
            ls="--",
            zorder=0,
        )
        ax.plot(
            (0, np.deg2rad(225)),
            (0, np.max(yy)),
            color="lightgray",
            linewidth=1,
            ls="--",
            zorder=0,
        )
        ax.plot(
            (0, np.deg2rad(315)),
            (0, np.max(yy)),
            color="lightgray",
            linewidth=1,
            ls="--",
            zorder=0,
        )
        # contours of DE
        cp = ax.contour(theta, r, r, colors="darkgray", levels=c_levels, zorder=1, linewidths=1)
        cl = ax.clabel(cp, inline=True, fmt="%1.1f", colors="dimgrey")
        # threshold efficiency for FBM
        eff_l = np.sqrt((limit) ** 2 + (limit) ** 2 + (limit) ** 2)
        # loop over each data point
        for (c1, c2, c3, eff) in zip(ll_comp1, ll_comp2, ll_comp3, ll_eff):
            # normalize threshold with mean flow becnhmark
            ang = np.arctan2(c1, c2)
            # convert temporal correlation to color
            cmap = cm.get_cmap('plasma_r')
            rgba_color = cmap(norm(c3))
            # relation of b_dir which explains the error
            if abs(c2) > 0:
                exp_err = c2
            elif abs(c2) == 0:
                exp_err = 0
            # diagnose the error
            if abs(c1) <= limit and exp_err > limit and eff > eff_l:
                c = ax.scatter(ang, eff, color=rgba_color, zorder=2)
            elif abs(c1) > limit and exp_err <= limit and eff > eff_l:
                c = ax.scatter(ang, eff, color=rgba_color, zorder=2)
            elif abs(c1) > limit and exp_err > limit and eff > eff_l:
                c = ax.scatter(ang, eff, color=rgba_color, zorder=2)
            # FBM
            elif abs(c1) <= limit and exp_err <= limit and eff > eff_l:
                ax.annotate(
                    "",
                    xytext=(0, 0),
                    xy=(0, eff),
                    arrowprops=dict(facecolor=rgba_color),
                )
                ax.annotate(
                    "",
                    xytext=(0, 0),
                    xy=(np.pi, eff),
                    arrowprops=dict(facecolor=rgba_color),
                )
            # FGM
            elif abs(c1) <= limit and exp_err <= limit and eff <= eff_l:
                c = ax.scatter(ang, eff, color=rgba_color, zorder=2)
        ax.set_rticks([])  # turn default ticks off
        ax.set_rmin(0)
        if eff_max <= 1:
            ax.set_rmax(1)
        elif eff_max > 1:
            ax.set_rmax(ax_lim)
        # turn labels and grid off
        ax.tick_params(
            labelleft=False,
            labelright=False,
            labeltop=False,
            labelbottom=True,
            grid_alpha=0.01,
        )
        ticks_loc = ax.get_xticks().tolist()
        ax.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
        ax.set_xticklabels(["", "", "", "", "", "", "", ""])
        ax.text(
            -0.04,
            0.5,
            r"Comp2 < 0",
            va="center",
            ha="center",
            rotation=90,
            rotation_mode="anchor",
            transform=ax.transAxes,
        )
        ax.text(
            1.04,
            0.5,
            r"Comp2 > 0",
            va="center",
            ha="center",
            rotation=90,
            rotation_mode="anchor",
            transform=ax.transAxes,
        )
        ax.text(
            0.5,
            -0.04,
            r"Comp1 < 0",
            va="center",
            ha="center",
            rotation=0,
            rotation_mode="anchor",
            transform=ax.transAxes,
        )
        ax.text(
            0.5,
            1.04,
            r"Comp1 > 0",
            va="center",
            ha="center",
            rotation=0,
            rotation_mode="anchor",
            transform=ax.transAxes,
        )
        # add colorbar for temporal correlation
        cbar = fig.colorbar(
            dummie_cax, ax=ax, orientation="horizontal", ticks=[1, 0.5, 0], shrink=0.8
        )
        cbar.set_label("Comp3", labelpad=4)
        cbar.set_ticklabels(["1", "0.5", "<0"])
        cbar.ax.tick_params(direction="in")

        # convert to degrees
        phi = np.arctan2(comp1, comp2)
        diag_deg = (phi * (180 / np.pi)) + 135
        diag_deg[diag_deg < 0] = 360 - diag_deg[diag_deg < 0]

        # 1-D density plot
        g = sns.kdeplot(x=diag_deg, color="k", ax=ax1)
        ax1.set_xticks([0, 90, 180, 270, 360])
        ax1.set_xlim(0, 360)
        ax1.set_ylim(0,)
        ax1.set(ylabel="Density", xlabel=r"[$^\circ$]")

        # 2-D density plot
        cmap = cm.get_cmap('plasma_r')
        c3_colors = cmap(norm(comp3))
        g = sns.jointplot(
            x=diag_deg,
            y=eff,
            kind="kde",
            zorder=1,
            n_levels=20,
            cmap="Greys",
            thresh=0.05,
            marginal_kws={"color": "k"},
        ).plot_joint(plt.scatter, c=c3_colors, alpha=0.4, zorder=2)
        g.set_axis_labels(r"[$^\circ$]", r"Eff [-]")
        g.ax_joint.set_xticks([0, 90, 180, 270, 360])
        g.ax_joint.set_xlim(0, 360)
        g.ax_joint.set_ylim(0, ax_lim)
        g.ax_marg_x.set_xticks([0, 90, 180, 270, 360])
        kde_data = g.ax_marg_y.get_lines()[0].get_data()
        kde_xx = kde_data[0]
        kde_yy = kde_data[1]
        norm = matplotlib.colors.Normalize(vmin=0, vmax=ax_lim)
        colors = cm.Greens_r(norm(kde_yy))
        npts = len(kde_xx)
        for i in range(npts - 1):
            g.ax_marg_y.fill_betweenx(
                [kde_yy[i], kde_yy[i + 1]], [kde_xx[i], kde_xx[i + 1]], color=colors[i]
            )
        g.fig.tight_layout()

        return fig, g.fig
