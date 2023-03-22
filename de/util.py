# -*- coding: utf-8 -*-

"""
de.util
~~~~~~~~~~~

:2021 by Robin Schwemmle.
:license: GNU GPLv3, see LICENSE for more details.
"""

import pandas as pd
import numpy as np
from matplotlib import cm
import matplotlib.dates as mdates
import scipy as sp
import matplotlib
import seaborn as sns
matplotlib.use("agg")
import matplotlib.pyplot as plt  # noqa: E402

# controlling figure aesthetics
sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})


_mmd = r"Q [mm $d^{-1}$]"
_m3s = r"Q [$m^{3}$ $s^{-1}$]"
_q_lab = _mmd
_sim_lab = "Simulated"


def import_ts(path, sep=","):
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
    df_ts = pd.read_csv(
        path, sep=sep, na_values=-9999, parse_dates=True, index_col=0, dayfirst=True
    )
    # drop nan values
    df_ts = df_ts.dropna()

    return df_ts

def import_camels_ts(path, sep=r"\s+", catch_area=None):
    r"""
    Import .txt-file with streamflow time series from CAMELS dataset (cubic
    feet per second).

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
    df_ts = pd.read_csv(
        path, sep=sep, na_values=-999, header=None, parse_dates=[[1, 2, 3]], index_col=0
    )
    df_ts.drop(df_ts.columns[[0, 2]], axis=1, inplace=True)
    df_ts.columns = ["Qobs"]
    df_ts = df_ts.dropna()

    # convert to m3/s
    df_ts["Qobs"] = df_ts["Qobs"].values / 35.3
    # convert to mm/day
    df_ts["Qobs"] = (df_ts["Qobs"].values * (24 * 60 * 60) * 1000) / (
        catch_area * 1000 * 1000
    )

    return df_ts


def import_camels_obs_sim(path, sep=r"\s+"):
    r"""
    Import .txt-file containing observed and simulated streamflow time series
    from CAMELS dataset (mm per day).

    Parameters
    ----------
    path : str
        Path to .csv-file which contains time series

    sep : str, optional
        Delimeter to use. The default is r"\s+".

    Returns
    ----------
    obs_sim : dataframe
        observed and simulated time series in mm/day
    """
    obs_sim = pd.read_csv(
        path, sep=sep, na_values=-999, header=0, parse_dates=[[0, 1, 2]], index_col=0
    )
    obs_sim.drop(
        ["HR", "SWE", "PRCP", "RAIM", "TAIR", "PET", "ET"], axis=1, inplace=True
    )
    obs_sim.columns = ["Qsim", "Qobs"]
    obs_sim = obs_sim[["Qobs", "Qsim"]]
    obs_sim = obs_sim.dropna()

    return obs_sim


def plot_ts(ts):
    """Plot time series.

    Parameters
    ----------
    ts : dataframe
        Dataframe which contains time series

    Returns
    ----------
    fig : Figure
        time series plot

    >>> from de import util
    >>> import pandas as pd
    >>> date_rng = pd.date_range(start='1/1/2018', periods=11)
    >>> arr = np.array([1.5, 1, 0.8, 0.85, 1.5, 2, 2.5, 3.5, 1.8, 1.5, 1.2])
    >>> ts = pd.Series(data=arr, index=date_rng)
    >>> util.plot_ts(ts)
    """
    fig, ax = plt.subplots()
    ax.plot(ts.index, ts.values, color="blue")
    ax.set(ylabel=_q_lab, xlabel="Time [Year]")
    ax.set_ylim(0,)
    ax.set_xlim(ts.index[0], ts.index[-1])
    years_5 = mdates.YearLocator(5)
    years = mdates.YearLocator()
    yearsFmt = mdates.DateFormatter("%Y")
    ax.xaxis.set_major_locator(years_5)
    ax.xaxis.set_major_formatter(yearsFmt)
    ax.xaxis.set_minor_locator(years)

    return fig


def plot_obs_sim(obs, sim):
    """Plot observed and simulated time series.

    Parameters
    ----------
    obs : series
        observed time series

    sim : series
        simulated time series

    Returns
    ----------
    fig : Figure
        time series plot

    Examples
    --------
    Provide arrays with equal length

    >>> from de import util
    >>> import pandas as pd
    >>> date_rng = pd.date_range(start='1/1/2018', periods=11)
    >>> obs = np.array([1.5, 1, 0.8, 0.85, 1.5, 2, 2.5, 3.5, 1.8, 1.5, 1.2])
    >>> ts_obs = pd.Series(data=obs, index=date_rng)
    >>> sim = np.array([1.4, .9, 1, 0.95, 1.4, 2.1, 2.6, 3.6, 1.9, 1.4, 1.1])
    >>> ts_sim = pd.Series(data=sim, index=date_rng)
    >>> util.plot_obs_sim(ts_obs, ts_sim)
    """
    fig, ax = plt.subplots(figsize=(3, 1.2))
    # observed time series
    ax.plot(obs.index, obs, lw=2, color="blue", label="Observed", alpha=0.8)
    # simulated time series
    ax.plot(sim.index, sim, lw=1, ls="-.", color="red", label=_sim_lab, alpha=0.9)
    ax.set(ylabel=_q_lab, xlabel="Time")
    ax.set_ylim(0,)
    ax.set_xlim(obs.index[0], obs.index[-1])

    # format the ticks
    years_20 = mdates.YearLocator(20)
    years_5 = mdates.YearLocator(5)
    yearsFmt = mdates.DateFormatter("%Y")
    ax.xaxis.set_major_locator(years_20)
    ax.xaxis.set_major_formatter(yearsFmt)
    ax.xaxis.set_minor_locator(years_5)

    return fig


def fdc_obs_sim_ax(obs, sim, ax, fig_num):  # pragma: no cover
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
    obs_sim = pd.DataFrame(index=obs.index, columns=["obs", "sim"])
    obs_sim.loc[:, "obs"] = obs.values
    obs_sim.loc[:, "sim"] = sim.values
    obs = obs_sim.sort_values(by=["obs"], ascending=True)
    sim = obs_sim.sort_values(by=["sim"], ascending=True)

    # calculate exceedence probability
    ranks_obs = sp.stats.rankdata(obs["obs"], method="ordinal")
    ranks_obs = ranks_obs[::-1]
    prob_obs = [(ranks_obs[i] / (len(obs["obs"]) + 1)) for i in range(len(obs["obs"]))]

    ranks_sim = sp.stats.rankdata(sim["sim"], method="ordinal")
    ranks_sim = ranks_sim[::-1]
    prob_sim = [(ranks_sim[i] / (len(sim["sim"]) + 1)) for i in range(len(sim["sim"]))]

    ax.plot(prob_obs, obs["obs"], lw=2, color="blue", alpha=0.5, label="Observed")
    ax.plot(prob_sim, sim["sim"], lw=2, ls="-.", color="red", label="Manipulated")
    ax.text(0.96, 0.95, fig_num, transform=ax.transAxes, ha="right", va="top")
    ax.set(yscale="log")
    ax.set_xlim(0, 1)


def plot_obs_sim_ax(obs, sim, ax, fig_num):  # pragma: no cover
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
    ax.plot(obs.index, obs, lw=2, color="blue", label="Observed", alpha=0.8)
    ax.plot(
        sim.index, sim, lw=1.5, ls="-.", color="red", alpha=0.9, label=_sim_lab
    )  # simulated time series
    ax.set_ylim(0,)
    ax.set_xlim(obs.index[0], obs.index[-1])
    ax.text(0.96, 0.95, fig_num, transform=ax.transAxes, ha="right", va="top")
    # format the ticks
    years_20 = mdates.YearLocator(20)
    years_5 = mdates.YearLocator(5)
    yearsFmt = mdates.DateFormatter("%Y")
    ax.xaxis.set_major_locator(years_20)
    ax.xaxis.set_major_formatter(yearsFmt)
    ax.xaxis.set_minor_locator(years_5)


def diag_polar_plot_multi_fc(
    brel_mean, temp_cor, eff_de, b_dir, phi, b_hf, b_lf, b_tot,
    err_hf, err_lf, fc, l=0.05, ax_lim=1.2
):  # pragma: no cover
    r"""Multiple polar plot of Diagnostic-Efficiency (DE)

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
        angle as 1-D array

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

    fc : list
        figure captions

    l : float, optional
        Threshold for which diagnosis can be made. The default is 0.05.

    Returns
    ----------
    fig : Figure
        diagnostic polar plot

    Notes
    ----------
    .. math::

        \varphi = arctan2(\overline{B_{rel}}, B_{slope})
    """
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
    yy = np.arange(0.01, ax_lim, delta)
    c_levels = np.arange(0, ax_lim, 0.2)

    len_yy = 360
    # len_yy1 = 90

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
    cl = ax.clabel(cp, inline=False, fmt="%1.1f", colors="dimgrey")
    # threshold efficiency for FBM
    eff_l = np.sqrt((l) ** 2 + (l) ** 2 + (l) ** 2)
    # loop over each data point
    zz = zip(ll_brel_mean, ll_temp_cor, ll_eff, ll_b_dir, ll_phi, ll_b_hf,
             ll_b_lf, ll_b_tot, ll_err_hf, ll_err_lf, fc)
    for (bm, r, eff, bd, ang, bhf, blf, btot, errhf, errlf, txt) in zz:
        # convert temporal correlation to color
        rgba_color = cm.plasma_r(norm(r))
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
        if abs(bm) <= l and exp_err > l and eff > eff_l:
            c0 = ax.scatter(ang, eff, marker=xy1, s=s1 * 50, facecolor=rgba_color, zorder=3)
            c1 = ax.scatter(ang, eff, marker=xy2, s=s2 * 50, facecolor=rgba_color, zorder=3)
            c = ax.scatter(ang, eff, s=4, facecolor=rgba_color, zorder=3)
            c2 = ax.scatter(ang, eff, s=1, facecolor='grey', zorder=3)
            ax.annotate(
                txt,
                xy=(ang, eff),
                color="black",
                xytext=(-8, 0),
                textcoords="offset points",
                ha="center",
                va="center",
            )
        elif abs(bm) > l and exp_err <= l and eff > eff_l:
            c0 = ax.scatter(ang, eff, marker=xy1, s=s1 * 50, facecolor=rgba_color, zorder=3)
            c1 = ax.scatter(ang, eff, marker=xy2, s=s2 * 50, facecolor=rgba_color, zorder=3)
            c = ax.scatter(ang, eff, s=4, facecolor=rgba_color, zorder=3)
            c2 = ax.scatter(ang, eff, s=1, facecolor='grey', zorder=3)
            ax.annotate(
                txt,
                xy=(ang, eff),
                color="black",
                xytext=(-8, 0),
                textcoords="offset points",
                ha="center",
                va="center",
            )
        elif abs(bm) > l and exp_err > l and eff > eff_l:
            c0 = ax.scatter(ang, eff, marker=xy1, s=s1 * 50, facecolor=rgba_color, zorder=3)
            c1 = ax.scatter(ang, eff, marker=xy2, s=s2 * 50, facecolor=rgba_color, zorder=3)
            c = ax.scatter(ang, eff, s=4, facecolor=rgba_color, zorder=3)
            c2 = ax.scatter(ang, eff, s=1, facecolor='grey', zorder=3)
            ax.annotate(
                txt,
                xy=(ang, eff),
                color="black",
                xytext=(-8, 0),
                textcoords="offset points",
                ha="center",
                va="center",
            )

        # FBM
        elif abs(bm) <= l and exp_err <= l and eff > eff_l:
            ax.annotate(
                "",
                xytext=(0, 0),
                xy=(0, eff),
                arrowprops=dict(edgecolor=rgba_color, facecolor="black", lw=3),
                zorder=2,
            )
            ax.annotate(
                "",
                xytext=(0, 0),
                xy=(np.pi, eff),
                arrowprops=dict(edgecolor=rgba_color, facecolor="black", lw=3),
                zorder=2,
            )
            ax.annotate(
                txt,
                xy=(ang, eff),
                color="black",
                xytext=(-8, 0),
                textcoords="offset points",
                ha="center",
                va="center",
            )
        # FGM
        elif abs(bm) <= l and exp_err <= l and eff <= eff_l:
            c = ax.scatter(ang, eff, color=rgba_color, zorder=3)
            d = ax.scatter(ang, eff, color="grey", marker=".", zorder=4)
            ax.annotate(
                txt,
                xy=(ang, eff),
                color="black",
                xytext=(-6, 0),
                textcoords="offset points",
                ha="center",
                va="center",
            )

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
    ax.set_rmax(ax_lim)
    ax.tick_params(
        labelleft=False,
        labelright=False,
        labeltop=False,
        labelbottom=True,
        grid_alpha=0.01,
    )  # turn labels and grid off
    ax.set(xticklabels=[])
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
    cbar.ax.tick_params(direction="in", labelsize=10)

    return fig


def polar_plot_multi_fc(
    kge_beta, alpha_or_gamma, kge_r, sig_kge, fc, ax_lim=-0.6
):  # pragma: no cover
    """Polar plot of Kling-Gupta efficiency (KGE) with multiple
    values.

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

    Returns
    ----------
    fig : Figure
        diagnostic polar plot
    """
    ll_kge_beta = kge_beta.tolist()
    ll_ag = alpha_or_gamma.tolist()
    ll_kge_r = kge_r.tolist()
    ll_eff = sig_kge.tolist()

    # convert temporal correlation to color
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1.0)

    delta = 0.01  # for spacing

    # determine axis limits
    yy = np.arange(ax_lim, 1.01, delta)[::-1]
    c_levels = np.arange(ax_lim, 1, 0.2)

    len_yy = 360
    # len_yy1 = 90

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
        (1, np.deg2rad(45)),
        (1, np.min(yy)),
        color="lightgray",
        linewidth=1,
        ls="--",
        zorder=0,
    )
    ax.plot(
        (1, np.deg2rad(135)),
        (1, np.min(yy)),
        color="lightgray",
        linewidth=1,
        ls="--",
        zorder=0,
    )
    ax.plot(
        (1, np.deg2rad(225)),
        (1, np.min(yy)),
        color="lightgray",
        linewidth=1,
        ls="--",
        zorder=0,
    )
    ax.plot(
        (1, np.deg2rad(315)),
        (1, np.min(yy)),
        color="lightgray",
        linewidth=1,
        ls="--",
        zorder=0,
    )
    # contours of KGE
    cp = ax.contour(theta, r, r, colors="darkgray", levels=c_levels, zorder=1, linewidths=1)
    cl = ax.clabel(cp, inline=False, fmt="%1.1f", colors="dimgrey")
    # loop over each data point
    for (b, ag, r, eff, txt) in zip(ll_kge_beta, ll_ag, ll_kge_r, ll_eff, fc):
        ang = np.arctan2(b - 1, ag - 1)
        # convert temporal correlation to color
        rgba_color = cm.plasma_r(norm(r))
        c = ax.scatter(ang, eff, s=75, color=rgba_color, zorder=2)
        d = ax.scatter(ang, eff, color="grey", marker=".", zorder=4)
        ax.annotate(
            txt,
            xy=(ang, eff),
            color="black",
            xytext=(8, 0),
            textcoords="offset points",
            ha="center",
            va="center",
        )
    ax.tick_params(
        labelleft=False,
        labelright=False,
        labeltop=False,
        labelbottom=True,
        grid_alpha=0.01,
    )  # turn labels and grid off
    ax.text(
        -0.11,
        0.5,
        "Variability underestimation",
        va="center",
        ha="center",
        rotation=90,
        rotation_mode="anchor",
        transform=ax.transAxes,
    )
    ax.text(
        -0.06,
        0.5,
        r"($\alpha$ - 1) < 0",
        va="center",
        ha="center",
        rotation=90,
        rotation_mode="anchor",
        transform=ax.transAxes,
    )
    ax.text(
        1.09,
        0.5,
        "Variability overestimation",
        va="center",
        ha="center",
        rotation=90,
        rotation_mode="anchor",
        transform=ax.transAxes,
    )
    ax.text(
        1.04,
        0.5,
        r"($\alpha$ - 1) > 0",
        va="center",
        ha="center",
        rotation=90,
        rotation_mode="anchor",
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        -0.09,
        "Mean underestimation",
        va="center",
        ha="center",
        rotation=0,
        rotation_mode="anchor",
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        -0.04,
        r"($\beta$ - 1) < 0",
        va="center",
        ha="center",
        rotation=0,
        rotation_mode="anchor",
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        1.09,
        "Mean overestimation",
        va="center",
        ha="center",
        rotation=0,
        rotation_mode="anchor",
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        1.04,
        r"($\beta$ - 1) > 0",
        va="center",
        ha="center",
        rotation=0,
        rotation_mode="anchor",
        transform=ax.transAxes,
    )
    ax.set(xticklabels=[])
    ax.set_rticks([])  # turn default ticks off
    ax.set_rmin(1)
    ax.set_rmax(ax_lim)
    # add colorbar for temporal correlation
    cbar = fig.colorbar(
        dummie_cax, ax=ax, orientation="horizontal", ticks=[1, 0.5, 0], shrink=0.8
    )
    cbar.set_label("r [-]\nTiming error", labelpad=4)
    cbar.set_ticklabels(["1", "0.5", "<0"])
    cbar.ax.tick_params(direction="in", labelsize=10)

    return fig
