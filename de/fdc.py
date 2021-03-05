# -*- coding: utf-8 -*-

"""
de.fdc
~~~~~~~~~~~
Flow duration curve
:2021 by Robin Schwemmle.
:license: GNU GPLv3, see LICENSE for more details.
"""

import numpy as np

# RunTimeWarning will not be displayed (division by zeros or NaN values)
np.seterr(divide="ignore", invalid="ignore")
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns

# controlling figure aesthetics
sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})
sns.set_context("paper", font_scale=1.5)


_mmd = r"[mm $d^{-1}$]"
_m3s = r"[$m^{3}$ $s^{-1}$]"
_q_lab = _mmd
_sim_lab = "Simulated"


def calc_fdc(ts, prob):
    """
    Calculate numeric values of the flow duration curve for a single
    hydrologic time series.

    Parameters
    ----------
    ts : (N,)array_like
        hydrologic time series

    prob : (N,)array_like
        Quantile or sequence of quantiles to compute.

    Returns
    ----------
    fdc_vals : (N,)array_like
        numeric values to plot flow duration curve
    """
    data = ts[np.logical_not(np.isnan(ts))]
    data = np.sort(data)  # sort values by ascending order
    fdc_vals = np.quantile(data, prob)[::-1]

    return fdc_vals


def fdc(ts):
    """
    Flow duration curve for a single hydrologic time series.

    Parameters
    ----------
    ts : series
        Containing a hydrologic time series

    Returns
    ----------
    fig : Figure
        Returns a single figure containing flow duration curve.

    Examples
    --------
    Provide arrays with equal length

    >>> from de import fdc
    >>> import pandas as pd
    >>> date_rng = pd.date_range(start='1/1/2018', periods=11)
    >>> arr = np.array([1.5, 1, 0.8, 0.85, 1.5, 2, 2.5, 3.5, 1.8, 1.5, 1.2])
    >>> ts = pd.Series(data=arr, index=date_rng)
    >>> fdc.fdc(ts)
    """
    data = ts.dropna()
    data = np.sort(data.values)  # sort values by ascending order
    ranks = sp.stats.rankdata(data, method="ordinal")  # rank data
    ranks = ranks[::-1]
    # calculate exceedence probability
    prob = [(ranks[i] / (len(data) + 1)) for i in range(len(data))]

    fig, ax = plt.subplots()
    ax.plot(prob, data, color="blue")
    ax.set(ylabel=_q_lab, xlabel="Exceedence probabilty [-]", yscale="log")
    fig.subplots_adjust(left=0.2)

    return fig


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

    Returns
    ----------
    fig : Figure
        Returns a single figure containing two flow duration curves.

    Examples
    --------
    Provide arrays with equal length

    >>> from de import fdc
    >>> import pandas as pd
    >>> date_rng = pd.date_range(start='1/1/2018', periods=11)
    >>> obs = np.array([1.5, 1, 0.8, 0.85, 1.5, 2, 2.5, 3.5, 1.8, 1.5, 1.2])
    >>> ts_obs = pd.Series(data=obs, index=date_rng)
    >>> sim = np.array([1.4, .9, 1, 0.95, 1.4, 2.1, 2.6, 3.6, 1.9, 1.4, 1.1])
    >>> ts_sim = pd.Series(data=sim, index=date_rng)
    >>> fdc.fdc_obs_sim(ts_obs, ts_sim)
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

    fig, ax = plt.subplots()
    ax.plot(prob_obs, obs["obs"], color="blue", lw=2, label="Observed")
    ax.plot(prob_sim, sim["sim"], color="red", lw=1, ls="-.", label=_sim_lab, alpha=0.8)
    ax.set(ylabel=_q_lab, xlabel="Exceedence probabilty [-]", yscale="log")
    ax.legend(loc=1)
    ax.set_xlim(0, 1)

    fig.subplots_adjust(left=0.2)

    return fig

def fdc_sort_obs(obs, sim):
    """
    Plotting the flow duration curves of two hydrologic time series (e.g.
    observed streamflow and simulated streamflow) sorted by observations.

    Parameters
    ----------
    obs : series
        observed time series
    sim : series
        simulated time series

    Returns
    ----------
    fig : Figure
        Returns a single figure containing two flow duration curves.

    Examples
    --------
    Provide arrays with equal length

    >>> from de import fdc
    >>> import pandas as pd
    >>> date_rng = pd.date_range(start='1/1/2018', periods=11)
    >>> obs = np.array([1.5, 1, 0.8, 0.85, 1.5, 2, 2.5, 3.5, 1.8, 1.5, 1.2])
    >>> ts_obs = pd.Series(data=obs, index=date_rng)
    >>> sim = np.array([1.4, .9, 1, 0.95, 1.4, 2.1, 2.6, 3.6, 1.9, 1.4, 1.1])
    >>> ts_sim = pd.Series(data=sim, index=date_rng)
    >>> fdc.fdc_obs_sim(ts_obs, ts_sim)
    """
    obs_sim = pd.DataFrame(index=obs.index, columns=["obs", "sim"])
    obs_sim.loc[:, "obs"] = obs.values
    obs_sim.loc[:, "sim"] = sim.values
    obs_sim.sort_values(by=['obs'], ascending=True)

    # calculate exceedence probability
    ranks_obs = sp.stats.rankdata(obs_sim["obs"], method="ordinal")
    ranks_obs = ranks_obs[::-1]
    prob_obs = [(ranks_obs[i] / (len(obs_sim["obs"]) + 1)) for i in range(len(obs_sim["obs"]))]

    fig, ax = plt.subplots()
    ax.plot(prob_obs, obs_sim["obs"], color="blue", lw=2, label="Observed")
    ax.plot(prob_obs, obs_sim["sim"], color="red", lw=1, ls="-.", label=_sim_lab, alpha=0.8)
    ax.set(ylabel=_q_lab, xlabel="Exceedence probabilty [-]", yscale="log")
    ax.legend(loc=1)
    ax.set_xlim(0, 1)

    fig.subplots_adjust(left=0.2)

    return fig
