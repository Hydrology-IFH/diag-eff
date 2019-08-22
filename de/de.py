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

import os
import datetime as dt
import numpy as np
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp
from sklearn import linear_model
#from oct2py import octave
from scipy.signal import find_peaks_cwt
import numpy as np
import seaborn as sns
sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})  # controlling figure aesthetics

__title__ = 'diagnostic_model_efficiency'
__version__ = '0.1'
#__build__ = 0x001201
__author__ = 'Robin Schwemmle'
__license__ = 'GNU GPLv3'
#__docformat__ = 'markdown'

#TODO: comparison to NSE and KGE
#TODO: match Qsim to Qobs
#TODO: normalize/standardize bias balance and slope

def import_ts(path, sep=','):
    """
    Import .csv-file with streamflow time series (m3/s).

    Args
    ----------
    path : str
        path to .csv-file which contains time series

    sep : str, default ‘,’
        delimeter to use

    Returns
    ----------
    df_ts : dataframe
        imported time series
    """
    df_ts = pd.read_csv(path, sep=sep, na_values=-9999, parse_dates=True, index_col=0, dayfirst=True)

    return df_ts

def plot_ts(ts):
    """
    Plot time series.

    Args
    ----------
    ts : dataframe
        dataframe with time series
    """
    fig, ax = plt.subplots()
    ax.plot(df_ts.index, df_ts['Qobs'], color='blue')
    ax.set(ylabel='[$\mathregular{m^{3}}$ $\mathregular{s^{-1}}$]',
           xlabel='Time [Days]')

def plot_obs_sim(obs, sim):
    """
    Plot observed and simulated time series.

    Args
    ----------
    obs : series
        observed time series

    sim : series
        simulated time series
    """
    fig, ax = plt.subplots()
    ax.plot(obs.index, obs, color='blue')
    ax.plot(sim.index, sim, color='red')
    ax.set(ylabel='[$\mathregular{m^{3}}$ $\mathregular{s^{-1}}$]',
           xlabel='Time [Days]')

def fdc(Q):
    """
    Generate a flow duration curve for an observed hydrologic time series.

    Args
    ----------
    Q : dataframe
        containing an observed hydrologic time series
    """
    data = Q.dropna()
    data = np.sort(data.values)
    ranks = sp.rankdata(data, method='ordinal')
    ranks = ranks[::-1]
    prob = [100*(ranks[i]/(len(data)+1)) for i in range(len(data))]

    fig, ax = plt.subplots()
    ax.plot(prob, data, color='blue')
    ax.set(ylabel='[mm $\mathregular{d^{-1}}$]',
           xlabel='Exceedence probabilty [%]', yscale='log')

def fdc_obs_sim(df_Q):
    """
    Plotting the flow duration curves of observed and simulated runoff.

    Args
    ----------
    Q : dataframe
        containing time series of Qobs and Qsim
    """
    df_Q = pd.DataFrame(data=Q)
    q_obs = df_Q.sort_values(by=['Qobs'], ascending=True)
    q_sim = df_Q.sort_values(by=['Qsim'], ascending=True)

    # calculate exceedence probability
    ranks_obs = sp.rankdata(q_obs['Qobs'], method='ordinal')
    ranks_obs = ranks_obs[::-1]
    prob_obs = [100*(ranks_obs[i]/(len(q_obs['Qobs'])+1)) for i in range(len(q_obs['Qobs']))]

    ranks_sim = sp.rankdata(q_sim['Qsim'], method='ordinal')
    ranks_sim = ranks_sim[::-1]
    prob_sim = [100*(ranks_sim[i]/(len(q_sim['Qsim'])+1)) for i in range(len(q_sim['Qsim']))]

    fig, ax = plt.subplots()
    ax.plot(prob_obs, q_obs['Qobs'], color='blue', label='Observed')
    ax.plot(prob_sim, q_sim['Qsim'], color='red', label='Simulated')
    ax.set(ylabel='[mm $\mathregular{d^{-1}}$]',
           xlabel='Exceedence probabilty [%]', yscale='log')
    ax.legend(loc=1)

def calc_fdc_bias_balance(obs, sim):
    """
    Calculate bias balance.

    Args
    ----------
    obs : series
        observed time series

    sim : series
        simulated time series

    Returns
    ----------
    sum_diff : float
        bias balance
    """
    sim_obs_diff = sim - obs
    sum_diff = np.sum(sim_obs_diff)

    return sum_diff

def calc_fdc_bias_slope(obs, sim):
    """
    Calculate slope of bias balance.

    Args
    ----------
    obs : array_like
        observed time series

    sim : array_like
        simulated time series

    Returns
    ----------
    slope : float
        slope of linear regression

    score : float
        score of linear regression model
    """
    y = sim - obs
    x = np.arange(len(y))

    lm = linear_model.LinearRegression()
    reg = lm.fit(x, y)
    y_reg = reg.predict(x)
    bias_slope = reg.coef_[0]
    score = reg.score(x, y)

    fig, ax = plt.subplots()
    ax.plot(x, y, 'b.', markersize=12)
    ax.plot(x, y_reg, 'r-')
    ax.set(ylabel='[mm $\mathregular{d^{-1}}$]',
           xlabel='Exceedence probabilty [%]', yscale='log')

    return bias_slope, score

def calc_temp_cor(obs, sim):
    """
    Calculate temporal correlation between observed and simulated
    time series.

    Args
    ----------
    obs : array_like
        observed time series

    sim : array_like
        simulated time series

    Returns
    ----------
    temp_cor : float
        rank correlation between observed and simulated time series
    """
    r = sp.spearmanr(obs, sim)
    temp_cor = r[0]

    return temp_cor

def calc_de(obs, sim):
    """
    Calculate Diagnostic-Efficiency (KGE).

    Args
    ----------
    obs : array_like
        observed time series

    sim : array_like
        simulated time series

    Returns
    ----------
    sig : float
        diagnostic efficiency measure
    """
    bias_bal = calc_fdc_bias_balance(obs, sim)
    bias_slope, _ = calc_fdc_bias_slope(obs, sim)
    temp_cor = calc_temp_cor(obs, sim)
    sig = 1 - np.sqrt((bias_bal - 1)**2 + (bias_slope - 1)**2  + (temp_cor - 1)**2)

    return sig

def calc_kge(obs, sim):
    """
    Calculate Kling-Gupta-Efficiency (KGE).

    Args
    ----------
    obs : array_like
        observed time series

    sim : array_like
        simulated time series

    Returns
    ----------
    sig : float
        Kling-Gupta-Efficiency measure
    """
    obs_mean = np.mean(obs)
    sim_mean = np.mean(sim)
    kge_beta = sim_mean/obs_mean

    obs_std = np.std(obs)
    sim_std = np.std(sim)
    obs_cv = obs_std/obs_mean
    sim_cv = sim_std/sim_mean
    kge_gamma = sim_cv/obs_cv

    temp_cor = calc_temp_cor(obs, sim)

    sig = 1 - np.sqrt((kge_beta - 1)**2 + (kge_gamma- 1)**2  + (temp_cor - 1)**2)

    return sig

def calc_nse(obs, sim):
    """
    Calculate Nash-Sutcliffe-Efficiency (NSE).

    Args
    ----------
    obs : array_like
        observed time series

    sim : array_like
        simulated time series

    Returns
    ----------
    sig : float
        Nash-Sutcliffe-Efficiency measure
    """
    sim_obs_diff = np.sum((sim - obs)**2)
    obs_mean = np.mean(obs)
    obs_diff_mean = np.sum((obs - obs_mean)**2)
    sig = 1 - (sim_obs_diff/obs_diff_mean)

    return sig

def vis2d_de(obs, sim):
    """
    2-D visualization of Diagnostic-Efficiency (DE)

    Args
    ----------
    obs : array_like
        observed time series

    sim : array_like
        simulated time series
    """
    bias_bal = calc_fdc_bias_balance(obs, sim)
    bias_slope, _ = calc_fdc_bias_slope(obs, sim)
    temp_cor = calc_temp_cor(obs, sim)
    sig = 1 - np.sqrt((bias_bal - 1)**2 + (bias_slope - 1)**2  + (temp_cor - 1)**2)

    y = np.array([0, bias_bal])
    x = np.array([0, bias_slope])
    
    # convert temporal correlation to color
    norm = matplotlib.colors.Normalize(vmin=-1.0, vmax=1.0)
    rgba_color = cm.RdYlGn(norm(temp_cor), bytes=True)

    fig, ax = plt.subplots()
    ax.plot(x, y, color=rgba_color)
    ax.set_xlim([-1, 10])
    ax.set_ylim([-1, 10])
    ax.set(ylabel='Bias balance',
           xlabel='Bias slope')

def vis2d_kge(obs, sim):
    """
    2-D visualization of Kling-Gupta-Efficiency (KGE)

    Args
    ----------
    obs : array_like
        observed time series

    sim : array_like
        simulated time series
    """
    obs_mean = np.mean(obs)
    sim_mean = np.mean(sim)
    kge_beta = sim_mean/obs_mean

    obs_std = np.std(obs)
    sim_std = np.std(sim)
    obs_cv = obs_std/obs_mean
    sim_cv = sim_std/sim_mean
    kge_gamma = sim_cv/obs_cv

    temp_cor = calc_temp_cor(obs, sim)

    sig = 1 - np.sqrt((kge_beta - 1)**2 + (kge_gamma- 1)**2  + (temp_cor - 1)**2)

    y = np.array([0, kge_beta])
    x = np.array([0, kge_gamma])
    
    # convert temporal correlation to color
    norm = matplotlib.colors.Normalize(vmin=-1.0, vmax=1.0)
    rgba_color = cm.RdYlGn(norm(temp_cor), bytes=True)

    fig, ax = plt.subplots()
    ax.plot(x, y, color=rgba_color)
    ax.set_xlim([-1, 10])
    ax.set_ylim([-1, 10])
    ax.set(ylabel='Beta',
           xlabel='Gamma')

def pos_shift_obs(obs):
    """
    Precipitation overestimation

    Args
    ----------
    obs : array_like
        observed time series

    Returns
    ----------
    shift_obs : dataframe
        time series with positve offset
    """
    shift_obs = obs + 1
    
    return shift_obs

def neg_shift_obs(obs):
    """
    Precipitation underestimation

    Args
    ----------
    obs : array_like
        observed time series

    Returns
    ----------
    df_ts : dataframe
        imported time series
    """
    shift_neg = obs - 1
    shift_neg[shift_neg < 0] = 0
    
    return shift_neg

def smooth_obs(obs):
    """
    Underestimate high flows - Overestimate low flows

    Args
    ----------
    obs : array_like
        observed time series

    Returns
    ----------
    smoothed_obs : series
        smoothed time series
    """
    smoothed_obs = obs.rolling(window=5).mean()
    smoothed_obs.fillna(method='bfill', inplace=True)
    
    return smoothed_obs

def disaggregate_obs():
    """
    Overestimate high flows - Underestimate low flows.

    Increase max values and decrease min values by maintaining balance within
    rolling window.

    Args
    ----------
    path : str
        path to file with meta informations on the catchments

    sep : str, default ‘,’
        delimeter to use

    Returns
    ----------
    df_ts : dataframe
        imported time series
    """
    pass

def sort_obs(df_Q):
    """
    Sort observed streamflow time series.

    Args
    ----------
    path : str
        path to file with meta informations on the catchments

    sep : str, default ‘,’
        delimeter to use

    Returns
    ----------
    df_ts : dataframe
        imported time series
    """
    df_Q = pd.DataFrame(data=Q)
    obs_sort = df_Q.sort_values(by=['Qobs'], ascending=True)
    
    return obs_sort

def _datacheck_peakdetect(x_axis, y_axis):
    if x_axis is None:
        x_axis = range(len(y_axis))
    
    if len(y_axis) != len(x_axis):
        raise ValueError( 
                "Input vectors y_axis and x_axis must have same length")
    
    #needs to be a numpy array
    y_axis = np.array(y_axis)
    x_axis = np.array(x_axis)
    
    return x_axis, y_axis

def peakdetect(y_axis, x_axis = None, lookahead = 200, delta=0):
    """
    Converted from/based on a MATLAB script at: 
    http://billauer.co.il/peakdet.html
    
    https://gist.github.com/sixtenbe/1178136
    
    Function for detecting local maxima and minima in a signal.
    Discovers peaks by searching for values which are surrounded by lower
    or larger values for maxima and minima respectively
    
    Args
    ----------
    y_axis : A list containing the signal over which to find peaks
    
    x_axis : A x-axis whose values correspond to the y_axis list and is used
        in the return to specify the position of the peaks. If omitted an
        index of the y_axis is used.
        (default: None)
    
    lookahead : distance to look ahead from a peak candidate to determine if
        it is the actual peak
        (default: 200) 
        '(samples / period) / f' where '4 >= f >= 1.25' might be a good value
    
    delta : this specifies a minimum difference between a peak and
        the following points, before a peak may be considered a peak. Useful
        to hinder the function from picking up false peaks towards to end of
        the signal. To work well delta should be set to delta >= RMSnoise * 5.
        (default: 0)
            When omitted delta function causes a 20% decrease in speed.
            When used Correctly it can double the speed of the function
    
    Returns
    ----------
    
    max_peaks : two lists [max_peaks, min_peaks] containing the positive and
        negative peaks respectively. Each cell of the lists contains a tuple
        of: (position, peak_value) 
        to get the average peak value do: np.mean(max_peaks, 0)[1] on the
        results to unpack one of the lists into x, y coordinates do: 
        x, y = zip(*max_peaks)
        
    min_peaks : two lists [max_peaks, min_peaks] containing the positive and
        negative peaks respectively. Each cell of the lists contains a tuple
        of: (position, peak_value) 
        to get the average peak value do: np.mean(max_peaks, 0)[1] on the
        results to unpack one of the lists into x, y coordinates do: 
        x, y = zip(*max_peaks)
    """
    max_peaks = []
    min_peaks = []
    dump = []   #Used to pop the first hit which almost always is false
       
    # check input data
    x_axis, y_axis = _datacheck_peakdetect(x_axis, y_axis)
    # store data length for later use
    length = len(y_axis)
    
    #perform some checks
    if lookahead < 1:
        raise ValueError("Lookahead must be '1' or above in value")
    if not (np.isscalar(delta) and delta >= 0):
        raise ValueError("delta must be a positive number")
    
    #maxima and minima candidates are temporarily stored in
    #mx and mn respectively
    mn, mx = np.Inf, -np.Inf
    
    #Only detect peak if there is 'lookahead' amount of points after it
    for index, (x, y) in enumerate(zip(x_axis[:-lookahead], 
                                        y_axis[:-lookahead])):
        if y > mx:
            mx = y
            mxpos = x
        if y < mn:
            mn = y
            mnpos = x
        
        ####look for max####
        if y < mx-delta and mx != np.Inf:
            #Maxima peak candidate found
            #look ahead in signal to ensure that this is a peak and not jitter
            if y_axis[index:index+lookahead].max() < mx:
                max_peaks.append([mxpos, mx])
                dump.append(True)
                #set algorithm to only find minima now
                mx = np.Inf
                mn = np.Inf
                if index+lookahead >= length:
                    #end is within lookahead no more peaks can be found
                    break
                continue
            #else:  #slows shit down this does
            #    mx = ahead
            #    mxpos = x_axis[np.where(y_axis[index:index+lookahead]==mx)]
        
        ####look for min####
        if y > mn+delta and mn != -np.Inf:
            #Minima peak candidate found 
            #look ahead in signal to ensure that this is a peak and not jitter
            if y_axis[index:index+lookahead].min() > mn:
                min_peaks.append([mnpos, mn])
                dump.append(False)
                #set algorithm to only find maxima now
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
        #no peaks were found, should the function return empty lists?
        pass
        
    return max_peaks, min_peaks


if __name__ == "__main__":
    path = '/Users/robinschwemmle/Desktop/PhD/diagnostic_model_efficiency/data/9960682_Q_1970_2012.csv'
    df_ts = import_ts(path, sep=';')
    ax = sns.lineplot(data=df_ts, x=df_ts.index, y='Qobs')
    test = df_ts['Qobs'].rolling(window=5).mean()
    test2 = df_ts['Qobs'].rolling(window=5).max()
    test3 = df_ts['Qobs'].rolling(window=5).min()
    test1 = test.interpolate(method='slinear', axis=0).ffill().bfill()
    arr = df_ts['Qobs'].values
    max_peaks, min_peaks = peakdetect(arr, lookahead=100)