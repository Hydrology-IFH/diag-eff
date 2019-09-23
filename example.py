#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from de import de
from de import util
import seaborn as sns
# controlling figure aesthetics
sns.set_style('ticks', {'xtick.major.size': 8, 'ytick.major.size': 8})

if __name__ == "__main__":
    path = '/Users/robinschwemmle/Desktop/PhD/diagnostic_model_efficiency/examples/data/9960682_Q_1970_2012.csv'
#    path = '/Users/robo/Desktop/PhD/de/examples/data/9960682_Q_1970_2012.csv'

    fig_num_fdc = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)', '(j)']
    fig_fdc, axes_fdc = plt.subplots(2, 5, sharey=True, sharex=True, figsize=(14,6))
    fig_fdc.text(0.5, 0.04, 'Exceedence probabilty [-]', ha='center', va='center')
    fig_fdc.text(0.09, 0.5, r'[$m^{3}$ $s^{-1}$]', ha='center', va='center', rotation='vertical')

    fig_num_ts = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)', '(j)', '(k)', '(l)', '(m)', '(n)']
    fig_ts, axes_ts = plt.subplots(3, 5, sharey=True, sharex=True, figsize=(14,9))
    fig_ts.text(0.5, 0.06, 'Time [Years]', ha='center', va='center')
    fig_ts.text(0.1, 0.5, r'[$m^{3}$ $s^{-1}$]', ha='center', va='center', rotation='vertical')
    axes_ts[2,4].remove()

    # dataframe efficiency measures
    idx = ['1', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n']
    cols = ['brel_mean', 'b_area', 'temp_cor', 'de', 'b_dir', 'b_slope', 'diag', 'kge', 'alpha', 'beta', 'nse']
    df_es = pd.DataFrame(index=idx, columns=cols, dtype=np.float64)

    # import observed time series
    df_ts = util.import_ts(path, sep=';')
    de.plot_ts(df_ts)

    ### perfect simulation ###
    obs_sim = pd.DataFrame(index=df_ts.index, columns=['Qobs', 'Qsim'])
    obs_sim.loc[:, 'Qobs'] = df_ts.loc[:, 'Qobs']
    obs_sim.loc[:, 'Qsim'] = df_ts.loc[:, 'Qobs']  # observed time series

    # make numpy arrays
    obs_arr = obs_sim['Qobs'].values  # observed time series
    sim_arr = obs_sim['Qsim'].values  # manipulated time series

    # mean relative bias
    brel_mean = de.calc_brel_mean(obs_arr, sim_arr)
    df_es.iloc[0, 0] = brel_mean
    # remaining relative bias
    brel_rest = de.calc_brel_rest(obs_arr, sim_arr)
    # area of relative remaing bias
    b_area = de.calc_b_area(brel_rest)
    df_es.iloc[0, 1] = b_area
    # temporal correlation
    temp_cor = de.calc_temp_cor(obs_arr, sim_arr)
    df_es.iloc[0, 2] = temp_cor
    # diagnostic efficiency
    df_es.iloc[0, 3] = 1 - np.sqrt((brel_mean)**2 + (b_area)**2 + (temp_cor - 1)**2)
    # direction of bias
    b_dir = de.calc_bias_dir(brel_rest)
    df_es.iloc[0, 4] = b_dir
    # slope of bias
    b_slope = de.calc_bias_slope(b_area, b_dir)
    df_es.iloc[0, 5] = b_slope
    # convert to radians
    # (y, x) Trigonometric inverse tangent
    df_es.iloc[0, 6] = np.arctan2(brel_mean, b_slope)

    # KGE
    df_es.iloc[0, 7] = de.calc_kge(obs_arr, sim_arr)
    # KGE alpha
    df_es.iloc[0, 8] = de.calc_kge_alpha(obs_arr, sim_arr)
    # KGE beta
    df_es.iloc[0, 9] = de.calc_kge_beta(obs_arr, sim_arr)

    # NSE
    df_es.iloc[0, 10] = de.calc_nse(obs_arr, sim_arr)

    ### increase high flows - decrease low flows ###
    obs_sim = pd.DataFrame(index=df_ts.index, columns=['Qobs', 'Qsim'])
    obs_sim.loc[:, 'Qobs'] = df_ts.loc[:, 'Qobs']
    tsd = de.highover_lowunder(df_ts.copy(), prop=0.5)
    obs_sim.loc[:, 'Qsim'] = tsd.loc[:, 'Qsim']  # disaggregated time series
    de.fdc_obs_sim_ax(obs_sim['Qobs'], obs_sim['Qsim'], axes_fdc[0, 0], fig_num_fdc[0])
    de.plot_obs_sim_ax(obs_sim['Qobs'], obs_sim['Qsim'], axes_ts[0, 0], fig_num_ts[0])

    # make arrays
    obs_arr = obs_sim['Qobs'].values
    sim_arr = obs_sim['Qsim'].values

    # mean relative bias
    brel_mean = de.calc_brel_mean(obs_arr, sim_arr)
    df_es.iloc[1, 0] = brel_mean
    # remaining relative bias
    brel_rest = de.calc_brel_rest(obs_arr, sim_arr)
    # area of relative remaing bias
    b_area = de.calc_b_area(brel_rest)
    df_es.iloc[1, 1] = b_area
    # temporal correlation
    temp_cor = de.calc_temp_cor(obs_arr, sim_arr)
    df_es.iloc[1, 2] = temp_cor
    # diagnostic efficiency
    df_es.iloc[1, 3] = 1 - np.sqrt((brel_mean)**2 + (b_area)**2 + (temp_cor - 1)**2)
    # direction of bias
    b_dir = de.calc_bias_dir(brel_rest)
    df_es.iloc[1, 4] = b_dir
    # slope of bias
    b_slope = de.calc_bias_slope(b_area, b_dir)
    df_es.iloc[1, 5] = b_slope
    # convert to radians
    # (y, x) Trigonometric inverse tangent
    df_es.iloc[1, 6] = np.arctan2(brel_mean, b_slope)

    # KGE
    df_es.iloc[1, 7] = de.calc_kge(obs_arr, sim_arr)
    # KGE alpha
    df_es.iloc[1, 8] = de.calc_kge_alpha(obs_arr, sim_arr)
    # KGE beta
    df_es.iloc[1, 9] = de.calc_kge_beta(obs_arr, sim_arr)

    # NSE
    df_es.iloc[1, 10] = de.calc_nse(obs_arr, sim_arr)

    ### decrease high flows - increase low flows ###
    obs_sim = pd.DataFrame(index=df_ts.index, columns=['Qobs', 'Qsim'])
    obs_sim.loc[:, 'Qobs'] = df_ts.loc[:, 'Qobs']
    tsd = de.highunder_lowover(df_ts.copy(), prop=0.5)
    obs_sim.loc[:, 'Qsim'] = tsd.loc[:, 'Qsim']  # smoothed time series
    de.fdc_obs_sim_ax(obs_sim['Qobs'], obs_sim['Qsim'], axes_fdc[0, 1], fig_num_fdc[1])
    de.plot_obs_sim_ax(obs_sim['Qobs'], obs_sim['Qsim'], axes_ts[0, 1], fig_num_ts[1])

    # make arrays
    obs_arr = obs_sim['Qobs'].values
    sim_arr = obs_sim['Qsim'].values

    # mean relative bias
    brel_mean = de.calc_brel_mean(obs_arr, sim_arr)
    df_es.iloc[2, 0] = brel_mean
    # remaining relative bias
    brel_rest = de.calc_brel_rest(obs_arr, sim_arr)
    # area of relative remaing bias
    b_area = de.calc_b_area(brel_rest)
    df_es.iloc[2, 1] = b_area
    # temporal correlation
    temp_cor = de.calc_temp_cor(obs_arr, sim_arr)
    df_es.iloc[2, 2] = temp_cor
    # diagnostic efficiency
    df_es.iloc[2, 3] = 1 - np.sqrt((brel_mean)**2 + (b_area)**2 + (temp_cor - 1)**2)
    # direction of bias
    b_dir = de.calc_bias_dir(brel_rest)
    df_es.iloc[2, 4] = b_dir
    # slope of bias
    b_slope = de.calc_bias_slope(b_area, b_dir)
    df_es.iloc[2, 5] = b_slope
    # convert to radians
    # (y, x) Trigonometric inverse tangent
    df_es.iloc[2, 6] = np.arctan2(brel_mean, b_slope)

    # KGE
    df_es.iloc[2, 7] = de.calc_kge(obs_arr, sim_arr)
    # KGE alpha
    df_es.iloc[2, 8] = de.calc_kge_alpha(obs_arr, sim_arr)
    # KGE beta
    df_es.iloc[2, 9] = de.calc_kge_beta(obs_arr, sim_arr)

    # NSE
    df_es.iloc[2, 10] = de.calc_nse(obs_arr, sim_arr)

    ### precipitation surplus ###
    obs_sim = pd.DataFrame(index=df_ts.index, columns=['Qobs', 'Qsim'])
    obs_sim.loc[:, 'Qobs'] = df_ts.loc[:, 'Qobs']
    obs_sim.loc[:, 'Qsim'] = de.pos_shift_ts(df_ts['Qobs'].values)  # positive offset
    de.fdc_obs_sim_ax(obs_sim['Qobs'], obs_sim['Qsim'], axes_fdc[0, 2], fig_num_fdc[2])
    de.plot_obs_sim_ax(obs_sim['Qobs'], obs_sim['Qsim'], axes_ts[0, 2], fig_num_ts[2])

    # make arrays
    obs_arr = obs_sim['Qobs'].values
    sim_arr = obs_sim['Qsim'].values

    # mean relative bias
    brel_mean = de.calc_brel_mean(obs_arr, sim_arr)
    df_es.iloc[3, 0] = brel_mean
    # remaining relative bias
    brel_rest = de.calc_brel_rest(obs_arr, sim_arr)
    # area of relative remaing bias
    b_area = de.calc_b_area(brel_rest)
    df_es.iloc[3, 1] = b_area
    # temporal correlation
    temp_cor = de.calc_temp_cor(obs_arr, sim_arr)
    df_es.iloc[3, 2] = temp_cor
    # diagnostic efficiency
    df_es.iloc[3, 3] = 1 - np.sqrt((brel_mean)**2 + (b_area)**2 + (temp_cor - 1)**2)
    # direction of bias
    b_dir = de.calc_bias_dir(brel_rest)
    df_es.iloc[3, 4] = b_dir
    # slope of bias
    b_slope = de.calc_bias_slope(b_area, b_dir)
    df_es.iloc[3, 5] = b_slope
    # convert to radians
    # (y, x) Trigonometric inverse tangent
    df_es.iloc[3, 6] = np.arctan2(brel_mean, b_slope)

    # KGE
    df_es.iloc[3, 7] = de.calc_kge(obs_arr, sim_arr)
    # KGE alpha
    df_es.iloc[3, 8] = de.calc_kge_alpha(obs_arr, sim_arr)
    # KGE beta
    df_es.iloc[3, 9] = de.calc_kge_beta(obs_arr, sim_arr)

    # NSE
    df_es.iloc[3, 10] = de.calc_nse(obs_arr, sim_arr)

    ### precipitation shortage ###
    obs_sim = pd.DataFrame(index=df_ts.index, columns=['Qobs', 'Qsim'])
    obs_sim.loc[:, 'Qobs'] = df_ts.loc[:, 'Qobs']
    obs_sim.loc[:, 'Qsim'] = de.neg_shift_ts(df_ts['Qobs'].values)  # negative offset
    de.fdc_obs_sim_ax(obs_sim['Qobs'], obs_sim['Qsim'], axes_fdc[0, 3], fig_num_fdc[3])
    de.plot_obs_sim_ax(obs_sim['Qobs'], obs_sim['Qsim'], axes_ts[0, 3], fig_num_ts[3])

    # make arrays
    obs_arr = obs_sim['Qobs'].values
    sim_arr = obs_sim['Qsim'].values

    # mean relative bias
    brel_mean = de.calc_brel_mean(obs_arr, sim_arr)
    df_es.iloc[4, 0] = brel_mean
    # remaining relative bias
    brel_rest = de.calc_brel_rest(obs_arr, sim_arr)
    # area of relative remaing bias
    b_area = de.calc_b_area(brel_rest)
    df_es.iloc[4, 1] = b_area
    # temporal correlation
    temp_cor = de.calc_temp_cor(obs_arr, sim_arr)
    df_es.iloc[4, 2] = temp_cor
    # diagnostic efficiency
    df_es.iloc[4, 3] = 1 - np.sqrt((brel_mean)**2 + (b_area)**2 + (temp_cor - 1)**2)
    # direction of bias
    b_dir = de.calc_bias_dir(brel_rest)
    df_es.iloc[4, 4] = b_dir
    # slope of bias
    b_slope = de.calc_bias_slope(b_area, b_dir)
    df_es.iloc[4, 5] = b_slope
    # convert to radians
    # (y, x) Trigonometric inverse tangent
    df_es.iloc[4, 6] = np.arctan2(brel_mean, b_slope)

    # KGE
    df_es.iloc[4, 7] = de.calc_kge(obs_arr, sim_arr)
    # KGE alpha
    df_es.iloc[4, 8] = de.calc_kge_alpha(obs_arr, sim_arr)
    # KGE beta
    df_es.iloc[4, 9] = de.calc_kge_beta(obs_arr, sim_arr)

    # NSE
    df_es.iloc[4, 10] = de.calc_nse(obs_arr, sim_arr)

    ### shuffling ###
    obs_sim = pd.DataFrame(index=df_ts.index, columns=['Qobs', 'Qsim'])
    obs_sim.loc[:, 'Qobs'] = df_ts.loc[:, 'Qobs']
    tss = de.time_shift(df_ts.copy(), random=True)  # shuffled time series
    obs_sim.loc[:, 'Qsim'] = tss.iloc[:, 0].values
    de.fdc_obs_sim_ax(obs_sim['Qobs'], obs_sim['Qsim'], axes_fdc[0, 4], fig_num_fdc[4])
    de.plot_obs_sim_ax(obs_sim['Qobs'], obs_sim['Qsim'], axes_ts[0, 4], fig_num_ts[4])

    # make arrays
    obs_arr = obs_sim['Qobs'].values
    sim_arr = obs_sim['Qsim'].values

    # mean relative bias
    brel_mean = de.calc_brel_mean(obs_arr, sim_arr)
    df_es.iloc[5, 0] = brel_mean
    # remaining relative bias
    brel_rest = de.calc_brel_rest(obs_arr, sim_arr)
    # area of relative remaing bias
    b_area = de.calc_b_area(brel_rest)
    df_es.iloc[5, 1] = b_area
    # temporal correlation
    temp_cor = de.calc_temp_cor(obs_arr, sim_arr)
    df_es.iloc[5, 2] = temp_cor
    # diagnostic efficiency
    df_es.iloc[5, 3] = 1 - np.sqrt((brel_mean)**2 + (b_area)**2 + (temp_cor - 1)**2)
    # direction of bias
    b_dir = de.calc_bias_dir(brel_rest)
    df_es.iloc[5, 4] = b_dir
    # slope of bias
    b_slope = de.calc_bias_slope(b_area, b_dir)
    df_es.iloc[5, 5] = b_slope
    # convert to radians
    # (y, x) Trigonometric inverse tangent
    df_es.iloc[5, 6] = np.arctan2(brel_mean, b_slope)

    # KGE
    df_es.iloc[5, 7] = de.calc_kge(obs_arr, sim_arr)
    # KGE alpha
    df_es.iloc[5, 8] = de.calc_kge_alpha(obs_arr, sim_arr)
    # KGE beta
    df_es.iloc[5, 9] = de.calc_kge_beta(obs_arr, sim_arr)

    # NSE
    df_es.iloc[5, 10] = de.calc_nse(obs_arr, sim_arr)

    ### Decrease high flows - Increase low flows and precipitation surplus ###
    obs_sim = pd.DataFrame(index=df_ts.index, columns=['Qobs', 'Qsim'])
    obs_sim.loc[:, 'Qobs'] = df_ts.loc[:, 'Qobs']
    tsd = de.highunder_lowover(df_ts.copy(), prop=0.34)  # smoothed time series
    obs_sim.loc[:, 'Qsim'] = de.pos_shift_ts(tsd.iloc[:, 0].values, offset=1.2)  # positive offset
    de.fdc_obs_sim_ax(obs_sim['Qobs'], obs_sim['Qsim'], axes_fdc[1, 0], fig_num_fdc[5])
    de.plot_obs_sim_ax(obs_sim['Qobs'], obs_sim['Qsim'], axes_ts[1, 0], fig_num_ts[5])

    # make arrays
    obs_arr = obs_sim['Qobs'].values
    sim_arr = obs_sim['Qsim'].values

    # mean relative bias
    brel_mean = de.calc_brel_mean(obs_arr, sim_arr)
    df_es.iloc[6, 0] = brel_mean
    # remaining relative bias
    brel_rest = de.calc_brel_rest(obs_arr, sim_arr)
    # area of relative remaing bias
    b_area = de.calc_b_area(brel_rest)
    df_es.iloc[6, 1] = b_area
    # temporal correlation
    temp_cor = de.calc_temp_cor(obs_arr, sim_arr)
    df_es.iloc[6, 2] = temp_cor
    # diagnostic efficiency
    df_es.iloc[6, 3] = 1 - np.sqrt((brel_mean)**2 + (b_area)**2 + (temp_cor - 1)**2)
    # direction of bias
    b_dir = de.calc_bias_dir(brel_rest)
    df_es.iloc[6, 4] = b_dir
    # slope of bias
    b_slope = de.calc_bias_slope(b_area, b_dir)
    df_es.iloc[6, 5] = b_slope
    # convert to radians
    # (y, x) Trigonometric inverse tangent
    df_es.iloc[6, 6] = np.arctan2(brel_mean, b_slope)

    # KGE
    df_es.iloc[6, 7] = de.calc_kge(obs_arr, sim_arr)
    # KGE alpha
    df_es.iloc[6, 8] = de.calc_kge_alpha(obs_arr, sim_arr)
    # KGE beta
    df_es.iloc[6, 9] = de.calc_kge_beta(obs_arr, sim_arr)

    # NSE
    df_es.iloc[6, 10] = de.calc_nse(obs_arr, sim_arr)

    ### Decrease high flows - Increase low flows and precipitation shortage ###
    obs_sim = pd.DataFrame(index=df_ts.index, columns=['Qobs', 'Qsim'])
    obs_sim.loc[:, 'Qobs'] = df_ts.loc[:, 'Qobs']
    tsd = de.highunder_lowover(df_ts.copy(), prop=0.5)  # smoothed time series
    obs_sim.loc[:, 'Qsim'] = de.neg_shift_ts(tsd.iloc[:, 0].values, offset=0.8)  # negative offset
    de.fdc_obs_sim_ax(obs_sim['Qobs'], obs_sim['Qsim'], axes_fdc[1, 1], fig_num_fdc[6])
    de.plot_obs_sim_ax(obs_sim['Qobs'], obs_sim['Qsim'], axes_ts[1, 1], fig_num_ts[6])

    # make arrays
    obs_arr = obs_sim['Qobs'].values
    sim_arr = obs_sim['Qsim'].values

    # mean relative bias
    brel_mean = de.calc_brel_mean(obs_arr, sim_arr)
    df_es.iloc[7, 0] = brel_mean
    # remaining relative bias
    brel_rest = de.calc_brel_rest(obs_arr, sim_arr)
    # area of relative remaing bias
    b_area = de.calc_b_area(brel_rest)
    df_es.iloc[7, 1] = b_area
    # temporal correlation
    temp_cor = de.calc_temp_cor(obs_arr, sim_arr)
    df_es.iloc[7, 2] = temp_cor
    # diagnostic efficiency
    df_es.iloc[7, 3] = 1 - np.sqrt((brel_mean)**2 + (b_area)**2 + (temp_cor - 1)**2)
    # direction of bias
    b_dir = de.calc_bias_dir(brel_rest)
    df_es.iloc[7, 4] = b_dir
    # slope of bias
    b_slope = de.calc_bias_slope(b_area, b_dir)
    df_es.iloc[7, 5] = b_slope
    # convert to radians
    # (y, x) Trigonometric inverse tangent
    df_es.iloc[7, 6] = np.arctan2(brel_mean, b_slope)

    # KGE
    df_es.iloc[7, 7] = de.calc_kge(obs_arr, sim_arr)
    # KGE alpha
    df_es.iloc[7, 8] = de.calc_kge_alpha(obs_arr, sim_arr)
    # KGE beta
    df_es.iloc[7, 9] = de.calc_kge_beta(obs_arr, sim_arr)

    # NSE
    df_es.iloc[7, 10] = de.calc_nse(obs_arr, sim_arr)

    ### Increase high flows - Decrease low flows and precipitation surplus ###
    obs_sim = pd.DataFrame(index=df_ts.index, columns=['Qobs', 'Qsim'])
    obs_sim.loc[:, 'Qobs'] = df_ts.loc[:, 'Qobs']
    tsd = de.highover_lowunder(df_ts.copy(), prop=0.34)  # disaggregated time series
    tsp = pd.DataFrame(index=df_ts.index, columns=['Qsim'])
    tsp.iloc[:, 0] = de.pos_shift_ts(tsd.iloc[:, 0].values, offset=1.2)  # positive offset
    obs_sim.loc[:, 'Qsim'] = tsp.iloc[:, 0].values  # positive offset
    de.fdc_obs_sim_ax(obs_sim['Qobs'], obs_sim['Qsim'], axes_fdc[1, 2], fig_num_fdc[7])
    de.plot_obs_sim_ax(obs_sim['Qobs'], obs_sim['Qsim'], axes_ts[1, 2], fig_num_ts[7])

    # make arrays
    obs_arr = obs_sim['Qobs'].values
    sim_arr = obs_sim['Qsim'].values

    # mean relative bias
    brel_mean = de.calc_brel_mean(obs_arr, sim_arr)
    df_es.iloc[8, 0] = brel_mean
    # remaining relative bias
    brel_rest = de.calc_brel_rest(obs_arr, sim_arr)
    # area of relative remaing bias
    b_area = de.calc_b_area(brel_rest)
    df_es.iloc[8, 1] = b_area
    # temporal correlation
    temp_cor = de.calc_temp_cor(obs_arr, sim_arr)
    df_es.iloc[8, 2] = temp_cor
    # diagnostic efficiency
    df_es.iloc[8, 3] = 1 - np.sqrt((brel_mean)**2 + (b_area)**2 + (temp_cor - 1)**2)
    # direction of bias
    b_dir = de.calc_bias_dir(brel_rest)
    df_es.iloc[8, 4] = b_dir
    # slope of bias
    b_slope = de.calc_bias_slope(b_area, b_dir)
    df_es.iloc[8, 5] = b_slope
    # convert to radians
    # (y, x) Trigonometric inverse tangent
    df_es.iloc[8, 6] = np.arctan2(brel_mean, b_slope)

    # KGE
    df_es.iloc[8, 7] = de.calc_kge(obs_arr, sim_arr)
    # KGE alpha
    df_es.iloc[8, 8] = de.calc_kge_alpha(obs_arr, sim_arr)
    # KGE beta
    df_es.iloc[8, 9] = de.calc_kge_beta(obs_arr, sim_arr)

    # NSE
    df_es.iloc[8, 10] = de.calc_nse(obs_arr, sim_arr)

    ### Increase high flows - Decrease low flows and precipitation shortage ###
    obs_sim = pd.DataFrame(index=df_ts.index, columns=['Qobs', 'Qsim'])
    obs_sim.loc[:, 'Qobs'] = df_ts.loc[:, 'Qobs']
    tsd = de.highover_lowunder(df_ts.copy(), prop=0.5) # disaggregated time series
    obs_sim.loc[:, 'Qsim'] = de.neg_shift_ts(tsd.iloc[:, 0].values, offset=0.8)  # negative offset
    de.fdc_obs_sim_ax(obs_sim['Qobs'], obs_sim['Qsim'], axes_fdc[1, 3], fig_num_fdc[8])
    de.plot_obs_sim_ax(obs_sim['Qobs'], obs_sim['Qsim'], axes_ts[1, 3], fig_num_ts[8])

    # make arrays
    obs_arr = obs_sim['Qobs'].values
    sim_arr = obs_sim['Qsim'].values

    # mean relative bias
    brel_mean = de.calc_brel_mean(obs_arr, sim_arr)
    df_es.iloc[9, 0] = brel_mean
    # remaining relative bias
    brel_rest = de.calc_brel_rest(obs_arr, sim_arr)
    # area of relative remaing bias
    b_area = de.calc_b_area(brel_rest)
    df_es.iloc[9, 1] = b_area
    # temporal correlation
    temp_cor = de.calc_temp_cor(obs_arr, sim_arr)
    df_es.iloc[9, 2] = temp_cor
    # diagnostic efficiency
    df_es.iloc[9, 3] = 1 - np.sqrt((brel_mean)**2 + (b_area)**2 + (temp_cor - 1)**2)
    # direction of bias
    b_dir = de.calc_bias_dir(brel_rest)
    df_es.iloc[9, 4] = b_dir
    # slope of bias
    b_slope = de.calc_bias_slope(b_area, b_dir)
    df_es.iloc[9, 5] = b_slope
    # convert to radians
    # (y, x) Trigonometric inverse tangent
    df_es.iloc[9, 6] = np.arctan2(brel_mean, b_slope)

    # KGE
    df_es.iloc[9, 7] = de.calc_kge(obs_arr, sim_arr)
    # KGE alpha
    df_es.iloc[9, 8] = de.calc_kge_alpha(obs_arr, sim_arr)
    # KGE beta
    df_es.iloc[9, 9] = de.calc_kge_beta(obs_arr, sim_arr)

    # NSE
    df_es.iloc[9, 10] = de.calc_nse(obs_arr, sim_arr)

    ### Decrease high flows - Increase low flows, precipitation surplus and shuffling ###
    obs_sim = pd.DataFrame(index=df_ts.index, columns=['Qobs', 'Qsim'])
    obs_sim.loc[:, 'Qobs'] = df_ts.loc[:, 'Qobs']
    tsd = de.highunder_lowover(df_ts.copy(), prop=0.34)  # smoothed time series
    tsp = pd.DataFrame(index=df_ts.index, columns=['Qsim'])
    tsp.iloc[:, 0] = de.pos_shift_ts(tsd.iloc[:, 0].values, offset=1.2)  # positive offset
    tst = de.time_shift(tsp, random=True)  # shuffling
    obs_sim.loc[:, 'Qsim'] = tst.iloc[:, 0].values
    de.plot_obs_sim_ax(obs_sim['Qobs'], obs_sim['Qsim'], axes_ts[2, 0], fig_num_ts[9])

    # make arrays
    obs_arr = obs_sim['Qobs'].values
    sim_arr = obs_sim['Qsim'].values

    # mean relative bias
    brel_mean = de.calc_brel_mean(obs_arr, sim_arr)
    df_es.iloc[10, 0] = brel_mean
    # remaining relative bias
    brel_rest = de.calc_brel_rest(obs_arr, sim_arr)
    # area of relative remaing bias
    b_area = de.calc_b_area(brel_rest)
    df_es.iloc[10, 1] = b_area
    # temporal correlation
    temp_cor = de.calc_temp_cor(obs_arr, sim_arr)
    df_es.iloc[10, 2] = temp_cor
    # diagnostic efficiency
    df_es.iloc[10, 3] = 1 - np.sqrt((brel_mean)**2 + (b_area)**2 + (temp_cor - 1)**2)
    # direction of bias
    b_dir = de.calc_bias_dir(brel_rest)
    df_es.iloc[10, 4] = b_dir
    # slope of bias
    b_slope = de.calc_bias_slope(b_area, b_dir)
    df_es.iloc[10, 5] = b_slope
    # convert to radians
    # (y, x) Trigonometric inverse tangent
    df_es.iloc[10, 6] = np.arctan2(brel_mean, b_slope)

    # KGE
    df_es.iloc[10, 7] = de.calc_kge(obs_arr, sim_arr)
    # KGE alpha
    df_es.iloc[10, 8] = de.calc_kge_alpha(obs_arr, sim_arr)
    # KGE beta
    df_es.iloc[10, 9] = de.calc_kge_beta(obs_arr, sim_arr)

    # NSE
    df_es.iloc[10, 10] = de.calc_nse(obs_arr, sim_arr)

    ### Decrease high flows - Increase low flows, precipitation shortage and shuffling ###
    obs_sim = pd.DataFrame(index=df_ts.index, columns=['Qobs', 'Qsim'])
    obs_sim.loc[:, 'Qobs'] = df_ts.loc[:, 'Qobs']
    tsd = de.highunder_lowover(df_ts.copy(), prop=0.5)  # smoothed time series
    tsn = pd.DataFrame(index=df_ts.index, columns=['Qsim'])
    tsn.iloc[:, 0]  = de.neg_shift_ts(tsd.iloc[:, 0].values, offset=0.8)  # negative offset
    tst = de.time_shift(tsn, random=True)  # shuffling
    obs_sim.loc[:, 'Qsim'] = tst.iloc[:, 0].values
    de.plot_obs_sim_ax(obs_sim['Qobs'], obs_sim['Qsim'], axes_ts[2, 1], fig_num_ts[10])

    # make arrays
    obs_arr = obs_sim['Qobs'].values
    sim_arr = obs_sim['Qsim'].values

    # mean relative bias
    brel_mean = de.calc_brel_mean(obs_arr, sim_arr)
    df_es.iloc[11, 0] = brel_mean
    # remaining relative bias
    brel_rest = de.calc_brel_rest(obs_arr, sim_arr)
    # area of relative remaing bias
    b_area = de.calc_b_area(brel_rest)
    df_es.iloc[11, 1] = b_area
    # temporal correlation
    temp_cor = de.calc_temp_cor(obs_arr, sim_arr)
    df_es.iloc[11, 2] = temp_cor
    # diagnostic efficiency
    df_es.iloc[11, 3] = 1 - np.sqrt((brel_mean)**2 + (b_area)**2 + (temp_cor - 1)**2)
    # direction of bias
    b_dir = de.calc_bias_dir(brel_rest)
    df_es.iloc[11, 4] = b_dir
    # slope of bias
    b_slope = de.calc_bias_slope(b_area, b_dir)
    df_es.iloc[11, 5] = b_slope
    # convert to radians
    # (y, x) Trigonometric inverse tangent
    df_es.iloc[11, 6] = np.arctan2(brel_mean, b_slope)

    # KGE
    df_es.iloc[11, 7] = de.calc_kge(obs_arr, sim_arr)
    # KGE alpha
    df_es.iloc[11, 8] = de.calc_kge_alpha(obs_arr, sim_arr)
    # KGE beta
    df_es.iloc[11, 9] = de.calc_kge_beta(obs_arr, sim_arr)

    # NSE
    df_es.iloc[11, 10] = de.calc_nse(obs_arr, sim_arr)

    ### Increase high flows - Decrease low flows, precipitation surplus and shuffling ###
    obs_sim = pd.DataFrame(index=df_ts.index, columns=['Qobs', 'Qsim'])
    obs_sim.loc[:, 'Qobs'] = df_ts.loc[:, 'Qobs']
    tsd = de.highover_lowunder(df_ts.copy(), prop=0.34)  # disaggregated time series
    tsp = pd.DataFrame(index=df_ts.index, columns=['Qsim'])
    tsp.iloc[:, 0] = de.pos_shift_ts(tsd.iloc[:, 0].values, offset=1.2)  # positive offset
    tst = de.time_shift(tsp, random=True)  # shuffling
    obs_sim.loc[:, 'Qsim'] = tst.iloc[:, 0].values
    de.plot_obs_sim_ax(obs_sim['Qobs'], obs_sim['Qsim'], axes_ts[2, 2], fig_num_ts[11])

    # make arrays
    obs_arr = obs_sim['Qobs'].values
    sim_arr = obs_sim['Qsim'].values

    # mean relative bias
    brel_mean = de.calc_brel_mean(obs_arr, sim_arr)
    df_es.iloc[12, 0] = brel_mean
    # remaining relative bias
    brel_rest = de.calc_brel_rest(obs_arr, sim_arr)
    # area of relative remaing bias
    b_area = de.calc_b_area(brel_rest)
    df_es.iloc[12, 1] = b_area
    # temporal correlation
    temp_cor = de.calc_temp_cor(obs_arr, sim_arr)
    df_es.iloc[12, 2] = temp_cor
    # diagnostic efficiency
    df_es.iloc[12, 3] = 1 - np.sqrt((brel_mean)**2 + (b_area)**2 + (temp_cor - 1)**2)
    # direction of bias
    b_dir = de.calc_bias_dir(brel_rest)
    df_es.iloc[12, 4] = b_dir
    # slope of bias
    b_slope = de.calc_bias_slope(b_area, b_dir)
    df_es.iloc[12, 5] = b_slope
    # convert to radians
    # (y, x) Trigonometric inverse tangent
    df_es.iloc[12, 6] = np.arctan2(brel_mean, b_slope)

    # KGE
    df_es.iloc[12, 7] = de.calc_kge(obs_arr, sim_arr)
    # KGE alpha
    df_es.iloc[12, 8] = de.calc_kge_alpha(obs_arr, sim_arr)
    # KGE beta
    df_es.iloc[12, 9] = de.calc_kge_beta(obs_arr, sim_arr)

    # NSE
    df_es.iloc[12, 10] = de.calc_nse(obs_arr, sim_arr)

    ### Increase high flows - Decrease low flows, precipitation shortage and shuffling ###
    obs_sim = pd.DataFrame(index=df_ts.index, columns=['Qobs', 'Qsim'])
    obs_sim.loc[:, 'Qobs'] = df_ts.loc[:, 'Qobs']
    tsd = de.highover_lowunder(df_ts.copy(), prop=0.5) # disaggregated time series
    tsn = pd.DataFrame(index=df_ts.index, columns=['Qsim'])
    tsn.iloc[:, 0] = de.neg_shift_ts(tsd.iloc[:, 0].values, offset=0.8)  # negative offset
    tst = de.time_shift(tsn, random=True)  # shuffling
    obs_sim.loc[:, 'Qsim'] = tst.iloc[:, 0].values
    de.plot_obs_sim_ax(obs_sim['Qobs'], obs_sim['Qsim'], axes_ts[2, 3], fig_num_ts[12])

    # make arrays
    obs_arr = obs_sim['Qobs'].values
    sim_arr = obs_sim['Qsim'].values

    # mean relative bias
    brel_mean = de.calc_brel_mean(obs_arr, sim_arr)
    df_es.iloc[13, 0] = brel_mean
    # remaining relative bias
    brel_rest = de.calc_brel_rest(obs_arr, sim_arr)
    # area of relative remaing bias
    b_area = de.calc_b_area(brel_rest)
    df_es.iloc[13, 1] = b_area
    # temporal correlation
    temp_cor = de.calc_temp_cor(obs_arr, sim_arr)
    df_es.iloc[13, 2] = temp_cor
    # diagnostic efficiency
    df_es.iloc[13, 3] = 1 - np.sqrt((brel_mean)**2 + (b_area)**2 + (temp_cor - 1)**2)
    # direction of bias
    b_dir = de.calc_bias_dir(brel_rest)
    df_es.iloc[13, 4] = b_dir
    # slope of bias
    b_slope = de.calc_bias_slope(b_area, b_dir)
    df_es.iloc[13, 5] = b_slope
    # convert to radians
    # (y, x) Trigonometric inverse tangent
    df_es.iloc[13, 6] = np.arctan2(brel_mean, b_slope)

    # KGE
    df_es.iloc[13, 7] = de.calc_kge(obs_arr, sim_arr)
    # KGE alpha
    df_es.iloc[13, 8] = de.calc_kge_alpha(obs_arr, sim_arr)
    # KGE beta
    df_es.iloc[13, 9] = de.calc_kge_beta(obs_arr, sim_arr)

    # NSE
    df_es.iloc[13, 10] = de.calc_nse(obs_arr, sim_arr)

    ### mean flow benchmark ###
    obs_sim = pd.DataFrame(index=df_ts.index, columns=['Qobs', 'Qsim'])
    obs_sim.loc[:, 'Qobs'] = df_ts.loc[:, 'Qobs']
    obs_mean = np.mean(obs_sim['Qobs'].values)
    obs_sim.loc[:, 'Qsim'] = np.repeat(obs_mean, len(obs_sim['Qobs'].values))
    de.fdc_obs_sim_ax(obs_sim['Qobs'], obs_sim['Qsim'], axes_fdc[1, 4], fig_num_fdc[9])
    de.plot_obs_sim_ax(obs_sim['Qobs'], obs_sim['Qsim'], axes_ts[1, 4], fig_num_ts[9])

    # make arrays
    obs_arr = obs_sim['Qobs'].values
    sim_arr = obs_sim['Qsim'].values

    # mean relative bias
    brel_mean = de.calc_brel_mean(obs_arr, sim_arr)
    df_es.iloc[14, 0] = brel_mean
    # remaining relative bias
    brel_rest = de.calc_brel_rest(obs_arr, sim_arr)
    # area of relative remaing bias
    b_area = de.calc_b_area(brel_rest)
    df_es.iloc[14, 1] = b_area
    # temporal correlation
    temp_cor = de.calc_temp_cor(obs_arr, sim_arr)
    df_es.iloc[14, 2] = temp_cor
    # diagnostic efficiency
    df_es.iloc[14, 3] = 1 - np.sqrt((brel_mean)**2 + (b_area)**2 + (temp_cor - 1)**2)
    # direction of bias
    b_dir = de.calc_bias_dir(brel_rest)
    df_es.iloc[14, 4] = b_dir
    # slope of bias
    b_slope = de.calc_bias_slope(b_area, b_dir)
    df_es.iloc[14, 5] = b_slope
    # convert to radians
    # (y, x) Trigonometric inverse tangent
    df_es.iloc[14, 6] = np.arctan2(brel_mean, b_slope)

    # KGE
    df_es.iloc[14, 7] = de.calc_kge(obs_arr, sim_arr)
    # KGE alpha
    df_es.iloc[14, 8] = de.calc_kge_alpha(obs_arr, sim_arr)
    # KGE beta
    df_es.iloc[14, 9] = de.calc_kge_beta(obs_arr, sim_arr)

    # NSE
    df_es.iloc[14, 10] = de.calc_nse(obs_arr, sim_arr)

    ### multi diagnostic plot ###
    # make arrays
    brel_mean_arr = df_es['brel_mean'].values
    b_area_arr = df_es['b_area'].values
    temp_cor_arr = df_es['temp_cor'].values
    b_dir_arr = df_es['b_dir'].values
    de_arr = df_es['de'].values
    diag_arr = df_es['diag'].values
    b_slope_arr = df_es['b_slope'].values

    de.vis2d_de_multi(brel_mean_arr, b_area_arr, temp_cor_arr, de_arr,
                      b_dir_arr, diag_arr, extended=False)

    ### multi KGE plot ###
    # make arrays
    alpha_arr = df_es['alpha'].values
    beta_arr = df_es['beta'].values
    kge_arr = df_es['kge'].values

    de.vis2d_kge_multi_fc(alpha_arr, beta_arr, temp_cor_arr, kge_arr, idx, extended=False)

    nse_arr = df_es['nse'].values
    # scatterplots efficiencies
    fig, ax = plt.subplots(figsize=(6,6))
    sc = sns.scatterplot(kge_arr, de_arr, color='black', ax=ax)
    sc1 = sns.scatterplot(nse_arr, de_arr, color='red', ax=ax)
    ax.plot([-1.05, 1.05], [-1.05, 1.05], ls="--", c=".3")
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlim(-1.05, 1.05)
    ax.set(ylabel='DE [-]', xlabel='KGE [-]')
    ax.text(.455, -.12, 'NSE [-]', color='red', transform=ax.transAxes)
    for i, txt in enumerate(df_es.index):
        ax.annotate(txt, (kge_arr[i], de_arr[i]), color='black', fontsize=15)
        ax.annotate(txt, (nse_arr[i], de_arr[i]), color='red', fontsize=15)

    # scatterplots components
    fig, ax = plt.subplots(figsize=(6,6))
    fig.subplots_adjust(left=.2)
    sc = sns.scatterplot(alpha_arr - 1, brel_mean_arr, color='black', ax=ax)
    sc1 = sns.scatterplot(beta_arr - 1, b_slope_arr, color='red', ax=ax)
    ax.plot([-1.05, 1.05], [-1.05, 1.05], ls="--", c=".3")
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlim(-1.05, 1.05)
    ax.set(ylabel=r'$\overline{B_{rel}}$ [-]', xlabel=r'$\alpha$ [-]')
    ax.text(-.22, .455, r'$B_{slope}$ [-]', color='red', transform=ax.transAxes, rotation=90)
    ax.text(.47, -.12, r'$\beta$ [-]', color='red', transform=ax.transAxes)
    for i, txt in enumerate(df_es.index):
        ax.annotate(txt, (alpha_arr[i] - 1, brel_mean_arr[i]), color='black', fontsize=15)
        ax.annotate(txt, (beta_arr[i] - 1, b_slope_arr[i]), color='red', fontsize=15)

    # export table
    path_csv = '/Users/robinschwemmle/Desktop/PhD/diagnostic_model_efficiency/figures/technical_note/table_eff.csv'
    df_es_t = df_es.T
    df_es_t = df_es_t.loc[['de', 'kge', 'nse'], :]
    df_es_t = df_es_t.round(2)
    df_es_t.to_csv(path_csv, header=True, index=True, sep=';')

#    ### Tier-1 ###
#    path_wrr1 = '/Users/robinschwemmle/Desktop/PhD/diagnostic_model_efficiency/examples/data/GRDC_4103631_wrr1.csv'
#    df_wrr1 = util.import_ts(path_wrr1, sep=';')
#    de.fdc_obs_sim(df_wrr1['Qobs'], df_wrr1['Qsim'])
#    de.plot_obs_sim(df_wrr1['Qobs'], df_wrr1['Qsim'])
#
#    obs_arr = df_wrr1['Qobs'].values
#    sim_arr = df_wrr1['Qsim'].values
#
#    sig_de = de.calc_de(obs_arr, sim_arr)
#    sig_kge = de.calc_kge(obs_arr, sim_arr)
#    sig_nse = de.calc_nse(obs_arr, sim_arr)
#
#    de.vis2d_de(obs_arr, sim_arr)
#    de.vis2d_kge(obs_arr, sim_arr)
#
#    ### Tier-2 ###
#    path_wrr2 = '/Users/robinschwemmle/Desktop/PhD/diagnostic_model_efficiency/examples/data/GRDC_4103631_wrr2.csv'
#    df_wrr2 = util.import_ts(path_wrr2, sep=';')
#    de.fdc_obs_sim(df_wrr2['Qobs'], df_wrr2['Qsim'])
#    de.plot_obs_sim(df_wrr2['Qobs'], df_wrr2['Qsim'])
#
#    obs_arr = df_wrr2['Qobs'].values
#    sim_arr = df_wrr2['Qsim'].values
#
#    sig_de = de.calc_de(obs_arr, sim_arr)
#    sig_kge = de.calc_kge(obs_arr, sim_arr)
#    sig_nse = de.calc_nse(obs_arr, sim_arr)
#
#    de.vis2d_de(obs_arr, sim_arr)
#    de.vis2d_kge(obs_arr, sim_arr)
