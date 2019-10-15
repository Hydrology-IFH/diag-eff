#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from de import de
from de import util
import seaborn as sns
# controlling figure aesthetics
sns.set_style('ticks', {'xtick.major.size': 8, 'ytick.major.size': 8})
sns.set_context("paper", font_scale=1.5)

if __name__ == "__main__":
    # 299.46 km2
    path = '/Users/robinschwemmle/Desktop/PhD/diagnostic_model_efficiency/examples/camels_example_data/13331500_streamflow_qc.txt'

    fig_num_fdc = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)', '(j)']
    fig_fdc, axes_fdc = plt.subplots(2, 5, sharey=True, sharex=True, figsize=(14,6))
    fig_fdc.text(0.5, 0.02, 'Exceedence probabilty [-]', ha='center', va='center')
    fig_fdc.text(0.08, 0.5, r'[mm $d^{-1}$]', ha='center', va='center', rotation='vertical')

    fig_num_ts = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)', '(j)', '(k)', '(l)', '(m)', '(n)']
    fig_ts, axes_ts = plt.subplots(3, 5, sharey=True, sharex=True, figsize=(14,9))
    fig_ts.text(0.5, 0.05, 'Time [Years]', ha='center', va='center')
    fig_ts.text(0.08, 0.5, r'[mm $d^{-1}$]', ha='center', va='center', rotation='vertical')
    axes_ts[2,4].remove()

    # dataframe efficiency measures
    idx = ['1', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n']
    cols = ['brel_mean', 'b_area', 'temp_cor', 'de', 'mfb', 'b_dir', 'b_slope', 'diag', 'kge_norm', 'alpha', 'beta', 'nse']
    df_es = pd.DataFrame(index=idx, columns=cols, dtype=np.float64)

    # import observed time series
    df_ts = util.import_camels_ts(path, sep=r"\s+")
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
    b_area = de.calc_bias_area(brel_rest)
    df_es.iloc[0, 1] = b_area
    # temporal correlation
    temp_cor = de.calc_temp_cor(obs_arr, sim_arr)
    df_es.iloc[0, 2] = temp_cor
    # diagnostic efficiency
    df_es.iloc[0, 3] = de.calc_de(obs_arr, sim_arr)
    # mean flow benchmark of diagnostic efficiency
    df_es.iloc[0, 4] = de.calc_de_mfb(obs_arr)
    # direction of bias
    b_dir = de.calc_bias_dir(brel_rest)
    df_es.iloc[0, 5] = b_dir
    # slope of bias
    b_slope = de.calc_bias_slope(b_area, b_dir)
    df_es.iloc[0, 6] = b_slope
    # convert to radians
    # (y, x) Trigonometric inverse tangent
    df_es.iloc[0, 7] = np.arctan2(brel_mean, b_slope)

    # KGE
    df_es.iloc[0, 8] = de.calc_kge_norm(obs_arr, sim_arr)
    # KGE alpha
    df_es.iloc[0, 9] = de.calc_kge_alpha(obs_arr, sim_arr)
    # KGE beta
    df_es.iloc[0, 10] = de.calc_kge_beta(obs_arr, sim_arr)

    # NSE
    df_es.iloc[0, 11] = de.calc_nse(obs_arr, sim_arr)

    ### increase high flows - decrease low flows ###
    obs_sim = pd.DataFrame(index=df_ts.index, columns=['Qobs', 'Qsim'])
    obs_sim.loc[:, 'Qobs'] = df_ts.loc[:, 'Qobs']
    tsd = de.highover_lowunder(df_ts.copy(), prop=0.5)
    obs_sim.loc[:, 'Qsim'] = tsd.loc[:, 'Qsim']  # disaggregated time series
    util.fdc_obs_sim_ax(obs_sim['Qobs'], obs_sim['Qsim'], axes_fdc[0, 0], fig_num_fdc[0])
    util.plot_obs_sim_ax(obs_sim['Qobs'], obs_sim['Qsim'], axes_ts[0, 0], fig_num_ts[0])

    # make arrays
    obs_arr = obs_sim['Qobs'].values
    sim_arr = obs_sim['Qsim'].values

    # mean relative bias
    brel_mean = de.calc_brel_mean(obs_arr, sim_arr)
    df_es.iloc[1, 0] = brel_mean
    # remaining relative bias
    brel_rest = de.calc_brel_rest(obs_arr, sim_arr)
    # area of relative remaing bias
    b_area = de.calc_bias_area(brel_rest)
    df_es.iloc[1, 1] = b_area
    # temporal correlation
    temp_cor = de.calc_temp_cor(obs_arr, sim_arr)
    df_es.iloc[1, 2] = temp_cor
    # diagnostic efficiency
    df_es.iloc[1, 3] = de.calc_de(obs_arr, sim_arr)
    # mean flow benchmark of diagnostic efficiency
    df_es.iloc[1, 4] = de.calc_de_mfb(obs_arr)
    # direction of bias
    b_dir = de.calc_bias_dir(brel_rest)
    df_es.iloc[1, 5] = b_dir
    # slope of bias
    b_slope = de.calc_bias_slope(b_area, b_dir)
    df_es.iloc[1, 6] = b_slope
    # convert to radians
    # (y, x) Trigonometric inverse tangent
    df_es.iloc[1, 7] = np.arctan2(brel_mean, b_slope)

    # KGE
    df_es.iloc[1, 8] = de.calc_kge_norm(obs_arr, sim_arr)
    # KGE alpha
    df_es.iloc[1, 9] = de.calc_kge_alpha(obs_arr, sim_arr)
    # KGE beta
    df_es.iloc[1, 10] = de.calc_kge_beta(obs_arr, sim_arr)

    # NSE
    df_es.iloc[1, 11] = de.calc_nse(obs_arr, sim_arr)

    ### decrease high flows - increase low flows ###
    obs_sim = pd.DataFrame(index=df_ts.index, columns=['Qobs', 'Qsim'])
    obs_sim.loc[:, 'Qobs'] = df_ts.loc[:, 'Qobs']
    tsd = de.highunder_lowover(df_ts.copy(), prop=0.5)
    obs_sim.loc[:, 'Qsim'] = tsd.loc[:, 'Qsim']  # smoothed time series
    util.fdc_obs_sim_ax(obs_sim['Qobs'], obs_sim['Qsim'], axes_fdc[0, 1], fig_num_fdc[1])
    util.plot_obs_sim_ax(obs_sim['Qobs'], obs_sim['Qsim'], axes_ts[0, 1], fig_num_ts[1])

    # make arrays
    obs_arr = obs_sim['Qobs'].values
    sim_arr = obs_sim['Qsim'].values

    # mean relative bias
    brel_mean = de.calc_brel_mean(obs_arr, sim_arr)
    df_es.iloc[2, 0] = brel_mean
    # remaining relative bias
    brel_rest = de.calc_brel_rest(obs_arr, sim_arr)
    # area of relative remaing bias
    b_area = de.calc_bias_area(brel_rest)
    df_es.iloc[2, 1] = b_area
    # temporal correlation
    temp_cor = de.calc_temp_cor(obs_arr, sim_arr)
    df_es.iloc[2, 2] = temp_cor
    # diagnostic efficiency
    df_es.iloc[2, 3] = de.calc_de(obs_arr, sim_arr)
    # mean flow benchmark of diagnostic efficiency
    df_es.iloc[2, 4] = de.calc_de_mfb(obs_arr)
    # direction of bias
    b_dir = de.calc_bias_dir(brel_rest)
    df_es.iloc[2, 5] = b_dir
    # slope of bias
    b_slope = de.calc_bias_slope(b_area, b_dir)
    df_es.iloc[2, 6] = b_slope
    # convert to radians
    # (y, x) Trigonometric inverse tangent
    df_es.iloc[2, 7] = np.arctan2(brel_mean, b_slope)

    # KGE
    df_es.iloc[2, 8] = de.calc_kge_norm(obs_arr, sim_arr)
    # KGE alpha
    df_es.iloc[2, 9] = de.calc_kge_alpha(obs_arr, sim_arr)
    # KGE beta
    df_es.iloc[2, 10] = de.calc_kge_beta(obs_arr, sim_arr)

    # NSE
    df_es.iloc[2, 11] = de.calc_nse(obs_arr, sim_arr)

    ### precipitation surplus ###
    obs_sim = pd.DataFrame(index=df_ts.index, columns=['Qobs', 'Qsim'])
    obs_sim.loc[:, 'Qobs'] = df_ts.loc[:, 'Qobs']
    obs_sim.loc[:, 'Qsim'] = de.shift_ts(df_ts['Qobs'].values, offset=1.5)  # positive offset
    util.fdc_obs_sim_ax(obs_sim['Qobs'], obs_sim['Qsim'], axes_fdc[0, 2], fig_num_fdc[2])
    util.plot_obs_sim_ax(obs_sim['Qobs'], obs_sim['Qsim'], axes_ts[0, 2], fig_num_ts[2])

    # make arrays
    obs_arr = obs_sim['Qobs'].values
    sim_arr = obs_sim['Qsim'].values

    # mean relative bias
    brel_mean = de.calc_brel_mean(obs_arr, sim_arr)
    df_es.iloc[3, 0] = brel_mean
    # remaining relative bias
    brel_rest = de.calc_brel_rest(obs_arr, sim_arr)
    # area of relative remaing bias
    b_area = de.calc_bias_area(brel_rest)
    df_es.iloc[3, 1] = b_area
    # temporal correlation
    temp_cor = de.calc_temp_cor(obs_arr, sim_arr)
    df_es.iloc[3, 2] = temp_cor
    # diagnostic efficiency
    df_es.iloc[3, 3] = de.calc_de(obs_arr, sim_arr)
    # mean flow benchmark of diagnostic efficiency
    df_es.iloc[3, 4] = de.calc_de_mfb(obs_arr)
    # direction of bias
    b_dir = de.calc_bias_dir(brel_rest)
    df_es.iloc[3, 5] = b_dir
    # slope of bias
    b_slope = de.calc_bias_slope(b_area, b_dir)
    df_es.iloc[3, 6] = b_slope
    # convert to radians
    # (y, x) Trigonometric inverse tangent
    df_es.iloc[3, 7] = np.arctan2(brel_mean, b_slope)

    # KGE
    df_es.iloc[3, 8] = de.calc_kge_norm(obs_arr, sim_arr)
    # KGE alpha
    df_es.iloc[3, 9] = de.calc_kge_alpha(obs_arr, sim_arr)
    # KGE beta
    df_es.iloc[3, 10] = de.calc_kge_beta(obs_arr, sim_arr)

    # NSE
    df_es.iloc[3, 11] = de.calc_nse(obs_arr, sim_arr)

    ### precipitation shortage ###
    obs_sim = pd.DataFrame(index=df_ts.index, columns=['Qobs', 'Qsim'])
    obs_sim.loc[:, 'Qobs'] = df_ts.loc[:, 'Qobs']
    obs_sim.loc[:, 'Qsim'] = de.shift_ts(df_ts['Qobs'].values, offset=.5)  # negative offset
    util.fdc_obs_sim_ax(obs_sim['Qobs'], obs_sim['Qsim'], axes_fdc[0, 3], fig_num_fdc[3])
    util.plot_obs_sim_ax(obs_sim['Qobs'], obs_sim['Qsim'], axes_ts[0, 3], fig_num_ts[3])

    # make arrays
    obs_arr = obs_sim['Qobs'].values
    sim_arr = obs_sim['Qsim'].values

    # mean relative bias
    brel_mean = de.calc_brel_mean(obs_arr, sim_arr)
    df_es.iloc[4, 0] = brel_mean
    # remaining relative bias
    brel_rest = de.calc_brel_rest(obs_arr, sim_arr)
    # area of relative remaing bias
    b_area = de.calc_bias_area(brel_rest)
    df_es.iloc[4, 1] = b_area
    # temporal correlation
    temp_cor = de.calc_temp_cor(obs_arr, sim_arr)
    df_es.iloc[4, 2] = temp_cor
    # diagnostic efficiency
    df_es.iloc[4, 3] = de.calc_de(obs_arr, sim_arr)
    # mean flow benchmark of diagnostic efficiency
    df_es.iloc[4, 4] = de.calc_de_mfb(obs_arr)
    # direction of bias
    b_dir = de.calc_bias_dir(brel_rest)
    df_es.iloc[4, 5] = b_dir
    # slope of bias
    b_slope = de.calc_bias_slope(b_area, b_dir)
    df_es.iloc[4, 6] = b_slope
    # convert to radians
    # (y, x) Trigonometric inverse tangent
    df_es.iloc[4, 7] = np.arctan2(brel_mean, b_slope)

    # KGE
    df_es.iloc[4, 8] = de.calc_kge_norm(obs_arr, sim_arr)
    # KGE alpha
    df_es.iloc[4, 9] = de.calc_kge_alpha(obs_arr, sim_arr)
    # KGE beta
    df_es.iloc[4, 10] = de.calc_kge_beta(obs_arr, sim_arr)

    # NSE
    df_es.iloc[4, 11] = de.calc_nse(obs_arr, sim_arr)

    ### shuffling ###
    obs_sim = pd.DataFrame(index=df_ts.index, columns=['Qobs', 'Qsim'])
    obs_sim.loc[:, 'Qobs'] = df_ts.loc[:, 'Qobs']
    tss = de.time_shift(df_ts.copy(), random=True)  # shuffled time series
    obs_sim.loc[:, 'Qsim'] = tss.iloc[:, 0].values
    util.fdc_obs_sim_ax(obs_sim['Qobs'], obs_sim['Qsim'], axes_fdc[0, 4], fig_num_fdc[4])
    axes_fdc[0, 4].legend(loc=2, frameon=False)
    util.plot_obs_sim_ax(obs_sim['Qobs'], obs_sim['Qsim'], axes_ts[0, 4], fig_num_ts[4])

    # make arrays
    obs_arr = obs_sim['Qobs'].values
    sim_arr = obs_sim['Qsim'].values

    # mean relative bias
    brel_mean = de.calc_brel_mean(obs_arr, sim_arr)
    df_es.iloc[5, 0] = brel_mean
    # remaining relative bias
    brel_rest = de.calc_brel_rest(obs_arr, sim_arr)
    # area of relative remaing bias
    b_area = de.calc_bias_area(brel_rest)
    df_es.iloc[5, 1] = b_area
    # temporal correlation
    temp_cor = de.calc_temp_cor(obs_arr, sim_arr)
    df_es.iloc[5, 2] = temp_cor
    # diagnostic efficiency
    df_es.iloc[5, 3] = de.calc_de(obs_arr, sim_arr)
    # mean flow benchmark of diagnostic efficiency
    df_es.iloc[5, 4] = de.calc_de_mfb(obs_arr)
    # direction of bias
    b_dir = de.calc_bias_dir(brel_rest)
    df_es.iloc[5, 5] = b_dir
    # slope of bias
    b_slope = de.calc_bias_slope(b_area, b_dir)
    df_es.iloc[5, 6] = b_slope
    # convert to radians
    # (y, x) Trigonometric inverse tangent
    df_es.iloc[5, 7] = np.arctan2(brel_mean, b_slope)

    # KGE
    df_es.iloc[5, 8] = de.calc_kge_norm(obs_arr, sim_arr)
    # KGE alpha
    df_es.iloc[5, 9] = de.calc_kge_alpha(obs_arr, sim_arr)
    # KGE beta
    df_es.iloc[5, 10] = de.calc_kge_beta(obs_arr, sim_arr)

    # NSE
    df_es.iloc[5, 11] = de.calc_nse(obs_arr, sim_arr)

    ### Decrease high flows - Increase low flows and precipitation surplus ###
    obs_sim = pd.DataFrame(index=df_ts.index, columns=['Qobs', 'Qsim'])
    obs_sim.loc[:, 'Qobs'] = df_ts.loc[:, 'Qobs']
    tsd = de.highunder_lowover(df_ts.copy(), prop=0.5)  # smoothed time series
    tso = de.shift_ts(obs_sim.iloc[:, 0].values, offset=.25)  # P offset
    obs_sim.loc[:, 'Qsim'] = tsd.iloc[:, 0].values + tso  # positive offset
    util.fdc_obs_sim_ax(obs_sim['Qobs'], obs_sim['Qsim'], axes_fdc[1, 0], fig_num_fdc[5])
    util.plot_obs_sim_ax(obs_sim['Qobs'], obs_sim['Qsim'], axes_ts[1, 0], fig_num_ts[5])

    # make arrays
    obs_arr = obs_sim['Qobs'].values
    sim_arr = obs_sim['Qsim'].values

    # mean relative bias
    brel_mean = de.calc_brel_mean(obs_arr, sim_arr)
    df_es.iloc[6, 0] = brel_mean
    # remaining relative bias
    brel_rest = de.calc_brel_rest(obs_arr, sim_arr)
    # area of relative remaing bias
    b_area = de.calc_bias_area(brel_rest)
    df_es.iloc[6, 1] = b_area
    # temporal correlation
    temp_cor = de.calc_temp_cor(obs_arr, sim_arr)
    df_es.iloc[6, 2] = temp_cor
    # diagnostic efficiency
    df_es.iloc[6, 3] = de.calc_de(obs_arr, sim_arr)
    # mean flow benchmark of diagnostic efficiency
    df_es.iloc[6, 4] = de.calc_de_mfb(obs_arr)
    # direction of bias
    b_dir = de.calc_bias_dir(brel_rest)
    df_es.iloc[6, 5] = b_dir
    # slope of bias
    b_slope = de.calc_bias_slope(b_area, b_dir)
    df_es.iloc[6, 6] = b_slope
    # convert to radians
    # (y, x) Trigonometric inverse tangent
    df_es.iloc[6, 7] = np.arctan2(brel_mean, b_slope)

    # KGE
    df_es.iloc[6, 8] = de.calc_kge_norm(obs_arr, sim_arr)
    # KGE alpha
    df_es.iloc[6, 9] = de.calc_kge_alpha(obs_arr, sim_arr)
    # KGE beta
    df_es.iloc[6, 10] = de.calc_kge_beta(obs_arr, sim_arr)

    # NSE
    df_es.iloc[6, 11] = de.calc_nse(obs_arr, sim_arr)

    ### Decrease high flows - Increase low flows and precipitation shortage ###
    obs_sim = pd.DataFrame(index=df_ts.index, columns=['Qobs', 'Qsim'])
    obs_sim.loc[:, 'Qobs'] = df_ts.loc[:, 'Qobs']
    tsd = de.highunder_lowover(df_ts.copy(), prop=0.5)  # smoothed time series
    tso = de.shift_ts(obs_sim.iloc[:, 0].values, offset=.25)  # P offset
    obs_sim.loc[:, 'Qsim'] = tsd.iloc[:, 0].values - tso  # negative offset
    util.fdc_obs_sim_ax(obs_sim['Qobs'], obs_sim['Qsim'], axes_fdc[1, 1], fig_num_fdc[6])
    util.plot_obs_sim_ax(obs_sim['Qobs'], obs_sim['Qsim'], axes_ts[1, 1], fig_num_ts[6])

    # make arrays
    obs_arr = obs_sim['Qobs'].values
    sim_arr = obs_sim['Qsim'].values

    # mean relative bias
    brel_mean = de.calc_brel_mean(obs_arr, sim_arr)
    df_es.iloc[7, 0] = brel_mean
    # remaining relative bias
    brel_rest = de.calc_brel_rest(obs_arr, sim_arr)
    # area of relative remaing bias
    b_area = de.calc_bias_area(brel_rest)
    df_es.iloc[7, 1] = b_area
    # temporal correlation
    temp_cor = de.calc_temp_cor(obs_arr, sim_arr)
    df_es.iloc[7, 2] = temp_cor
    # diagnostic efficiency
    df_es.iloc[7, 3] = de.calc_de(obs_arr, sim_arr)
    # mean flow benchmark of diagnostic efficiency
    df_es.iloc[7, 4] = de.calc_de_mfb(obs_arr)
    # direction of bias
    b_dir = de.calc_bias_dir(brel_rest)
    df_es.iloc[7, 5] = b_dir
    # slope of bias
    b_slope = de.calc_bias_slope(b_area, b_dir)
    df_es.iloc[7, 6] = b_slope
    # convert to radians
    # (y, x) Trigonometric inverse tangent
    df_es.iloc[7, 7] = np.arctan2(brel_mean, b_slope)

    # KGE
    df_es.iloc[7, 8] = de.calc_kge_norm(obs_arr, sim_arr)
    # KGE alpha
    df_es.iloc[7, 9] = de.calc_kge_alpha(obs_arr, sim_arr)
    # KGE beta
    df_es.iloc[7, 10] = de.calc_kge_beta(obs_arr, sim_arr)

    # NSE
    df_es.iloc[7, 11] = de.calc_nse(obs_arr, sim_arr)

    ### Increase high flows - Decrease low flows and precipitation surplus ###
    obs_sim = pd.DataFrame(index=df_ts.index, columns=['Qobs', 'Qsim'])
    obs_sim.loc[:, 'Qobs'] = df_ts.loc[:, 'Qobs']
    tsd = de.highover_lowunder(df_ts.copy(), prop=0.5)  # disaggregated time series
    tso = de.shift_ts(obs_sim.iloc[:, 0].values, offset=.25)  # P offset
    obs_sim.loc[:, 'Qsim'] = tsd.iloc[:, 0].values + tso  # positive offset
    util.fdc_obs_sim_ax(obs_sim['Qobs'], obs_sim['Qsim'], axes_fdc[1, 2], fig_num_fdc[7])
    util.plot_obs_sim_ax(obs_sim['Qobs'], obs_sim['Qsim'], axes_ts[1, 2], fig_num_ts[7])

    # make arrays
    obs_arr = obs_sim['Qobs'].values
    sim_arr = obs_sim['Qsim'].values

    # mean relative bias
    brel_mean = de.calc_brel_mean(obs_arr, sim_arr)
    df_es.iloc[8, 0] = brel_mean
    # remaining relative bias
    brel_rest = de.calc_brel_rest(obs_arr, sim_arr)
    # area of relative remaing bias
    b_area = de.calc_bias_area(brel_rest)
    df_es.iloc[8, 1] = b_area
    # temporal correlation
    temp_cor = de.calc_temp_cor(obs_arr, sim_arr)
    df_es.iloc[8, 2] = temp_cor
    # diagnostic efficiency
    df_es.iloc[8, 3] = de.calc_de(obs_arr, sim_arr)
    # mean flow benchmark of diagnostic efficiency
    df_es.iloc[8, 4] = de.calc_de_mfb(obs_arr)
    # direction of bias
    b_dir = de.calc_bias_dir(brel_rest)
    df_es.iloc[8, 5] = b_dir
    # slope of bias
    b_slope = de.calc_bias_slope(b_area, b_dir)
    df_es.iloc[8, 6] = b_slope
    # convert to radians
    # (y, x) Trigonometric inverse tangent
    df_es.iloc[8, 7] = np.arctan2(brel_mean, b_slope)

    # KGE
    df_es.iloc[8, 8] = de.calc_kge_norm(obs_arr, sim_arr)
    # KGE alpha
    df_es.iloc[8, 9] = de.calc_kge_alpha(obs_arr, sim_arr)
    # KGE beta
    df_es.iloc[8, 10] = de.calc_kge_beta(obs_arr, sim_arr)

    # NSE
    df_es.iloc[8, 11] = de.calc_nse(obs_arr, sim_arr)

    ### Increase high flows - Decrease low flows and precipitation shortage ###
    obs_sim = pd.DataFrame(index=df_ts.index, columns=['Qobs', 'Qsim'])
    obs_sim.loc[:, 'Qobs'] = df_ts.loc[:, 'Qobs']
    tsd = de.highover_lowunder(df_ts.copy(), prop=0.5) # disaggregated time series
    tso = de.shift_ts(obs_sim.iloc[:, 0].values, offset=.25)  # P offset
    obs_sim.loc[:, 'Qsim'] = tsd.iloc[:, 0].values - tso  # negative offset
    util.fdc_obs_sim_ax(obs_sim['Qobs'], obs_sim['Qsim'], axes_fdc[1, 3], fig_num_fdc[8])
    util.plot_obs_sim_ax(obs_sim['Qobs'], obs_sim['Qsim'], axes_ts[1, 3], fig_num_ts[8])

    # make arrays
    obs_arr = obs_sim['Qobs'].values
    sim_arr = obs_sim['Qsim'].values

    # mean relative bias
    brel_mean = de.calc_brel_mean(obs_arr, sim_arr)
    df_es.iloc[9, 0] = brel_mean
    # remaining relative bias
    brel_rest = de.calc_brel_rest(obs_arr, sim_arr)
    # area of relative remaing bias
    b_area = de.calc_bias_area(brel_rest)
    df_es.iloc[9, 1] = b_area
    # temporal correlation
    temp_cor = de.calc_temp_cor(obs_arr, sim_arr)
    df_es.iloc[9, 2] = temp_cor
    # diagnostic efficiency
    df_es.iloc[9, 3] = de.calc_de(obs_arr, sim_arr)
    # mean flow benchmark of diagnostic efficiency
    df_es.iloc[9, 4] = de.calc_de_mfb(obs_arr)
    # direction of bias
    b_dir = de.calc_bias_dir(brel_rest)
    df_es.iloc[9, 5] = b_dir
    # slope of bias
    b_slope = de.calc_bias_slope(b_area, b_dir)
    df_es.iloc[9, 6] = b_slope
    # convert to radians
    # (y, x) Trigonometric inverse tangent
    df_es.iloc[9, 7] = np.arctan2(brel_mean, b_slope)

    # KGE
    df_es.iloc[9, 8] = de.calc_kge_norm(obs_arr, sim_arr)
    # KGE alpha
    df_es.iloc[9, 9] = de.calc_kge_alpha(obs_arr, sim_arr)
    # KGE beta
    df_es.iloc[9, 10] = de.calc_kge_beta(obs_arr, sim_arr)

    # NSE
    df_es.iloc[9, 11] = de.calc_nse(obs_arr, sim_arr)

    ### mean flow benchmark ###
    obs_sim = pd.DataFrame(index=df_ts.index, columns=['Qobs', 'Qsim'])
    obs_sim.loc[:, 'Qobs'] = df_ts.loc[:, 'Qobs']
    obs_mean = np.mean(obs_sim['Qobs'].values)
    obs_sim.loc[:, 'Qsim'] = np.repeat(obs_mean, len(obs_sim['Qobs'].values))
    util.fdc_obs_sim_ax(obs_sim['Qobs'], obs_sim['Qsim'], axes_fdc[1, 4], fig_num_fdc[9])
    util.plot_obs_sim_ax(obs_sim['Qobs'], obs_sim['Qsim'], axes_ts[1, 4], fig_num_ts[9])

    # make arrays
    obs_arr = obs_sim['Qobs'].values
    sim_arr = obs_sim['Qsim'].values

    # mean relative bias
    brel_mean = de.calc_brel_mean(obs_arr, sim_arr)
    df_es.iloc[10, 0] = brel_mean
    # remaining relative bias
    brel_rest = de.calc_brel_rest(obs_arr, sim_arr)
    # area of relative remaing bias
    b_area = de.calc_bias_area(brel_rest)
    df_es.iloc[10, 1] = b_area
    # temporal correlation
    temp_cor = de.calc_temp_cor(obs_arr, sim_arr)
    df_es.iloc[10, 2] = temp_cor
    # diagnostic efficiency
    df_es.iloc[10, 3] = de.calc_de(obs_arr, sim_arr)
    # mean flow benchmark of diagnostic efficiency
    df_es.iloc[10, 4] = de.calc_de_mfb(obs_arr)
    # direction of bias
    b_dir = de.calc_bias_dir(brel_rest)
    df_es.iloc[10, 5] = b_dir
    # slope of bias
    b_slope = de.calc_bias_slope(b_area, b_dir)
    df_es.iloc[10, 6] = b_slope
    # convert to radians
    # (y, x) Trigonometric inverse tangent
    df_es.iloc[10, 7] = np.arctan2(brel_mean, b_slope)

    # KGE
    df_es.iloc[10, 8] = de.calc_kge_norm(obs_arr, sim_arr)
    # KGE alpha
    df_es.iloc[10, 9] = de.calc_kge_alpha(obs_arr, sim_arr)
    # KGE beta
    df_es.iloc[10, 10] = de.calc_kge_beta(obs_arr, sim_arr)

    # NSE
    df_es.iloc[10, 11] = de.calc_nse(obs_arr, sim_arr)

    ### Decrease high flows - Increase low flows, precipitation surplus and shuffling ###
    obs_sim = pd.DataFrame(index=df_ts.index, columns=['Qobs', 'Qsim'])
    obs_sim.loc[:, 'Qobs'] = df_ts.loc[:, 'Qobs']
    tsd = de.highunder_lowover(df_ts.copy(), prop=0.5)  # smoothed time series
    tsp = pd.DataFrame(index=df_ts.index, columns=['Qsim'])
    tso = de.shift_ts(obs_sim.iloc[:, 0].values, offset=.25)  # P offset
    tsp.iloc[:, 0]  = tsd.iloc[:, 0].values + tso  # negative offset
    tst = de.time_shift(tsp, random=True)  # shuffling
    obs_sim.loc[:, 'Qsim'] = tst.iloc[:, 0].values
    util.plot_obs_sim_ax(obs_sim['Qobs'], obs_sim['Qsim'], axes_ts[2, 0], fig_num_ts[9])

    # make arrays
    obs_arr = obs_sim['Qobs'].values
    sim_arr = obs_sim['Qsim'].values

    # mean relative bias
    brel_mean = de.calc_brel_mean(obs_arr, sim_arr)
    df_es.iloc[11, 0] = brel_mean
    # remaining relative bias
    brel_rest = de.calc_brel_rest(obs_arr, sim_arr)
    # area of relative remaing bias
    b_area = de.calc_bias_area(brel_rest)
    df_es.iloc[11, 1] = b_area
    # temporal correlation
    temp_cor = de.calc_temp_cor(obs_arr, sim_arr)
    df_es.iloc[11, 2] = temp_cor
    # diagnostic efficiency
    df_es.iloc[11, 3] = de.calc_de(obs_arr, sim_arr)
    # mean flow benchmark of diagnostic efficiency
    df_es.iloc[11, 4] = de.calc_de_mfb(obs_arr)
    # direction of bias
    b_dir = de.calc_bias_dir(brel_rest)
    df_es.iloc[11, 5] = b_dir
    # slope of bias
    b_slope = de.calc_bias_slope(b_area, b_dir)
    df_es.iloc[11, 6] = b_slope
    # convert to radians
    # (y, x) Trigonometric inverse tangent
    df_es.iloc[11, 7] = np.arctan2(brel_mean, b_slope)

    # KGE
    df_es.iloc[11, 8] = de.calc_kge_norm(obs_arr, sim_arr)
    # KGE alpha
    df_es.iloc[11, 9] = de.calc_kge_alpha(obs_arr, sim_arr)
    # KGE beta
    df_es.iloc[11, 10] = de.calc_kge_beta(obs_arr, sim_arr)

    # NSE
    df_es.iloc[11, 11] = de.calc_nse(obs_arr, sim_arr)

    ### Decrease high flows - Increase low flows, precipitation shortage and shuffling ###
    obs_sim = pd.DataFrame(index=df_ts.index, columns=['Qobs', 'Qsim'])
    obs_sim.loc[:, 'Qobs'] = df_ts.loc[:, 'Qobs']
    tsd = de.highunder_lowover(df_ts.copy(), prop=0.5)  # smoothed time series
    tsn = pd.DataFrame(index=df_ts.index, columns=['Qsim'])
    tso = de.shift_ts(obs_sim.iloc[:, 0].values, offset=.25)  # P offset
    tsn.iloc[:, 0]  = tsd.iloc[:, 0].values - tso  # negative offset
    tst = de.time_shift(tsn, random=True)  # shuffling
    obs_sim.loc[:, 'Qsim'] = tst.iloc[:, 0].values
    util.plot_obs_sim_ax(obs_sim['Qobs'], obs_sim['Qsim'], axes_ts[2, 1], fig_num_ts[10])

    # make arrays
    obs_arr = obs_sim['Qobs'].values
    sim_arr = obs_sim['Qsim'].values

    # mean relative bias
    brel_mean = de.calc_brel_mean(obs_arr, sim_arr)
    df_es.iloc[12, 0] = brel_mean
    # remaining relative bias
    brel_rest = de.calc_brel_rest(obs_arr, sim_arr)
    # area of relative remaing bias
    b_area = de.calc_bias_area(brel_rest)
    df_es.iloc[12, 1] = b_area
    # temporal correlation
    temp_cor = de.calc_temp_cor(obs_arr, sim_arr)
    df_es.iloc[12, 2] = temp_cor
    # diagnostic efficiency
    df_es.iloc[12, 3] = de.calc_de(obs_arr, sim_arr)
    # mean flow benchmark of diagnostic efficiency
    df_es.iloc[12, 4] = de.calc_de_mfb(obs_arr)
    # direction of bias
    b_dir = de.calc_bias_dir(brel_rest)
    df_es.iloc[12, 5] = b_dir
    # slope of bias
    b_slope = de.calc_bias_slope(b_area, b_dir)
    df_es.iloc[12, 6] = b_slope
    # convert to radians
    # (y, x) Trigonometric inverse tangent
    df_es.iloc[12, 7] = np.arctan2(brel_mean, b_slope)

    # KGE
    df_es.iloc[12, 8] = de.calc_kge_norm(obs_arr, sim_arr)
    # KGE alpha
    df_es.iloc[12, 9] = de.calc_kge_alpha(obs_arr, sim_arr)
    # KGE beta
    df_es.iloc[12, 10] = de.calc_kge_beta(obs_arr, sim_arr)

    # NSE
    df_es.iloc[12, 11] = de.calc_nse(obs_arr, sim_arr)

    ### Increase high flows - Decrease low flows, precipitation surplus and shuffling ###
    obs_sim = pd.DataFrame(index=df_ts.index, columns=['Qobs', 'Qsim'])
    obs_sim.loc[:, 'Qobs'] = df_ts.loc[:, 'Qobs']
    tsd = de.highover_lowunder(df_ts.copy(), prop=0.5)  # disaggregated time series
    tsp = pd.DataFrame(index=df_ts.index, columns=['Qsim'])
    tso = de.shift_ts(obs_sim.iloc[:, 0].values, offset=.25)  # P offset
    tsp.iloc[:, 0]  = tsd.iloc[:, 0].values + tso  # positve offset
    tst = de.time_shift(tsp, random=True)  # shuffling
    obs_sim.loc[:, 'Qsim'] = tst.iloc[:, 0].values
    util.plot_obs_sim_ax(obs_sim['Qobs'], obs_sim['Qsim'], axes_ts[2, 2], fig_num_ts[11])

    # make arrays
    obs_arr = obs_sim['Qobs'].values
    sim_arr = obs_sim['Qsim'].values

    # mean relative bias
    brel_mean = de.calc_brel_mean(obs_arr, sim_arr)
    df_es.iloc[13, 0] = brel_mean
    # remaining relative bias
    brel_rest = de.calc_brel_rest(obs_arr, sim_arr)
    # area of relative remaing bias
    b_area = de.calc_bias_area(brel_rest)
    df_es.iloc[13, 1] = b_area
    # temporal correlation
    temp_cor = de.calc_temp_cor(obs_arr, sim_arr)
    df_es.iloc[13, 2] = temp_cor
    # diagnostic efficiency
    df_es.iloc[13, 3] = de.calc_de(obs_arr, sim_arr)
    # mean flow benchmark of diagnostic efficiency
    df_es.iloc[13, 4] = de.calc_de_mfb(obs_arr)
    # direction of bias
    b_dir = de.calc_bias_dir(brel_rest)
    df_es.iloc[13, 5] = b_dir
    # slope of bias
    b_slope = de.calc_bias_slope(b_area, b_dir)
    df_es.iloc[13, 6] = b_slope
    # convert to radians
    # (y, x) Trigonometric inverse tangent
    df_es.iloc[13, 7] = np.arctan2(brel_mean, b_slope)

    # KGE
    df_es.iloc[13, 8] = de.calc_kge_norm(obs_arr, sim_arr)
    # KGE alpha
    df_es.iloc[13, 9] = de.calc_kge_alpha(obs_arr, sim_arr)
    # KGE beta
    df_es.iloc[13, 10] = de.calc_kge_beta(obs_arr, sim_arr)

    # NSE
    df_es.iloc[13, 11] = de.calc_nse(obs_arr, sim_arr)

    ### Increase high flows - Decrease low flows, precipitation shortage and shuffling ###
    obs_sim = pd.DataFrame(index=df_ts.index, columns=['Qobs', 'Qsim'])
    obs_sim.loc[:, 'Qobs'] = df_ts.loc[:, 'Qobs']
    tsd = de.highover_lowunder(df_ts.copy(), prop=0.5) # disaggregated time series
    tsn = pd.DataFrame(index=df_ts.index, columns=['Qsim'])
    tso = de.shift_ts(obs_sim.iloc[:, 0].values, offset=.25)  # P offset
    tsn.iloc[:, 0]  = tsd.iloc[:, 0].values - tso  # negative offset
    tst = de.time_shift(tsn, random=True)  # shuffling
    obs_sim.loc[:, 'Qsim'] = tst.iloc[:, 0].values
    util.plot_obs_sim_ax(obs_sim['Qobs'], obs_sim['Qsim'], axes_ts[2, 3], fig_num_ts[12])

    # make arrays
    obs_arr = obs_sim['Qobs'].values
    sim_arr = obs_sim['Qsim'].values

    # mean relative bias
    brel_mean = de.calc_brel_mean(obs_arr, sim_arr)
    df_es.iloc[14, 0] = brel_mean
    # remaining relative bias
    brel_rest = de.calc_brel_rest(obs_arr, sim_arr)
    # area of relative remaing bias
    b_area = de.calc_bias_area(brel_rest)
    df_es.iloc[14, 1] = b_area
    # temporal correlation
    temp_cor = de.calc_temp_cor(obs_arr, sim_arr)
    df_es.iloc[14, 2] = temp_cor
    # diagnostic efficiency
    df_es.iloc[14, 3] = de.calc_de(obs_arr, sim_arr)
    # mean flow benchmark of diagnostic efficiency
    df_es.iloc[14, 4] = de.calc_de_mfb(obs_arr)
    # direction of bias
    b_dir = de.calc_bias_dir(brel_rest)
    df_es.iloc[14, 5] = b_dir
    # slope of bias
    b_slope = de.calc_bias_slope(b_area, b_dir)
    df_es.iloc[14, 6] = b_slope
    # convert to radians
    # (y, x) Trigonometric inverse tangent
    df_es.iloc[14, 7] = np.arctan2(brel_mean, b_slope)

    # KGE
    df_es.iloc[14, 8] = de.calc_kge_norm(obs_arr, sim_arr)
    # KGE alpha
    df_es.iloc[14, 9] = de.calc_kge_alpha(obs_arr, sim_arr)
    # KGE beta
    df_es.iloc[14, 10] = de.calc_kge_beta(obs_arr, sim_arr)

    # NSE
    df_es.iloc[14, 11] = de.calc_nse(obs_arr, sim_arr)

    axes_ts[2, 3].legend(loc=6, bbox_to_anchor=(1.18, .85))

    fig_fdc.savefig('/Users/robinschwemmle/Desktop/PhD/diagnostic_model_efficiency/figures/technical_note/fdc_errors.png', dpi=250)
    fig_ts.savefig('/Users/robinschwemmle/Desktop/PhD/diagnostic_model_efficiency/figures/technical_note/ts_errors.png', dpi=250)

    ### multi diagnostic plot ###
    # make arrays
    brel_mean_arr = df_es['brel_mean'].values
    b_area_arr = df_es['b_area'].values
    temp_cor_arr = df_es['temp_cor'].values
    b_dir_arr = df_es['b_dir'].values
    de_arr = df_es['de'].values
    mfb_arr = df_es['mfb'].values
    diag_arr = df_es['diag'].values
    b_slope_arr = df_es['b_slope'].values

    fig_de = util.vis2d_de_multi_fc(brel_mean_arr, b_area_arr, temp_cor_arr,
                                    de_arr, mfb_arr, b_dir_arr, diag_arr, idx)
    fig_de.savefig('/Users/robinschwemmle/Desktop/PhD/diagnostic_model_efficiency/figures/technical_note/de_diag.pdf', dpi=250)

    # plot without 'j'
    idx = ['1', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm', 'n']
    ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14]
    brel_mean_arr = df_es['brel_mean'].values[ids]
    b_area_arr = df_es['b_area'].values[ids]
    temp_cor_arr = df_es['temp_cor'].values[ids]
    b_dir_arr = df_es['b_dir'].values[ids]
    de_arr = df_es['de'].values[ids]
    mfb_arr = df_es['mfb'].values[ids]
    diag_arr = df_es['diag'].values[ids]
    b_slope_arr = df_es['b_slope'].values[ids]

    fig_de = util.vis2d_de_multi_fc(brel_mean_arr, b_area_arr, temp_cor_arr,
                                    de_arr, mfb_arr, b_dir_arr, diag_arr, idx)
    fig_de.savefig('/Users/robinschwemmle/Desktop/PhD/diagnostic_model_efficiency/figures/technical_note/de_diag_1.pdf', dpi=250)

    # plot only 'j'
    idx = ['j']
    ids = [10]
    brel_mean_arr = df_es['brel_mean'].values[ids]
    b_area_arr = df_es['b_area'].values[ids]
    temp_cor_arr = df_es['temp_cor'].values[ids]
    b_dir_arr = df_es['b_dir'].values[ids]
    de_arr = df_es['de'].values[ids]
    mfb_arr = df_es['mfb'].values[ids]
    diag_arr = df_es['diag'].values[ids]
    b_slope_arr = df_es['b_slope'].values[ids]

    fig_de = util.vis2d_de_multi_fc(brel_mean_arr, b_area_arr, temp_cor_arr,
                                    de_arr, mfb_arr, b_dir_arr, diag_arr, idx)
    fig_de.savefig('/Users/robinschwemmle/Desktop/PhD/diagnostic_model_efficiency/figures/technical_note/de_diag_2.pdf', dpi=250)

    ### multi KGE plot ###
    # make arrays
    idx = ['1', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n']
    alpha_arr = df_es['alpha'].values
    beta_arr = df_es['beta'].values
    temp_cor_arr = df_es['temp_cor'].values
    kge_arr = df_es['kge_norm'].values

    fig_kge = util.vis2d_kge_norm_multi_fc(alpha_arr, beta_arr, temp_cor_arr,
                                           kge_arr, idx)
    fig_kge.savefig('/Users/robinschwemmle/Desktop/PhD/diagnostic_model_efficiency/figures/technical_note/kge_diag.pdf', dpi=250)


    nse_arr = df_es['nse'].values
    brel_mean_arr = df_es['brel_mean'].values
    b_slope_arr = df_es['b_slope'].values
    de_arr = df_es['de'].values
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))
    # scatterplots efficiencies
    sc = sns.scatterplot(kge_arr, de_arr, color='black', ax=ax1)
    sc1 = sns.scatterplot(nse_arr, de_arr, color='red', ax=ax1)
    ax1.plot([-2.05, 1.05], [-2.05, 1.05], ls="--", c=".3")
    ax1.set_ylim(-2.05, 1.05)
    ax1.set_xlim(-2.05, 1.05)
    ax1.set(ylabel='DE [-]', xlabel='KGE [-]')
    ax1.text(.42, -.22, 'NSE [-]', color='red', transform=ax1.transAxes)
    ax1.text(.025, .93, '(a)', transform=ax1.transAxes)
    ax1.text(.05, .1, '1:1', rotation=45, transform=ax1.transAxes)
    # for i, txt in enumerate(df_es.index):
    #     ax.annotate(txt, (kge_arr[i], de_arr[i]), color='black', fontsize=15)
    #     ax.annotate(txt, (nse_arr[i], de_arr[i]), color='red', fontsize=15)

    # scatterplots components
    sc = sns.scatterplot(alpha_arr - 1, brel_mean_arr, color='black', ax=ax2)
    sc1 = sns.scatterplot(beta_arr - 1, b_slope_arr, color='red', ax=ax2)
    ax2.plot([-2.05, 2.05], [-2.05, 2.05], ls="--", c=".3")
    ax2.set_ylim(-2.05, 2.05)
    ax2.set_xlim(-2.05, 2.05)
    ax2.set(ylabel=r'$\overline{B_{rel}}$ [-]', xlabel=r'$\alpha$ - 1 [-]')
    ax2.text(-.29, .415, r'$B_{slope}$ [-]', color='red', transform=ax2.transAxes, rotation=90)
    ax2.text(.42, -.22, r'$\beta$ - 1 [-]', color='red', transform=ax2.transAxes)
    ax2.text(.03, .93, '(b)', transform=ax2.transAxes)
    ax2.text(.05, .1, '1:1', rotation=45, transform=ax2.transAxes)
    fig.subplots_adjust(wspace=.35, bottom=.2)
    fig.savefig('/Users/robinschwemmle/Desktop/PhD/diagnostic_model_efficiency/figures/technical_note/scatter_eff_comp.png', dpi=250)
    # for i, txt in enumerate(df_es.index):
    #     ax.annotate(txt, (alpha_arr[i] - 1, brel_mean_arr[i]), color='black', fontsize=15)
    #     ax.annotate(txt, (beta_arr[i] - 1, b_slope_arr[i]), color='red', fontsize=15)

    # export table
    path_csv = '/Users/robinschwemmle/Desktop/PhD/diagnostic_model_efficiency/figures/technical_note/table_eff.csv'
    df_es_t = df_es.T
    df_es_t = df_es_t.loc[['de', 'kge_norm', 'nse'], :]
    df_es_t = df_es_t.round(2)
    df_es_t.to_csv(path_csv, header=True, index=True, sep=';')

    ### camels
    # dataframe efficiency measures
    idx = ['05', '48', '94']
    cols = ['brel_mean', 'b_area', 'temp_cor', 'de', 'mfb', 'b_dir', 'b_slope', 'diag', 'kge_norm', 'alpha', 'beta', 'nse']
    df_es_cam = pd.DataFrame(index=idx, columns=cols, dtype=np.float64)

    path_cam1 = '/Users/robinschwemmle/Desktop/PhD/diagnostic_model_efficiency/examples/camels_example_data/13331500_05_model_output.txt'
    path_cam2 = '/Users/robinschwemmle/Desktop/PhD/diagnostic_model_efficiency/examples/camels_example_data/13331500_48_model_output.txt'
    path_cam3 = '/Users/robinschwemmle/Desktop/PhD/diagnostic_model_efficiency/examples/camels_example_data/13331500_94_model_output.txt'
    df_cam1 = util.import_camels_obs_sim(path_cam1)
    df_cam2 = util.import_camels_obs_sim(path_cam2)
    df_cam3 = util.import_camels_obs_sim(path_cam3)

    fig, axes = plt.subplots(3, 2, figsize=(10,10), sharex='col')
    fig.text(0.06, 0.5, r'[mm $d^{-1}$]', ha='center', va='center', rotation='vertical')
    fig.text(0.5, 0.5, r'[mm $d^{-1}$]', ha='center', va='center', rotation='vertical')
    fig.text(0.25, 0.05, 'Time [Years]', ha='center', va='center')
    fig.text(0.75, 0.05, 'Exceedence probabilty [-]', ha='center', va='center')

    util.plot_obs_sim_ax(df_cam1['Qobs'], df_cam1['Qsim'], axes[0,0], '')
    axes[0,0].text(.95, .95, '(a; {})'.format(idx[0]),
                   transform=axes[0,0].transAxes, ha='right', va='top')
    # format the ticks
    years_10 = mdates.YearLocator(10)
    years_5 = mdates.YearLocator(5)
    yearsFmt = mdates.DateFormatter('%Y')
    axes[0,0].xaxis.set_major_locator(years_10)
    axes[0,0].xaxis.set_major_formatter(yearsFmt)
    axes[0,0].xaxis.set_minor_locator(years_5)
    util.fdc_obs_sim_ax(df_cam1['Qobs'], df_cam1['Qsim'], axes[0,1], '')
    axes[0,1].text(.95, .95, '(b; {})'.format(idx[0]),
                   transform=axes[0,1].transAxes, ha='right', va='top')
    # legend above plot
    axes[0,1].legend(loc=2, labels=['Observed', 'Simulated'], ncol=2, frameon=False, bbox_to_anchor=(-0.6, 1.2))

    util.plot_obs_sim_ax(df_cam2['Qobs'], df_cam2['Qsim'], axes[1,0], '')
    axes[1,0].text(.95, .95, '(c; {})'.format(idx[1]),
                   transform=axes[1,0].transAxes, ha='right', va='top')
    # format the ticks
    years_10 = mdates.YearLocator(10)
    years_5 = mdates.YearLocator(5)
    yearsFmt = mdates.DateFormatter('%Y')
    axes[1,0].xaxis.set_major_locator(years_10)
    axes[1,0].xaxis.set_major_formatter(yearsFmt)
    axes[1,0].xaxis.set_minor_locator(years_5)
    util.fdc_obs_sim_ax(df_cam1['Qobs'], df_cam1['Qsim'], axes[1,1], '')
    axes[1,1].text(.95, .95, '(d; {})'.format(idx[1]),
                   transform=axes[1,1].transAxes, ha='right', va='top')

    util.plot_obs_sim_ax(df_cam1['Qobs'], df_cam1['Qsim'], axes[2,0], '')
    axes[2,0].text(.95, .95, '(e; {})'.format(idx[2]),
                   transform=axes[2,0].transAxes, ha='right', va='top')
    # format the ticks
    years_10 = mdates.YearLocator(10)
    years_5 = mdates.YearLocator(5)
    yearsFmt = mdates.DateFormatter('%Y')
    axes[2,0].xaxis.set_major_locator(years_10)
    axes[2,0].xaxis.set_major_formatter(yearsFmt)
    axes[2,0].xaxis.set_minor_locator(years_5)
    util.fdc_obs_sim_ax(df_cam1['Qobs'], df_cam1['Qsim'], axes[2,1], '')
    axes[2,1].text(.95, .95, '(f; {})'.format(idx[2]),
                   transform=axes[2,1].transAxes, ha='right', va='top')

    fig.subplots_adjust(wspace=0.3)
    fig.savefig('/Users/robinschwemmle/Desktop/PhD/diagnostic_model_efficiency/figures/technical_note/ts_fdc_real_case.png', dpi=250)

    obs_arr = df_cam1['Qobs'].values
    sim_arr = df_cam1['Qsim'].values

    # mean relative bias
    brel_mean = de.calc_brel_mean(obs_arr, sim_arr)
    df_es_cam.iloc[0, 0] = brel_mean
    # remaining relative bias
    brel_rest = de.calc_brel_rest(obs_arr, sim_arr)
    # area of relative remaing bias
    b_area = de.calc_bias_area(brel_rest)
    df_es_cam.iloc[0, 1] = b_area
    # temporal correlation
    temp_cor = de.calc_temp_cor(obs_arr, sim_arr)
    df_es_cam.iloc[0, 2] = temp_cor
    # diagnostic efficiency
    df_es_cam.iloc[0, 3] = de.calc_de(obs_arr, sim_arr)
    # mean flow benchmark of diagnostic efficiency
    df_es_cam.iloc[0, 4] = de.calc_de_mfb(obs_arr)
    # direction of bias
    b_dir = de.calc_bias_dir(brel_rest)
    df_es_cam.iloc[0, 5] = b_dir
    # slope of bias
    b_slope = de.calc_bias_slope(b_area, b_dir)
    df_es_cam.iloc[0, 6] = b_slope
    # convert to radians
    # (y, x) Trigonometric inverse tangent
    df_es_cam.iloc[0, 7] = np.arctan2(brel_mean, b_slope)

    # KGE
    df_es_cam.iloc[0, 8] = de.calc_kge_norm(obs_arr, sim_arr)
    # KGE alpha
    df_es_cam.iloc[0, 9] = de.calc_kge_alpha(obs_arr, sim_arr)
    # KGE beta
    df_es_cam.iloc[0, 10] = de.calc_kge_beta(obs_arr, sim_arr)

    # NSE
    df_es_cam.iloc[0, 11] = de.calc_nse(obs_arr, sim_arr)

    obs_arr = df_cam2['Qobs'].values
    sim_arr = df_cam2['Qsim'].values

    # mean relative bias
    brel_mean = de.calc_brel_mean(obs_arr, sim_arr)
    df_es_cam.iloc[1, 0] = brel_mean
    # remaining relative bias
    brel_rest = de.calc_brel_rest(obs_arr, sim_arr)
    # area of relative remaing bias
    b_area = de.calc_bias_area(brel_rest)
    df_es_cam.iloc[1, 1] = b_area
    # temporal correlation
    temp_cor = de.calc_temp_cor(obs_arr, sim_arr)
    df_es_cam.iloc[1, 2] = temp_cor
    # diagnostic efficiency
    df_es_cam.iloc[1, 3] = de.calc_de(obs_arr, sim_arr)
    # mean flow benchmark of diagnostic efficiency
    df_es_cam.iloc[1, 4] = de.calc_de_mfb(obs_arr)
    # direction of bias
    b_dir = de.calc_bias_dir(brel_rest)
    df_es_cam.iloc[1, 5] = b_dir
    # slope of bias
    b_slope = de.calc_bias_slope(b_area, b_dir)
    df_es_cam.iloc[1, 6] = b_slope
    # convert to radians
    # (y, x) Trigonometric inverse tangent
    df_es_cam.iloc[1, 7] = np.arctan2(brel_mean, b_slope)

    # KGE
    df_es_cam.iloc[1, 8] = de.calc_kge_norm(obs_arr, sim_arr)
    # KGE alpha
    df_es_cam.iloc[1, 9] = de.calc_kge_alpha(obs_arr, sim_arr)
    # KGE beta
    df_es_cam.iloc[1, 10] = de.calc_kge_beta(obs_arr, sim_arr)

    # NSE
    df_es_cam.iloc[1, 11] = de.calc_nse(obs_arr, sim_arr)

    obs_arr = df_cam3['Qobs'].values
    sim_arr = df_cam3['Qsim'].values

    # mean relative bias
    brel_mean = de.calc_brel_mean(obs_arr, sim_arr)
    df_es_cam.iloc[2, 0] = brel_mean
    # remaining relative bias
    brel_rest = de.calc_brel_rest(obs_arr, sim_arr)
    # area of relative remaing bias
    b_area = de.calc_bias_area(brel_rest)
    df_es_cam.iloc[2, 1] = b_area
    # temporal correlation
    temp_cor = de.calc_temp_cor(obs_arr, sim_arr)
    df_es_cam.iloc[2, 2] = temp_cor
    # diagnostic efficiency
    df_es_cam.iloc[2, 3] = de.calc_de(obs_arr, sim_arr)
    # mean flow benchmark of diagnostic efficiency
    df_es_cam.iloc[2, 4] = de.calc_de_mfb(obs_arr)
    # direction of bias
    b_dir = de.calc_bias_dir(brel_rest)
    df_es_cam.iloc[2, 5] = b_dir
    # slope of bias
    b_slope = de.calc_bias_slope(b_area, b_dir)
    df_es_cam.iloc[2, 6] = b_slope
    # convert to radians
    # (y, x) Trigonometric inverse tangent
    df_es_cam.iloc[2, 7] = np.arctan2(brel_mean, b_slope)

    # KGE
    df_es_cam.iloc[2, 8] = de.calc_kge_norm(obs_arr, sim_arr)
    # KGE alpha
    df_es_cam.iloc[2, 9] = de.calc_kge_alpha(obs_arr, sim_arr)
    # KGE beta
    df_es_cam.iloc[2, 10] = de.calc_kge_beta(obs_arr, sim_arr)

    # NSE
    df_es_cam.iloc[2, 11] = de.calc_nse(obs_arr, sim_arr)

    path_csv = '/Users/robinschwemmle/Desktop/PhD/diagnostic_model_efficiency/figures/technical_note/table_eff_real_case.csv'
    df_es_cam.to_csv(path_csv, header=True, index=True, sep=';')

    ### multi diagnostic plot ###
    # make arrays
    brel_mean_arr = df_es_cam['brel_mean'].values
    b_area_arr = df_es_cam['b_area'].values
    temp_cor_arr = df_es_cam['temp_cor'].values
    b_dir_arr = df_es_cam['b_dir'].values
    de_arr = df_es_cam['de'].values
    mfb_arr = df_es_cam['mfb'].values
    diag_arr = df_es_cam['diag'].values
    b_slope_arr = df_es_cam['b_slope'].values

    fig_de = util.vis2d_de_multi_fc(brel_mean_arr, b_area_arr, temp_cor_arr,
                                    de_arr, mfb_arr, b_dir_arr, diag_arr, idx)
    fig_de.savefig('/Users/robinschwemmle/Desktop/PhD/diagnostic_model_efficiency/figures/technical_note/de_diag_real_case.pdf', dpi=250)

    ### multi KGE plot ###
    # make arrays
    alpha_arr = df_es_cam['alpha'].values
    beta_arr = df_es_cam['beta'].values
    kge_arr = df_es_cam['kge_norm'].values

    fig_kge = util.vis2d_kge_norm_multi_fc(alpha_arr, beta_arr, temp_cor_arr,
                                      kge_arr, idx)
    fig_kge.savefig('/Users/robinschwemmle/Desktop/PhD/diagnostic_model_efficiency/figures/technical_note/kge_diag_real_case.pdf', dpi=250)
