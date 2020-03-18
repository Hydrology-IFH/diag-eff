#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os  # load modules first before importing .spydata
PATH = '/Users/robinschwemmle/Desktop/PhD/diagnostic_efficiency/'
os.chdir(PATH)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from de import de
from de import generate_errors
from de import kge
from de import nse
from de import util
import seaborn as sns

# controlling figure aesthetics
sns.set_style('ticks', {'xtick.major.size': 8, 'ytick.major.size': 8})
sns.set_context("paper", font_scale=1.5)

if __name__ == "__main__":
    #==========================================================
    # import observed streamflow time series
    #==========================================================
    # 619.11 km2; AI: 0.82
    area = 619.11
    path = os.path.join(os.getcwd(),
                        'examples/camels_example_data/13331500_streamflow_qc.txt')
    ## 191.55 km2; AI: 2.04
    # area = 191.55
    # path = os.path.join(os.getcwd(),
    #                     'examples/camels_example_data/06332515_streamflow_qc.txt')
    ## 190.65 km2; AI: 2.98
    # area = 190.65
    # path = os.path.join(os.getcwd(),
    #                     'examples/camels_example_data/09512280_streamflow_qc.txt')
    ## 66.57 km2; AI: 0.27
    # area = 66.57
    # path = os.path.join(os.getcwd(),
    #                     'examples/camels_example_data/12114500_streamflow_qc.txt')

    # import observed time series
    df_ts = util.import_camels_ts(path, sep=r"\s+", catch_area=area)
    util.plot_ts(df_ts)

    #==========================================================
    # proof of concept
    #==========================================================
    # prepare figures
    fig_num_fdc = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)',
                   '(i)']
    fig_fdc, axes_fdc = plt.subplots(2, 5, sharey=True, sharex=True,
                                     figsize=(14, 6))
    fig_fdc.text(0.5, 0.02, 'Exceedence probabilty [-]', ha='center',
                 va='center')
    fig_fdc.text(0.08, 0.5, r'[mm $d^{-1}$]', ha='center', va='center',
                 rotation='vertical')
    axes_fdc[1, 4].remove()

    fig_num_ts = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)',
                  '(i)', '(j)', '(k)', '(l)', '(m)']
    fig_ts, axes_ts = plt.subplots(3, 5, sharey=True, sharex=True,
                                   figsize=(14, 9))
    fig_ts.text(0.5, 0.05, 'Time [Years]', ha='center', va='center')
    fig_ts.text(0.08, 0.5, r'[mm $d^{-1}$]', ha='center', va='center',
                rotation='vertical')
    axes_ts[2, 3].remove()
    axes_ts[2, 4].remove()

    # dataframe to compare DE, KGE and NSE
    idx = ['1', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
           'm']
    cols = ['brel_mean', 'b_area', 'temp_cor', 'de', 'b_dir', 'b_slope',
            'phi', 'beta', 'alpha', 'kge', 'nse']
    df_es = pd.DataFrame(index=idx, columns=cols, dtype=np.float32)

    #----------------------------------------------------------
    # perfect simulation
    #----------------------------------------------------------
    obs_sim = pd.DataFrame(index=df_ts.index, columns=['Qobs', 'Qsim'])
    obs_sim.loc[:, 'Qobs'] = df_ts.loc[:, 'Qobs']
    obs_sim.loc[:, 'Qsim'] = df_ts.loc[:, 'Qobs']  # observed time series

    # make numpy arrays
    obs_arr = obs_sim['Qobs'].values  # observed time series
    sim_arr = obs_sim['Qsim'].values  # manipulated time series

    # mean relative bias
    brel_mean = de.calc_brel_mean(obs_arr, sim_arr)
    df_es.iloc[0, 0] = brel_mean
    # residual relative bias
    brel_rest = de.calc_brel_rest(obs_arr, sim_arr)
    # area of relative remaing bias
    b_area = de.calc_bias_area(brel_rest)
    df_es.iloc[0, 1] = b_area
    # temporal correlation
    temp_cor = de.calc_temp_cor(obs_arr, sim_arr)
    df_es.iloc[0, 2] = temp_cor
    # diagnostic efficiency
    df_es.iloc[0, 3] = de.calc_de(obs_arr, sim_arr)
    # direction of bias
    b_dir = de.calc_bias_dir(brel_rest)
    df_es.iloc[0, 4] = b_dir
    # slope of bias
    b_slope = de.calc_bias_slope(b_area, b_dir)
    df_es.iloc[0, 5] = b_slope
    # convert to radians
    # (y, x) Trigonometric inverse tangent
    df_es.iloc[0, 6] = np.arctan2(brel_mean, b_slope)

    # KGE beta
    df_es.iloc[0, 7] = kge.calc_kge_beta(obs_arr, sim_arr)
    # KGE alpha
    df_es.iloc[0, 8] = kge.calc_kge_alpha(obs_arr, sim_arr)
    # # KGE gamma
    # df_es.iloc[0, 8] = kge.calc_kge_gamma(obs_arr, sim_arr)
    # KGE
    df_es.iloc[0, 9] = kge.calc_kge(obs_arr, sim_arr)

    # NSE
    df_es.iloc[0, 10] = nse.calc_nse(obs_arr, sim_arr)

    #----------------------------------------------------------
    # positive constant error
    #----------------------------------------------------------
    obs_sim = pd.DataFrame(index=df_ts.index, columns=['Qobs', 'Qsim'])
    obs_sim.loc[:, 'Qobs'] = df_ts.loc[:, 'Qobs']
    # generate positive constant error
    obs_sim.loc[:, 'Qsim'] = generate_errors.constant(df_ts['Qobs'].values,
                                                      offset=1.25)
    util.fdc_obs_sim_ax(obs_sim['Qobs'], obs_sim['Qsim'], axes_fdc[0, 0],
                        fig_num_fdc[0])
    util.plot_obs_sim_ax(obs_sim['Qobs'], obs_sim['Qsim'], axes_ts[0, 0],
                         fig_num_ts[0])

    # make arrays
    obs_arr = obs_sim['Qobs'].values
    sim_arr = obs_sim['Qsim'].values

    # mean relative bias
    brel_mean = de.calc_brel_mean(obs_arr, sim_arr)
    df_es.iloc[1, 0] = brel_mean
    # residual relative bias
    brel_rest = de.calc_brel_rest(obs_arr, sim_arr)
    # area of relative remaing bias
    b_area = de.calc_bias_area(brel_rest)
    df_es.iloc[1, 1] = b_area
    # temporal correlation
    temp_cor = de.calc_temp_cor(obs_arr, sim_arr)
    df_es.iloc[1, 2] = temp_cor
    # diagnostic efficiency
    df_es.iloc[1, 3] = de.calc_de(obs_arr, sim_arr)
    # direction of bias
    b_dir = de.calc_bias_dir(brel_rest)
    df_es.iloc[1, 4] = b_dir
    # slope of bias
    b_slope = de.calc_bias_slope(b_area, b_dir)
    df_es.iloc[1, 5] = b_slope
    # convert to radians
    # (y, x) Trigonometric inverse tangent
    df_es.iloc[1, 6] = np.arctan2(brel_mean, b_slope)

    # KGE beta
    df_es.iloc[1, 7] = kge.calc_kge_beta(obs_arr, sim_arr)
    # KGE alpha
    df_es.iloc[1, 8] = kge.calc_kge_alpha(obs_arr, sim_arr)
    # # KGE gamma
    # df_es.iloc[0, 8] = kge.calc_kge_gamma(obs_arr, sim_arr)
    # KGE
    df_es.iloc[1, 9] = kge.calc_kge(obs_arr, sim_arr)

    # NSE
    df_es.iloc[1, 10] = nse.calc_nse(obs_arr, sim_arr)

    #----------------------------------------------------------
    # negative constant error
    #----------------------------------------------------------
    obs_sim = pd.DataFrame(index=df_ts.index, columns=['Qobs', 'Qsim'])
    obs_sim.loc[:, 'Qobs'] = df_ts.loc[:, 'Qobs']
    # generate negative constant error
    obs_sim.loc[:, 'Qsim'] = generate_errors.constant(df_ts['Qobs'].values,
                                                      offset=.75)
    util.fdc_obs_sim_ax(obs_sim['Qobs'], obs_sim['Qsim'], axes_fdc[0, 1],
                        fig_num_fdc[1])
    util.plot_obs_sim_ax(obs_sim['Qobs'], obs_sim['Qsim'], axes_ts[0, 1],
                         fig_num_ts[1])

    # make arrays
    obs_arr = obs_sim['Qobs'].values
    sim_arr = obs_sim['Qsim'].values

    # mean relative bias
    brel_mean = de.calc_brel_mean(obs_arr, sim_arr)
    df_es.iloc[2, 0] = brel_mean
    # residual relative bias
    brel_rest = de.calc_brel_rest(obs_arr, sim_arr)
    # area of relative remaing bias
    b_area = de.calc_bias_area(brel_rest)
    df_es.iloc[2, 1] = b_area
    # temporal correlation
    temp_cor = de.calc_temp_cor(obs_arr, sim_arr)
    df_es.iloc[2, 2] = temp_cor
    # diagnostic efficiency
    df_es.iloc[2, 3] = de.calc_de(obs_arr, sim_arr)
    # direction of bias
    b_dir = de.calc_bias_dir(brel_rest)
    df_es.iloc[2, 4] = b_dir
    # slope of bias
    b_slope = de.calc_bias_slope(b_area, b_dir)
    df_es.iloc[2, 5] = b_slope
    # convert to radians
    # (y, x) Trigonometric inverse tangent
    df_es.iloc[2, 6] = np.arctan2(brel_mean, b_slope)

    # KGE beta
    df_es.iloc[2, 7] = kge.calc_kge_beta(obs_arr, sim_arr)
    # KGE alpha
    df_es.iloc[2, 8] = kge.calc_kge_alpha(obs_arr, sim_arr)
    # # KGE gamma
    # df_es.iloc[0, 8] = kge.calc_kge_gamma(obs_arr, sim_arr)
    # KGE
    df_es.iloc[2, 9] = kge.calc_kge(obs_arr, sim_arr)

    # NSE
    df_es.iloc[2, 10] = nse.calc_nse(obs_arr, sim_arr)

    #----------------------------------------------------------
    # positive dynamic error
    #----------------------------------------------------------
    obs_sim = pd.DataFrame(index=df_ts.index, columns=['Qobs', 'Qsim'])
    obs_sim.loc[:, 'Qobs'] = df_ts.loc[:, 'Qobs']
    # generate positive dynamic error
    tsd = generate_errors.positive_dynamic(df_ts.copy(), prop=0.5)
    obs_sim.loc[:, 'Qsim'] = tsd.loc[:, 'Qsim']
    util.fdc_obs_sim_ax(obs_sim['Qobs'], obs_sim['Qsim'], axes_fdc[0, 2],
                        fig_num_fdc[2])
    util.plot_obs_sim_ax(obs_sim['Qobs'], obs_sim['Qsim'], axes_ts[0, 2],
                         fig_num_ts[2])

    # make arrays
    obs_arr = obs_sim['Qobs'].values
    sim_arr = obs_sim['Qsim'].values

    # mean relative bias
    brel_mean = de.calc_brel_mean(obs_arr, sim_arr)
    df_es.iloc[3, 0] = brel_mean
    # residual relative bias
    brel_rest = de.calc_brel_rest(obs_arr, sim_arr)
    # area of relative remaing bias
    b_area = de.calc_bias_area(brel_rest)
    df_es.iloc[3, 1] = b_area
    # temporal correlation
    temp_cor = de.calc_temp_cor(obs_arr, sim_arr)
    df_es.iloc[3, 2] = temp_cor
    # diagnostic efficiency
    df_es.iloc[3, 3] = de.calc_de(obs_arr, sim_arr)
    # direction of bias
    b_dir = de.calc_bias_dir(brel_rest)
    df_es.iloc[3, 4] = b_dir
    # slope of bias
    b_slope = de.calc_bias_slope(b_area, b_dir)
    df_es.iloc[3, 5] = b_slope
    # convert to radians
    # (y, x) Trigonometric inverse tangent
    df_es.iloc[3, 6] = np.arctan2(brel_mean, b_slope)

    # KGE beta
    df_es.iloc[3, 7] = kge.calc_kge_beta(obs_arr, sim_arr)
    # KGE alpha
    df_es.iloc[3, 8] = kge.calc_kge_alpha(obs_arr, sim_arr)
    # # KGE gamma
    # df_es.iloc[0, 8] = kge.calc_kge_gamma(obs_arr, sim_arr)
    # KGE
    df_es.iloc[3, 9] = kge.calc_kge(obs_arr, sim_arr)

    # NSE
    df_es.iloc[3, 10] = nse.calc_nse(obs_arr, sim_arr)

    #----------------------------------------------------------
    # negative dynamic error
    #----------------------------------------------------------
    obs_sim = pd.DataFrame(index=df_ts.index, columns=['Qobs', 'Qsim'])
    obs_sim.loc[:, 'Qobs'] = df_ts.loc[:, 'Qobs']
    # generate negative dynamic error
    tsd = generate_errors.negative_dynamic(df_ts.copy(), prop=0.5)
    obs_sim.loc[:, 'Qsim'] = tsd.loc[:, 'Qsim']
    util.fdc_obs_sim_ax(obs_sim['Qobs'], obs_sim['Qsim'], axes_fdc[0, 3],
                        fig_num_fdc[3])
    util.plot_obs_sim_ax(obs_sim['Qobs'], obs_sim['Qsim'], axes_ts[0, 3],
                         fig_num_ts[3])

    # make arrays
    obs_arr = obs_sim['Qobs'].values
    sim_arr = obs_sim['Qsim'].values

    # mean relative bias
    brel_mean = de.calc_brel_mean(obs_arr, sim_arr)
    df_es.iloc[4, 0] = brel_mean
    # residual relative bias
    brel_rest = de.calc_brel_rest(obs_arr, sim_arr)
    # area of relative remaing bias
    b_area = de.calc_bias_area(brel_rest)
    df_es.iloc[4, 1] = b_area
    # temporal correlation
    temp_cor = de.calc_temp_cor(obs_arr, sim_arr)
    df_es.iloc[4, 2] = temp_cor
    # diagnostic efficiency
    df_es.iloc[4, 3] = de.calc_de(obs_arr, sim_arr)
    # direction of bias
    b_dir = de.calc_bias_dir(brel_rest)
    df_es.iloc[4, 4] = b_dir
    # slope of bias
    b_slope = de.calc_bias_slope(b_area, b_dir)
    df_es.iloc[4, 5] = b_slope
    # convert to radians
    # (y, x) Trigonometric inverse tangent
    df_es.iloc[4, 6] = np.arctan2(brel_mean, b_slope)

    # KGE beta
    df_es.iloc[4, 7] = kge.calc_kge_beta(obs_arr, sim_arr)
    # KGE alpha
    df_es.iloc[4, 8] = kge.calc_kge_alpha(obs_arr, sim_arr)
    # # KGE gamma
    # df_es.iloc[0, 8] = kge.calc_kge_gamma(obs_arr, sim_arr)
    # KGE
    df_es.iloc[4, 9] = kge.calc_kge(obs_arr, sim_arr)

    # NSE
    df_es.iloc[4, 10] = nse.calc_nse(obs_arr, sim_arr)

    #----------------------------------------------------------
    # timing error
    #----------------------------------------------------------
    obs_sim = pd.DataFrame(index=df_ts.index, columns=['Qobs', 'Qsim'])
    obs_sim.loc[:, 'Qobs'] = df_ts.loc[:, 'Qobs']
    # generate timing error
    tss = generate_errors.timing(df_ts.copy(), random=True)  # shuffling
    obs_sim.loc[:, 'Qsim'] = tss.iloc[:, 0].values
    util.fdc_obs_sim_ax(obs_sim['Qobs'], obs_sim['Qsim'], axes_fdc[0, 4],
                        fig_num_fdc[4])
    util.plot_obs_sim_ax(obs_sim['Qobs'], obs_sim['Qsim'], axes_ts[0, 4],
                         fig_num_ts[4])

    # make arrays
    obs_arr = obs_sim['Qobs'].values
    sim_arr = obs_sim['Qsim'].values

    # mean relative bias
    brel_mean = de.calc_brel_mean(obs_arr, sim_arr)
    df_es.iloc[5, 0] = brel_mean
    # residual relative bias
    brel_rest = de.calc_brel_rest(obs_arr, sim_arr)
    # area of relative remaing bias
    b_area = de.calc_bias_area(brel_rest)
    df_es.iloc[5, 1] = b_area
    # temporal correlation
    temp_cor = de.calc_temp_cor(obs_arr, sim_arr)
    df_es.iloc[5, 2] = temp_cor
    # diagnostic efficiency
    df_es.iloc[5, 3] = de.calc_de(obs_arr, sim_arr)
    # direction of bias
    b_dir = de.calc_bias_dir(brel_rest)
    df_es.iloc[5, 4] = b_dir
    # slope of bias
    b_slope = de.calc_bias_slope(b_area, b_dir)
    df_es.iloc[5, 5] = b_slope
    # convert to radians
    # (y, x) Trigonometric inverse tangent
    df_es.iloc[5, 6] = np.arctan2(brel_mean, b_slope)

    # KGE beta
    df_es.iloc[5, 7] = kge.calc_kge_beta(obs_arr, sim_arr)
    # KGE alpha
    df_es.iloc[5, 8] = kge.calc_kge_alpha(obs_arr, sim_arr)
    # # KGE gamma
    # df_es.iloc[0, 8] = kge.calc_kge_gamma(obs_arr, sim_arr)
    # KGE
    df_es.iloc[5, 9] = kge.calc_kge(obs_arr, sim_arr)

    # NSE
    df_es.iloc[5, 10] = nse.calc_nse(obs_arr, sim_arr)

    #----------------------------------------------------------
    # negative dynamic error and negative constant error
    #----------------------------------------------------------
    obs_sim = pd.DataFrame(index=df_ts.index, columns=['Qobs', 'Qsim'])
    obs_sim.loc[:, 'Qobs'] = df_ts.loc[:, 'Qobs']
    # generate negative dynamic error
    tsd = generate_errors.negative_dynamic(df_ts.copy(), prop=0.5)
    # generate positive constant error
    tso = generate_errors.constant(obs_sim.iloc[:, 0].values, offset=.25)
    obs_sim.loc[:, 'Qsim'] = tsd.iloc[:, 0].values - tso  # negative offset
    util.fdc_obs_sim_ax(obs_sim['Qobs'], obs_sim['Qsim'], axes_fdc[1, 0],
                        fig_num_fdc[5])
    util.plot_obs_sim_ax(obs_sim['Qobs'], obs_sim['Qsim'], axes_ts[1, 0],
                         fig_num_ts[5])

    # make arrays
    obs_arr = obs_sim['Qobs'].values
    sim_arr = obs_sim['Qsim'].values

    # mean relative bias
    brel_mean = de.calc_brel_mean(obs_arr, sim_arr)
    df_es.iloc[6, 0] = brel_mean
    # residual relative bias
    brel_rest = de.calc_brel_rest(obs_arr, sim_arr)
    # area of relative remaing bias
    b_area = de.calc_bias_area(brel_rest)
    df_es.iloc[6, 1] = b_area
    # temporal correlation
    temp_cor = de.calc_temp_cor(obs_arr, sim_arr)
    df_es.iloc[6, 2] = temp_cor
    # diagnostic efficiency
    df_es.iloc[6, 3] = de.calc_de(obs_arr, sim_arr)
    # direction of bias
    b_dir = de.calc_bias_dir(brel_rest)
    df_es.iloc[6, 4] = b_dir
    # slope of bias
    b_slope = de.calc_bias_slope(b_area, b_dir)
    df_es.iloc[6, 5] = b_slope
    # convert to radians
    # (y, x) Trigonometric inverse tangent
    df_es.iloc[6, 6] = np.arctan2(brel_mean, b_slope)

    # KGE beta
    df_es.iloc[6, 7] = kge.calc_kge_beta(obs_arr, sim_arr)
    # KGE alpha
    df_es.iloc[6, 8] = kge.calc_kge_alpha(obs_arr, sim_arr)
    # # KGE gamma
    # df_es.iloc[0, 8] = kge.calc_kge_gamma(obs_arr, sim_arr)
    # KGE
    df_es.iloc[6, 9] = kge.calc_kge(obs_arr, sim_arr)

    # NSE
    df_es.iloc[6, 10] = nse.calc_nse(obs_arr, sim_arr)

    #----------------------------------------------------------
    # negative dynamic error and positive constant error
    #----------------------------------------------------------
    obs_sim = pd.DataFrame(index=df_ts.index, columns=['Qobs', 'Qsim'])
    obs_sim.loc[:, 'Qobs'] = df_ts.loc[:, 'Qobs']
    # generate negative dynamic error
    tsd = generate_errors.negative_dynamic(df_ts.copy(), prop=0.5)
    # generate positive constant error
    tso = generate_errors.constant(obs_sim.iloc[:, 0].values, offset=.25)
    obs_sim.loc[:, 'Qsim'] = tsd.iloc[:, 0].values + tso  # positive offset
    util.fdc_obs_sim_ax(obs_sim['Qobs'], obs_sim['Qsim'], axes_fdc[1, 1],
                        fig_num_fdc[6])
    util.plot_obs_sim_ax(obs_sim['Qobs'], obs_sim['Qsim'], axes_ts[1, 1],
                         fig_num_ts[6])

    # make arrays
    obs_arr = obs_sim['Qobs'].values
    sim_arr = obs_sim['Qsim'].values

    # mean relative bias
    brel_mean = de.calc_brel_mean(obs_arr, sim_arr)
    df_es.iloc[7, 0] = brel_mean
    # residual relative bias
    brel_rest = de.calc_brel_rest(obs_arr, sim_arr)
    # area of relative remaing bias
    b_area = de.calc_bias_area(brel_rest)
    df_es.iloc[7, 1] = b_area
    # temporal correlation
    temp_cor = de.calc_temp_cor(obs_arr, sim_arr)
    df_es.iloc[7, 2] = temp_cor
    # diagnostic efficiency
    df_es.iloc[7, 3] = de.calc_de(obs_arr, sim_arr)
    # direction of bias
    b_dir = de.calc_bias_dir(brel_rest)
    df_es.iloc[7, 4] = b_dir
    # slope of bias
    b_slope = de.calc_bias_slope(b_area, b_dir)
    df_es.iloc[7, 5] = b_slope
    # convert to radians
    # (y, x) Trigonometric inverse tangent
    df_es.iloc[7, 6] = np.arctan2(brel_mean, b_slope)

    # KGE beta
    df_es.iloc[7, 7] = kge.calc_kge_beta(obs_arr, sim_arr)
    # KGE alpha
    df_es.iloc[7, 8] = kge.calc_kge_alpha(obs_arr, sim_arr)
    # # KGE gamma
    # df_es.iloc[0, 8] = kge.calc_kge_gamma(obs_arr, sim_arr)
    # KGE
    df_es.iloc[7, 9] = kge.calc_kge(obs_arr, sim_arr)

    # NSE
    df_es.iloc[7, 10] = nse.calc_nse(obs_arr, sim_arr)

    #----------------------------------------------------------
    # postive dynamic error and negative constant error
    #----------------------------------------------------------
    obs_sim = pd.DataFrame(index=df_ts.index, columns=['Qobs', 'Qsim'])
    obs_sim.loc[:, 'Qobs'] = df_ts.loc[:, 'Qobs']
    # generate positve dynamic error
    tsd = generate_errors.positive_dynamic(df_ts.copy(), prop=0.5)
    # generate negative constant error
    tso = generate_errors.constant(obs_sim.iloc[:, 0].values, offset=.25)
    obs_sim.loc[:, 'Qsim'] = tsd.iloc[:, 0].values - tso  # negative offset
    util.fdc_obs_sim_ax(obs_sim['Qobs'], obs_sim['Qsim'], axes_fdc[1, 2],
                        fig_num_fdc[7])
    util.plot_obs_sim_ax(obs_sim['Qobs'], obs_sim['Qsim'], axes_ts[1, 2],
                         fig_num_ts[7])

    # make arrays
    obs_arr = obs_sim['Qobs'].values
    sim_arr = obs_sim['Qsim'].values

    # mean relative bias
    brel_mean = de.calc_brel_mean(obs_arr, sim_arr)
    df_es.iloc[8, 0] = brel_mean
    # residual relative bias
    brel_rest = de.calc_brel_rest(obs_arr, sim_arr)
    # area of relative remaing bias
    b_area = de.calc_bias_area(brel_rest)
    df_es.iloc[8, 1] = b_area
    # temporal correlation
    temp_cor = de.calc_temp_cor(obs_arr, sim_arr)
    df_es.iloc[8, 2] = temp_cor
    # diagnostic efficiency
    df_es.iloc[8, 3] = de.calc_de(obs_arr, sim_arr)
    # direction of bias
    b_dir = de.calc_bias_dir(brel_rest)
    df_es.iloc[8, 4] = b_dir
    # slope of bias
    b_slope = de.calc_bias_slope(b_area, b_dir)
    df_es.iloc[8, 5] = b_slope
    # convert to radians
    # (y, x) Trigonometric inverse tangent
    df_es.iloc[8, 6] = np.arctan2(brel_mean, b_slope)

    # KGE beta
    df_es.iloc[8, 7] = kge.calc_kge_beta(obs_arr, sim_arr)
    # KGE alpha
    df_es.iloc[8, 8] = kge.calc_kge_alpha(obs_arr, sim_arr)
    # # KGE gamma
    # df_es.iloc[0, 8] = kge.calc_kge_gamma(obs_arr, sim_arr)
    # KGE
    df_es.iloc[8, 9] = kge.calc_kge(obs_arr, sim_arr)

    # NSE
    df_es.iloc[8, 10] = nse.calc_nse(obs_arr, sim_arr)

    #----------------------------------------------------------
    # positive dynamic error and positive constant error
    #----------------------------------------------------------
    obs_sim = pd.DataFrame(index=df_ts.index, columns=['Qobs', 'Qsim'])
    obs_sim.loc[:, 'Qobs'] = df_ts.loc[:, 'Qobs']
    # generate positive dynamic error
    tsd = generate_errors.positive_dynamic(df_ts.copy(), prop=0.5)
    # generate positive constant error
    tso = generate_errors.constant(obs_sim.iloc[:, 0].values, offset=.25)
    obs_sim.loc[:, 'Qsim'] = tsd.iloc[:, 0].values + tso  # positive offset
    util.fdc_obs_sim_ax(obs_sim['Qobs'], obs_sim['Qsim'], axes_fdc[1, 3],
                        fig_num_fdc[8])
    util.plot_obs_sim_ax(obs_sim['Qobs'], obs_sim['Qsim'], axes_ts[1, 3],
                         fig_num_ts[8])

    # make arrays
    obs_arr = obs_sim['Qobs'].values
    sim_arr = obs_sim['Qsim'].values

    # mean relative bias
    brel_mean = de.calc_brel_mean(obs_arr, sim_arr)
    df_es.iloc[9, 0] = brel_mean
    # residual relative bias
    brel_rest = de.calc_brel_rest(obs_arr, sim_arr)
    # area of relative remaing bias
    b_area = de.calc_bias_area(brel_rest)
    df_es.iloc[9, 1] = b_area
    # temporal correlation
    temp_cor = de.calc_temp_cor(obs_arr, sim_arr)
    df_es.iloc[9, 2] = temp_cor
    # diagnostic efficiency
    df_es.iloc[9, 3] = de.calc_de(obs_arr, sim_arr)
    # direction of bias
    b_dir = de.calc_bias_dir(brel_rest)
    df_es.iloc[9, 4] = b_dir
    # slope of bias
    b_slope = de.calc_bias_slope(b_area, b_dir)
    df_es.iloc[9, 5] = b_slope
    # convert to radians
    # (y, x) Trigonometric inverse tangent
    df_es.iloc[9, 6] = np.arctan2(brel_mean, b_slope)

    # KGE beta
    df_es.iloc[9, 7] = kge.calc_kge_beta(obs_arr, sim_arr)
    # KGE alpha
    df_es.iloc[9, 8] = kge.calc_kge_alpha(obs_arr, sim_arr)
    # # KGE gamma
    # df_es.iloc[0, 8] = kge.calc_kge_gamma(obs_arr, sim_arr)
    # KGE
    df_es.iloc[9, 9] = kge.calc_kge(obs_arr, sim_arr)

    # NSE
    df_es.iloc[9, 10] = nse.calc_nse(obs_arr, sim_arr)

    #-----------------------------------------------------------------
    # negative dynamic error, negative constant error and timing error
    #-----------------------------------------------------------------
    obs_sim = pd.DataFrame(index=df_ts.index, columns=['Qobs', 'Qsim'])
    obs_sim.loc[:, 'Qobs'] = df_ts.loc[:, 'Qobs']
    # generate negative dynamic error
    tsd = generate_errors.negative_dynamic(df_ts.copy(), prop=0.5)
    tsn = pd.DataFrame(index=df_ts.index, columns=['Qsim'])
    # generate negative constant error
    tso = generate_errors.constant(obs_sim.iloc[:, 0].values, offset=.25)
    tsn.iloc[:, 0] = tsd.iloc[:, 0].values - tso  # negative offset
    tst = generate_errors.timing(tsn, random=True)  # shuffling
    obs_sim.loc[:, 'Qsim'] = tst.iloc[:, 0].values
    util.plot_obs_sim_ax(obs_sim['Qobs'], obs_sim['Qsim'], axes_ts[1, 4],
                         fig_num_ts[9])

    # make arrays
    obs_arr = obs_sim['Qobs'].values
    sim_arr = obs_sim['Qsim'].values

    # mean relative bias
    brel_mean = de.calc_brel_mean(obs_arr, sim_arr)
    df_es.iloc[10, 0] = brel_mean
    # residual relative bias
    brel_rest = de.calc_brel_rest(obs_arr, sim_arr)
    # area of relative remaing bias
    b_area = de.calc_bias_area(brel_rest)
    df_es.iloc[10, 1] = b_area
    # temporal correlation
    temp_cor = de.calc_temp_cor(obs_arr, sim_arr)
    df_es.iloc[10, 2] = temp_cor
    # diagnostic efficiency
    df_es.iloc[10, 3] = de.calc_de(obs_arr, sim_arr)
    # direction of bias
    b_dir = de.calc_bias_dir(brel_rest)
    df_es.iloc[10, 4] = b_dir
    # slope of bias
    b_slope = de.calc_bias_slope(b_area, b_dir)
    df_es.iloc[10, 5] = b_slope
    # convert to radians
    # (y, x) Trigonometric inverse tangent
    df_es.iloc[10, 6] = np.arctan2(brel_mean, b_slope)

    # KGE beta
    df_es.iloc[10, 7] = kge.calc_kge_beta(obs_arr, sim_arr)
    # KGE alpha
    df_es.iloc[10, 8] = kge.calc_kge_alpha(obs_arr, sim_arr)
    # # KGE gamma
    # df_es.iloc[0, 8] = kge.calc_kge_gamma(obs_arr, sim_arr)
    # KGE
    df_es.iloc[10, 9] = kge.calc_kge(obs_arr, sim_arr)

    # NSE
    df_es.iloc[10, 10] = nse.calc_nse(obs_arr, sim_arr)

    #-----------------------------------------------------------------
    # negative dynamic error, positive constant error and timing error
    #-----------------------------------------------------------------
    obs_sim = pd.DataFrame(index=df_ts.index, columns=['Qobs', 'Qsim'])
    obs_sim.loc[:, 'Qobs'] = df_ts.loc[:, 'Qobs']
    # generate negative dynamic error
    tsd = generate_errors.negative_dynamic(df_ts.copy(), prop=0.5)
    tsn = pd.DataFrame(index=df_ts.index, columns=['Qsim'])
    # generate positive constant error
    tso = generate_errors.constant(obs_sim.iloc[:, 0].values, offset=.25)
    tsn.iloc[:, 0] = tsd.iloc[:, 0].values + tso  # negative offset
    tst = generate_errors.timing(tsn, random=True)  # shuffling
    obs_sim.loc[:, 'Qsim'] = tst.iloc[:, 0].values
    util.plot_obs_sim_ax(obs_sim['Qobs'], obs_sim['Qsim'], axes_ts[2, 0],
                         fig_num_ts[10])

    # make arrays
    obs_arr = obs_sim['Qobs'].values
    sim_arr = obs_sim['Qsim'].values

    # mean relative bias
    brel_mean = de.calc_brel_mean(obs_arr, sim_arr)
    df_es.iloc[11, 0] = brel_mean
    # residual relative bias
    brel_rest = de.calc_brel_rest(obs_arr, sim_arr)
    # area of relative remaing bias
    b_area = de.calc_bias_area(brel_rest)
    df_es.iloc[11, 1] = b_area
    # temporal correlation
    temp_cor = de.calc_temp_cor(obs_arr, sim_arr)
    df_es.iloc[11, 2] = temp_cor
    # diagnostic efficiency
    df_es.iloc[11, 3] = de.calc_de(obs_arr, sim_arr)
    # direction of bias
    b_dir = de.calc_bias_dir(brel_rest)
    df_es.iloc[11, 4] = b_dir
    # slope of bias
    b_slope = de.calc_bias_slope(b_area, b_dir)
    df_es.iloc[11, 5] = b_slope
    # convert to radians
    # (y, x) Trigonometric inverse tangent
    df_es.iloc[11, 6] = np.arctan2(brel_mean, b_slope)

    # KGE beta
    df_es.iloc[11, 7] = kge.calc_kge_beta(obs_arr, sim_arr)
    # KGE alpha
    df_es.iloc[11, 8] = kge.calc_kge_alpha(obs_arr, sim_arr)
    # # KGE gamma
    # df_es.iloc[0, 8] = kge.calc_kge_gamma(obs_arr, sim_arr)
    # KGE
    df_es.iloc[11, 9] = kge.calc_kge(obs_arr, sim_arr)

    # NSE
    df_es.iloc[11, 10] = nse.calc_nse(obs_arr, sim_arr)

    #-----------------------------------------------------------------
    # positive dynamic error, negative constant error and timing error
    #-----------------------------------------------------------------
    obs_sim = pd.DataFrame(index=df_ts.index, columns=['Qobs', 'Qsim'])
    obs_sim.loc[:, 'Qobs'] = df_ts.loc[:, 'Qobs']
    # generate positive dynamic error
    tsd = generate_errors.positive_dynamic(df_ts.copy(), prop=0.5)
    tsp = pd.DataFrame(index=df_ts.index, columns=['Qsim'])
    # generate positive constant offset
    tso = generate_errors.constant(obs_sim.iloc[:, 0].values, offset=.25)
    tsp.iloc[:, 0] = tsd.iloc[:, 0].values - tso  # negative offset
    tst = generate_errors.timing(tsp, random=True)  # shuffling
    obs_sim.loc[:, 'Qsim'] = tst.iloc[:, 0].values
    util.plot_obs_sim_ax(obs_sim['Qobs'], obs_sim['Qsim'], axes_ts[2, 1],
                         fig_num_ts[11])

    # make arrays
    obs_arr = obs_sim['Qobs'].values
    sim_arr = obs_sim['Qsim'].values

    # mean relative bias
    brel_mean = de.calc_brel_mean(obs_arr, sim_arr)
    df_es.iloc[12, 0] = brel_mean
    # residual relative bias
    brel_rest = de.calc_brel_rest(obs_arr, sim_arr)
    # area of relative remaing bias
    b_area = de.calc_bias_area(brel_rest)
    df_es.iloc[12, 1] = b_area
    # temporal correlation
    temp_cor = de.calc_temp_cor(obs_arr, sim_arr)
    df_es.iloc[12, 2] = temp_cor
    # diagnostic efficiency
    df_es.iloc[12, 3] = de.calc_de(obs_arr, sim_arr)
    # direction of bias
    b_dir = de.calc_bias_dir(brel_rest)
    df_es.iloc[12, 4] = b_dir
    # slope of bias
    b_slope = de.calc_bias_slope(b_area, b_dir)
    df_es.iloc[12, 5] = b_slope
    # convert to radians
    # (y, x) Trigonometric inverse tangent
    df_es.iloc[12, 6] = np.arctan2(brel_mean, b_slope)

    # KGE beta
    df_es.iloc[12, 7] = kge.calc_kge_beta(obs_arr, sim_arr)
    # KGE alpha
    df_es.iloc[12, 8] = kge.calc_kge_alpha(obs_arr, sim_arr)
    # # KGE gamma
    # df_es.iloc[0, 8] = kge.calc_kge_gamma(obs_arr, sim_arr)
    # KGE
    df_es.iloc[12, 9] = kge.calc_kge(obs_arr, sim_arr)

    # NSE
    df_es.iloc[12, 10] = nse.calc_nse(obs_arr, sim_arr)

    #-----------------------------------------------------------------
    # positive dynamic error, positive constant error and timing error
    #-----------------------------------------------------------------
    obs_sim = pd.DataFrame(index=df_ts.index, columns=['Qobs', 'Qsim'])
    obs_sim.loc[:, 'Qobs'] = df_ts.loc[:, 'Qobs']
    # generate positive dynamic error
    tsd = generate_errors.positive_dynamic(df_ts.copy(), prop=0.5)
    tsp = pd.DataFrame(index=df_ts.index, columns=['Qsim'])
    # generate positve constant offset
    tso = generate_errors.constant(obs_sim.iloc[:, 0].values, offset=.25)
    tsp.iloc[:, 0] = tsd.iloc[:, 0].values + tso  # positve offset
    tst = generate_errors.timing(tsp, random=True)  # shuffling
    obs_sim.loc[:, 'Qsim'] = tst.iloc[:, 0].values
    util.plot_obs_sim_ax(obs_sim['Qobs'], obs_sim['Qsim'], axes_ts[2, 2],
                         fig_num_ts[12])

    # make arrays
    obs_arr = obs_sim['Qobs'].values
    sim_arr = obs_sim['Qsim'].values

    # mean relative bias
    brel_mean = de.calc_brel_mean(obs_arr, sim_arr)
    df_es.iloc[13, 0] = brel_mean
    # residual relative bias
    brel_rest = de.calc_brel_rest(obs_arr, sim_arr)
    # area of relative remaing bias
    b_area = de.calc_bias_area(brel_rest)
    df_es.iloc[13, 1] = b_area
    # temporal correlation
    temp_cor = de.calc_temp_cor(obs_arr, sim_arr)
    df_es.iloc[13, 2] = temp_cor
    # diagnostic efficiency
    df_es.iloc[13, 3] = de.calc_de(obs_arr, sim_arr)
    # direction of bias
    b_dir = de.calc_bias_dir(brel_rest)
    df_es.iloc[13, 4] = b_dir
    # slope of bias
    b_slope = de.calc_bias_slope(b_area, b_dir)
    df_es.iloc[13, 5] = b_slope
    # convert to radians
    # (y, x) Trigonometric inverse tangent
    df_es.iloc[13, 6] = np.arctan2(brel_mean, b_slope)

    # KGE beta
    df_es.iloc[13, 7] = kge.calc_kge_beta(obs_arr, sim_arr)
    # KGE alpha
    df_es.iloc[13, 8] = kge.calc_kge_alpha(obs_arr, sim_arr)
    # # KGE gamma
    # df_es.iloc[0, 8] = kge.calc_kge_gamma(obs_arr, sim_arr)
    # KGE
    df_es.iloc[13, 9] = kge.calc_kge(obs_arr, sim_arr)

    # NSE
    df_es.iloc[13, 10] = nse.calc_nse(obs_arr, sim_arr)

    axes_fdc[1, 3].legend(loc=6, bbox_to_anchor=(1.18, .85))
    axes_ts[2, 2].legend(loc=6, bbox_to_anchor=(1.18, .85))

    path_png = os.path.join(os.getcwd(),
                            'figures/technical_note/fdc_errors.png')
    fig_fdc.savefig(path_png, dpi=250)
    path_png = os.path.join(os.getcwd(),
                            'figures/technical_note/ts_errors.png')
    fig_ts.savefig(path_png, dpi=250)

    #-----------------------------------------------------------------
    # diagnostic plolar plot
    #-----------------------------------------------------------------
    ### multi DE plot ###
    # make arrays
    idx = ['1', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
           'm']
    ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    brel_mean_arr = df_es['brel_mean'].values[ids]
    b_area_arr = df_es['b_area'].values[ids]
    temp_cor_arr = df_es['temp_cor'].values[ids]
    b_dir_arr = df_es['b_dir'].values[ids]
    de_arr = df_es['de'].values[ids]
    phi_arr = df_es['phi'].values[ids]
    b_slope_arr = df_es['b_slope'].values[ids]

    fig_de = util.diag_polar_plot_multi_fc(brel_mean_arr, b_area_arr,
                                           temp_cor_arr, de_arr, b_dir_arr,
                                           phi_arr, idx)
    fig_de.savefig('/Users/robinschwemmle/Desktop/PhD/diagnostic_efficiency/figures/technical_note/de_diag.pdf', dpi=250)

    ### multi KGE plot ###
    # make arrays
    idx = ['1', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
           'm']
    beta_arr = df_es['beta'].values
    alpha_arr = df_es['alpha'].values
    # gamma_arr = df_es['gamma'].values
    temp_cor_arr = df_es['temp_cor'].values
    kge_arr = df_es['kge'].values

    fig_kge = util.diag_polar_plot_kge_multi_fc(beta_arr, alpha_arr,
                                                temp_cor_arr, kge_arr, idx)
    path = os.path.join(os.getcwd(), 'figures/technical_note/kge_diag.pdf')
    fig_kge.savefig(path, dpi=250)

    ### scatterplots DE, KGE and NSE ###
    nse_arr = df_es['nse'].values
    brel_mean_arr = df_es['brel_mean'].values
    b_slope_arr = df_es['b_slope'].values
    de_arr = df_es['de'].values

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
    # scatterplots DE, KGE and NSE
    sc = sns.scatterplot(kge_arr, de_arr, color='black', s=60, ax=ax1)
    sc1 = sns.scatterplot(nse_arr, de_arr, color='red', s=30, marker="X",
                          ax=ax1)
    ax1.plot([-2.05, 1.05], [-2.05, 1.05], ls="--", c=".3")
    ax1.set_ylim(-2.05, 1.05)
    ax1.set_xlim(-2.05, 1.05)
    ax1.set(ylabel='DE [-]', xlabel='KGE [-]')
    ax1.text(.40, -.22, 'NSE [-]', color='red', transform=ax1.transAxes)
    ax1.text(.025, .93, '(a)', transform=ax1.transAxes)
    ax1.text(.05, .1, '1:1', rotation=45, transform=ax1.transAxes)
    # for i, txt in enumerate(df_es.index):
    #     ax.annotate(txt, (kge_arr[i], de_arr[i]), color='black', fontsize=15)
    #     ax.annotate(txt, (nse_arr[i], de_arr[i]), color='red', fontsize=15)

    # scatterplot DE and KGE components inter-comparison
    sc = sns.scatterplot(beta_arr - 1, brel_mean_arr, color='black', s=60,
                         ax=ax2)
    sc1 = sns.scatterplot(alpha_arr - 1, b_slope_arr, color='red', s=30,
                          marker="X", ax=ax2)
    ax2.plot([-2.05, 2.05], [-2.05, 2.05], ls="--", c=".3")
    ax2.set_ylim(-2.05, 2.05)
    ax2.set_xlim(-2.05, 2.05)
    ax2.set(ylabel=r'$\overline{B_{rel}}$ [-]', xlabel=r'$\beta$ - 1 [-]')
    ax2.text(-.33, .415, r'$B_{slope}$ [-]', color='red',
             transform=ax2.transAxes, rotation=90)
    ax2.text(.4, -.22, r'$\alpha$ - 1 [-]', color='red',
             transform=ax2.transAxes)
    ax2.text(.03, .93, '(b)', transform=ax2.transAxes)
    ax2.text(.05, .1, '1:1', rotation=45, transform=ax2.transAxes)

    # scatterplot DE and KGE components intra-comparison
    sc = sns.scatterplot(b_slope_arr, brel_mean_arr, color='black', s=60,
                         ax=ax3)
    sc1 = sns.scatterplot(alpha_arr - 1, beta_arr - 1, color='red', s=30,
                          marker="X", ax=ax3)
    ax3.plot([-2.05, 2.05], [-2.05, 2.05], ls="--", c=".3")
    ax3.set_ylim(-2.05, 2.05)
    ax3.set_xlim(-2.05, 2.05)
    ax3.set(ylabel=r'$B_{slope}$ [-]', xlabel=r'$\overline{B_{rel}}$ [-]')
    ax3.text(-.33, .425, r'$\beta$ - 1 [-]', color='red',
             transform=ax3.transAxes, rotation=90)
    ax3.text(.4, -.23, r'$\alpha$ - 1 [-]', color='red',
             transform=ax3.transAxes)
    ax3.text(.03, .93, '(c)', transform=ax3.transAxes)
    ax3.text(.05, .1, '1:1', rotation=45, transform=ax3.transAxes)

    fig.subplots_adjust(wspace=.45, bottom=.2)
    path_png = os.path.join(os.getcwd(),
                            'figures/technical_note/scatter_eff_comp.png')
    fig.savefig(path_png, dpi=250)
    # for i, txt in enumerate(df_es.index):
    #     ax.annotate(txt, (alpha_arr[i] - 1, brel_mean_arr[i]), color='black',
    #                 fontsize=15)
    #     ax.annotate(txt, (beta_arr[i] - 1, b_slope_arr[i]), color='red',
    #                 fontsize=15)

    # export table
    df_es_t = df_es.T
    path_csv = os.path.join(os.getcwd(),
                            'figures/technical_note/table_eff_comp.csv')
    df_es_t = df_es_t.round(2)
    df_es_t.to_csv(path_csv, header=True, index=True, sep=';')
    df_es_t = df_es_t.loc[['de', 'kge', 'nse'], :]
    df_es_t = df_es_t.round(2)
    path_csv = os.path.join(os.getcwd(),
                            'figures/technical_note/table_eff_eff.csv')
    df_es_t.to_csv(path_csv, header=True, index=True, sep=';')

    #==========================================================
    # Real case example - CAMELS
    #==========================================================
    # dataframe DE, KGE and NSE
    idx = ['05', '48', '94']
    cols = ['brel_mean', 'b_area', 'temp_cor', 'de', 'b_dir', 'b_slope',
            'phi', 'beta', 'alpha', 'kge', 'nse']
    df_eff_cam = pd.DataFrame(index=idx, columns=cols, dtype=np.float64)

    path_cam1 = os.path.join(os.getcwd(),
                             'examples/camels_example_data/13331500_05_model_output.txt')
    path_cam2 = os.path.join(os.getcwd(),
                             'examples/camels_example_data/13331500_48_model_output.txt')
    path_cam3 = os.path.join(os.getcwd(),
                             'examples/camels_example_data/13331500_94_model_output.txt')
    df_cam1 = util.import_camels_obs_sim(path_cam1)
    df_cam2 = util.import_camels_obs_sim(path_cam2)
    df_cam3 = util.import_camels_obs_sim(path_cam3)

    fig, axes = plt.subplots(3, 2, figsize=(10, 10), sharex='col')
    fig.text(0.06, 0.5, r'[mm $d^{-1}$]', ha='center', va='center',
             rotation='vertical')
    fig.text(0.5, 0.5, r'[mm $d^{-1}$]', ha='center', va='center',
             rotation='vertical')
    fig.text(0.25, 0.05, 'Time [Years]', ha='center', va='center')
    fig.text(0.75, 0.05, 'Exceedence probabilty [-]', ha='center', va='center')

    util.plot_obs_sim_ax(df_cam1['Qobs'], df_cam1['Qsim'], axes[0, 0], '')
    axes[0, 0].text(.95, .95, '(a; set_id: {})'.format(idx[0]),
                    transform=axes[0, 0].transAxes, ha='right', va='top')
    # format the ticks
    years_10 = mdates.YearLocator(10)
    years_5 = mdates.YearLocator(5)
    yearsFmt = mdates.DateFormatter('%Y')
    axes[0, 0].xaxis.set_major_locator(years_10)
    axes[0, 0].xaxis.set_major_formatter(yearsFmt)
    axes[0, 0].xaxis.set_minor_locator(years_5)
    util.fdc_obs_sim_ax(df_cam1['Qobs'], df_cam1['Qsim'], axes[0, 1], '')
    axes[0, 1].text(.95, .95, '(b; set_id: {})'.format(idx[0]),
                    transform=axes[0, 1].transAxes, ha='right', va='top')
    # legend above plot
    axes[0, 1].legend(loc=2, labels=['Observed', 'Simulated'], ncol=2,
                      frameon=False, bbox_to_anchor=(-0.6, 1.2))

    util.plot_obs_sim_ax(df_cam2['Qobs'], df_cam2['Qsim'], axes[1, 0], '')
    axes[1, 0].text(.95, .95, '(c; set_id: {})'.format(idx[1]),
                    transform=axes[1, 0].transAxes, ha='right', va='top')
    # format the ticks
    years_10 = mdates.YearLocator(10)
    years_5 = mdates.YearLocator(5)
    yearsFmt = mdates.DateFormatter('%Y')
    axes[1, 0].xaxis.set_major_locator(years_10)
    axes[1, 0].xaxis.set_major_formatter(yearsFmt)
    axes[1, 0].xaxis.set_minor_locator(years_5)
    util.fdc_obs_sim_ax(df_cam1['Qobs'], df_cam1['Qsim'], axes[1, 1], '')
    axes[1, 1].text(.95, .95, '(d; set_id: {})'.format(idx[1]),
                    transform=axes[1, 1].transAxes, ha='right', va='top')

    util.plot_obs_sim_ax(df_cam1['Qobs'], df_cam1['Qsim'], axes[2, 0], '')
    axes[2, 0].text(.95, .95, '(e; set_id: {})'.format(idx[2]),
                    transform=axes[2, 0].transAxes, ha='right', va='top')
    # format the ticks
    years_10 = mdates.YearLocator(10)
    years_5 = mdates.YearLocator(5)
    yearsFmt = mdates.DateFormatter('%Y')
    axes[2, 0].xaxis.set_major_locator(years_10)
    axes[2, 0].xaxis.set_major_formatter(yearsFmt)
    axes[2, 0].xaxis.set_minor_locator(years_5)
    util.fdc_obs_sim_ax(df_cam1['Qobs'], df_cam1['Qsim'], axes[2, 1], '')
    axes[2, 1].text(.95, .95, '(f; set_id: {})'.format(idx[2]),
                    transform=axes[2, 1].transAxes, ha='right', va='top')

    fig.subplots_adjust(wspace=0.3)
    path_png = os.path.join(os.getcwd(),
                            'figures/technical_note/ts_fdc_real_case.png')
    fig.savefig(path_png, dpi=250)
    path_pdf = os.path.join(os.getcwd(),
                            'figures/technical_note/ts_fdc_real_case.pdf')
    fig.savefig(path_pdf, dpi=250)

    # make arrays
    obs_arr = df_cam1['Qobs'].values
    sim_arr = df_cam1['Qsim'].values

    # mean relative bias
    brel_mean = de.calc_brel_mean(obs_arr, sim_arr)
    df_eff_cam.iloc[0, 0] = brel_mean
    # residual relative bias
    brel_rest = de.calc_brel_rest(obs_arr, sim_arr)
    # area of relative remaing bias
    b_area = de.calc_bias_area(brel_rest)
    df_eff_cam.iloc[0, 1] = b_area
    # temporal correlation
    temp_cor = de.calc_temp_cor(obs_arr, sim_arr)
    df_eff_cam.iloc[0, 2] = temp_cor
    # diagnostic efficiency
    df_eff_cam.iloc[0, 3] = de.calc_de(obs_arr, sim_arr)
    # direction of bias
    b_dir = de.calc_bias_dir(brel_rest)
    df_eff_cam.iloc[0, 4] = b_dir
    # slope of bias
    b_slope = de.calc_bias_slope(b_area, b_dir)
    df_eff_cam.iloc[0, 5] = b_slope
    # convert to radians
    # (y, x) Trigonometric inverse tangent
    df_eff_cam.iloc[0, 6] = np.arctan2(brel_mean, b_slope)

    # KGE beta
    df_eff_cam.iloc[0, 7] = kge.calc_kge_beta(obs_arr, sim_arr)
    # KGE alpha
    df_eff_cam.iloc[0, 8] = kge.calc_kge_alpha(obs_arr, sim_arr)
    # KGE
    df_eff_cam.iloc[0, 9] = kge.calc_kge(obs_arr, sim_arr)

    # NSE
    df_eff_cam.iloc[0, 10] = nse.calc_nse(obs_arr, sim_arr)

    # make arrays
    obs_arr = df_cam2['Qobs'].values
    sim_arr = df_cam2['Qsim'].values

    # mean relative bias
    brel_mean = de.calc_brel_mean(obs_arr, sim_arr)
    df_eff_cam.iloc[1, 0] = brel_mean
    # residual relative bias
    brel_rest = de.calc_brel_rest(obs_arr, sim_arr)
    # area of relative remaing bias
    b_area = de.calc_bias_area(brel_rest)
    df_eff_cam.iloc[1, 1] = b_area
    # temporal correlation
    temp_cor = de.calc_temp_cor(obs_arr, sim_arr)
    df_eff_cam.iloc[1, 2] = temp_cor
    # diagnostic efficiency
    df_eff_cam.iloc[1, 3] = de.calc_de(obs_arr, sim_arr)
    # direction of bias
    b_dir = de.calc_bias_dir(brel_rest)
    df_eff_cam.iloc[1, 4] = b_dir
    # slope of bias
    b_slope = de.calc_bias_slope(b_area, b_dir)
    df_eff_cam.iloc[1, 5] = b_slope
    # convert to radians
    # (y, x) Trigonometric inverse tangent
    df_eff_cam.iloc[1, 6] = np.arctan2(brel_mean, b_slope)

    # KGE beta
    df_eff_cam.iloc[1, 7] = kge.calc_kge_beta(obs_arr, sim_arr)
    # KGE alpha
    df_eff_cam.iloc[1, 8] = kge.calc_kge_alpha(obs_arr, sim_arr)
    # KGE
    df_eff_cam.iloc[1, 9] = kge.calc_kge(obs_arr, sim_arr)

    # NSE
    df_eff_cam.iloc[1, 10] = nse.calc_nse(obs_arr, sim_arr)

    # make arrays
    obs_arr = df_cam3['Qobs'].values
    sim_arr = df_cam3['Qsim'].values

    # mean relative bias
    brel_mean = de.calc_brel_mean(obs_arr, sim_arr)
    df_eff_cam.iloc[2, 0] = brel_mean
    # residual relative bias
    brel_rest = de.calc_brel_rest(obs_arr, sim_arr)
    # area of relative remaing bias
    b_area = de.calc_bias_area(brel_rest)
    df_eff_cam.iloc[2, 1] = b_area
    # temporal correlation
    temp_cor = de.calc_temp_cor(obs_arr, sim_arr)
    df_eff_cam.iloc[2, 2] = temp_cor
    # diagnostic efficiency
    df_eff_cam.iloc[2, 3] = de.calc_de(obs_arr, sim_arr)
    # direction of bias
    b_dir = de.calc_bias_dir(brel_rest)
    df_eff_cam.iloc[2, 4] = b_dir
    # slope of bias
    b_slope = de.calc_bias_slope(b_area, b_dir)
    df_eff_cam.iloc[2, 5] = b_slope
    # convert to radians
    # (y, x) Trigonometric inverse tangent
    df_eff_cam.iloc[2, 6] = np.arctan2(brel_mean, b_slope)

    # KGE beta
    df_eff_cam.iloc[2, 7] = kge.calc_kge_beta(obs_arr, sim_arr)
    # KGE alpha
    df_eff_cam.iloc[2, 8] = kge.calc_kge_alpha(obs_arr, sim_arr)
    # KGE
    df_eff_cam.iloc[2, 9] = kge.calc_kge(obs_arr, sim_arr)

    # NSE
    df_eff_cam.iloc[2, 10] = nse.calc_nse(obs_arr, sim_arr)

    path_csv = os.path.join(os.getcwd(),
                            'figures/technical_note/table_eff_real_case.csv')
    df_eff_cam.to_csv(path_csv, header=True, index=True, sep=';')

    ### multi diagnostic plot ###
    # make arrays
    brel_mean_arr = df_eff_cam['brel_mean'].values
    b_area_arr = df_eff_cam['b_area'].values
    temp_cor_arr = df_eff_cam['temp_cor'].values
    b_dir_arr = df_eff_cam['b_dir'].values
    de_arr = df_eff_cam['de'].values
    phi_arr = df_eff_cam['phi'].values
    b_slope_arr = df_eff_cam['b_slope'].values

    fig_de = util.diag_polar_plot_multi_fc(brel_mean_arr, b_area_arr,
                                           temp_cor_arr, de_arr, b_dir_arr,
                                           phi_arr, idx, ax_lim=0)

    path_pdf = os.path.join(os.getcwd(),
                            'figures/technical_note/de_diag_real_case.pdf')
    fig_de.savefig(path_pdf, dpi=250)

    ### multi KGE plot ###
    # make arrays
    alpha_arr = df_eff_cam['alpha'].values
    beta_arr = df_eff_cam['beta'].values
    kge_arr = df_eff_cam['kge'].values

    fig_kge = util.diag_polar_plot_kge_multi_fc(beta_arr, alpha_arr,
                                                temp_cor_arr, kge_arr, idx,
                                                ax_lim=0)
    path_pdf = os.path.join(os.getcwd(),
                            'figures/technical_note/kge_diag_real_case.pdf')
    fig_kge.savefig(path_pdf, dpi=250)
