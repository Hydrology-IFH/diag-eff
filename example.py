#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from de import de
from de import util

if __name__ == "__main__":
    path = '/Users/robinschwemmle/Desktop/PhD/diagnostic_model_efficiency/examples/data/9960682_Q_1970_2012.csv'
#    path = '/Users/robo/Desktop/PhD/de/examples/data/9960682_Q_1970_2012.csv'

    # dataframe efficiency measures
    idx = np.arange(11)
    cols = ['brel_mean', 'b_area', 'temp_cor', 'de', 'b_dir', 'b_slope', 'diag', 'kge', 'alpha', 'beta', 'nse']
    df_es = pd.DataFrame(index=df_ts.index, columns=cols)

    # import observed time series
    df_ts = util.import_ts(path, sep=';')
    de.plot_ts(df_ts

    ### perfect simulation ###
    obs_sim = pd.DataFrame(index=df_ts.index, columns=['Qobs', 'Qsim'])
    obs_sim.loc[:, 'Qobs'] = df_ts.loc[:, 'Qobs']
    obs_sim.loc[:, 'Qsim'] = df_ts.loc[:, 'Qobs']  # observed time series

    # make numpy arrays
    obs_arr = obs_sim['Qobs'].values  # observed time series
    sim_arr = obs_sim['Qsim'].values  # manipulated time series

    # mean relative bias
    brel_mean = de.calc_brel_mean(obs_arr, sim_arr)
    meta.iloc[0, 0] = brel_mean
    # remaining relative bias
    brel_rest = de.calc_brel_rest(obs_arr, sim_arr)
    # area of relative remaing bias
    b_area = de.calc_b_area(brel_rest)
    meta.iloc[0, 1] = b_area
    # temporal correlation
    temp_cor = de.calc_temp_cor(obs_arr, sim_arr)
    meta.iloc[0, 2] = temp_cor
    # diagnostic efficiency
    meta.iloc[0, 3] = 1 - np.sqrt((brel_mean)**2 + (b_area)**2 + (temp_cor - 1)**2)
    # direction of bias
    b_dir = de.calc_bias_dir(brel_rest)
    meta.iloc[0, 4] = b_dir
    # slope of bias
    b_slope = de.calc_bias_slope(b_area, b_dir)
    meta.iloc[0, 5] = b_slope
    # convert to radians
    # (y, x) Trigonometric inverse tangent
    meta.iloc[0, 6] = np.arctan2(brel_mean, b_slope)

    # KGE
    meta.iloc[0, 7] = de.calc_kge(obs_arr, sim_arr)
    # KGE alpha
    meta.iloc[0, 8] = de.calc_kge_alpha(obs_arr, sim_arr)
    # KGE beta
    meta.iloc[0, 9] = de.calc_kge_beta(obs_arr, sim_arr)

    # NSE
    meta.iloc[0, 10] = de.calc_nse(obs_arr, sim_arr)

    ### increase high flows - decrease low flows ###
    obs_sim = pd.DataFrame(index=df_ts.index, columns=['Qobs', 'Qsim'])
    obs_sim.loc[:, 'Qobs'] = df_ts.loc[:, 'Qobs']
    tsd = de.highover_lowunder(df_ts.copy(), prop=0.5)
    obs_sim.loc[:, 'Qsim'] = tsd.loc[:, 'Qsim']  # disaggregated time series
    de.fdc_obs_sim(obs_sim['Qobs'], obs_sim['Qsim'])
    de.plot_obs_sim(obs_sim['Qobs'], obs_sim['Qsim'])

    # make arrays
    obs_arr = obs_sim['Qobs'].values
    sim_arr = obs_sim['Qsim'].values

    # mean relative bias
    brel_mean = de.calc_brel_mean(obs_arr, sim_arr)
    meta.iloc[1, 0] = brel_mean
    # remaining relative bias
    brel_rest = de.calc_brel_rest(obs_arr, sim_arr)
    # area of relative remaing bias
    b_area = de.calc_b_area(brel_rest)
    meta.iloc[1, 1] = b_area
    # temporal correlation
    temp_cor = de.calc_temp_cor(obs_arr, sim_arr)
    meta.iloc[1, 2] = temp_cor
    # diagnostic efficiency
    meta.iloc[1, 3] = 1 - np.sqrt((brel_mean)**2 + (b_area)**2 + (temp_cor - 1)**2)
    # direction of bias
    b_dir = de.calc_bias_dir(brel_rest)
    meta.iloc[1, 4] = b_dir
    # slope of bias
    b_slope = de.calc_bias_slope(b_area, b_dir)
    meta.iloc[1, 5] = b_slope
    # convert to radians
    # (y, x) Trigonometric inverse tangent
    meta.iloc[1, 6] = np.arctan2(brel_mean, b_slope)

    # KGE
    meta.iloc[1, 7] = de.calc_kge(obs_arr, sim_arr)
    # KGE alpha
    meta.iloc[1, 8] = de.calc_kge_alpha(obs_arr, sim_arr)
    # KGE beta
    meta.iloc[1, 9] = de.calc_kge_beta(obs_arr, sim_arr)

    # NSE
    meta.iloc[1, 10] = de.calc_nse(obs_arr, sim_arr)

    ### decrease high flows - increase low flows ###
    obs_sim = pd.DataFrame(index=df_ts.index, columns=['Qobs', 'Qsim'])
    obs_sim.loc[:, 'Qobs'] = df_ts.loc[:, 'Qobs']
    tsd = de.highunder_lowover(df_ts.copy(), prop=0.5)
    obs_sim.loc[:, 'Qsim'] = tsd.loc[:, 'Qsim']  # smoothed time series
    de.fdc_obs_sim(obs_sim['Qobs'], obs_sim['Qsim'])
    de.plot_obs_sim(obs_sim['Qobs'], obs_sim['Qsim'])

    # make arrays
    obs_arr = obs_sim['Qobs'].values
    sim_arr = obs_sim['Qsim'].values

    # mean relative bias
    brel_mean = de.calc_brel_mean(obs_arr, sim_arr)
    meta.iloc[2, 0] = brel_mean
    # remaining relative bias
    brel_rest = de.calc_brel_rest(obs_arr, sim_arr)
    # area of relative remaing bias
    b_area = de.calc_b_area(brel_rest)
    meta.iloc[2, 1] = b_area
    # temporal correlation
    temp_cor = de.calc_temp_cor(obs_arr, sim_arr)
    meta.iloc[2, 2] = temp_cor
    # diagnostic efficiency
    meta.iloc[2, 3] = 1 - np.sqrt((brel_mean)**2 + (b_area)**2 + (temp_cor - 1)**2)
    # direction of bias
    b_dir = de.calc_bias_dir(brel_rest)
    meta.iloc[2, 4] = b_dir
    # slope of bias
    b_slope = de.calc_bias_slope(b_area, b_dir)
    meta.iloc[2, 5] = b_slope
    # convert to radians
    # (y, x) Trigonometric inverse tangent
    meta.iloc[2, 6] = np.arctan2(brel_mean, b_slope)

    # KGE
    meta.iloc[2, 7] = de.calc_kge(obs_arr, sim_arr)
    # KGE alpha
    meta.iloc[2, 8] = de.calc_kge_alpha(obs_arr, sim_arr)
    # KGE beta
    meta.iloc[2, 9] = de.calc_kge_beta(obs_arr, sim_arr)

    # NSE
    meta.iloc[2, 10] = de.calc_nse(obs_arr, sim_arr)

    ### precipitation surplus ###
    obs_sim = pd.DataFrame(index=df_ts.index, columns=['Qobs', 'Qsim'])
    obs_sim.loc[:, 'Qobs'] = df_ts.loc[:, 'Qobs']
    obs_sim.loc[:, 'Qsim'] = de.pos_shift_ts(df_ts['Qobs'].values)  # positive offset
    de.plot_obs_sim(obs_sim['Qobs'], obs_sim['Qsim'])
    de.fdc_obs_sim(obs_sim['Qobs'], obs_sim['Qsim'])

    # make arrays
    obs_arr = obs_sim['Qobs'].values
    sim_arr = obs_sim['Qsim'].values

    # mean relative bias
    brel_mean = de.calc_brel_mean(obs_arr, sim_arr)
    meta.iloc[3, 0] = brel_mean
    # remaining relative bias
    brel_rest = de.calc_brel_rest(obs_arr, sim_arr)
    # area of relative remaing bias
    b_area = de.calc_b_area(brel_rest)
    meta.iloc[3, 1] = b_area
    # temporal correlation
    temp_cor = de.calc_temp_cor(obs_arr, sim_arr)
    meta.iloc[3, 2] = temp_cor
    # diagnostic efficiency
    meta.iloc[3, 3] = 1 - np.sqrt((brel_mean)**2 + (b_area)**2 + (temp_cor - 1)**2)
    # direction of bias
    b_dir = de.calc_bias_dir(brel_rest)
    meta.iloc[3, 4] = b_dir
    # slope of bias
    b_slope = de.calc_bias_slope(b_area, b_dir)
    meta.iloc[3, 5] = b_slope
    # convert to radians
    # (y, x) Trigonometric inverse tangent
    meta.iloc[3, 6] = np.arctan2(brel_mean, b_slope)

    # KGE
    meta.iloc[3, 7] = de.calc_kge(obs_arr, sim_arr)
    # KGE alpha
    meta.iloc[3, 8] = de.calc_kge_alpha(obs_arr, sim_arr)
    # KGE beta
    meta.iloc[3, 9] = de.calc_kge_beta(obs_arr, sim_arr)

    # NSE
    meta.iloc[3, 10] = de.calc_nse(obs_arr, sim_arr)

    ### precipitation shortage ###
    obs_sim = pd.DataFrame(index=df_ts.index, columns=['Qobs', 'Qsim'])
    obs_sim.loc[:, 'Qobs'] = df_ts.loc[:, 'Qobs']
    obs_sim.loc[:, 'Qsim'] = de.neg_shift_ts(df_ts['Qobs'].values)  # negative offset
    de.plot_obs_sim(obs_sim['Qobs'], obs_sim['Qsim'])
    de.fdc_obs_sim(obs_sim['Qobs'], obs_sim['Qsim'])

    # make arrays
    obs_arr = obs_sim['Qobs'].values
    sim_arr = obs_sim['Qsim'].values

    # mean relative bias
    brel_mean = de.calc_brel_mean(obs_arr, sim_arr)
    meta.iloc[4, 0] = brel_mean
    # remaining relative bias
    brel_rest = de.calc_brel_rest(obs_arr, sim_arr)
    # area of relative remaing bias
    b_area = de.calc_b_area(brel_rest)
    meta.iloc[4, 1] = b_area
    # temporal correlation
    temp_cor = de.calc_temp_cor(obs_arr, sim_arr)
    meta.iloc[4, 2] = temp_cor
    # diagnostic efficiency
    meta.iloc[4, 3] = 1 - np.sqrt((brel_mean)**2 + (b_area)**2 + (temp_cor - 1)**2)
    # direction of bias
    b_dir = de.calc_bias_dir(brel_rest)
    meta.iloc[4, 4] = b_dir
    # slope of bias
    b_slope = de.calc_bias_slope(b_area, b_dir)
    meta.iloc[4, 5] = b_slope
    # convert to radians
    # (y, x) Trigonometric inverse tangent
    meta.iloc[4, 6] = np.arctan2(brel_mean, b_slope)

    # KGE
    meta.iloc[4, 7] = de.calc_kge(obs_arr, sim_arr)
    # KGE alpha
    meta.iloc[4, 8] = de.calc_kge_alpha(obs_arr, sim_arr)
    # KGE beta
    meta.iloc[4, 9] = de.calc_kge_beta(obs_arr, sim_arr)

    # NSE
    meta.iloc[4, 10] = de.calc_nse(obs_arr, sim_arr)

    ### shuffling ###
    obs_sim = pd.DataFrame(index=df_ts.index, columns=['Qobs', 'Qsim'])
    obs_sim.loc[:, 'Qobs'] = df_ts.loc[:, 'Qobs']
    tss = de.time_shift(df_ts.copy(), random=True)  # shuffled time series
    obs_sim.loc[:, 'Qsim'] = tss.iloc[:, 0].values
    de.fdc_obs_sim(obs_sim['Qobs'], obs_sim['Qsim'])
    de.plot_obs_sim(obs_sim['Qobs'], obs_sim['Qsim'])

    # make arrays
    obs_arr = obs_sim['Qobs'].values
    sim_arr = obs_sim['Qsim'].values

    # mean relative bias
    brel_mean = de.calc_brel_mean(obs_arr, sim_arr)
    meta.iloc[5, 0] = brel_mean
    # remaining relative bias
    brel_rest = de.calc_brel_rest(obs_arr, sim_arr)
    # area of relative remaing bias
    b_area = de.calc_b_area(brel_rest)
    meta.iloc[5, 1] = b_area
    # temporal correlation
    temp_cor = de.calc_temp_cor(obs_arr, sim_arr)
    meta.iloc[5, 2] = temp_cor
    # diagnostic efficiency
    meta.iloc[5, 3] = 1 - np.sqrt((brel_mean)**2 + (b_area)**2 + (temp_cor - 1)**2)
    # direction of bias
    b_dir = de.calc_bias_dir(brel_rest)
    meta.iloc[5, 4] = b_dir
    # slope of bias
    b_slope = de.calc_bias_slope(b_area, b_dir)
    meta.iloc[5, 5] = b_slope
    # convert to radians
    # (y, x) Trigonometric inverse tangent
    meta.iloc[5, 6] = np.arctan2(brel_mean, b_slope)

    # KGE
    meta.iloc[5, 7] = de.calc_kge(obs_arr, sim_arr)
    # KGE alpha
    meta.iloc[5, 8] = de.calc_kge_alpha(obs_arr, sim_arr)
    # KGE beta
    meta.iloc[5, 9] = de.calc_kge_beta(obs_arr, sim_arr)

    # NSE
    meta.iloc[5, 10] = de.calc_nse(obs_arr, sim_arr)

    ### Decrease high flows - Increase low flows and precipitation surplus ###
    obs_sim = pd.DataFrame(index=df_ts.index, columns=['Qobs', 'Qsim'])
    obs_sim.loc[:, 'Qobs'] = df_ts.loc[:, 'Qobs']
    tsd = de.highunder_lowover(df_ts.copy(), prop=0.34)  # smoothed time series
    obs_sim.loc[:, 'Qsim'] = de.pos_shift_ts(tsd.iloc[:, 0].values, offset=1.2)  # positive offset
    de.fdc_obs_sim(obs_sim['Qobs'], obs_sim['Qsim'])
    de.plot_obs_sim(obs_sim['Qobs'], obs_sim['Qsim'])

    # make arrays
    obs_arr = obs_sim['Qobs'].values
    sim_arr = obs_sim['Qsim'].values

    # mean relative bias
    brel_mean = de.calc_brel_mean(obs_arr, sim_arr)
    meta.iloc[6, 0] = brel_mean
    # remaining relative bias
    brel_rest = de.calc_brel_rest(obs_arr, sim_arr)
    # area of relative remaing bias
    b_area = de.calc_b_area(brel_rest)
    meta.iloc[6, 1] = b_area
    # temporal correlation
    temp_cor = de.calc_temp_cor(obs_arr, sim_arr)
    meta.iloc[6, 2] = temp_cor
    # diagnostic efficiency
    meta.iloc[6, 3] = 1 - np.sqrt((brel_mean)**2 + (b_area)**2 + (temp_cor - 1)**2)
    # direction of bias
    b_dir = de.calc_bias_dir(brel_rest)
    meta.iloc[6, 4] = b_dir
    # slope of bias
    b_slope = de.calc_bias_slope(b_area, b_dir)
    meta.iloc[6, 5] = b_slope
    # convert to radians
    # (y, x) Trigonometric inverse tangent
    meta.iloc[6, 6] = np.arctan2(brel_mean, b_slope)

    # KGE
    meta.iloc[6, 7] = de.calc_kge(obs_arr, sim_arr)
    # KGE alpha
    meta.iloc[6, 8] = de.calc_kge_alpha(obs_arr, sim_arr)
    # KGE beta
    meta.iloc[6, 9] = de.calc_kge_beta(obs_arr, sim_arr)

    # NSE
    meta.iloc[6, 10] = de.calc_nse(obs_arr, sim_arr)

    ### Decrease high flows - Increase low flows and precipitation shortage ###
    obs_sim = pd.DataFrame(index=df_ts.index, columns=['Qobs', 'Qsim'])
    obs_sim.loc[:, 'Qobs'] = df_ts.loc[:, 'Qobs']
    tsd = de.highunder_lowover(df_ts.copy(), prop=0.5)  # smoothed time series
    obs_sim.loc[:, 'Qsim'] = de.neg_shift_ts(tsd.iloc[:, 0].values, offset=0.8)  # negative offset
    de.fdc_obs_sim(obs_sim['Qobs'], obs_sim['Qsim'])
    de.plot_obs_sim(obs_sim['Qobs'], obs_sim['Qsim'])

    # make arrays
    obs_arr = obs_sim['Qobs'].values
    sim_arr = obs_sim['Qsim'].values

    # mean relative bias
    brel_mean = de.calc_brel_mean(obs_arr, sim_arr)
    meta.iloc[7, 0] = brel_mean
    # remaining relative bias
    brel_rest = de.calc_brel_rest(obs_arr, sim_arr)
    # area of relative remaing bias
    b_area = de.calc_b_area(brel_rest)
    meta.iloc[7, 1] = b_area
    # temporal correlation
    temp_cor = de.calc_temp_cor(obs_arr, sim_arr)
    meta.iloc[7, 2] = temp_cor
    # diagnostic efficiency
    meta.iloc[7, 3] = 1 - np.sqrt((brel_mean)**2 + (b_area)**2 + (temp_cor - 1)**2)
    # direction of bias
    b_dir = de.calc_bias_dir(brel_rest)
    meta.iloc[7, 4] = b_dir
    # slope of bias
    b_slope = de.calc_bias_slope(b_area, b_dir)
    meta.iloc[7, 5] = b_slope
    # convert to radians
    # (y, x) Trigonometric inverse tangent
    meta.iloc[7, 6] = np.arctan2(brel_mean, b_slope)

    # KGE
    meta.iloc[7, 7] = de.calc_kge(obs_arr, sim_arr)
    # KGE alpha
    meta.iloc[7, 8] = de.calc_kge_alpha(obs_arr, sim_arr)
    # KGE beta
    meta.iloc[7, 9] = de.calc_kge_beta(obs_arr, sim_arr)

    # NSE
    meta.iloc[7, 10] = de.calc_nse(obs_arr, sim_arr)

    ### Increase high flows - Decrease low flows and precipitation surplus ###
    obs_sim = pd.DataFrame(index=df_ts.index, columns=['Qobs', 'Qsim'])
    obs_sim.loc[:, 'Qobs'] = df_ts.loc[:, 'Qobs']
    tsp = df_ts.copy()
    tsp.iloc[:, 0] = de.pos_shift_ts(tsp.iloc[:, 0].values, offset=1.2)  # positive offset
    tsd = de.highover_lowunder(tsp.copy(), prop=0.34)  # disaggregated time series
    obs_sim.loc[:, 'Qsim'] = tsd.iloc[:, 0].values  # positive offset
    de.fdc_obs_sim(obs_sim['Qobs'], obs_sim['Qsim'])
    de.plot_obs_sim(obs_sim['Qobs'], obs_sim['Qsim'])

    # make arrays
    obs_arr = obs_sim['Qobs'].values
    sim_arr = obs_sim['Qsim'].values

    # mean relative bias
    brel_mean = de.calc_brel_mean(obs_arr, sim_arr)
    meta.iloc[8, 0] = brel_mean
    # remaining relative bias
    brel_rest = de.calc_brel_rest(obs_arr, sim_arr)
    # area of relative remaing bias
    b_area = de.calc_b_area(brel_rest)
    meta.iloc[8, 1] = b_area
    # temporal correlation
    temp_cor = de.calc_temp_cor(obs_arr, sim_arr)
    meta.iloc[8, 2] = temp_cor
    # diagnostic efficiency
    meta.iloc[8, 3] = 1 - np.sqrt((brel_mean)**2 + (b_area)**2 + (temp_cor - 1)**2)
    # direction of bias
    b_dir = de.calc_bias_dir(brel_rest)
    meta.iloc[8, 4] = b_dir
    # slope of bias
    b_slope = de.calc_bias_slope(b_area, b_dir)
    meta.iloc[8, 5] = b_slope
    # convert to radians
    # (y, x) Trigonometric inverse tangent
    meta.iloc[8, 6] = np.arctan2(brel_mean, b_slope)

    # KGE
    meta.iloc[8, 7] = de.calc_kge(obs_arr, sim_arr)
    # KGE alpha
    meta.iloc[8, 8] = de.calc_kge_alpha(obs_arr, sim_arr)
    # KGE beta
    meta.iloc[8, 9] = de.calc_kge_beta(obs_arr, sim_arr)

    # NSE
    meta.iloc[8, 10] = de.calc_nse(obs_arr, sim_arr)

    ### Increase high flows - Decrease low flows and precipitation shortage ###
    obs_sim = pd.DataFrame(index=df_ts.index, columns=['Qobs', 'Qsim'])
    obs_sim.loc[:, 'Qobs'] = df_ts.loc[:, 'Qobs']
    tsd = de.highover_lowunder(df_ts.copy(), prop=0.5) # disaggregated time series
    obs_sim.loc[:, 'Qsim'] = de.neg_shift_ts(tsd.iloc[:, 0].values, offset=0.8)  # negative offset
    de.fdc_obs_sim(obs_sim['Qobs'], obs_sim['Qsim'])
    de.plot_obs_sim(obs_sim['Qobs'], obs_sim['Qsim'])

    # make arrays
    obs_arr = obs_sim['Qobs'].values
    sim_arr = obs_sim['Qsim'].values

    # mean relative bias
    brel_mean = de.calc_brel_mean(obs_arr, sim_arr)
    meta.iloc[9, 0] = brel_mean
    # remaining relative bias
    brel_rest = de.calc_brel_rest(obs_arr, sim_arr)
    # area of relative remaing bias
    b_area = de.calc_b_area(brel_rest)
    meta.iloc[9, 1] = b_area
    # temporal correlation
    temp_cor = de.calc_temp_cor(obs_arr, sim_arr)
    meta.iloc[9, 2] = temp_cor
    # diagnostic efficiency
    meta.iloc[9, 3] = 1 - np.sqrt((brel_mean)**2 + (b_area)**2 + (temp_cor - 1)**2)
    # direction of bias
    b_dir = de.calc_bias_dir(brel_rest)
    meta.iloc[9, 4] = b_dir
    # slope of bias
    b_slope = de.calc_bias_slope(b_area, b_dir)
    meta.iloc[9, 5] = b_slope
    # convert to radians
    # (y, x) Trigonometric inverse tangent
    meta.iloc[9, 6] = np.arctan2(brel_mean, b_slope)

    # KGE
    meta.iloc[9, 7] = de.calc_kge(obs_arr, sim_arr)
    # KGE alpha
    meta.iloc[9, 8] = de.calc_kge_alpha(obs_arr, sim_arr)
    # KGE beta
    meta.iloc[9, 9] = de.calc_kge_beta(obs_arr, sim_arr)

    # NSE
    meta.iloc[9, 10] = de.calc_nse(obs_arr, sim_arr)

    ### mean flow benchmark ###
    obs_sim = pd.DataFrame(index=df_ts.index, columns=['Qobs', 'Qsim'])
    obs_sim.loc[:, 'Qobs'] = df_ts.loc[:, 'Qobs']
    obs_mean = np.mean(obs_sim['Qobs'].values)
    obs_sim.loc[:, 'Qsim'] = np.repeat(obs_mean, len(obs_sim['Qobs'].values))
    de.plot_obs_sim(obs_sim['Qobs'], obs_sim['Qsim'])
    de.fdc_obs_sim(obs_sim['Qobs'], obs_sim['Qsim'])

    # make arrays
    obs_arr = obs_sim['Qobs'].values
    sim_arr = obs_sim['Qsim'].values

    # mean relative bias
    brel_mean = de.calc_brel_mean(obs_arr, sim_arr)
    meta.iloc[10, 0] = brel_mean
    # remaining relative bias
    brel_rest = de.calc_brel_rest(obs_arr, sim_arr)
    # area of relative remaing bias
    b_area = de.calc_b_area(brel_rest)
    meta.iloc[10, 1] = b_area
    # temporal correlation
    temp_cor = de.calc_temp_cor(obs_arr, sim_arr)
    meta.iloc[10, 2] = temp_cor
    # diagnostic efficiency
    meta.iloc[10, 3] = 1 - np.sqrt((brel_mean)**2 + (b_area)**2 + (temp_cor - 1)**2)
    # direction of bias
    b_dir = de.calc_bias_dir(brel_rest)
    meta.iloc[10, 4] = b_dir
    # slope of bias
    b_slope = de.calc_bias_slope(b_area, b_dir)
    meta.iloc[10, 5] = b_slope
    # convert to radians
    # (y, x) Trigonometric inverse tangent
    meta.iloc[10, 6] = np.arctan2(brel_mean, b_slope)

    # KGE
    meta.iloc[10, 7] = de.calc_kge(obs_arr, sim_arr)
    # KGE alpha
    meta.iloc[10, 8] = de.calc_kge_alpha(obs_arr, sim_arr)
    # KGE beta
    meta.iloc[10, 9] = de.calc_kge_beta(obs_arr, sim_arr)

    # NSE
    meta.iloc[10, 10] = de.calc_nse(obs_arr, sim_arr)

#
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
