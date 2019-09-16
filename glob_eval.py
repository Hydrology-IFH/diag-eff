#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from de import de
from de import util

if __name__ == "__main__":
    # RunTimeWarning will not be displayed (division by zeros or NaN values)
    np.seterr(divide='ignore', invalid='ignore')

    # define operation system
    # 'win' = windows machine, 'unix_local' = local unix machine, 'unix' = acces data with unix machine from file server
    OS = 'unix_server'
    if OS == 'win':
        # windows directories
        db_Qobs_meta_dir = '//fuhys013/Schwemmle/Data/Runoff/observed/' \
                           'summary_near_nat.csv'
        Q_dir = '//fuhys013/Schwemmle/Data/Runoff/Q_paired/wrr2/'

    elif OS == 'unix_server':
        # unix directories server
        db_Qobs_meta_dir = '/Volumes/Schwemmle/Data/Runoff/observed/' \
                           'summary_near_nat.csv'
        Q_dir = '/Volumes/Schwemmle/Data/Runoff/Q_paired/wrr1/'

    elif OS == 'unix_local':
        # unix directories local
        db_Qobs_meta_dir = '/Users/robinschwemmle/Desktop/MSc_Thesis/Data'\
                           '/Runoff/observed/summary_near_nat.csv'
        Q_dir = '/Users/robinschwemmle/Desktop/MSc_Thesis/Data/Runoff/Q_paired/wrr1/'

    df_meta = pd.read_csv(db_Qobs_meta_dir, sep=';', na_values=-9999, index_col=0)
    meta = df_meta.loc[ :, ['lat', 'lon']].copy()
    meta['brel_mean'] = np.nan
    meta['b_area'] = np.nan
    meta['temp_cor'] = np.nan
    meta['de'] = np.nan
    meta['b_dir'] = np.nan
    meta['b_slope'] = np.nan
    meta['diag'] = np.nan

    for i, catch in enumerate(meta.index):
        path_csv = '{}{}.csv'.format(Q_dir, str(catch))
        # import observed time series
        obs_sim = util.import_ts(path_csv, sep=';', na_values=-9999)
        obs_arr = obs_sim['Qobs'].values
        sim_arr = obs_sim['Qsim'].values

        # mean relative bias
        brel_mean = de.calc_brel_mean(obs_arr, sim_arr, sort=sort)
        meta.iloc[i, 0] = brel_mean

        # remaining relative bias
        brel_rest = de.calc_brel_rest(obs_arr, sim_arr, sort=sort)
        # area of relative remaing bias
        b_area = de.calc_b_area(brel_rest)
        meta.iloc[i, 1] = b_area
        # temporal correlation
        temp_cor = de.calc_temp_cor(obs_arr, sim_arr)
        meta.iloc[i, 2] = temp_cor
        # diagnostic efficiency
        meta.iloc[i, 3] = 1 - np.sqrt((brel_mean)**2 + (b_area)**2 + (temp_cor - 1)**2)

        # direction of bias
        b_dir = calc_bias_dir(brel_rest)

        meta.iloc[i, 4] = b_dir

        # slope of bias
        b_slope = calc_bias_slope(b_area, b_dir)
        meta.iloc[i, 5] = b_slope

        # convert to radians
        # (y, x) Trigonometric inverse tangent
        meta.iloc[i, 6] = np.arctan2(brel_mean, b_slope)
