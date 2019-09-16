#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from de import de
from de import util

if __name__ == "__main__":
    path = '/Users/robinschwemmle/Desktop/PhD/diagnostic_model_efficiency/examples/data/9960682_Q_1970_2012.csv'
#    path = '/Users/robo/Desktop/PhD/de/examples/data/9960682_Q_1970_2012.csv'

    # import observed time series
    df_ts = util.import_ts(path, sep=';')
    de.plot_ts(df_ts)

    ### increase high flows - decrease low flows ###
    obs_sim = pd.DataFrame(index=df_ts.index, columns=['Qobs', 'Qsim'])
    obs_sim.loc[:, 'Qobs'] = df_ts.loc[:, 'Qobs']
    tsd = de.highover_lowunder(df_ts.copy(), prop=0.5)
    obs_sim.loc[:, 'Qsim'] = tsd.loc[:, 'Qsim']  # disaggregated time series
    de.fdc_obs_sim(obs_sim['Qobs'], obs_sim['Qsim'])
    de.plot_obs_sim(obs_sim['Qobs'], obs_sim['Qsim'])

    obs_arr = obs_sim['Qobs'].values
    sim_arr = obs_sim['Qsim'].values

    sig_de = de.calc_de(obs_arr, sim_arr)
    sig_kge = de.calc_kge(obs_arr, sim_arr)
    sig_nse = de.calc_nse(obs_arr, sim_arr)

    de.vis2d_de(obs_arr, sim_arr)
    de.vis2d_kge(obs_arr, sim_arr)

#    ### decrease high flows - increase low flows ###
#    obs_sim = pd.DataFrame(index=df_ts.index, columns=['Qobs', 'Qsim'])
#    obs_sim.loc[:, 'Qobs'] = df_ts.loc[:, 'Qobs']
#    tsd = de.highunder_lowover(df_ts.copy(), prop=0.5)
#    obs_sim.loc[:, 'Qsim'] = tsd.loc[:, 'Qsim']  # smoothed time series
#    de.fdc_obs_sim(obs_sim['Qobs'], obs_sim['Qsim'])
#    de.plot_obs_sim(obs_sim['Qobs'], obs_sim['Qsim'])
#
#    obs_arr = obs_sim['Qobs'].values
#    sim_arr = obs_sim['Qsim'].values
#
#    sig_de = de.calc_de(obs_arr, sim_arr)
#    sig_kge = de.calc_kge(obs_arr, sim_arr)
#    sig_nse = de.calc_nse(obs_arr, sim_arr)
#
#    de.vis2d_de(obs_arr, sim_arr, extended=True)
#    de.vis2d_kge(obs_arr, sim_arr)
#
#    ### precipitation surplus ###
#    obs_sim = pd.DataFrame(index=df_ts.index, columns=['Qobs', 'Qsim'])
#    obs_sim.loc[:, 'Qobs'] = df_ts.loc[:, 'Qobs']
#    obs_sim.loc[:, 'Qsim'] = de.pos_shift_ts(df_ts['Qobs'].values)  # positive offset
#    de.plot_obs_sim(obs_sim['Qobs'], obs_sim['Qsim'])
#    de.fdc_obs_sim(obs_sim['Qobs'], obs_sim['Qsim'])
#
#    obs_arr = obs_sim['Qobs'].values
#    sim_arr = obs_sim['Qsim'].values
#
#    sig_de = de.calc_de(obs_arr, sim_arr)
#    sig_kge = de.calc_kge(obs_arr, sim_arr)
#    sig_nse = de.calc_nse(obs_arr, sim_arr)
#
#    de.vis2d_de(obs_arr, sim_arr)
#    de.vis2d_kge(obs_arr, sim_arr)
#
#    ### precipitation shortage ###
#    obs_sim = pd.DataFrame(index=df_ts.index, columns=['Qobs', 'Qsim'])
#    obs_sim.loc[:, 'Qobs'] = df_ts.loc[:, 'Qobs']
#    obs_sim.loc[:, 'Qsim'] = de.neg_shift_ts(df_ts['Qobs'].values)  # negative offset
#    de.plot_obs_sim(obs_sim['Qobs'], obs_sim['Qsim'])
#    de.fdc_obs_sim(obs_sim['Qobs'], obs_sim['Qsim'])
#
#    obs_arr = obs_sim['Qobs'].values
#    sim_arr = obs_sim['Qsim'].values
#
#    sig_de = de.calc_de(obs_arr, sim_arr)
#    sig_kge = de.calc_kge(obs_arr, sim_arr)
#    sig_nse = de.calc_nse(obs_arr, sim_arr)
#
#    de.vis2d_de(obs_arr, sim_arr)
#    de.vis2d_kge(obs_arr, sim_arr)
#
#    ### mean flow benchmark ###
#    obs_sim = pd.DataFrame(index=df_ts.index, columns=['Qobs', 'Qsim'])
#    obs_sim.loc[:, 'Qobs'] = df_ts.loc[:, 'Qobs']
#    obs_mean = np.mean(obs_sim['Qobs'].values)
#    obs_sim.loc[:, 'Qsim'] = np.repeat(obs_mean, len(obs_sim['Qobs'].values))
#    de.plot_obs_sim(obs_sim['Qobs'], obs_sim['Qsim'])
#    de.fdc_obs_sim(obs_sim['Qobs'], obs_sim['Qsim'])
#
#    obs_arr = obs_sim['Qobs'].values
#    sim_arr = obs_sim['Qsim'].values
#
#    sig_de = de.calc_de(obs_arr, sim_arr)
#    sig_kge = de.calc_kge(obs_arr, sim_arr)
#    sig_nse = de.calc_nse(obs_arr, sim_arr)
#
#    de.vis2d_de(obs_arr, sim_arr)
#    de.vis2d_kge(obs_arr, sim_arr)
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
