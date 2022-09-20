#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os  # load modules first before importing .spydata
from pathlib import Path

PATH = Path(__file__).parent
sys.path.append(str(PATH))
PATH_FIG = PATH.parent.parent / "diagnostic_efficiency" / "figures"
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
sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})
sns.set_context("paper", font_scale=1.5)

if __name__ == "__main__":
    # ==========================================================
    # import observed streamflow time series
    # ==========================================================
    # 619.11 km2; AI: 0.82
    area = 619.11
    path = PATH / "examples" / "13331500_streamflow_qc.txt"
    ## 191.55 km2; AI: 2.04
    # area = 191.55
    # path = PATH / "examples" / "06332515_streamflow_qc.txt"
    #
    ## 190.65 km2; AI: 2.98
    # area = 190.65
    # path = PATH / "examples" / "09512280_streamflow_qc.txt"
    #
    ## 66.57 km2; AI: 0.27
    # area = 66.57
    # path = PATH / "examples" / "12114500_streamflow_qc.txt"
    #

    # import observed time series
    df_ts = util.import_camels_ts(path, sep=r"\s+", catch_area=area)
    df_ts0 = df_ts.copy()
    cond0 = (df_ts0['Qobs'] < 0.15)
    df_ts0.loc[cond0, 'Qobs'] = 0
    no0 = np.sum(cond0.values)
    # fig_ts_obs = util.plot_ts(df_ts)
    # path = Path(os.path.join(PATH_FIG, "original_ts.pdf"))
    # fig_ts_obs.savefig(path, dpi=250)

    # # ==========================================================
    # # proof of concept
    # # ==========================================================
    # # prepare figures
    # fig_num_fdc = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)", "(i)"]
    # fig_fdc, axes_fdc = plt.subplots(2, 5, sharey=True, sharex=True, figsize=(14, 6))
    # fig_fdc.text(0.5, 0.02, "Exceedence probabilty [-]", ha="center", va="center")
    # fig_fdc.text(
    #     0.08, 0.5, r"Q [mm $d^{-1}$]", ha="center", va="center", rotation="vertical"
    # )
    # axes_fdc[1, 4].remove()

    # fig_num_ts = [
    #     "(a)",
    #     "(b)",
    #     "(c)",
    #     "(d)",
    #     "(e)",
    #     "(f)",
    #     "(g)",
    #     "(h)",
    #     "(i)",
    #     "(j)",
    #     "(k)",
    #     "(l)",
    #     "(m)",
    # ]
    # fig_ts, axes_ts = plt.subplots(3, 5, sharey=True, sharex=True,
    #                                 figsize=(14, 9))
    # fig_ts.text(0.5, 0.05, "Time [Years]", ha="center", va="center")
    # fig_ts.text(
    #     0.08, 0.5, r"Q [mm $d^{-1}$]", ha="center", va="center", rotation="vertical"
    # )
    # axes_ts[2, 3].remove()
    # axes_ts[2, 4].remove()

    # # extract a single year from the time series
    # yy_str_1 = '2000-1-1 00:00:00'  # start date
    # yy_str_2 = '2000-12-31 00:00:00'  # end date
    # fig_ts_yy, axes_ts_yy = plt.subplots(3, 5, sharey=True, sharex=True,
    #                                       figsize=(14, 9))
    # fig_ts_yy.text(0.5, 0.05, "Time [Years]", ha="center", va="center")
    # fig_ts_yy.text(
    #     0.08, 0.5, r"Q [mm $d^{-1}$]", ha="center", va="center", rotation="vertical"
    # )
    # axes_ts_yy[2, 3].remove()
    # axes_ts_yy[2, 4].remove()

    # # dataframe to compare DE, KGE and NSE
    # idx = ["0", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m"]
    # cols = [
    #     "brel_mean",
    #     "b_area",
    #     "temp_cor",
    #     "de",
    #     "b_dir",
    #     "b_slope",
    #     "phi",
    #     "b_hf",
    #     "b_lf",
    #     "b_tot",
    #     "err_hf",
    #     "err_lf",
    #     "beta",
    #     "alpha",
    #     "kge",
    #     "nse",
    # ]
    # df_es = pd.DataFrame(index=idx, columns=cols, dtype=np.float32)

    # # ----------------------------------------------------------
    # # perfect simulation
    # # ----------------------------------------------------------
    # obs_sim = pd.DataFrame(index=df_ts.index, columns=["Qobs", "Qsim"])
    # obs_sim.loc[:, "Qobs"] = df_ts.loc[:, "Qobs"]
    # obs_sim.loc[:, "Qsim"] = df_ts.loc[:, "Qobs"]  # observed time series

    # # make numpy arrays
    # obs_arr = obs_sim["Qobs"].values  # observed time series
    # sim_arr = obs_sim["Qsim"].values  # manipulated time series

    # # mean relative bias
    # brel_mean = de.calc_brel_mean(obs_arr, sim_arr)
    # df_es.loc["0", "brel_mean"] = brel_mean
    # # residual relative bias
    # brel_res = de.calc_brel_res(obs_arr, sim_arr)
    # # area of relative remaing bias
    # b_area = de.calc_bias_area(brel_res)
    # df_es.loc["0", "b_area"] = b_area
    # # temporal correlation
    # temp_cor = de.calc_temp_cor(obs_arr, sim_arr)
    # df_es.loc["0", "temp_cor"] = temp_cor
    # # diagnostic efficiency
    # df_es.loc["0", "de"] = de.calc_de(obs_arr, sim_arr)
    # # relative bias
    # brel = de.calc_brel(obs_arr, sim_arr)
    # # total bias
    # b_tot = de.calc_bias_tot(brel)
    # df_es.loc["0", "b_tot"] = b_tot
    # # bias of high flows
    # b_hf = de.calc_bias_hf(brel)
    # df_es.loc["0", "b_hf"] = b_hf
    # # contribution of high flow errors
    # err_hf = de.calc_err_hf(b_hf, b_tot)
    # df_es.loc["0", "err_hf"] = err_hf
    # # bias of low flows
    # b_lf = de.calc_bias_lf(brel)
    # df_es.loc["0", "b_lf"] = b_lf
    # # contribution of low flow errors
    # err_lf = de.calc_err_lf(b_lf, b_tot)
    # df_es.loc["0", "err_lf"] = err_lf
    # # direction of bias
    # b_dir = de.calc_bias_dir(brel_res)
    # df_es.loc["0", "b_dir"] = b_dir
    # # slope of bias
    # b_slope = de.calc_bias_slope(b_area, b_dir)
    # df_es.loc["0", "b_slope"] = b_slope
    # # convert to radians
    # # (y, x) Trigonometric inverse tangent
    # df_es.loc["0", "phi"] = de.calc_phi(brel_mean, b_slope)

    # # KGE beta
    # df_es.loc["0", "beta"] = kge.calc_kge_beta(obs_arr, sim_arr)
    # # KGE alpha
    # df_es.loc["0", "alpha"] = kge.calc_kge_alpha(obs_arr, sim_arr)
    # # # KGE gamma
    # # df_es.iloc[0, 8] = kge.calc_kge_gamma(obs_arr, sim_arr)
    # # KGE
    # df_es.loc["0", "kge"] = kge.calc_kge(obs_arr, sim_arr)

    # # NSE
    # df_es.loc["0", "nse"] = nse.calc_nse(obs_arr, sim_arr)

    # # ----------------------------------------------------------
    # # positive constant error
    # # ----------------------------------------------------------
    # obs_sim = pd.DataFrame(index=df_ts.index, columns=["Qobs", "Qsim"])
    # obs_sim.loc[:, "Qobs"] = df_ts.loc[:, "Qobs"]
    # # generate positive constant error
    # obs_sim.loc[:, "Qsim"] = generate_errors.constant(df_ts["Qobs"].values,
    #                                                   offset=1.25)
    # util.fdc_obs_sim_ax(
    #     obs_sim["Qobs"], obs_sim["Qsim"], axes_fdc[0, 0], fig_num_fdc[0]
    # )
    # util.plot_obs_sim_ax(obs_sim["Qobs"], obs_sim["Qsim"], axes_ts[0, 0],
    #                       fig_num_ts[0])

    # obs_sim_yy = obs_sim.loc[yy_str_1:yy_str_2, :]
    # util.plot_obs_sim_ax(obs_sim_yy["Qobs"], obs_sim_yy["Qsim"],
    #                       axes_ts_yy[0, 0], fig_num_ts[0])

    # # make arrays
    # obs_arr = obs_sim["Qobs"].values
    # sim_arr = obs_sim["Qsim"].values

    # # mean relative bias
    # brel_mean = de.calc_brel_mean(obs_arr, sim_arr)
    # df_es.loc["a", "brel_mean"] = brel_mean
    # # residual relative bias
    # brel_res = de.calc_brel_res(obs_arr, sim_arr)
    # # area of relative remaing bias
    # b_area = de.calc_bias_area(brel_res)
    # df_es.loc["a", "b_area"] = b_area
    # # temporal correlation
    # temp_cor = de.calc_temp_cor(obs_arr, sim_arr)
    # df_es.loc["a", "temp_cor"] = temp_cor
    # # diagnostic efficiency
    # df_es.loc["a", "de"] = de.calc_de(obs_arr, sim_arr)
    # # relative bias
    # brel = de.calc_brel(obs_arr, sim_arr)
    # # total bias
    # b_tot = de.calc_bias_tot(brel)
    # df_es.loc["a", "b_tot"] = b_tot
    # # bias of high flows
    # b_hf = de.calc_bias_hf(brel)
    # df_es.loc["a", "b_hf"] = b_hf
    # # contribution of high flow errors
    # err_hf = de.calc_err_hf(b_hf, b_tot)
    # df_es.loc["a", "err_hf"] = err_hf
    # # bias of low flows
    # b_lf = de.calc_bias_lf(brel)
    # df_es.loc["a", "b_lf"] = b_lf
    # # contribution of low flow errors
    # err_lf = de.calc_err_lf(b_lf, b_tot)
    # df_es.loc["a", "err_lf"] = err_lf
    # # direction of bias
    # b_dir = de.calc_bias_dir(brel_res)
    # df_es.loc["a", "b_dir"] = b_dir
    # # slope of bias
    # b_slope = de.calc_bias_slope(b_area, b_dir)
    # df_es.loc["a", "b_slope"] = b_slope
    # # convert to radians
    # # (y, x) Trigonometric inverse tangent
    # df_es.loc["a", "phi"] = de.calc_phi(brel_mean, b_slope)

    # # KGE beta
    # df_es.loc["a", "beta"] = kge.calc_kge_beta(obs_arr, sim_arr)
    # # KGE alpha
    # df_es.loc["a", "alpha"] = kge.calc_kge_alpha(obs_arr, sim_arr)
    # # # KGE gamma
    # # df_es.iloc[0, 8] = kge.calc_kge_gamma(obs_arr, sim_arr)
    # # KGE
    # df_es.loc["a", "kge"] = kge.calc_kge(obs_arr, sim_arr)

    # # NSE
    # df_es.loc["a", "nse"] = nse.calc_nse(obs_arr, sim_arr)

    # # ----------------------------------------------------------
    # # negative constant error
    # # ----------------------------------------------------------
    # obs_sim = pd.DataFrame(index=df_ts.index, columns=["Qobs", "Qsim"])
    # obs_sim.loc[:, "Qobs"] = df_ts.loc[:, "Qobs"]
    # # generate negative constant error
    # obs_sim.loc[:, "Qsim"] = generate_errors.constant(df_ts["Qobs"].values,
    #                                                   offset=0.75)
    # util.fdc_obs_sim_ax(
    #     obs_sim["Qobs"], obs_sim["Qsim"], axes_fdc[0, 1], fig_num_fdc[1]
    # )
    # util.plot_obs_sim_ax(obs_sim["Qobs"], obs_sim["Qsim"], axes_ts[0, 1],
    #                       fig_num_ts[1])

    # obs_sim_yy = obs_sim.loc[yy_str_1:yy_str_2, :]
    # util.plot_obs_sim_ax(obs_sim_yy["Qobs"], obs_sim_yy["Qsim"],
    #                       axes_ts_yy[0, 1], fig_num_ts[1])

    # # make arrays
    # obs_arr = obs_sim["Qobs"].values
    # sim_arr = obs_sim["Qsim"].values

    # # mean relative bias
    # brel_mean = de.calc_brel_mean(obs_arr, sim_arr)
    # df_es.loc["b", "brel_mean"] = brel_mean
    # # residual relative bias
    # brel_res = de.calc_brel_res(obs_arr, sim_arr)
    # # area of relative remaing bias
    # b_area = de.calc_bias_area(brel_res)
    # df_es.loc["b", "b_area"] = b_area
    # # temporal correlation
    # temp_cor = de.calc_temp_cor(obs_arr, sim_arr)
    # df_es.loc["b", "temp_cor"] = temp_cor
    # # diagnostic efficiency
    # df_es.loc["b", "de"] = de.calc_de(obs_arr, sim_arr)
    # # relative bias
    # brel = de.calc_brel(obs_arr, sim_arr)
    # # total bias
    # b_tot = de.calc_bias_tot(brel)
    # df_es.loc["b", "b_tot"] = b_tot
    # # bias of high flows
    # b_hf = de.calc_bias_hf(brel)
    # df_es.loc["b", "b_hf"] = b_hf
    # # contribution of high flow errors
    # err_hf = de.calc_err_hf(b_hf, b_tot)
    # df_es.loc["b", "err_hf"] = err_hf
    # # bias of low flows
    # b_lf = de.calc_bias_lf(brel)
    # df_es.loc["b", "b_lf"] = b_lf
    # # contribution of low flow errors
    # err_lf = de.calc_err_lf(b_lf, b_tot)
    # df_es.loc["b", "err_lf"] = err_lf
    # # direction of bias
    # b_dir = de.calc_bias_dir(brel_res)
    # df_es.loc["b", "b_dir"] = b_dir
    # # slope of bias
    # b_slope = de.calc_bias_slope(b_area, b_dir)
    # df_es.loc["b", "b_slope"] = b_slope
    # # convert to radians
    # # (y, x) Trigonometric inverse tangent
    # df_es.loc["b", "phi"] = de.calc_phi(brel_mean, b_slope)

    # # KGE beta
    # df_es.loc["b", "beta"] = kge.calc_kge_beta(obs_arr, sim_arr)
    # # KGE alpha
    # df_es.loc["b", "alpha"] = kge.calc_kge_alpha(obs_arr, sim_arr)
    # # # KGE gamma
    # # df_es.iloc[0, 8] = kge.calc_kge_gamma(obs_arr, sim_arr)
    # # KGE
    # df_es.loc["b", "kge"] = kge.calc_kge(obs_arr, sim_arr)

    # # NSE
    # df_es.loc["b", "nse"] = nse.calc_nse(obs_arr, sim_arr)

    # # ----------------------------------------------------------
    # # positive dynamic error
    # # ----------------------------------------------------------
    # obs_sim = pd.DataFrame(index=df_ts.index, columns=["Qobs", "Qsim"])
    # obs_sim.loc[:, "Qobs"] = df_ts.loc[:, "Qobs"]
    # # generate positive dynamic error
    # tsd = generate_errors.positive_dynamic(df_ts.copy(), prop=0.5)
    # obs_sim.loc[:, "Qsim"] = tsd.loc[:, "Qsim"]
    # util.fdc_obs_sim_ax(
    #     obs_sim["Qobs"], obs_sim["Qsim"], axes_fdc[0, 2], fig_num_fdc[2]
    # )
    # util.plot_obs_sim_ax(obs_sim["Qobs"], obs_sim["Qsim"], axes_ts[0, 2],
    #                       fig_num_ts[2])

    # obs_sim_yy = obs_sim.loc[yy_str_1:yy_str_2, :]
    # util.plot_obs_sim_ax(obs_sim_yy["Qobs"], obs_sim_yy["Qsim"],
    #                       axes_ts_yy[0, 2], fig_num_ts[2])

    # # make arrays
    # obs_arr = obs_sim["Qobs"].values
    # sim_arr = obs_sim["Qsim"].values

    # # mean relative bias
    # brel_mean = de.calc_brel_mean(obs_arr, sim_arr)
    # df_es.loc["c", "brel_mean"] = brel_mean
    # # residual relative bias
    # brel_res = de.calc_brel_res(obs_arr, sim_arr)
    # # area of relative remaing bias
    # b_area = de.calc_bias_area(brel_res)
    # df_es.loc["c", "b_area"] = b_area
    # # temporal correlation
    # temp_cor = de.calc_temp_cor(obs_arr, sim_arr)
    # df_es.loc["c", "temp_cor"] = temp_cor
    # # diagnostic efficiency
    # df_es.loc["c", "de"] = de.calc_de(obs_arr, sim_arr)
    # # relative bias
    # brel = de.calc_brel(obs_arr, sim_arr)
    # # total bias
    # b_tot = de.calc_bias_tot(brel)
    # df_es.loc["c", "b_tot"] = b_tot
    # # bias of high flows
    # b_hf = de.calc_bias_hf(brel)
    # df_es.loc["c", "b_hf"] = b_hf
    # # contribution of high flow errors
    # err_hf = de.calc_err_hf(b_hf, b_tot)
    # df_es.loc["c", "err_hf"] = err_hf
    # # bias of low flows
    # b_lf = de.calc_bias_lf(brel)
    # df_es.loc["c", "b_lf"] = b_lf
    # # contribution of low flow errors
    # err_lf = de.calc_err_lf(b_lf, b_tot)
    # df_es.loc["c", "err_lf"] = err_lf
    # # direction of bias
    # b_dir = de.calc_bias_dir(brel_res)
    # df_es.loc["c", "b_dir"] = b_dir
    # # slope of bias
    # b_slope = de.calc_bias_slope(b_area, b_dir)
    # df_es.loc["c", "b_slope"] = b_slope
    # # convert to radians
    # # (y, x) Trigonometric inverse tangent
    # df_es.loc["c", "phi"] = de.calc_phi(brel_mean, b_slope)

    # # KGE beta
    # df_es.loc["c", "beta"] = kge.calc_kge_beta(obs_arr, sim_arr)
    # # KGE alpha
    # df_es.loc["c", "alpha"] = kge.calc_kge_alpha(obs_arr, sim_arr)
    # # # KGE gamma
    # # df_es.iloc[0, 8] = kge.calc_kge_gamma(obs_arr, sim_arr)
    # # KGE
    # df_es.loc["c", "kge"] = kge.calc_kge(obs_arr, sim_arr)

    # # NSE
    # df_es.loc["c", "nse"] = nse.calc_nse(obs_arr, sim_arr)

    # # ----------------------------------------------------------
    # # negative dynamic error
    # # ----------------------------------------------------------
    # obs_sim = pd.DataFrame(index=df_ts.index, columns=["Qobs", "Qsim"])
    # obs_sim.loc[:, "Qobs"] = df_ts.loc[:, "Qobs"]
    # # generate negative dynamic error
    # tsd = generate_errors.negative_dynamic(df_ts.copy(), prop=0.5)
    # obs_sim.loc[:, "Qsim"] = tsd.loc[:, "Qsim"]
    # util.fdc_obs_sim_ax(
    #     obs_sim["Qobs"], obs_sim["Qsim"], axes_fdc[0, 3], fig_num_fdc[3]
    # )
    # util.plot_obs_sim_ax(obs_sim["Qobs"], obs_sim["Qsim"], axes_ts[0, 3],
    #                       fig_num_ts[3])

    # obs_sim_yy = obs_sim.loc[yy_str_1:yy_str_2, :]
    # util.plot_obs_sim_ax(obs_sim_yy["Qobs"], obs_sim_yy["Qsim"],
    #                       axes_ts_yy[0, 3], fig_num_ts[3])

    # # make arrays
    # obs_arr = obs_sim["Qobs"].values
    # sim_arr = obs_sim["Qsim"].values

    # # mean relative bias
    # brel_mean = de.calc_brel_mean(obs_arr, sim_arr)
    # df_es.loc["d", "brel_mean"] = brel_mean
    # # residual relative bias
    # brel_res = de.calc_brel_res(obs_arr, sim_arr)
    # # area of relative remaing bias
    # b_area = de.calc_bias_area(brel_res)
    # df_es.loc["d", "b_area"] = b_area
    # # temporal correlation
    # temp_cor = de.calc_temp_cor(obs_arr, sim_arr)
    # df_es.loc["d", "temp_cor"] = temp_cor
    # # diagnostic efficiency
    # df_es.loc["d", "de"] = de.calc_de(obs_arr, sim_arr)
    # # relative bias
    # brel = de.calc_brel(obs_arr, sim_arr)
    # # total bias
    # b_tot = de.calc_bias_tot(brel)
    # df_es.loc["d", "b_tot"] = b_tot
    # # bias of high flows
    # b_hf = de.calc_bias_hf(brel)
    # df_es.loc["d", "b_hf"] = b_hf
    # # contribution of high flow errors
    # err_hf = de.calc_err_hf(b_hf, b_tot)
    # df_es.loc["d", "err_hf"] = err_hf
    # # bias of low flows
    # b_lf = de.calc_bias_lf(brel)
    # df_es.loc["d", "b_lf"] = b_lf
    # # contribution of low flow errors
    # err_lf = de.calc_err_lf(b_lf, b_tot)
    # df_es.loc["d", "err_lf"] = err_lf
    # # direction of bias
    # b_dir = de.calc_bias_dir(brel_res)
    # df_es.loc["d", "b_dir"] = b_dir
    # # slope of bias
    # b_slope = de.calc_bias_slope(b_area, b_dir)
    # df_es.loc["d", "b_slope"] = b_slope
    # # convert to radians
    # # (y, x) Trigonometric inverse tangent
    # df_es.loc["d", "phi"] = de.calc_phi(brel_mean, b_slope)

    # # KGE beta
    # df_es.loc["d", "beta"] = kge.calc_kge_beta(obs_arr, sim_arr)
    # # KGE alpha
    # df_es.loc["d", "alpha"] = kge.calc_kge_alpha(obs_arr, sim_arr)
    # # # KGE gamma
    # # df_es.iloc[0, 8] = kge.calc_kge_gamma(obs_arr, sim_arr)
    # # KGE
    # df_es.loc["d", "kge"] = kge.calc_kge(obs_arr, sim_arr)

    # # NSE
    # df_es.loc["d", "nse"] = nse.calc_nse(obs_arr, sim_arr)

    # # ----------------------------------------------------------
    # # timing error
    # # ----------------------------------------------------------
    # obs_sim = pd.DataFrame(index=df_ts.index, columns=["Qobs", "Qsim"])
    # obs_sim.loc[:, "Qobs"] = df_ts.loc[:, "Qobs"]
    # # generate timing error
    # tss = generate_errors.timing(df_ts.copy(), shuffle=True)  # shuffling
    # obs_sim.loc[:, "Qsim"] = tss.iloc[:, 0].values
    # util.fdc_obs_sim_ax(
    #     obs_sim["Qobs"], obs_sim["Qsim"], axes_fdc[0, 4], fig_num_fdc[4]
    # )
    # util.plot_obs_sim_ax(obs_sim["Qobs"], obs_sim["Qsim"], axes_ts[0, 4],
    #                       fig_num_ts[4])

    # obs_sim_yy = obs_sim.loc[yy_str_1:yy_str_2, :]
    # util.plot_obs_sim_ax(obs_sim_yy["Qobs"], obs_sim_yy["Qsim"],
    #                       axes_ts_yy[0, 4], fig_num_ts[4])

    # # make arrays
    # obs_arr = obs_sim["Qobs"].values
    # sim_arr = obs_sim["Qsim"].values

    # # mean relative bias
    # brel_mean = de.calc_brel_mean(obs_arr, sim_arr)
    # df_es.loc["e", "brel_mean"] = brel_mean
    # # residual relative bias
    # brel_res = de.calc_brel_res(obs_arr, sim_arr)
    # # area of relative remaing bias
    # b_area = de.calc_bias_area(brel_res)
    # df_es.loc["e", "b_area"] = b_area
    # # temporal correlation
    # temp_cor = de.calc_temp_cor(obs_arr, sim_arr)
    # df_es.loc["e", "temp_cor"] = temp_cor
    # # diagnostic efficiency
    # df_es.loc["e", "de"] = de.calc_de(obs_arr, sim_arr)
    # # relative bias
    # brel = de.calc_brel(obs_arr, sim_arr)
    # # total bias
    # b_tot = de.calc_bias_tot(brel)
    # df_es.loc["e", "b_tot"] = b_tot
    # # bias of high flows
    # b_hf = de.calc_bias_hf(brel)
    # df_es.loc["e", "b_hf"] = b_hf
    # # contribution of high flow errors
    # err_hf = de.calc_err_hf(b_hf, b_tot)
    # df_es.loc["e", "err_hf"] = err_hf
    # # bias of low flows
    # b_lf = de.calc_bias_lf(brel)
    # df_es.loc["e", "b_lf"] = b_lf
    # # contribution of low flow errors
    # err_lf = de.calc_err_lf(b_lf, b_tot)
    # df_es.loc["e", "err_lf"] = err_lf
    # # direction of bias
    # b_dir = de.calc_bias_dir(brel_res)
    # df_es.loc["e", "b_dir"] = b_dir
    # # slope of bias
    # b_slope = de.calc_bias_slope(b_area, b_dir)
    # df_es.loc["e", "b_slope"] = b_slope
    # # convert to radians
    # # (y, x) Trigonometric inverse tangent
    # df_es.loc["e", "phi"] = de.calc_phi(brel_mean, b_slope)

    # # KGE beta
    # df_es.loc["e", "beta"] = kge.calc_kge_beta(obs_arr, sim_arr)
    # # KGE alpha
    # df_es.loc["e", "alpha"] = kge.calc_kge_alpha(obs_arr, sim_arr)
    # # # KGE gamma
    # # df_es.iloc[0, 8] = kge.calc_kge_gamma(obs_arr, sim_arr)
    # # KGE
    # df_es.loc["e", "kge"] = kge.calc_kge(obs_arr, sim_arr)

    # # NSE
    # df_es.loc["e", "nse"] = nse.calc_nse(obs_arr, sim_arr)

    # # ----------------------------------------------------------
    # # negative dynamic error and negative constant error
    # # ----------------------------------------------------------
    # obs_sim = pd.DataFrame(index=df_ts.index, columns=["Qobs", "Qsim"])
    # obs_sim.loc[:, "Qobs"] = df_ts.loc[:, "Qobs"]
    # # generate negative dynamic error
    # tsd = generate_errors.negative_dynamic(df_ts.copy(), prop=0.5)
    # # generate positive constant error
    # tso = generate_errors.constant(obs_sim.iloc[:, 0].values, offset=0.25)
    # obs_sim.loc[:, "Qsim"] = tsd.iloc[:, 0].values - tso  # negative offset
    # util.fdc_obs_sim_ax(
    #     obs_sim["Qobs"], obs_sim["Qsim"], axes_fdc[1, 0], fig_num_fdc[5]
    # )
    # util.plot_obs_sim_ax(obs_sim["Qobs"], obs_sim["Qsim"], axes_ts[1, 0],
    #                       fig_num_ts[5])

    # obs_sim_yy = obs_sim.loc[yy_str_1:yy_str_2, :]
    # util.plot_obs_sim_ax(obs_sim_yy["Qobs"], obs_sim_yy["Qsim"],
    #                       axes_ts_yy[1, 0], fig_num_ts[5])

    # # make arrays
    # obs_arr = obs_sim["Qobs"].values
    # sim_arr = obs_sim["Qsim"].values

    # # mean relative bias
    # brel_mean = de.calc_brel_mean(obs_arr, sim_arr)
    # df_es.loc["f", "brel_mean"] = brel_mean
    # # residual relative bias
    # brel_res = de.calc_brel_res(obs_arr, sim_arr)
    # # area of relative remaing bias
    # b_area = de.calc_bias_area(brel_res)
    # df_es.loc["f", "b_area"] = b_area
    # # temporal correlation
    # temp_cor = de.calc_temp_cor(obs_arr, sim_arr)
    # df_es.loc["f", "temp_cor"] = temp_cor
    # # diagnostic efficiency
    # df_es.loc["f", "de"] = de.calc_de(obs_arr, sim_arr)
    # # relative bias
    # brel = de.calc_brel(obs_arr, sim_arr)
    # # total bias
    # b_tot = de.calc_bias_tot(brel)
    # df_es.loc["f", "b_tot"] = b_tot
    # # bias of high flows
    # b_hf = de.calc_bias_hf(brel)
    # df_es.loc["f", "b_hf"] = b_hf
    # # contribution of high flow errors
    # err_hf = de.calc_err_hf(b_hf, b_tot)
    # df_es.loc["f", "err_hf"] = err_hf
    # # bias of low flows
    # b_lf = de.calc_bias_lf(brel)
    # df_es.loc["f", "b_lf"] = b_lf
    # # contribution of low flow errors
    # err_lf = de.calc_err_lf(b_lf, b_tot)
    # df_es.loc["f", "err_lf"] = err_lf
    # # direction of bias
    # b_dir = de.calc_bias_dir(brel_res)
    # df_es.loc["f", "b_dir"] = b_dir
    # # slope of bias
    # b_slope = de.calc_bias_slope(b_area, b_dir)
    # df_es.loc["f", "b_slope"] = b_slope
    # # convert to radians
    # # (y, x) Trigonometric inverse tangent
    # df_es.loc["f", "phi"] = de.calc_phi(brel_mean, b_slope)

    # # KGE beta
    # df_es.loc["f", "beta"] = kge.calc_kge_beta(obs_arr, sim_arr)
    # # KGE alpha
    # df_es.loc["f", "alpha"] = kge.calc_kge_alpha(obs_arr, sim_arr)
    # # # KGE gamma
    # # df_es.iloc[0, 8] = kge.calc_kge_gamma(obs_arr, sim_arr)
    # # KGE
    # df_es.loc["f", "kge"] = kge.calc_kge(obs_arr, sim_arr)

    # # NSE
    # df_es.loc["f", "nse"] = nse.calc_nse(obs_arr, sim_arr)

    # # ----------------------------------------------------------
    # # negative dynamic error and positive constant error
    # # ----------------------------------------------------------
    # obs_sim = pd.DataFrame(index=df_ts.index, columns=["Qobs", "Qsim"])
    # obs_sim.loc[:, "Qobs"] = df_ts.loc[:, "Qobs"]
    # # generate negative dynamic error
    # tsd = generate_errors.negative_dynamic(df_ts.copy(), prop=0.5)
    # # generate positive constant error
    # tso = generate_errors.constant(obs_sim.iloc[:, 0].values, offset=0.25)
    # obs_sim.loc[:, "Qsim"] = tsd.iloc[:, 0].values + tso  # positive offset
    # util.fdc_obs_sim_ax(
    #     obs_sim["Qobs"], obs_sim["Qsim"], axes_fdc[1, 1], fig_num_fdc[6]
    # )
    # util.plot_obs_sim_ax(obs_sim["Qobs"], obs_sim["Qsim"], axes_ts[1, 1],
    #                       fig_num_ts[6])

    # obs_sim_yy = obs_sim.loc[yy_str_1:yy_str_2, :]
    # util.plot_obs_sim_ax(obs_sim_yy["Qobs"], obs_sim_yy["Qsim"],
    #                       axes_ts_yy[1, 1], fig_num_ts[6])

    # # make arrays
    # obs_arr = obs_sim["Qobs"].values
    # sim_arr = obs_sim["Qsim"].values

    # # mean relative bias
    # brel_mean = de.calc_brel_mean(obs_arr, sim_arr)
    # df_es.loc["g", "brel_mean"] = brel_mean
    # # residual relative bias
    # brel_res = de.calc_brel_res(obs_arr, sim_arr)
    # # area of relative remaing bias
    # b_area = de.calc_bias_area(brel_res)
    # df_es.loc["g", "b_area"] = b_area
    # # temporal correlation
    # temp_cor = de.calc_temp_cor(obs_arr, sim_arr)
    # df_es.loc["g", "temp_cor"] = temp_cor
    # # diagnostic efficiency
    # df_es.loc["g", "de"] = de.calc_de(obs_arr, sim_arr)
    # # relative bias
    # brel = de.calc_brel(obs_arr, sim_arr)
    # # total bias
    # b_tot = de.calc_bias_tot(brel)
    # df_es.loc["g", "b_tot"] = b_tot
    # # bias of high flows
    # b_hf = de.calc_bias_hf(brel)
    # df_es.loc["g", "b_hf"] = b_hf
    # # contribution of high flow errors
    # err_hf = de.calc_err_hf(b_hf, b_tot)
    # df_es.loc["g", "err_hf"] = err_hf
    # # bias of low flows
    # b_lf = de.calc_bias_lf(brel)
    # df_es.loc["g", "b_lf"] = b_lf
    # # contribution of low flow errors
    # err_lf = de.calc_err_lf(b_lf, b_tot)
    # df_es.loc["g", "err_lf"] = err_lf
    # # direction of bias
    # b_dir = de.calc_bias_dir(brel_res)
    # df_es.loc["g", "b_dir"] = b_dir
    # # slope of bias
    # b_slope = de.calc_bias_slope(b_area, b_dir)
    # df_es.loc["g", "b_slope"] = b_slope
    # # convert to radians
    # # (y, x) Trigonometric inverse tangent
    # df_es.loc["g", "phi"] = de.calc_phi(brel_mean, b_slope)

    # # KGE beta
    # df_es.loc["g", "beta"] = kge.calc_kge_beta(obs_arr, sim_arr)
    # # KGE alpha
    # df_es.loc["g", "alpha"] = kge.calc_kge_alpha(obs_arr, sim_arr)
    # # # KGE gamma
    # # df_es.iloc[0, 8] = kge.calc_kge_gamma(obs_arr, sim_arr)
    # # KGE
    # df_es.loc["g", "kge"] = kge.calc_kge(obs_arr, sim_arr)

    # # NSE
    # df_es.loc["g", "nse"] = nse.calc_nse(obs_arr, sim_arr)

    # # ----------------------------------------------------------
    # # postive dynamic error and negative constant error
    # # ----------------------------------------------------------
    # obs_sim = pd.DataFrame(index=df_ts.index, columns=["Qobs", "Qsim"])
    # obs_sim.loc[:, "Qobs"] = df_ts.loc[:, "Qobs"]
    # # generate positve dynamic error
    # tsd = generate_errors.positive_dynamic(df_ts.copy(), prop=0.5)
    # # generate negative constant error
    # tso = generate_errors.constant(obs_sim.iloc[:, 0].values, offset=0.25)
    # obs_sim.loc[:, "Qsim"] = tsd.iloc[:, 0].values - tso  # negative offset
    # util.fdc_obs_sim_ax(
    #     obs_sim["Qobs"], obs_sim["Qsim"], axes_fdc[1, 2], fig_num_fdc[7]
    # )
    # util.plot_obs_sim_ax(obs_sim["Qobs"], obs_sim["Qsim"], axes_ts[1, 2],
    #                       fig_num_ts[7])

    # obs_sim_yy = obs_sim.loc[yy_str_1:yy_str_2, :]
    # util.plot_obs_sim_ax(obs_sim_yy["Qobs"], obs_sim_yy["Qsim"],
    #                       axes_ts_yy[1, 2], fig_num_ts[7])

    # # make arrays
    # obs_arr = obs_sim["Qobs"].values
    # sim_arr = obs_sim["Qsim"].values

    # # mean relative bias
    # brel_mean = de.calc_brel_mean(obs_arr, sim_arr)
    # df_es.loc["h", "brel_mean"] = brel_mean
    # # residual relative bias
    # brel_res = de.calc_brel_res(obs_arr, sim_arr)
    # # area of relative remaing bias
    # b_area = de.calc_bias_area(brel_res)
    # df_es.loc["h", "b_area"] = b_area
    # # temporal correlation
    # temp_cor = de.calc_temp_cor(obs_arr, sim_arr)
    # df_es.loc["h", "temp_cor"] = temp_cor
    # # diagnostic efficiency
    # df_es.loc["h", "de"] = de.calc_de(obs_arr, sim_arr)
    # # relative bias
    # brel = de.calc_brel(obs_arr, sim_arr)
    # # total bias
    # b_tot = de.calc_bias_tot(brel)
    # df_es.loc["h", "b_tot"] = b_tot
    # # bias of high flows
    # b_hf = de.calc_bias_hf(brel)
    # df_es.loc["h", "b_hf"] = b_hf
    # # contribution of high flow errors
    # err_hf = de.calc_err_hf(b_hf, b_tot)
    # df_es.loc["h", "err_hf"] = err_hf
    # # bias of low flows
    # b_lf = de.calc_bias_lf(brel)
    # df_es.loc["h", "b_lf"] = b_lf
    # # contribution of low flow errors
    # err_lf = de.calc_err_lf(b_lf, b_tot)
    # df_es.loc["h", "err_lf"] = err_lf
    # # direction of bias
    # b_dir = de.calc_bias_dir(brel_res)
    # df_es.loc["h", "b_dir"] = b_dir
    # # slope of bias
    # b_slope = de.calc_bias_slope(b_area, b_dir)
    # df_es.loc["h", "b_slope"] = b_slope
    # # convert to radians
    # # (y, x) Trigonometric inverse tangent
    # df_es.loc["h", "phi"] = de.calc_phi(brel_mean, b_slope)

    # # KGE beta
    # df_es.loc["h", "beta"] = kge.calc_kge_beta(obs_arr, sim_arr)
    # # KGE alpha
    # df_es.loc["h", "alpha"] = kge.calc_kge_alpha(obs_arr, sim_arr)
    # # # KGE gamma
    # # df_es.iloc[0, 8] = kge.calc_kge_gamma(obs_arr, sim_arr)
    # # KGE
    # df_es.loc["h", "kge"] = kge.calc_kge(obs_arr, sim_arr)

    # # NSE
    # df_es.loc["h", "nse"] = nse.calc_nse(obs_arr, sim_arr)

    # # ----------------------------------------------------------
    # # positive dynamic error and positive constant error
    # # ----------------------------------------------------------
    # obs_sim = pd.DataFrame(index=df_ts.index, columns=["Qobs", "Qsim"])
    # obs_sim.loc[:, "Qobs"] = df_ts.loc[:, "Qobs"]
    # # generate positive dynamic error
    # tsd = generate_errors.positive_dynamic(df_ts.copy(), prop=0.5)
    # # generate positive constant error
    # tso = generate_errors.constant(obs_sim.iloc[:, 0].values, offset=0.25)
    # obs_sim.loc[:, "Qsim"] = tsd.iloc[:, 0].values + tso  # positive offset
    # util.fdc_obs_sim_ax(
    #     obs_sim["Qobs"], obs_sim["Qsim"], axes_fdc[1, 3], fig_num_fdc[8]
    # )
    # util.plot_obs_sim_ax(obs_sim["Qobs"], obs_sim["Qsim"], axes_ts[1, 3],
    #                       fig_num_ts[8])

    # obs_sim_yy = obs_sim.loc[yy_str_1:yy_str_2, :]
    # util.plot_obs_sim_ax(obs_sim_yy["Qobs"], obs_sim_yy["Qsim"],
    #                       axes_ts_yy[1, 3], fig_num_ts[8])

    # # make arrays
    # obs_arr = obs_sim["Qobs"].values
    # sim_arr = obs_sim["Qsim"].values

    # # mean relative bias
    # brel_mean = de.calc_brel_mean(obs_arr, sim_arr)
    # df_es.loc["i", "brel_mean"] = brel_mean
    # # residual relative bias
    # brel_res = de.calc_brel_res(obs_arr, sim_arr)
    # # area of relative remaing bias
    # b_area = de.calc_bias_area(brel_res)
    # df_es.loc["i", "b_area"] = b_area
    # # temporal correlation
    # temp_cor = de.calc_temp_cor(obs_arr, sim_arr)
    # df_es.loc["i", "temp_cor"] = temp_cor
    # # diagnostic efficiency
    # df_es.loc["i", "de"] = de.calc_de(obs_arr, sim_arr)
    # # relative bias
    # brel = de.calc_brel(obs_arr, sim_arr)
    # # total bias
    # b_tot = de.calc_bias_tot(brel)
    # df_es.loc["i", "b_tot"] = b_tot
    # # bias of high flows
    # b_hf = de.calc_bias_hf(brel)
    # df_es.loc["i", "b_hf"] = b_hf
    # # contribution of high flow errors
    # err_hf = de.calc_err_hf(b_hf, b_tot)
    # df_es.loc["i", "err_hf"] = err_hf
    # # bias of low flows
    # b_lf = de.calc_bias_lf(brel)
    # df_es.loc["i", "b_lf"] = b_lf
    # # contribution of low flow errors
    # err_lf = de.calc_err_lf(b_lf, b_tot)
    # df_es.loc["i", "err_lf"] = err_lf
    # # direction of bias
    # b_dir = de.calc_bias_dir(brel_res)
    # df_es.loc["i", "b_dir"] = b_dir
    # # slope of bias
    # b_slope = de.calc_bias_slope(b_area, b_dir)
    # df_es.loc["i", "b_slope"] = b_slope
    # # convert to radians
    # # (y, x) Trigonometric inverse tangent
    # df_es.loc["i", "phi"] = de.calc_phi(brel_mean, b_slope)

    # # KGE beta
    # df_es.loc["i", "beta"] = kge.calc_kge_beta(obs_arr, sim_arr)
    # # KGE alpha
    # df_es.loc["i", "alpha"] = kge.calc_kge_alpha(obs_arr, sim_arr)
    # # # KGE gamma
    # # df_es.iloc[0, 8] = kge.calc_kge_gamma(obs_arr, sim_arr)
    # # KGE
    # df_es.loc["i", "kge"] = kge.calc_kge(obs_arr, sim_arr)

    # # NSE
    # df_es.loc["i", "nse"] = nse.calc_nse(obs_arr, sim_arr)

    # # -----------------------------------------------------------------
    # # negative dynamic error, negative constant error and timing error
    # # -----------------------------------------------------------------
    # obs_sim = pd.DataFrame(index=df_ts.index, columns=["Qobs", "Qsim"])
    # obs_sim.loc[:, "Qobs"] = df_ts.loc[:, "Qobs"]
    # # generate negative dynamic error
    # tsd = generate_errors.negative_dynamic(df_ts.copy(), prop=0.5)
    # tsn = pd.DataFrame(index=df_ts.index, columns=["Qsim"])
    # # generate negative constant error
    # tso = generate_errors.constant(obs_sim.iloc[:, 0].values, offset=0.25)
    # tsn.iloc[:, 0] = tsd.iloc[:, 0].values - tso  # negative offset
    # tst = generate_errors.timing(tsn, shuffle=True)  # shuffling
    # obs_sim.loc[:, "Qsim"] = tst.iloc[:, 0].values
    # util.plot_obs_sim_ax(obs_sim["Qobs"], obs_sim["Qsim"], axes_ts[1, 4],
    #                       fig_num_ts[9])

    # obs_sim_yy = obs_sim.loc[yy_str_1:yy_str_2, :]
    # util.plot_obs_sim_ax(obs_sim_yy["Qobs"], obs_sim_yy["Qsim"],
    #                       axes_ts_yy[1, 4], fig_num_ts[9])

    # # make arrays
    # obs_arr = obs_sim["Qobs"].values
    # sim_arr = obs_sim["Qsim"].values

    # # mean relative bias
    # brel_mean = de.calc_brel_mean(obs_arr, sim_arr)
    # df_es.loc["j", "brel_mean"] = brel_mean
    # # residual relative bias
    # brel_res = de.calc_brel_res(obs_arr, sim_arr)
    # # area of relative remaing bias
    # b_area = de.calc_bias_area(brel_res)
    # df_es.loc["j", "b_area"] = b_area
    # # temporal correlation
    # temp_cor = de.calc_temp_cor(obs_arr, sim_arr)
    # df_es.loc["j", "temp_cor"] = temp_cor
    # # diagnostic efficiency
    # df_es.loc["j", "de"] = de.calc_de(obs_arr, sim_arr)
    # # relative bias
    # brel = de.calc_brel(obs_arr, sim_arr)
    # # total bias
    # b_tot = de.calc_bias_tot(brel)
    # df_es.loc["j", "b_tot"] = b_tot
    # # bias of high flows
    # b_hf = de.calc_bias_hf(brel)
    # df_es.loc["j", "b_hf"] = b_hf
    # # contribution of high flow errors
    # err_hf = de.calc_err_hf(b_hf, b_tot)
    # df_es.loc["j", "err_hf"] = err_hf
    # # bias of low flows
    # b_lf = de.calc_bias_lf(brel)
    # df_es.loc["j", "b_lf"] = b_lf
    # # contribution of low flow errors
    # err_lf = de.calc_err_lf(b_lf, b_tot)
    # df_es.loc["j", "err_lf"] = err_lf
    # # direction of bias
    # b_dir = de.calc_bias_dir(brel_res)
    # df_es.loc["j", "b_dir"] = b_dir
    # # slope of bias
    # b_slope = de.calc_bias_slope(b_area, b_dir)
    # df_es.loc["j", "b_slope"] = b_slope
    # # convert to radians
    # # (y, x) Trigonometric inverse tangent
    # df_es.loc["j", "phi"] = de.calc_phi(brel_mean, b_slope)

    # # KGE beta
    # df_es.loc["j", "beta"] = kge.calc_kge_beta(obs_arr, sim_arr)
    # # KGE alpha
    # df_es.loc["j", "alpha"] = kge.calc_kge_alpha(obs_arr, sim_arr)
    # # # KGE gamma
    # # df_es.iloc[0, 8] = kge.calc_kge_gamma(obs_arr, sim_arr)
    # # KGE
    # df_es.loc["j", "kge"] = kge.calc_kge(obs_arr, sim_arr)

    # # NSE
    # df_es.loc["j", "nse"] = nse.calc_nse(obs_arr, sim_arr)

    # # -----------------------------------------------------------------
    # # negative dynamic error, positive constant error and timing error
    # # -----------------------------------------------------------------
    # obs_sim = pd.DataFrame(index=df_ts.index, columns=["Qobs", "Qsim"])
    # obs_sim.loc[:, "Qobs"] = df_ts.loc[:, "Qobs"]
    # # generate negative dynamic error
    # tsd = generate_errors.negative_dynamic(df_ts.copy(), prop=0.5)
    # tsn = pd.DataFrame(index=df_ts.index, columns=["Qsim"])
    # # generate positive constant error
    # tso = generate_errors.constant(obs_sim.iloc[:, 0].values, offset=0.25)
    # tsn.iloc[:, 0] = tsd.iloc[:, 0].values + tso  # negative offset
    # tst = generate_errors.timing(tsn, shuffle=True)  # shuffling
    # obs_sim.loc[:, "Qsim"] = tst.iloc[:, 0].values
    # util.plot_obs_sim_ax(
    #     obs_sim["Qobs"], obs_sim["Qsim"], axes_ts[2, 0], fig_num_ts[10]
    # )

    # obs_sim_yy = obs_sim.loc[yy_str_1:yy_str_2, :]
    # util.plot_obs_sim_ax(obs_sim_yy["Qobs"], obs_sim_yy["Qsim"],
    #                       axes_ts_yy[2, 0], fig_num_ts[10])

    # # make arrays
    # obs_arr = obs_sim["Qobs"].values
    # sim_arr = obs_sim["Qsim"].values

    # # mean relative bias
    # brel_mean = de.calc_brel_mean(obs_arr, sim_arr)
    # df_es.loc["k", "brel_mean"] = brel_mean
    # # residual relative bias
    # brel_res = de.calc_brel_res(obs_arr, sim_arr)
    # # area of relative remaing bias
    # b_area = de.calc_bias_area(brel_res)
    # df_es.loc["k", "b_area"] = b_area
    # # temporal correlation
    # temp_cor = de.calc_temp_cor(obs_arr, sim_arr)
    # df_es.loc["k", "temp_cor"] = temp_cor
    # # diagnostic efficiency
    # df_es.loc["k", "de"] = de.calc_de(obs_arr, sim_arr)
    # # relative bias
    # brel = de.calc_brel(obs_arr, sim_arr)
    # # total bias
    # b_tot = de.calc_bias_tot(brel)
    # df_es.loc["k", "b_tot"] = b_tot
    # # bias of high flows
    # b_hf = de.calc_bias_hf(brel)
    # df_es.loc["k", "b_hf"] = b_hf
    # # contribution of high flow errors
    # err_hf = de.calc_err_hf(b_hf, b_tot)
    # df_es.loc["k", "err_hf"] = err_hf
    # # bias of low flows
    # b_lf = de.calc_bias_lf(brel)
    # df_es.loc["k", "b_lf"] = b_lf
    # # contribution of low flow errors
    # err_lf = de.calc_err_lf(b_lf, b_tot)
    # df_es.loc["k", "err_lf"] = err_lf
    # # direction of bias
    # b_dir = de.calc_bias_dir(brel_res)
    # df_es.loc["k", "b_dir"] = b_dir
    # # slope of bias
    # b_slope = de.calc_bias_slope(b_area, b_dir)
    # df_es.loc["k", "b_slope"] = b_slope
    # # convert to radians
    # # (y, x) Trigonometric inverse tangent
    # df_es.loc["k", "phi"] = de.calc_phi(brel_mean, b_slope)

    # # KGE beta
    # df_es.loc["k", "beta"] = kge.calc_kge_beta(obs_arr, sim_arr)
    # # KGE alpha
    # df_es.loc["k", "alpha"] = kge.calc_kge_alpha(obs_arr, sim_arr)
    # # # KGE gamma
    # # df_es.iloc[0, 8] = kge.calc_kge_gamma(obs_arr, sim_arr)
    # # KGE
    # df_es.loc["k", "kge"] = kge.calc_kge(obs_arr, sim_arr)

    # # NSE
    # df_es.loc["k", "nse"] = nse.calc_nse(obs_arr, sim_arr)

    # # -----------------------------------------------------------------
    # # positive dynamic error, negative constant error and timing error
    # # -----------------------------------------------------------------
    # obs_sim = pd.DataFrame(index=df_ts.index, columns=["Qobs", "Qsim"])
    # obs_sim.loc[:, "Qobs"] = df_ts.loc[:, "Qobs"]
    # # generate positive dynamic error
    # tsd = generate_errors.positive_dynamic(df_ts.copy(), prop=0.5)
    # tsp = pd.DataFrame(index=df_ts.index, columns=["Qsim"])
    # # generate positive constant offset
    # tso = generate_errors.constant(obs_sim.iloc[:, 0].values, offset=0.25)
    # tsp.iloc[:, 0] = tsd.iloc[:, 0].values - tso  # negative offset
    # tst = generate_errors.timing(tsp, shuffle=True)  # shuffling
    # obs_sim.loc[:, "Qsim"] = tst.iloc[:, 0].values
    # util.plot_obs_sim_ax(
    #     obs_sim["Qobs"], obs_sim["Qsim"], axes_ts[2, 1], fig_num_ts[11]
    # )

    # obs_sim_yy = obs_sim.loc[yy_str_1:yy_str_2, :]
    # util.plot_obs_sim_ax(obs_sim_yy["Qobs"], obs_sim_yy["Qsim"],
    #                       axes_ts_yy[2, 1], fig_num_ts[11])

    # # make arrays
    # obs_arr = obs_sim["Qobs"].values
    # sim_arr = obs_sim["Qsim"].values

    # # mean relative bias
    # brel_mean = de.calc_brel_mean(obs_arr, sim_arr)
    # df_es.loc["l", "brel_mean"] = brel_mean
    # # residual relative bias
    # brel_res = de.calc_brel_res(obs_arr, sim_arr)
    # # area of relative remaing bias
    # b_area = de.calc_bias_area(brel_res)
    # df_es.loc["l", "b_area"] = b_area
    # # temporal correlation
    # temp_cor = de.calc_temp_cor(obs_arr, sim_arr)
    # df_es.loc["l", "temp_cor"] = temp_cor
    # # diagnostic efficiency
    # df_es.loc["l", "de"] = de.calc_de(obs_arr, sim_arr)
    # # relative bias
    # brel = de.calc_brel(obs_arr, sim_arr)
    # # total bias
    # b_tot = de.calc_bias_tot(brel)
    # df_es.loc["l", "b_tot"] = b_tot
    # # bias of high flows
    # b_hf = de.calc_bias_hf(brel)
    # df_es.loc["l", "b_hf"] = b_hf
    # # contribution of high flow errors
    # err_hf = de.calc_err_hf(b_hf, b_tot)
    # df_es.loc["l", "err_hf"] = err_hf
    # # bias of low flows
    # b_lf = de.calc_bias_lf(brel)
    # df_es.loc["l", "b_lf"] = b_lf
    # # contribution of low flow errors
    # err_lf = de.calc_err_lf(b_lf, b_tot)
    # df_es.loc["l", "err_lf"] = err_lf
    # # direction of bias
    # b_dir = de.calc_bias_dir(brel_res)
    # df_es.loc["l", "b_dir"] = b_dir
    # # slope of bias
    # b_slope = de.calc_bias_slope(b_area, b_dir)
    # df_es.loc["l", "b_slope"] = b_slope
    # # convert to radians
    # # (y, x) Trigonometric inverse tangent
    # df_es.loc["l", "phi"] = de.calc_phi(brel_mean, b_slope)

    # # KGE beta
    # df_es.loc["l", "beta"] = kge.calc_kge_beta(obs_arr, sim_arr)
    # # KGE alpha
    # df_es.loc["l", "alpha"] = kge.calc_kge_alpha(obs_arr, sim_arr)
    # # # KGE gamma
    # # df_es.iloc[0, 8] = kge.calc_kge_gamma(obs_arr, sim_arr)
    # # KGE
    # df_es.loc["l", "kge"] = kge.calc_kge(obs_arr, sim_arr)

    # # NSE
    # df_es.loc["l", "nse"] = nse.calc_nse(obs_arr, sim_arr)

    # # -----------------------------------------------------------------
    # # positive dynamic error, positive constant error and timing error
    # # -----------------------------------------------------------------
    # obs_sim = pd.DataFrame(index=df_ts.index, columns=["Qobs", "Qsim"])
    # obs_sim.loc[:, "Qobs"] = df_ts.loc[:, "Qobs"]
    # # generate positive dynamic error
    # tsd = generate_errors.positive_dynamic(df_ts.copy(), prop=0.5)
    # tsp = pd.DataFrame(index=df_ts.index, columns=["Qsim"])
    # # generate positve constant offset
    # tso = generate_errors.constant(obs_sim.iloc[:, 0].values, offset=0.25)
    # tsp.iloc[:, 0] = tsd.iloc[:, 0].values + tso  # positve offset
    # tst = generate_errors.timing(tsp, shuffle=True)  # shuffling
    # obs_sim.loc[:, "Qsim"] = tst.iloc[:, 0].values
    # util.plot_obs_sim_ax(
    #     obs_sim["Qobs"], obs_sim["Qsim"], axes_ts[2, 2], fig_num_ts[12]
    # )

    # obs_sim_yy = obs_sim.loc[yy_str_1:yy_str_2, :]
    # util.plot_obs_sim_ax(obs_sim_yy["Qobs"], obs_sim_yy["Qsim"],
    #                       axes_ts_yy[2, 2], fig_num_ts[12])

    # # make arrays
    # obs_arr = obs_sim["Qobs"].values
    # sim_arr = obs_sim["Qsim"].values

    # # mean relative bias
    # brel_mean = de.calc_brel_mean(obs_arr, sim_arr)
    # df_es.loc["m", "brel_mean"] = brel_mean
    # # residual relative bias
    # brel_res = de.calc_brel_res(obs_arr, sim_arr)
    # # area of relative remaing bias
    # b_area = de.calc_bias_area(brel_res)
    # df_es.loc["m", "b_area"] = b_area
    # # temporal correlation
    # temp_cor = de.calc_temp_cor(obs_arr, sim_arr)
    # df_es.loc["m", "temp_cor"] = temp_cor
    # # diagnostic efficiency
    # df_es.loc["m", "de"] = de.calc_de(obs_arr, sim_arr)
    # # relative bias
    # brel = de.calc_brel(obs_arr, sim_arr)
    # # total bias
    # b_tot = de.calc_bias_tot(brel)
    # df_es.loc["m", "b_tot"] = b_tot
    # # bias of high flows
    # b_hf = de.calc_bias_hf(brel)
    # df_es.loc["m", "b_hf"] = b_hf
    # # contribution of high flow errors
    # err_hf = de.calc_err_hf(b_hf, b_tot)
    # df_es.loc["m", "err_hf"] = err_hf
    # # bias of low flows
    # b_lf = de.calc_bias_lf(brel)
    # df_es.loc["m", "b_lf"] = b_lf
    # # contribution of low flow errors
    # err_lf = de.calc_err_lf(b_lf, b_tot)
    # df_es.loc["m", "err_lf"] = err_lf
    # # direction of bias
    # b_dir = de.calc_bias_dir(brel_res)
    # df_es.loc["m", "b_dir"] = b_dir
    # # slope of bias
    # b_slope = de.calc_bias_slope(b_area, b_dir)
    # df_es.loc["m", "b_slope"] = b_slope
    # # convert to radians
    # # (y, x) Trigonometric inverse tangent
    # df_es.loc["m", "phi"] = de.calc_phi(brel_mean, b_slope)

    # # KGE beta
    # df_es.loc["m", "beta"] = kge.calc_kge_beta(obs_arr, sim_arr)
    # # KGE alpha
    # df_es.loc["m", "alpha"] = kge.calc_kge_alpha(obs_arr, sim_arr)
    # # # KGE gamma
    # # df_es.iloc[0, 8] = kge.calc_kge_gamma(obs_arr, sim_arr)
    # # KGE
    # df_es.loc["m", "kge"] = kge.calc_kge(obs_arr, sim_arr)

    # # NSE
    # df_es.loc["m", "nse"] = nse.calc_nse(obs_arr, sim_arr)

    # axes_fdc[1, 3].legend(loc=6, bbox_to_anchor=(1.18, 0.85))
    # axes_ts[2, 2].legend(loc=6, bbox_to_anchor=(1.18, 0.85))

    # path_png = Path(os.path.join(PATH_FIG, "fdc_errors.png"))
    # fig_fdc.savefig(path_png, dpi=250)
    # path_pdf = Path(os.path.join(PATH_FIG, "fdc_errors.pdf"))
    # fig_fdc.savefig(path_pdf, dpi=250)
    # path_png = Path(os.path.join(PATH_FIG, "ts_errors.png"))
    # fig_ts.savefig(path_png, dpi=250)
    # path_pdf = Path(os.path.join(PATH_FIG, "ts_errors.pdf"))
    # fig_ts.savefig(path_pdf, dpi=250)
    # path_png = Path(os.path.join(PATH_FIG, "ts_errors_yy.png"))
    # fig_ts_yy.savefig(path_png, dpi=250)
    # path_pdf = Path(os.path.join(PATH_FIG, "ts_errors_yy.pdf"))
    # fig_ts_yy.savefig(path_pdf, dpi=250)

    # # -----------------------------------------------------------------
    # # diagnostic plolar plot
    # # -----------------------------------------------------------------
    # ### multi DE plot ###
    # # make arrays
    # idx = ["0", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m"]
    # ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    # brel_mean_arr = df_es["brel_mean"].values[ids]
    # temp_cor_arr = df_es["temp_cor"].values[ids]
    # b_dir_arr = df_es["b_dir"].values[ids]
    # de_arr = df_es["de"].values[ids]
    # phi_arr = df_es["phi"].values[ids]
    # b_slope_arr = df_es["b_slope"].values[ids]
    # b_hf_arr = df_es["b_hf"].values[ids]
    # b_lf_arr = df_es["b_lf"].values[ids]
    # b_tot_arr = df_es["b_tot"].values[ids]
    # err_hf_arr = df_es["err_hf"].values[ids]
    # err_lf_arr = df_es["err_lf"].values[ids]

    # fig_de = util.diag_polar_plot_multi_fc(
    #     brel_mean_arr, temp_cor_arr, de_arr, b_dir_arr, phi_arr,
    #     b_hf_arr, b_lf_arr, b_tot_arr, err_hf_arr, err_lf_arr, idx
    # )
    # path = Path(os.path.join(PATH_FIG, "de_diag.pdf"))
    # fig_de.savefig(path, dpi=250)

    # ### multi KGE plot ###
    # # make arrays
    # idx = ["0", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m"]
    # beta_arr = df_es["beta"].values
    # alpha_arr = df_es["alpha"].values
    # # gamma_arr = df_es['gamma'].values
    # temp_cor_arr = df_es["temp_cor"].values
    # kge_arr = df_es["kge"].values

    # fig_kge = util.polar_plot_multi_fc(
    #     beta_arr, alpha_arr, temp_cor_arr, kge_arr, idx
    # )
    # path = Path(os.path.join(PATH_FIG, "kge_diag.pdf"))
    # fig_kge.savefig(path, dpi=250)

    # ### scatterplots DE, KGE and NSE ###
    # nse_arr = df_es["nse"].values
    # brel_mean_arr = df_es["brel_mean"].values
    # b_slope_arr = df_es["b_slope"].values
    # de_arr = df_es["de"].values

    # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
    # # scatterplots DE, KGE and NSE
    # sc = sns.scatterplot(kge_arr, de_arr, color="black", s=60, ax=ax1)
    # sc1 = sns.scatterplot(nse_arr, de_arr, color="red", s=30, marker="X",
    #                       ax=ax1)
    # ax1.plot([-2.05, 1.55], [-2.05, 1.55], ls="--", c=".3")
    # ax1.set_ylim(-2.05, 1.55)
    # ax1.set_xlim(-2.05, 1.55)
    # ax1.set(ylabel="DE [-]", xlabel="KGE [-]")
    # ax1.text(0.40, -0.22, "NSE [-]", color="red", transform=ax1.transAxes)
    # ax1.text(0.025, 0.93, "(a)", transform=ax1.transAxes)
    # ax1.text(0.05, 0.1, "1:1", rotation=45, transform=ax1.transAxes)
    # # for i, txt in enumerate(df_es.index):
    # #     ax.annotate(txt, (kge_arr[i], de_arr[i]), color='black', fontsize=15)
    # #     ax.annotate(txt, (nse_arr[i], de_arr[i]), color='red', fontsize=15)

    # # scatterplot DE and KGE components inter-comparison
    # sc = sns.scatterplot(beta_arr - 1, brel_mean_arr, color="black", s=60,
    #                       ax=ax2)
    # sc1 = sns.scatterplot(
    #     alpha_arr - 1, b_slope_arr, color="red", s=30, marker="X", ax=ax2
    # )
    # ax2.plot([-2.05, 2.05], [-2.05, 2.05], ls="--", c=".3")
    # ax2.set_ylim(-2.05, 2.05)
    # ax2.set_xlim(-2.05, 2.05)
    # ax2.set(ylabel=r"$\overline{B_{rel}}$ [-]", xlabel=r"$\beta$ - 1 [-]")
    # ax2.text(
    #     -0.33,
    #     0.415,
    #     r"$B_{slope}$ [-]",
    #     color="red",
    #     transform=ax2.transAxes,
    #     rotation=90,
    # )
    # ax2.text(0.4, -0.22, r"$\alpha$ - 1 [-]", color="red",
    #           transform=ax2.transAxes)
    # ax2.text(0.03, 0.93, "(b)", transform=ax2.transAxes)
    # ax2.text(0.05, 0.1, "1:1", rotation=45, transform=ax2.transAxes)

    # # scatterplot DE and KGE components intra-comparison
    # sc = sns.scatterplot(b_slope_arr, brel_mean_arr, color="black", s=60,
    #                       ax=ax3)
    # sc1 = sns.scatterplot(
    #     alpha_arr - 1, beta_arr - 1, color="red", s=30, marker="X", ax=ax3
    # )
    # ax3.plot([-2.05, 2.05], [-2.05, 2.05], ls="--", c=".3")
    # ax3.set_ylim(-2.05, 2.05)
    # ax3.set_xlim(-2.05, 2.05)
    # ax3.set(ylabel=r"$B_{slope}$ [-]", xlabel=r"$\overline{B_{rel}}$ [-]")
    # ax3.text(
    #     -0.33,
    #     0.425,
    #     r"$\beta$ - 1 [-]",
    #     color="red",
    #     transform=ax3.transAxes,
    #     rotation=90,
    # )
    # ax3.text(0.4, -0.23, r"$\alpha$ - 1 [-]", color="red",
    #           transform=ax3.transAxes)
    # ax3.text(0.03, 0.93, "(c)", transform=ax3.transAxes)
    # ax3.text(0.05, 0.1, "1:1", rotation=45, transform=ax3.transAxes)

    # fig.subplots_adjust(wspace=0.45, bottom=0.2)
    # path_png = Path(os.path.join(PATH_FIG, "scatter_eff_comp.png"))
    # fig.savefig(path_png, dpi=250)
    # path_pdf = Path(os.path.join(PATH_FIG, "scatter_eff_comp.pdf"))
    # fig.savefig(path_pdf, dpi=250)
    # # for i, txt in enumerate(df_es.index):
    # #     ax.annotate(txt, (alpha_arr[i] - 1, brel_mean_arr[i]), color='black',
    # #                 fontsize=15)
    # #     ax.annotate(txt, (beta_arr[i] - 1, b_slope_arr[i]), color='red',
    # #                 fontsize=15)

    # # export table
    # df_es = df_es.round(2)
    # df_es_t = df_es.T
    # path_csv = Path(os.path.join(PATH_FIG, "table_eff_comp.csv"))
    # df_es_t = df_es_t.round(2)
    # df_es_t.to_csv(path_csv, header=True, index=True, sep=";")
    # df_es_t = df_es_t.loc[["de", "kge", "nse"], :]
    # df_es_t = df_es_t.round(2)
    # path_csv = Path(os.path.join(PATH_FIG, "table_eff.csv"))
    # df_es_t.to_csv(path_csv, header=True, index=True, sep=";")

    # # ==========================================================
    # # Modelling example - CAMELS
    # # ==========================================================
    # # dataframe DE, KGE and NSE
    # idx = ["05", "48", "94"]
    # cols = [
    #     "brel_mean",
    #     "b_area",
    #     "temp_cor",
    #     "de",
    #     "b_dir",
    #     "b_slope",
    #     "phi",
    #     "b_hf",
    #     "b_lf",
    #     "b_tot",
    #     "err_hf",
    #     "err_lf",
    #     "beta",
    #     "alpha",
    #     "kge",
    #     "nse",
    # ]
    # df_eff_cam = pd.DataFrame(index=idx, columns=cols, dtype=np.float64)

    # path_cam1 = Path(os.path.join(
    #     os.getcwd(), "examples/13331500_05_model_output.txt")
    # )
    # path_cam2 = Path(os.path.join(
    #     os.getcwd(), "examples/13331500_48_model_output.txt")
    # )
    # path_cam3 = Path(os.path.join(
    #     os.getcwd(), "examples/13331500_94_model_output.txt")
    # )
    # # entire time series
    # df_cam1 = util.import_camels_obs_sim(path_cam1)
    # df_cam2 = util.import_camels_obs_sim(path_cam2)
    # df_cam3 = util.import_camels_obs_sim(path_cam3)

    # fig, axes = plt.subplots(3, 2, figsize=(10, 10), sharex="col")
    # fig.text(0.06, 0.5, r"Q [mm $d^{-1}$]", ha="center", va="center",
    #           rotation="vertical")
    # fig.text(0.5, 0.5, r"Q [mm $d^{-1}$]", ha="center", va="center",
    #           rotation="vertical")
    # fig.text(0.25, 0.05, "Time [Years]", ha="center", va="center")
    # fig.text(0.75, 0.05, "Exceedence probabilty [-]", ha="center", va="center")
    # # format the ticks
    # years_10 = mdates.YearLocator(10)
    # years_5 = mdates.YearLocator(5)
    # yearsFmt = mdates.DateFormatter("%Y")

    # util.plot_obs_sim_ax(df_cam1["Qobs"], df_cam1["Qsim"], axes[0, 0], "")
    # axes[0, 0].text(
    #     0.95,
    #     0.95,
    #     "(a; set_id: {})".format(idx[0]),
    #     transform=axes[0, 0].transAxes,
    #     ha="right",
    #     va="top",
    # )
    # axes[0, 0].xaxis.set_major_locator(years_10)
    # axes[0, 0].xaxis.set_minor_locator(years_5)
    # axes[0, 0].xaxis.set_major_formatter(yearsFmt)

    # util.fdc_obs_sim_ax(df_cam1["Qobs"], df_cam1["Qsim"], axes[0, 1], "")
    # axes[0, 1].text(
    #     0.95,
    #     0.95,
    #     "(b; set_id: {})".format(idx[0]),
    #     transform=axes[0, 1].transAxes,
    #     ha="right",
    #     va="top",
    # )
    # # legend above plot
    # axes[0, 1].legend(
    #     loc=2,
    #     labels=["Observed", "Simulated"],
    #     ncol=2,
    #     frameon=False,
    #     bbox_to_anchor=(-0.6, 1.2),
    # )

    # util.plot_obs_sim_ax(df_cam2["Qobs"], df_cam2["Qsim"], axes[1, 0], "")
    # axes[1, 0].text(
    #     0.95,
    #     0.95,
    #     "(c; set_id: {})".format(idx[1]),
    #     transform=axes[1, 0].transAxes,
    #     ha="right",
    #     va="top",
    # )
    # axes[1, 0].xaxis.set_major_locator(years_10)
    # axes[1, 0].xaxis.set_minor_locator(years_5)
    # axes[1, 0].xaxis.set_major_formatter(yearsFmt)

    # util.fdc_obs_sim_ax(df_cam1["Qobs"], df_cam1["Qsim"], axes[1, 1], "")
    # axes[1, 1].text(
    #     0.95,
    #     0.95,
    #     "(d; set_id: {})".format(idx[1]),
    #     transform=axes[1, 1].transAxes,
    #     ha="right",
    #     va="top",
    # )

    # util.plot_obs_sim_ax(df_cam1["Qobs"], df_cam1["Qsim"], axes[2, 0], "")
    # axes[2, 0].text(
    #     0.95,
    #     0.95,
    #     "(e; set_id: {})".format(idx[2]),
    #     transform=axes[2, 0].transAxes,
    #     ha="right",
    #     va="top",
    # )
    # axes[2, 0].xaxis.set_major_locator(years_10)
    # axes[2, 0].xaxis.set_minor_locator(years_5)
    # axes[2, 0].xaxis.set_major_formatter(yearsFmt)

    # util.fdc_obs_sim_ax(df_cam1["Qobs"], df_cam1["Qsim"], axes[2, 1], "")
    # axes[2, 1].text(
    #     0.95,
    #     0.95,
    #     "(f; set_id: {})".format(idx[2]),
    #     transform=axes[2, 1].transAxes,
    #     ha="right",
    #     va="top",
    # )

    # fig.subplots_adjust(wspace=0.3)
    # path_png = Path(os.path.join(PATH_FIG, "ts_fdc_real_case.png"))
    # fig.savefig(path_png, dpi=250)
    # path_pdf = Path(os.path.join(PATH_FIG, "ts_fdc_real_case.pdf"))
    # fig.savefig(path_pdf, dpi=250)

    # # time series of a single year
    # df_cam1_yy = df_cam1.loc[yy_str_1:yy_str_2, :]
    # df_cam2_yy = df_cam2.loc[yy_str_1:yy_str_2, :]
    # df_cam3_yy = df_cam3.loc[yy_str_1:yy_str_2, :]

    # fig, axes = plt.subplots(3, 2, figsize=(10, 10), sharex="col")
    # fig.text(0.06, 0.5, r"Q [mm $d^{-1}$]", ha="center", va="center",
    #           rotation="vertical")
    # fig.text(0.5, 0.5, r"Q [mm $d^{-1}$]", ha="center", va="center",
    #           rotation="vertical")
    # fig.text(0.25, 0.05, "2000", ha="center", va="center")
    # fig.text(0.75, 0.05, "Exceedence probabilty [-]", ha="center", va="center")
    # # format the ticks
    # months_1 = mdates.MonthLocator()
    # days_15 = mdates.MonthLocator(bymonthday=15)
    # fmt = mdates.DateFormatter("%m")

    # util.plot_obs_sim_ax(df_cam1_yy["Qobs"], df_cam1_yy["Qsim"], axes[0, 0], "")
    # axes[0, 0].text(
    #     0.95,
    #     0.95,
    #     "(a; set_id: {})".format(idx[0]),
    #     transform=axes[0, 0].transAxes,
    #     ha="right",
    #     va="top",
    # )
    # axes[0, 0].xaxis.set_major_locator(months_1)
    # axes[0, 0].xaxis.set_minor_locator(days_15)
    # axes[0, 0].xaxis.set_major_formatter(fmt)

    # util.fdc_obs_sim_ax(df_cam1["Qobs"], df_cam1["Qsim"], axes[0, 1], "")
    # axes[0, 1].text(
    #     0.95,
    #     0.95,
    #     "(b; set_id: {})".format(idx[0]),
    #     transform=axes[0, 1].transAxes,
    #     ha="right",
    #     va="top",
    # )
    # # legend above plot
    # axes[0, 1].legend(
    #     loc=2,
    #     labels=["Observed", "Simulated"],
    #     ncol=2,
    #     frameon=False,
    #     bbox_to_anchor=(-0.6, 1.2),
    # )

    # util.plot_obs_sim_ax(df_cam2_yy["Qobs"], df_cam2_yy["Qsim"], axes[1, 0], "")
    # axes[1, 0].text(
    #     0.95,
    #     0.95,
    #     "(c; set_id: {})".format(idx[1]),
    #     transform=axes[1, 0].transAxes,
    #     ha="right",
    #     va="top",
    # )
    # axes[1, 0].xaxis.set_major_locator(months_1)
    # axes[1, 0].xaxis.set_minor_locator(days_15)
    # axes[1, 0].xaxis.set_major_formatter(fmt)

    # util.fdc_obs_sim_ax(df_cam1["Qobs"], df_cam1["Qsim"], axes[1, 1], "")
    # axes[1, 1].text(
    #     0.95,
    #     0.95,
    #     "(d; set_id: {})".format(idx[1]),
    #     transform=axes[1, 1].transAxes,
    #     ha="right",
    #     va="top",
    # )

    # util.plot_obs_sim_ax(df_cam1_yy["Qobs"], df_cam1_yy["Qsim"], axes[2, 0], "")
    # axes[2, 0].text(
    #     0.95,
    #     0.95,
    #     "(e; set_id: {})".format(idx[2]),
    #     transform=axes[2, 0].transAxes,
    #     ha="right",
    #     va="top",
    # )
    # axes[2, 0].xaxis.set_major_locator(months_1)
    # axes[2, 0].xaxis.set_minor_locator(days_15)
    # axes[2, 0].xaxis.set_major_formatter(fmt)

    # util.fdc_obs_sim_ax(df_cam1["Qobs"], df_cam1["Qsim"], axes[2, 1], "")
    # axes[2, 1].text(
    #     0.95,
    #     0.95,
    #     "(f; set_id: {})".format(idx[2]),
    #     transform=axes[2, 1].transAxes,
    #     ha="right",
    #     va="top",
    # )

    # fig.subplots_adjust(wspace=0.3)
    # path_png = Path(os.path.join(PATH_FIG, "ts_fdc_real_case_yy.png"))
    # fig.savefig(path_png, dpi=250)
    # path_pdf = Path(os.path.join(PATH_FIG, "ts_fdc_real_case_yy.pdf"))
    # fig.savefig(path_pdf, dpi=250)

    # # make arrays
    # obs_arr = df_cam1["Qobs"].values
    # sim_arr = df_cam1["Qsim"].values

    # # mean relative bias
    # brel_mean = de.calc_brel_mean(obs_arr, sim_arr)
    # df_eff_cam.loc["05", "brel_mean"] = brel_mean
    # # residual relative bias
    # brel_res = de.calc_brel_res(obs_arr, sim_arr)
    # # area of relative remaing bias
    # b_area = de.calc_bias_area(brel_res)
    # df_eff_cam.loc["05", "b_area"] = b_area
    # # temporal correlation
    # temp_cor = de.calc_temp_cor(obs_arr, sim_arr)
    # df_eff_cam.loc["05", "temp_cor"] = temp_cor
    # # diagnostic efficiency
    # df_eff_cam.loc["05", "de"] = de.calc_de(obs_arr, sim_arr)
    # # relative bias
    # brel = de.calc_brel(obs_arr, sim_arr)
    # # total bias
    # b_tot = de.calc_bias_tot(brel)
    # df_eff_cam.loc["05", "b_tot"] = de.calc_bias_tot(brel)
    # # bias of high flows
    # b_hf = de.calc_bias_hf(brel)
    # df_eff_cam.loc["05", "b_hf"] = b_hf
    # # contribution of high flow errors
    # err_hf = de.calc_err_hf(b_hf, b_tot)
    # df_eff_cam.loc["05", "err_hf"] = err_hf
    # # bias of low flows
    # b_lf = de.calc_bias_lf(brel)
    # df_eff_cam.loc["05", "b_lf"] = b_lf
    # # contribution of low flow errors
    # err_lf = de.calc_err_lf(b_lf, b_tot)
    # df_eff_cam.loc["05", "err_lf"] = err_lf
    # # direction of bias
    # b_dir = de.calc_bias_dir(brel_res)
    # df_eff_cam.loc["05", "b_dir"] = b_dir
    # # slope of bias
    # b_slope = de.calc_bias_slope(b_area, b_dir)
    # df_eff_cam.loc["05", "b_slope"] = b_slope
    # # convert to radians
    # # (y, x) Trigonometric inverse tangent
    # df_eff_cam.loc["05", "phi"] = de.calc_phi(brel_mean, b_slope)

    # # KGE beta
    # df_eff_cam.loc["05", "beta"] = kge.calc_kge_beta(obs_arr, sim_arr)
    # # KGE alpha
    # df_eff_cam.loc["05", "alpha"] = kge.calc_kge_alpha(obs_arr, sim_arr)
    # # KGE
    # df_eff_cam.loc["05", "kge"] = kge.calc_kge(obs_arr, sim_arr)

    # # NSE
    # df_eff_cam.loc["05", "nse"] = nse.calc_nse(obs_arr, sim_arr)

    # # make arrays
    # obs_arr = df_cam2["Qobs"].values
    # sim_arr = df_cam2["Qsim"].values

    # # mean relative bias
    # brel_mean = de.calc_brel_mean(obs_arr, sim_arr)
    # df_eff_cam.loc["48", "brel_mean"] = brel_mean
    # # residual relative bias
    # brel_res = de.calc_brel_res(obs_arr, sim_arr)
    # # area of relative remaing bias
    # b_area = de.calc_bias_area(brel_res)
    # df_eff_cam.loc["48", "b_area"] = b_area
    # # temporal correlation
    # temp_cor = de.calc_temp_cor(obs_arr, sim_arr)
    # df_eff_cam.loc["48", "temp_cor"] = temp_cor
    # # diagnostic efficiency
    # df_eff_cam.loc["48", "de"] = de.calc_de(obs_arr, sim_arr)
    # # relative bias
    # brel = de.calc_brel(obs_arr, sim_arr)
    # # total bias
    # b_tot = de.calc_bias_tot(brel)
    # df_eff_cam.loc["48", "b_tot"] = de.calc_bias_tot(brel)
    # # bias of high flows
    # b_hf = de.calc_bias_hf(brel)
    # df_eff_cam.loc["48", "b_hf"] = b_hf
    # # contribution of high flow errors
    # err_hf = de.calc_err_hf(b_hf, b_tot)
    # df_eff_cam.loc["48", "err_hf"] = err_hf
    # # bias of low flows
    # b_lf = de.calc_bias_lf(brel)
    # df_eff_cam.loc["48", "b_lf"] = b_lf
    # # contribution of low flow errors
    # err_lf = de.calc_err_lf(b_lf, b_tot)
    # df_eff_cam.loc["48", "err_lf"] = err_lf
    # # direction of bias
    # b_dir = de.calc_bias_dir(brel_res)
    # df_eff_cam.loc["48", "b_dir"] = b_dir
    # # slope of bias
    # b_slope = de.calc_bias_slope(b_area, b_dir)
    # df_eff_cam.loc["48", "b_slope"] = b_slope
    # # convert to radians
    # # (y, x) Trigonometric inverse tangent
    # df_eff_cam.loc["48", "phi"] = de.calc_phi(brel_mean, b_slope)

    # # KGE beta
    # df_eff_cam.loc["48", "beta"] = kge.calc_kge_beta(obs_arr, sim_arr)
    # # KGE alpha
    # df_eff_cam.loc["48", "alpha"] = kge.calc_kge_alpha(obs_arr, sim_arr)
    # # KGE
    # df_eff_cam.loc["48", "kge"] = kge.calc_kge(obs_arr, sim_arr)

    # # NSE
    # df_eff_cam.loc["48", "nse"] = nse.calc_nse(obs_arr, sim_arr)

    # # make arrays
    # obs_arr = df_cam3["Qobs"].values
    # sim_arr = df_cam3["Qsim"].values

    # # mean relative bias
    # brel_mean = de.calc_brel_mean(obs_arr, sim_arr)
    # df_eff_cam.loc["94", "brel_mean"] = brel_mean
    # # residual relative bias
    # brel_res = de.calc_brel_res(obs_arr, sim_arr)
    # # area of relative remaing bias
    # b_area = de.calc_bias_area(brel_res)
    # df_eff_cam.loc["94", "b_area"] = b_area
    # # temporal correlation
    # temp_cor = de.calc_temp_cor(obs_arr, sim_arr)
    # df_eff_cam.loc["94", "temp_cor"] = temp_cor
    # # diagnostic efficiency
    # df_eff_cam.loc["94", "de"] = de.calc_de(obs_arr, sim_arr)
    # # relative bias
    # brel = de.calc_brel(obs_arr, sim_arr)
    # # total bias
    # b_tot = de.calc_bias_tot(brel)
    # df_eff_cam.loc["94", "b_tot"] = de.calc_bias_tot(brel)
    # # bias of high flows
    # b_hf = de.calc_bias_hf(brel)
    # df_eff_cam.loc["94", "b_hf"] = b_hf
    # # contribution of high flow errors
    # err_hf = de.calc_err_hf(b_hf, b_tot)
    # df_eff_cam.loc["94", "err_hf"] = err_hf
    # # bias of low flows
    # b_lf = de.calc_bias_lf(brel)
    # df_eff_cam.loc["94", "b_lf"] = b_lf
    # # contribution of low flow errors
    # err_lf = de.calc_err_lf(b_lf, b_tot)
    # df_eff_cam.loc["94", "err_lf"] = err_lf
    # # direction of bias
    # b_dir = de.calc_bias_dir(brel_res)
    # df_eff_cam.loc["94", "b_dir"] = b_dir
    # # slope of bias
    # b_slope = de.calc_bias_slope(b_area, b_dir)
    # df_eff_cam.loc["94", "b_slope"] = b_slope
    # # convert to radians
    # # (y, x) Trigonometric inverse tangent
    # df_eff_cam.loc["94", "phi"] = de.calc_phi(brel_mean, b_slope)


    # # KGE beta
    # df_eff_cam.loc["94", "beta"] = kge.calc_kge_beta(obs_arr, sim_arr)
    # # KGE alpha
    # df_eff_cam.loc["94", "alpha"] = kge.calc_kge_alpha(obs_arr, sim_arr)
    # # KGE
    # df_eff_cam.loc["94", "kge"] = kge.calc_kge(obs_arr, sim_arr)

    # # NSE
    # df_eff_cam.loc["94", "nse"] = nse.calc_nse(obs_arr, sim_arr)

    # path_csv = Path(os.path.join(
    #     PATH_FIG, "table_eff_real_case.csv"
    # ))
    # df_eff_cam.to_csv(path_csv, header=True, index=True, sep=";")

    # ### multi diagnostic polar plot ###
    # # make arrays
    # brel_mean_arr = df_eff_cam["brel_mean"].values
    # temp_cor_arr = df_eff_cam["temp_cor"].values
    # b_dir_arr = df_eff_cam["b_dir"].values
    # de_arr = df_eff_cam["de"].values
    # phi_arr = df_eff_cam["phi"].values
    # b_slope_arr = df_eff_cam["b_slope"].values
    # b_hf_arr = df_eff_cam["b_hf"].values
    # b_lf_arr = df_eff_cam["b_lf"].values
    # b_tot_arr = df_eff_cam["b_tot"].values
    # err_hf_arr = df_eff_cam["err_hf"].values
    # err_lf_arr = df_eff_cam["err_lf"].values

    # fig_de = util.diag_polar_plot_multi_fc(
    #     brel_mean_arr,
    #     temp_cor_arr,
    #     de_arr,
    #     b_dir_arr,
    #     phi_arr,
    #     b_hf_arr,
    #     b_lf_arr,
    #     b_tot_arr,
    #     err_hf_arr,
    #     err_lf_arr,
    #     idx,
    #     ax_lim=0.6,
    # )

    # path_pdf = Path(os.path.join(PATH_FIG, "de_diag_real_case.pdf"))
    # fig_de.savefig(path_pdf, dpi=250)

    # ### multi KGE plot ###
    # # make arrays
    # alpha_arr = df_eff_cam["alpha"].values
    # beta_arr = df_eff_cam["beta"].values
    # kge_arr = df_eff_cam["kge"].values

    # fig_kge = util.polar_plot_multi_fc(
    #     beta_arr, alpha_arr, temp_cor_arr, kge_arr, idx, ax_lim=0.4
    # )
    # path_pdf = Path(os.path.join(
    #     PATH_FIG, "kge_diag_real_case.pdf"
    # ))
    # fig_kge.savefig(path_pdf, dpi=250)