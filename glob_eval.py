#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
# set path to fix bug in basemap lib
os.environ['PROJ_LIB'] = '/Users/robinschwemmle/anaconda3/envs/de/share/proj/'
PATH = '/Users/robinschwemmle/Desktop/PhD/diagnostic_efficiency'
os.chdir(PATH)
import pandas as pd
import numpy as np
import scipy as sp
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon
import seaborn as sns
sns.set_style('ticks')
# from bokeh.plotting import figure, output_file, show
# from bokeh.models import ColumnDataSource
# from bokeh.layouts import gridplot
from de import de
from de import kge
from de import nse
from de import util

if __name__ == "__main__":
    calc = True
    tier = 'wrr2/'

    # define operation system
    # 'win' = windows machine, 'unix_local' = local unix machine, 'unix' = acces data with unix machine from file server
    OS = 'unix_server'
    if OS == 'win':
        # windows directories
        db_Qobs_meta_dir = '//fuhys013/Schwemmle/Data/Runoff/observed/' \
                           'summary_near_nat.csv'
        Q_dir = '//fuhys013/Schwemmle/Data/Runoff/Q_paired/%s' % (tier)

    elif OS == 'unix_server':
        # unix directories server
        db_Qobs_meta_dir = '/Volumes/Schwemmle/Data/Runoff/observed/' \
                           'summary_near_nat.csv'
        Q_dir = '/Volumes/Schwemmle/Data/Runoff/Q_paired/%s' % (tier)

    elif OS == 'unix_local':
        # unix directories local
        db_Qobs_meta_dir = '/Users/robinschwemmle/Desktop/MSc_Thesis/data'\
                           '/summary_near_nat.csv'
        Q_dir = '/Users/robinschwemmle/Desktop/MSc_Thesis/data/Runoff/Q_paired/%s' % (tier)

    if calc:
        df_meta = pd.read_csv(db_Qobs_meta_dir, sep=',', na_values=-9999, index_col=0)
        ll_catchs = os.listdir(Q_dir)
        try:
            ll_catchs.remove('.DS_Store')
        except:
            pass

        meta = df_meta.loc[ll_catchs, ['lat', 'lon', 'catchsize']]
        meta['brel_mean'] = np.nan
        meta['b_area'] = np.nan
        meta['temp_cor'] = np.nan
        meta['de'] = np.nan
        meta['b_dir'] = np.nan
        meta['b_slope'] = np.nan
        meta['diag'] = np.nan

        meta['kge'] = np.nan
        meta['alpha'] = np.nan
        meta['beta'] = np.nan

        meta['nse'] = np.nan

        meta['obs_mean'] = np.nan
        meta['obs_std'] = np.nan
        meta['obs_cv'] = np.nan
        meta['obs_min'] = np.nan
        meta['obs_max'] = np.nan
        meta['obs_skew'] = np.nan
        meta['obs_kurt'] = np.nan

        meta['perennial'] = True

        prob = np.arange(0, 1.01, .01)
        df_fdc_sim = pd.DataFrame(index=prob, columns=meta.index)
        df_fdc_obs = pd.DataFrame(index=prob, columns=meta.index)

        with tqdm(total=len(meta.index)) as pbar:
            for i, catch in enumerate(meta.index):
                path_csv = '{}{}/Q_sdQsim.csv'.format(Q_dir, str(catch))
                # import observed time series
                obs_sim = util.import_ts(path_csv, sep=',')
                obs_arr = obs_sim['Qobs'].values
                sim_arr = obs_sim['Qsim'].values
                # replace negative values with zero
                sim_arr = np.where(sim_arr<0, 0, sim_arr)
                # observed flow duration curve
                df_fdc_obs.iloc[:, i] = util.fdc(obs_arr)[1]
                # simulated flow duration curve
                df_fdc_sim.iloc[:, i] = util.fdc(sim_arr)[1]
                # count zero values to check whether streamflow is perennial
                zeros = np.sum(np.where(obs_arr == 0))
                if zeros > 33:
                    meta.iloc[i, 21] = False
                # set zero values in observations to constant value
                catch_area = meta.iloc[i, 2]  # catchment area in km2
                # convert 1 l/s to mm/d
                mm_d = 0.001*(1000*24*60*60)/(catch_area * 1000 * 1000)
                const_idx = (sim_arr != 0) & (obs_arr == 0)
                obs_arr[const_idx] = mm_d

                # mean relative bias
                brel_mean = de.calc_brel_mean(obs_arr, sim_arr)
                meta.iloc[i, 3] = brel_mean
                # remaining relative bias
                brel_rest = de.calc_brel_rest(obs_arr, sim_arr)
                # area of relative remaing bias
                b_area = de.calc_bias_area(brel_rest)
                meta.iloc[i, 4] = b_area
                # temporal correlation
                temp_cor = de.calc_temp_cor(obs_arr, sim_arr)
                meta.iloc[i, 5] = temp_cor
                # diagnostic efficiency
                meta.iloc[i, 6] = de.calc_de(obs_arr, sim_arr)
                # direction of bias
                b_dir = de.calc_bias_dir(brel_rest)
                meta.iloc[i, 7] = b_dir
                # slope of bias
                b_slope = de.calc_bias_slope(b_area, b_dir)
                meta.iloc[i, 8] = b_slope
                # convert to radians
                # (y, x) Trigonometric inverse tangent
                meta.iloc[i, 9] = np.arctan2(brel_mean, b_slope)

                # KGE
                meta.iloc[i, 10] = kge.calc_kge(obs_arr, sim_arr)
                # KGE alpha
                meta.iloc[i, 11] = kge.calc_kge_alpha(obs_arr, sim_arr)
                # KGE beta
                meta.iloc[i, 12] = kge.calc_kge_beta(obs_arr, sim_arr)

                # NSE
                meta.iloc[i, 13] = nse.calc_nse(obs_arr, sim_arr)

                # mean, std, min and max of obs
                meta.iloc[i, 14] = np.mean(obs_arr)
                meta.iloc[i, 15] = np.std(obs_arr)
                meta.iloc[i, 16] = np.std(obs_arr)/np.mean(obs_arr)
                meta.iloc[i, 17] = np.min(obs_arr)
                meta.iloc[i, 18] = np.max(obs_arr)
                meta.iloc[i, 19] = sp.stats.skew(obs_arr)
                meta.iloc[i, 20] = sp.stats.kurtosis(obs_arr)
                pbar.update(1)

        meta_all = meta.copy()
        # export metrics
        ll_vars = ['brel_mean', 'b_area', 'temp_cor', 'de', 'b_dir', 'b_slope',
           'diag', 'kge', 'alpha', 'beta', 'nse', 'perennial']
        meta_eff = meta_all.loc[:, ll_vars].join(df_meta)
        path_csv = '/Users/robinschwemmle/Desktop/PhD/diagnostic_efficiency/figures/glob_eval/%smeta_eff.csv' % (tier)
        meta_eff.to_csv(path_csv, sep=';')

    # # generate index first (see make_index.py)
    # # import common index
    # path_idx = '/Users/robinschwemmle/Desktop/PhD/diagnostic_efficiency/figures/glob_eval/meta_idx.csv'
    # df_idx = pd.read_csv(path_idx, sep=';', na_values=-9999, index_col=0)

    # # # remove outliers
    # meta = meta[(meta['de'] >= -1) & (meta['perennial'] == True)]
    # meta = df_idx.join(meta)
    # ll_catchs = meta.index.tolist()

    # # make dataframe for FDC plots
    # fdc_obs = df_fdc_obs.loc[:, ll_catchs]
    # fdc_sim = df_fdc_sim.loc[:, ll_catchs]

    # # interactive FDC plots for visual inspection
    # source = ColumnDataSource(dict(
    #         xs=[fdc_sim.index.values for i, p in enumerate(fdc_sim.columns)],
    #         ys=[fdc_sim[catch].values for catch in fdc_sim.columns],
    #         xo=[fdc_sim.index.values for i, p in enumerate(fdc_obs.columns)],
    #         yo=[fdc_obs[catch].values for catch in fdc_obs.columns],
    #         name=[catch for catch in fdc_sim.columns]
    #     )
    # )

    # TOOLS = "pan,wheel_zoom,reset,hover,save"

    # p1 = figure(plot_width=600, plot_height=300, x_range=[-.05, 1.05],
    #             title="observed FDCs", tools=TOOLS,
    #             y_axis_type="log", x_axis_label='Exceedence probabilty [-]',
    #             y_axis_label='[mm d-1]', background_fill_color="#fafafa",
    #             tooltips=[
    #                 ("(Name, Prob, Q)", "(@name, $x, $y)")
    # ])
    # p1.hover.point_policy = "follow_mouse"

    # p1.multi_line(xs="xo", ys="yo", source=source,
    #               line_color='blue',
    #               line_width=2)

    # p2 = figure(plot_width=600, plot_height=300, x_range=[-.05, 1.05],
    #             title="simulated FDCs", tools=TOOLS,
    #             background_fill_color="#fafafa",
    #             y_axis_type="log", x_axis_label='Exceedence probabilty [-]',
    #             y_axis_label='[mm d−1]',
    #             tooltips=[
    #                 ("(Name, Prob, Q)", "(@name, $x, $y)")
    # ])
    # p2.hover.point_policy = "follow_mouse"

    # p2.multi_line(xs="xs", ys="ys", source=source,
    #               line_color='red',
    #               line_width=2)
    # output_file("/Users/robinschwemmle/Downloads/interactive_fdc.html",
    #             title="interactive FDCs")
    # show(gridplot([p1, p2], ncols=2))

    # # make dataframes for FDC plots with seaborn
    # fdc_obs['prob'] = fdc_obs.index
    # fdc_sim['prob'] = fdc_sim.index
    # sim = fdc_sim.melt(id_vars=['prob'])
    # obs = fdc_obs.melt(id_vars=['prob'])

    # # plot mean FDC with confidence intervals 95%
    # sns.set_style('ticks', {'xtick.major.size': 8, 'ytick.major.size': 8})
    # sns.set_context("paper", font_scale=2)
    # fig, ax = plt.subplots(figsize=(10, 6))
    # g = sns.lineplot(x="prob", y="value", data=sim, color='r',
    #                   estimator='mean', seed=42, ax=ax)
    # g = sns.lineplot(x="prob", y="value", data=obs, seed=42, color='b',
    #                   estimator='mean', ax=ax)
    # g.set(yscale="log")
    # ax.set_xlim([0, 1])
    # ax.set_ylim([0, 100])
    # ax.set(ylabel=r'[mm $d^{-1}$]', xlabel='Exceedence probabilty [-]')

    # # plot mean FDC with confidence intervals 95% and 100%
    # fig, ax = plt.subplots(figsize=(10, 6))
    # g = sns.lineplot(x="prob", y="value", data=sim, color='r', ci=100,
    #                  estimator='mean', seed=42, alpha=.5, ax=ax)
    # g = sns.lineplot(x="prob", y="value", data=obs, seed=42, ci=100,
    #                  color='b', estimator='mean', alpha=.5, ax=ax)
    # g = sns.lineplot(x="prob", y="value", data=sim, color='r',
    #                  estimator='mean', seed=42, ax=ax)
    # g = sns.lineplot(x="prob", y="value", data=obs, seed=42, color='b',
    #                  estimator='mean', ax=ax)
    # g.set(yscale="log")
    # ax.set_xlim([0, 1])
    # ax.set_ylim([0, 100])
    # ax.set(ylabel=r'[mm $d^{-1}$]', xlabel='Exceedence probabilty [-]')
    # sns.reset_defaults()

    # make arrays
    brel_mean_arr = meta['brel_mean'].values
    b_area_arr = meta['b_area'].values
    temp_cor_arr = meta['temp_cor'].values
    b_dir_arr = meta['b_dir'].values
    eff_de_arr = meta['de'].values
    phi_arr = meta['diag'].values
    b_slope_arr = meta['b_slope'].values

    # multi polar plot
    fig1, fig2 = de.diag_polar_plot_multi(brel_mean_arr, b_area_arr, temp_cor_arr,
                                          eff_de_arr, b_dir_arr, phi_arr, extended=True)
    fig1_png = '/Users/robinschwemmle/Desktop/PhD/diagnostic_efficiency/figures/glob_eval/%polar_de.png' % (tier)
    fig1.savefig(fig1_png, dpi=250)
    fig1_pdf = '/Users/robinschwemmle/Desktop/PhD/diagnostic_efficiency/figures/glob_eval/%spolar_de.pdf' % (tier)
    fig1.savefig(fig1_pdf, dpi=250)
    fig2_png = '/Users/robinschwemmle/Desktop/PhD/diagnostic_efficiency/figures/glob_eval/%sde_kde_bi.png' % (tier)
    fig2.savefig(fig2_png, dpi=250)
    fig2_pdf = '/Users/robinschwemmle/Desktop/PhD/diagnostic_efficiency/figures/glob_eval/%sde_kde_bi.pdf' % (tier)
    fig2.savefig(fig2_pdf, dpi=250)

    # # make arrays
    # alpha_arr = meta['alpha'].values
    # beta_arr = meta['beta'].values
    # kge_arr = meta['kge'].values

    # # multi KGE plot
    # fig1, fig2 = kge.diag_polar_plot_kge_multi(beta_arr, alpha_arr, temp_cor_arr, kge_arr,
    #                                            extended=True)

    # # global map
    # x = meta['lon'].values
    # y = meta['lat'].values
    # de_arr = meta['de'].values
    # r = meta['temp_cor'].values
    # norm_de = colors.Normalize(vmin=-1.0, vmax=1.0)
    # norm_r = colors.Normalize(vmin=0, vmax=1.0)

#        # map visualizing the spatial distribution of DE
#        fig, ax = plt.subplots(figsize=(12, 6))
#        ax.set_title(r'$DE$ [-]', fontsize=30, pad=12)
#        m = Basemap(resolution='i',
#                    projection='robin',
#                    lon_0=0)
#        # draw continents
#        m.drawmapboundary(fill_color='white')
#        m.drawcoastlines(linewidth=0.5)
#        # plot grid
#        m.drawmeridians(np.arange(0, 360, 30))
#        m.drawparallels(np.arange(-90, 90, 30))
#        lons, lats = m(x, y)  # projecting the coordinates
##        sc = m.scatter(lons, lats, c=de_arr, s=5, cmap='YlGnBu', vmin=-2, vmax=1)
#        q = m.quiver(lons, lats, b_slope_arr, brel_mean_arr, de_arr, cmap='Reds_r',
#                     scale=5, scale_units='inches', angles='xy', norm=norm_de)
#
#        cbar_ax = fig.add_axes([.3, 0.02, .4, .04], frameon=False)
#        cbar = fig.colorbar(q, cax=cbar_ax, orientation='horizontal',
#                            ticks=[-2, -1.5, -1, -0.5, 0, 0.5, 1])
#        cbar.set_ticklabels(['-2', '-1.5', '-1', '-0.5', '0', '0.5', '1'])
#        cbar.ax.tick_params(direction='in', labelsize=24)
#        cbar.ax.xaxis.set_label_position('top')
#        cbar.ax.xaxis.set_ticks_position('top')
#        fig.tight_layout(rect=[0, .13, 1, 1])
#        fig_path = '/Users/robinschwemmle/Desktop/PhD/diagnostic_model_efficiency/figures/glob_eval/%sde_glob.png' % (tier)
#        fig.savefig(fig_path, dpi=250)
#
#        # EU map visualizing the spatial distribution of DE
#        fig, ax = plt.subplots(figsize=(12, 6))
#        m = Basemap(llcrnrlon=-15,llcrnrlat=35,urcrnrlon=50,urcrnrlat=75,
#                    lon_0=17.5, lat_0=55, resolution='i')
#        # draw continents
#        m.drawmapboundary(fill_color='white')
#        m.drawcoastlines(linewidth=0.5)
#        # plot grid
#        m.drawmeridians(np.arange(-15, 50, 10))
#        m.drawparallels(np.arange(40, 75, 10))
#        lons, lats = m(x, y)  # projecting the coordinates
#        q = m.quiver(lons, lats, b_slope_arr, brel_mean_arr, de_arr, cmap='Reds_r',
#                     scale=2, scale_units='inches', angles='xy', norm=norm_de)
#        fig.tight_layout(rect=[0, .13, 1, 1])
#        fig_path = '/Users/robinschwemmle/Desktop/PhD/diagnostic_model_efficiency/figures/glob_eval/%sde_eu.png' % (tier)
#        fig.savefig(fig_path, dpi=250)
#
#
#        # US map visualizing the spatial distribution of DE
#        fig, ax = plt.subplots(figsize=(12, 6))
#        m = Basemap(llcrnrlon=-150,llcrnrlat=20,urcrnrlon=-50,urcrnrlat=75,
#                    lon_0=-100, lat_0=47.5, resolution='i')
#        # draw continents
#        m.drawmapboundary(fill_color='white')
#        m.drawcoastlines(linewidth=0.5)
#        # plot grid
#        m.drawmeridians(np.arange(-150, -50, 10))
#        m.drawparallels(np.arange(20, 75, 10))
#        lons, lats = m(x, y)  # projecting the coordinates
#        q = m.quiver(lons, lats, b_slope_arr, brel_mean_arr, de_arr, cmap='Reds_r',
#                     scale=2, scale_units='inches', angles='xy', norm=norm_de)
#        fig.tight_layout(rect=[0, .13, 1, 1])
#        fig_path = '/Users/robinschwemmle/Desktop/PhD/diagnostic_model_efficiency/figures/glob_eval/%sde_us.png' % (tier)
#        fig.savefig(fig_path, dpi=250)
#
    # # DE global map with DE colorbar
    # fig = plt.figure(figsize=(10, 10))
    # ax1 = plt.subplot2grid((2,2), (1,0), colspan=2)
    # ax2 = plt.subplot2grid((2,2), (0,0))
    # ax3 = plt.subplot2grid((2,2), (0,1))

    # # define EU outcrop
    # lats_eu = [40, 75, 75, 40]
    # lons_eu = [-15, -15, 50, 50]

    # # define US outcrop
    # lats_us = [20, 75, 75, 20]
    # lons_us = [-150, -150, -50, -50]

    # # global map
    # m = Basemap(resolution='i',
    #             projection='robin',
    #             lon_0=0, ax=ax1)
    # # draw continents
    # m.drawmapboundary(fill_color='white')
    # m.drawcoastlines(linewidth=0.5)
    # # plot grid
    # m.drawmeridians(np.arange(0, 360, 30))
    # m.drawparallels(np.arange(-90, 90, 30))
    # lons, lats = m(x, y)  # projecting the coordinates
    # q = m.quiver(lons, lats, b_slope_arr, brel_mean_arr, de_arr, cmap='Reds_r',
    #               scale=5, angles='xy', scale_units='inches', norm=norm_de)

    # # mark EU inset
    # x_eu, y_eu = m(lons_eu, lats_eu)
    # xy_eu = zip(x_eu, y_eu)
    # poly_eu = Polygon(list(xy_eu), edgecolor='grey', lw=2, fill=False,
    #                   alpha=.5)
    # m.ax.add_patch(poly_eu)

    # # mark US inset
    # x_us, y_us = m(lons_us, lats_us)
    # xy_us = zip(x_us, y_us)
    # poly_us = Polygon(list(xy_us), edgecolor='grey', lw=2, fill=False,
    #                   alpha=.5)
    # m.ax.add_patch(poly_us)

    # cbar_ax = fig.add_axes([.3, 0.02, .4, .03], frameon=False)
    # cbar = fig.colorbar(q, cax=cbar_ax, orientation='horizontal',
    #                     ticks=[-1, -0.5, 0, 0.5, 1])
    # cbar.set_label(r'DE [-]', fontsize=16, labelpad=8)
    # cbar.set_ticklabels(['-1', '-0.5', '0', '0.5', '1'])
    # cbar.ax.tick_params(direction='in', labelsize=14)
    # cbar.ax.xaxis.set_label_position('top')
    # cbar.ax.xaxis.set_ticks_position('top')

    # # EU map
    # m = Basemap(llcrnrlon=lons_eu[0], llcrnrlat=lats_eu[0], urcrnrlon=lons_eu[-1], urcrnrlat=lats_eu[2],
    #             lon_0=(lons_eu[0] - lons_eu[-1])/2, lat_0=(lats_eu[0] - lats_eu[2])/2, resolution='i', ax=ax3)
    # # draw continents
    # m.drawmapboundary(fill_color='white')
    # m.drawcoastlines(linewidth=0.5)
    # lons, lats = m(x, y)  # projecting the coordinates
    # q = m.quiver(lons, lats, b_slope_arr, brel_mean_arr, de_arr, cmap='Reds_r',
    #               scale=3, scale_units='inches', width=0.01, angles='xy',
    #               norm=norm_de)

    # # US map
    # m = Basemap(llcrnrlon=lons_us[0], llcrnrlat=lats_us[0], urcrnrlon=lons_us[-1], urcrnrlat=lats_us[2],
    #             lon_0=(lons_us[0] - lons_us[-1])/2, lat_0=(lats_us[0] - lats_us[2])/2, resolution='i', ax=ax2)
    # # draw continents
    # m.drawmapboundary(fill_color='white')
    # m.drawcoastlines(linewidth=0.5)
    # lons, lats = m(x, y)  # projecting the coordinates
    # q = m.quiver(lons, lats, b_slope_arr, brel_mean_arr, de_arr, cmap='Reds_r',
    #               scale=3, scale_units='inches', width=0.01, angles='xy',
    #               norm=norm_de)

    # fig.tight_layout(rect=[0, .13, 1, 1])
    # fig_path = '/Users/robinschwemmle/Desktop/PhD/diagnostic_efficiency/figures/glob_eval/%sde_de_glob_us_eu.png' % (tier)
    # fig.savefig(fig_path, dpi=250)
    # fig_path = '/Users/robinschwemmle/Desktop/PhD/diagnostic_efficiency/figures/glob_eval/%sde_de_glob_us_eu.pdf' % (tier)
    # fig.savefig(fig_path, dpi=250)

    # # DE global map with r colorbar
    # fig = plt.figure(figsize=(10, 10))
    # ax1 = plt.subplot2grid((2,2), (1,0), colspan=2)
    # ax2 = plt.subplot2grid((2,2), (0,0))
    # ax3 = plt.subplot2grid((2,2), (0,1))

    # # define EU outcrop
    # lats_eu = [40, 75, 75, 40]
    # lons_eu = [-15, -15, 50, 50]

    # # define US outcrop
    # lats_us = [20, 75, 75, 20]
    # lons_us = [-150, -150, -50, -50]

    # # global map
    # m = Basemap(resolution='i',
    #             projection='robin',
    #             lon_0=0, ax=ax1)
    # # draw continents
    # m.drawmapboundary(fill_color='white')
    # m.drawcoastlines(linewidth=0.5)
    # # plot grid
    # m.drawmeridians(np.arange(0, 360, 30))
    # m.drawparallels(np.arange(-90, 90, 30))
    # lons, lats = m(x, y)  # projecting the coordinates
    # q = m.quiver(lons, lats, b_slope_arr, brel_mean_arr, r, cmap='plasma_r',
    #               scale=5, angles='xy', scale_units='inches', norm=norm_r)

    # # mark EU inset
    # x_eu, y_eu = m(lons_eu, lats_eu)
    # xy_eu = zip(x_eu, y_eu)
    # poly_eu = Polygon(list(xy_eu), edgecolor='grey', lw=2, fill=False,
    #                   alpha=.5)
    # m.ax.add_patch(poly_eu)

    # # mark US inset
    # x_us, y_us = m(lons_us, lats_us)
    # xy_us = zip(x_us, y_us)
    # poly_us = Polygon(list(xy_us), edgecolor='grey', lw=2, fill=False,
    #                   alpha=.5)
    # m.ax.add_patch(poly_us)

    # cbar_ax = fig.add_axes([.3, 0.02, .4, .03], frameon=False)
    # cbar = fig.colorbar(q, cax=cbar_ax, orientation='horizontal',
    #                     ticks=[0, 0.5, 1])
    # cbar.set_label('r [-]', fontsize=16, labelpad=8)
    # cbar.set_ticklabels(['<0', '0.5', '1'])
    # cbar.ax.tick_params(direction='in', labelsize=14)
    # cbar.ax.xaxis.set_label_position('top')
    # cbar.ax.xaxis.set_ticks_position('top')

    # # EU map
    # m = Basemap(llcrnrlon=lons_eu[0], llcrnrlat=lats_eu[0], urcrnrlon=lons_eu[-1], urcrnrlat=lats_eu[2],
    #             lon_0=(lons_eu[0] - lons_eu[-1])/2, lat_0=(lats_eu[0] - lats_eu[2])/2, resolution='i', ax=ax3)
    # # draw continents
    # m.drawmapboundary(fill_color='white')
    # m.drawcoastlines(linewidth=0.5)
    # lons, lats = m(x, y)  # projecting the coordinates
    # q = m.quiver(lons, lats, b_slope_arr, brel_mean_arr, r, cmap='plasma_r',
    #               scale=3, scale_units='inches', width=0.01, angles='xy',
    #               norm=norm_de)

    # # US map
    # m = Basemap(llcrnrlon=lons_us[0], llcrnrlat=lats_us[0], urcrnrlon=lons_us[-1], urcrnrlat=lats_us[2],
    #             lon_0=(lons_us[0] - lons_us[-1])/2, lat_0=(lats_us[0] - lats_us[2])/2, resolution='i', ax=ax2)
    # # draw continents
    # m.drawmapboundary(fill_color='white')
    # m.drawcoastlines(linewidth=0.5)
    # lons, lats = m(x, y)  # projecting the coordinates
    # q = m.quiver(lons, lats, b_slope_arr, brel_mean_arr, r, cmap='plasma_r',
    #               scale=3, scale_units='inches', width=0.01, angles='xy',
    #               norm=norm_de)

    # fig.tight_layout(rect=[0, .13, 1, 1])
    # fig_path = '/Users/robinschwemmle/Desktop/PhD/diagnostic_efficiency/figures/glob_eval/%sde_r_glob_us_eu.png' % (tier)
    # fig.savefig(fig_path, dpi=250)
    # fig_path = '/Users/robinschwemmle/Desktop/PhD/diagnostic_efficiency/figures/glob_eval/%sde_r_glob_us_eu.pdf' % (tier)
    # fig.savefig(fig_path, dpi=250)
