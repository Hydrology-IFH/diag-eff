#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
# set path to fix bug in basemap lib
os.environ['PROJ_LIB'] = '/Users/robinschwemmle/anaconda3/envs/de/share/proj/'
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon
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
        Q_dir = '//fuhys013/Schwemmle/Data/Runoff/Q_paired/wrr1/'

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

    with tqdm(total=len(meta.index)) as pbar:
        for i, catch in enumerate(meta.index):
            path_csv = '{}{}/Q_sdQsim.csv'.format(Q_dir, str(catch))
            # import observed time series
            obs_sim = util.import_ts(path_csv, sep=',')
            obs_arr = obs_sim['Qobs'].values
            sim_arr = obs_sim['Qsim'].values
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
            b_area = de.calc_b_area(brel_rest)
            meta.iloc[i, 4] = b_area
            # temporal correlation
            temp_cor = de.calc_temp_cor(obs_arr, sim_arr)
            meta.iloc[i, 5] = temp_cor
            # diagnostic efficiency
            meta.iloc[i, 6] = 1 - np.sqrt((brel_mean)**2 + (b_area)**2 + (temp_cor - 1)**2)
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
            meta.iloc[i, 10] = de.calc_kge(obs_arr, sim_arr)
            # KGE alpha
            meta.iloc[i, 11] = de.calc_kge_alpha(obs_arr, sim_arr)
            # KGE beta
            meta.iloc[i, 12] = de.calc_kge_beta(obs_arr, sim_arr)
            pbar.update(1)

        # remove outliers
        sig_de_arr = meta['de'].values
        meta = meta.loc[sig_de_arr > -2, :]

        # make arrays
        brel_mean_arr = meta['brel_mean'].values
        b_area_arr = meta['b_area'].values
        temp_cor_arr = meta['temp_cor'].values
        b_dir_arr = meta['b_dir'].values
        sig_de_arr = meta['de'].values
        diag_arr = meta['diag'].values

        # multi polar plot
        de.vis2d_de_multi(brel_mean_arr, b_area_arr, temp_cor_arr, sig_de_arr,
                          b_dir_arr, diag_arr, extended=True)

        b_slope_arr = meta['b_slope'].values

        # global map
        x = meta['lon'].values
        y = meta['lat'].values
        z = meta['de'].values

        # map visualizing the spatial distribution of DE
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_title(r'$DE$ [-]', fontsize=30, pad=12)
        m = Basemap(resolution='i',
                    projection='robin',
                    lon_0=0)
        # draw continents
        m.drawmapboundary(fill_color='white')
        m.drawcoastlines(linewidth=0.5)
        # plot grid
        m.drawmeridians(np.arange(0, 360, 30))
        m.drawparallels(np.arange(-90, 90, 30))
        lons, lats = m(x, y)  # projecting the coordinates
#        sc = m.scatter(lons, lats, c=z, s=5, cmap='YlGnBu', vmin=-2, vmax=1)
        q = m.quiver(lons, lats, b_slope_arr, brel_mean_arr, z, cmap='Reds_r',
                     scale=5, angles='xy', scale_units='inches')

        cbar_ax = fig.add_axes([.3, 0.02, .4, .04], frameon=False)
        cbar = fig.colorbar(q, cax=cbar_ax, orientation='horizontal',
                            ticks=[-2, -1.5, -1, -0.5, 0, 0.5, 1])
        cbar.set_ticklabels(['-2', '-1.5', '-1', '-0.5', '0', '0.5', '1'])
        cbar.ax.tick_params(direction='in', labelsize=24)
        cbar.ax.xaxis.set_label_position('top')
        cbar.ax.xaxis.set_ticks_position('top')
        fig.tight_layout(rect=[0, .13, 1, 1])


        # EU map visualizing the spatial distribution of DE
        fig, ax = plt.subplots(figsize=(12, 6))
        m = Basemap(llcrnrlon=-15,llcrnrlat=35,urcrnrlon=50,urcrnrlat=75,
                    lon_0=17.5, lat_0=55, resolution='i')
        # draw continents
        m.drawmapboundary(fill_color='white')
        m.drawcoastlines(linewidth=0.5)
        # plot grid
        m.drawmeridians(np.arange(-15, 50, 10))
        m.drawparallels(np.arange(40, 75, 10))
        lons, lats = m(x, y)  # projecting the coordinates
        q = m.quiver(lons, lats, b_slope_arr, brel_mean_arr, z, cmap='Reds_r',
                     scale=2, angles='xy', scale_units='inches')
        fig.tight_layout(rect=[0, .13, 1, 1])


        # US map visualizing the spatial distribution of DE
        fig, ax = plt.subplots(figsize=(12, 6))
        m = Basemap(llcrnrlon=-150,llcrnrlat=20,urcrnrlon=-50,urcrnrlat=75,
                    lon_0=-100, lat_0=47.5, resolution='i')
        # draw continents
        m.drawmapboundary(fill_color='white')
        m.drawcoastlines(linewidth=0.5)
        # plot grid
        m.drawmeridians(np.arange(-150, -50, 10))
        m.drawparallels(np.arange(20, 75, 10))
        lons, lats = m(x, y)  # projecting the coordinates
        q = m.quiver(lons, lats, b_slope_arr, brel_mean_arr, z, cmap='Reds_r',
                     scale=2, angles='xy', scale_units='inches')
        fig.tight_layout(rect=[0, .13, 1, 1])

        # global map with outsets
        fig = plt.figure(figsize=(10, 10))
        ax1 = plt.subplot2grid((2,2), (1,0), colspan=2)
        ax2 = plt.subplot2grid((2,2), (0,0))
        ax3 = plt.subplot2grid((2,2), (0,1))

        # define EU outcrop
        lats_eu = [40, 75, 75, 40]
        lons_eu = [-15, -15, 50, 50]

        # define US outcrop
        lats_us = [20, 75, 75, 20]
        lons_us = [-150, -150, -50, -50]

        # global map
        m = Basemap(resolution='i',
                    projection='robin',
                    lon_0=0, ax=ax1)
        # draw continents
        m.drawmapboundary(fill_color='white')
        m.drawcoastlines(linewidth=0.5)
        # plot grid
        m.drawmeridians(np.arange(0, 360, 30))
        m.drawparallels(np.arange(-90, 90, 30))
        lons, lats = m(x, y)  # projecting the coordinates
        q = m.quiver(lons, lats, b_slope_arr, brel_mean_arr, z, cmap='Reds_r',
                     scale=5, angles='xy', scale_units='inches')

        # mark EU inset
        x_eu, y_eu = m(lons_eu, lats_eu)
        xy_eu = zip(x_eu, y_eu)
        poly_eu = Polygon(list(xy_eu), edgecolor='grey', lw=2, fill=False,
                          alpha=.5)
        m.ax.add_patch(poly_eu)

        # mark US inset
        x_us, y_us = m(lons_us, lats_us)
        xy_us = zip(x_us, y_us)
        poly_us = Polygon(list(xy_us), edgecolor='grey', lw=2, fill=False,
                          alpha=.5)
        m.ax.add_patch(poly_us)

        cbar_ax = fig.add_axes([.3, 0.02, .4, .03], frameon=False)
        cbar = fig.colorbar(q, cax=cbar_ax, orientation='horizontal',
                            ticks=[-2, -1.5, -1, -0.5, 0, 0.5, 1])
        cbar.set_label(r'DE [-]', fontsize=16, labelpad=8)
        cbar.set_ticklabels(['-2', '-1.5', '-1', '-0.5', '0', '0.5', '1'])
        cbar.ax.tick_params(direction='in', labelsize=14)
        cbar.ax.xaxis.set_label_position('top')
        cbar.ax.xaxis.set_ticks_position('top')

        # EU map
        m = Basemap(llcrnrlon=lons_eu[0], llcrnrlat=lats_eu[0], urcrnrlon=lons_eu[-1], urcrnrlat=lats_eu[2],
                    lon_0=(lons_eu[0] - lons_eu[-1])/2, lat_0=(lats_eu[0] - lats_eu[2])/2, resolution='i', ax=ax3)
        # draw continents
        m.drawmapboundary(fill_color='white')
        m.drawcoastlines(linewidth=0.5)
        lons, lats = m(x, y)  # projecting the coordinates
        q = m.quiver(lons, lats, b_slope_arr, brel_mean_arr, z, cmap='Reds_r',
                     scale=3, angles='xy', scale_units='inches', width=0.01)

        # US map
        m = Basemap(llcrnrlon=lons_us[0], llcrnrlat=lats_us[0], urcrnrlon=lons_us[-1], urcrnrlat=lats_us[2],
                    lon_0=(lons_us[0] - lons_us[-1])/2, lat_0=(lats_us[0] - lats_us[2])/2, resolution='i', ax=ax2)
        # draw continents
        m.drawmapboundary(fill_color='white')
        m.drawcoastlines(linewidth=0.5)
        lons, lats = m(x, y)  # projecting the coordinates
        q = m.quiver(lons, lats, b_slope_arr, brel_mean_arr, z, cmap='Reds_r',
                     scale=3, angles='xy', scale_units='inches', width=0.01)

        fig.tight_layout(rect=[0, .13, 1, 1])
