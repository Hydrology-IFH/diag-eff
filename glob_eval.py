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
        
        ll_b_area = b_area_arr.tolist()
        ll_b_dir = b_dir_arr.tolist()
        ll_b_slope = []
        for (ba, bd) in zip(ll_b_area, ll_b_dir):
            bs = de.calc_bias_slope(ba, bd)
            ll_b_slope.append(bs)
        
        b_slope_arr = np.array(ll_b_slope)
                
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
        q = m.quiver(lons, lats, b_slope_arr, brel_mean_arr, z, cmap='Reds_r', scale=50)

        cbar_ax = fig.add_axes([.3, 0.02, .4, .04], frameon=False)
        cbar = fig.colorbar(q, cax=cbar_ax, orientation='horizontal', ticks=[-2, -1.5, -1, -0.5, 0, 0.5, 1])
        cbar.set_ticklabels(['-2', '-1.5', '-1', '-0.5', '0', '0.5', '1'])
        cbar.ax.tick_params(direction='in', labelsize=24)
        cbar.ax.xaxis.set_label_position('top')
        cbar.ax.xaxis.set_ticks_position('top')
        fig.tight_layout(rect=[0, .13, 1, 1])