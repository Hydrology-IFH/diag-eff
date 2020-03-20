# -*- coding: utf-8 -*-
import pandas as pd

if __name__ == "__main__":
    # make common index based on efficiency
    # select only perennial rivers
    path_wrr1 = '/Users/robinschwemmle/Desktop/PhD/diagnostic_efficiency/figures/glob_eval/wrr1/meta_eff.csv'
    path_wrr2 = '/Users/robinschwemmle/Desktop/PhD/diagnostic_efficiency/figures/glob_eval/wrr2/meta_eff.csv'

    df_wrr1 = pd.read_csv(path_wrr1, sep=';', na_values=-9999, index_col=0)
    df_wrr2 = pd.read_csv(path_wrr2, sep=';', na_values=-9999, index_col=0)

    meta1 = df_wrr1[(df_wrr1['de'] >= -1) & (df_wrr1['perennial']==True)]
    meta2 = df_wrr2[(df_wrr2['de'] >= -1) & (df_wrr2['perennial']==True)]
    meta_idx = pd.DataFrame(columns=['catchment'])
    meta_idx.loc[:, 'catchment'] = meta2.index.join(meta1.index, how='inner')
    
    
    path_csv = '/Users/robinschwemmle/Desktop/PhD/diagnostic_efficiency/figures/glob_eval/meta_idx.csv'
    meta_idx.to_csv(path_csv, sep=';', index=False)
