# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 18:24:07 2022

Author: Rounak Meyur
Description: Plots the results of stability studies for outliers
"""

import sys, os
import matplotlib.pyplot as plt
import numpy as np


FN = FLAT_NORM = "\\mathbb{{F}}_{{\\lambda}}"
FNN = NORMALIZED_FLAT_NORM = "\\widetilde{{\\mathbb{{F}}}}_{{\\lambda}}"
FNM = FLAT_NORM_MEAN = "\\widehat{{\\mathbb{{F}}}}_{{\\lambda}}"
FNI = FLAT_NORM_INDEX = "\\widetilde{{\\mathbb{{F}}}}_{{\\lambda}}"
HAUS = HAUSDORFF_DISTANCE = "\\mathbb{{D}}_{{H}}"
HAUSM = HAUSDORFF_DISTANCE_MEAN = "\\widehat{{\\mathbb{{D}}}}_{{H}}"

workpath = os.getcwd()
sys.path.append(workpath+'/libs/')

from libs.pyFlatNormFixture import FlatNormFixture, close_fig


# get fixture
fx = FlatNormFixture('runTest')
fx.out_dir = "out/stability"
fx.fig_dir = "figs/stability"
fx.area = "mcbryde"



# read geometries
area = 'mcbryde'

# label defnitions
ind_label = {
    994: 'ex1',
    1003:'ex2',
    # 967: 'ex3',
    # 930: 'ex4'
    }

lambda_ = 0.001
# radius_list = [10, 20, 30, 40, 50]
radius_list = [10, 30, 50]
num_networks = 1000
df_fn, df_ind = fx.read_stability_stats(
    f"{area}-L{lambda_}_FN_STABILITY_STAT_OUTLIER_N{num_networks}_R5",
    f"{area}-L{lambda_}_FN_STAT_INDEX",
    in_dir=fx.out_dir)



L = len(radius_list)

for i,index in enumerate(ind_label):
    fig, axs = plt.subplots(1, L, figsize=(L*12, 12), constrained_layout=True)
    fn_means, hd_means, fn_index, hd_index = list(), list(), list(), list()
    
    for e,radius in enumerate(radius_list):
        fni, hdi = fx.plot_stability_outlier(
            df_fn, df_ind, index, radius = radius, ax=axs[e],
            xylabel_fontsize=40, tick_fontsize=35,
            scatter_size=500,
        )
    
    # fnm_prefix = f"(${HAUS}$,${FNN}$) for perturbed networks in region {ind_label[index]}"
    fnm_prefix = f"(${HAUS}$,${FNN}$) for perturbed networks with outliers"
    region_original = f"original ${FNN}={fni:0.3g}$, ${HAUS}={hdi:0.3g}$"
    fig.suptitle(f"{fnm_prefix}  :  {region_original}", fontsize=50)

    file_name = f"{fx.area}-L{lambda_}_stability_outlier_{ind_label[index]}_selected"
    close_fig(fig, to_file=f"{fx.fig_dir}/{file_name}.png", show=False)
