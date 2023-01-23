# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 11:45:25 2022

Author: Rounak Meyur

Description: Chooses particular regions in a geographic location to compare the
networks inside it.
"""

import sys, os
from shapely.geometry import Point
import pandas as pd
import csv


FN = FLAT_NORM = "\\mathbb{{F}}_{{\\lambda}}"
FNN = NORMALIZED_FLAT_NORM = "\\widetilde{{\\mathbb{{F}}}}_{{\\lambda}}"
HAUS = HAUSDORFF_DISTANCE = "\\mathbb{{D}}_{{Haus}}"
MIN_X, MIN_Y, MAX_X, MAX_Y = 0, 1, 2, 3


workpath = os.getcwd()
sys.path.append(workpath+'/libs/')

from libs.pyFlatNormFixture import FlatNormFixture
from libs.pyFlatNormlib import get_structure


# get fixture
fx = FlatNormFixture('runTest')
fx.out_dir = "out/test"
fx.fig_dir = "figs/test"



# read geometries
area = 'mcbryde'
act_geom, synt_geom, hull = fx.read_networks(area)
struct = get_structure(act_geom)


# label defnitions
ind_label = {
    994: 'ex1',
    1003:'ex2',
    967: 'ex3',
    930: 'ex4'}

#%% Flat norm computation
def compute_flat_norm_region(ind,point,eps,lamb_,plot_result=True):
    # Tag the point
    region = ind_label[ind]
    
    # compute flat norm
    norm, enorm, tnorm, w, plot_data = fx.compute_region_flatnorm(
        fx.get_region(point, eps),
        act_geom, synt_geom,
        lambda_=lamb_,
        normalized=True,
        plot=True
    )
    
    # compute hausdorff distance
    hd, hd_geom = fx.compute_region_hausdorff(
        fx.get_region(point, eps),
        act_geom, synt_geom,
        distance = "geodesic",
        )
    plot_data["hd_geom"] = hd_geom
    
    
    titles = {
        'lambda': f"$\\lambda = {lambda_:d}$",
        'epsilon': f"$\\epsilon = {epsilon:0.4f}$",
        'ratio': f"$|T|/\\epsilon = {w/epsilon :0.3g}$",
        'fn': f"${FNN}={norm:0.3g}$",
        'haus': f"${HAUS}={hd:0.3g}$"
    }
    title = ", ".join([titles[t_name] for t_name in titles])
    
    # plot flat norm
    if plot_result:
        fx.plot_triangulated_region_flatnorm(
            epsilon=eps, lambda_=lamb_,
            to_file=f"{area}-L{lamb_}_fn_region_{region}",
            suptitle=title,
            show_figs = ["haus", "fn"],
            do_return=False, show=True,
            figsize=(26,14),
            legend_location = "upper left",
            **plot_data
            )
    return norm, hd, w



#%% compute flat norm

# parameters
epsilon = 1e-3
lambda_ = 1


# compute stats
flatnorm_data = {
    'flatnorms': [], 'input_ratios': [], 'index': [], 'hausdorff': [],
    'MIN_X': [], 'MIN_Y': [], 'MAX_X': [], 'MAX_Y': []
}

for ind in ind_label:
    pt = Point(struct["vertices"][ind])
    region = fx.get_region(pt, epsilon)
    norm, hd, w = compute_flat_norm_region(ind, pt, epsilon, lambda_,plot_result=True)
    # sys.exit(0)
    # store the data
    flatnorm_data['index'].append(ind)
    flatnorm_data['flatnorms'].append(norm)
    flatnorm_data['input_ratios'].append(w/epsilon)
    flatnorm_data['hausdorff'].append(hd)
    region_bounds = region.exterior.bounds
    flatnorm_data['MIN_X'].append(region_bounds[MIN_X])
    flatnorm_data['MIN_Y'].append(region_bounds[MIN_Y])
    flatnorm_data['MAX_X'].append(region_bounds[MAX_X])
    flatnorm_data['MAX_Y'].append(region_bounds[MAX_Y])
    

df = pd.DataFrame(flatnorm_data)
filename = f"{area}-L{lambda_}_FN_STAT_INDEX"
with open(f"{fx.out_dir}/{filename}.csv", "w") as outfile:
    df.to_csv(outfile, sep=",", index=False,
              header=True, quoting=csv.QUOTE_NONNUMERIC)

