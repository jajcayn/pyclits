#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm

matplotlib.rcParams['text.usetex'] = True

plt.rc('text', usetex=True)
plt.rc('font', family='serif')


def figures3d(dataset, dimensions, title, zlabel, zaxis_selector, row_sizes, filename, suffix, view=(30, 30), scale=lambda N: N, dpi=300):

    fig = plt.figure(figsize=(13, 8), tight_layout=True)
    ax = fig.add_subplot(111, projection='3d')

    # For each set of style and range settings, plot n random points in
    # the box
    # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
    colors = ["r", "g", "b", "c", "m", "y", "k", "orange", "pink"]
    markers = ['b', '^']

    ax.set_title(title)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$N$")
    ax.set_zlabel(zlabel)
    #ax.set_yticks([1, 2, 3, 4, 5], ["10", "100", "1000", "10000", "100000"])
    plt.yticks((1.0, 2.0, 3.0, 4.0, 5.0), ("10", "100", "1000", "10000", "100000"))

    for dimension, color, row_size in zip(dimensions, colors, row_sizes):
        dataset_selected = dataset.loc[dataset["dimension"] == dimension][['alpha', 'sample size', zaxis_selector]]

        xs = dataset_selected['alpha']
        ys = dataset_selected['sample size']
        zs = dataset_selected[zaxis_selector]/scale(dimension)

        try:
            ax.plot_wireframe(np.reshape(xs.values, (-1, row_size)), np.reshape(np.log10(ys.values), (-1, row_size)), np.reshape(zs.values, (-1, row_size)), rstride=1, cstride=1, color=color, label=f"D={dimension}", linewidth=1)
        except:
            print(f"Problem D={dimension}, {zaxis_selector}")
    #surf = ax.plot_surface(np.reshape(xs.values, (-1, row_size)), np.reshape(np.log10(ys.values), (-1, row_size)), np.reshape(zs.values, (-1, row_size)), rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    # Add a color bar which maps values to colors.
    #fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.legend(loc=1)
    ax.view_init(view[0], view[1])

    plt.savefig(filename+"."+suffix, dpi=dpi)
    #plt.draw()
    #plt.show()
    plt.close()


if __name__ == "__main__":
    list_of_tables = []
    for root, dirs, files in os.walk("."):
        for filename in files:
            if "statistics" in filename and "complete" not in filename:
                print(filename)
                colnames = ['alpha', 'sample size', 'sigma', 'mean Renyi entropy', 'std Renyi entropy', 'mean computer time', 'std computer time', 'mean difference', 'std of difference']
                table = pd.read_csv(filepath_or_buffer=filename, sep=" ", names=colnames, header=None)
                number = int(filename.split("_")[1].split(".")[0])
                table["dimension"] = number
                list_of_tables.append(table)
                #print(table)
                #print(table.reindex(columns=['alpha', 'sample size', 'sigma', 'dimension']))

    complete_table = pd.concat(list_of_tables, join="inner")
    complete_table_indexed = complete_table.set_index(['dimension', 'sigma', 'alpha', 'sample size'])

    #print(complete_table)
    #mean = complete_table_indexed.loc[(2,), "mean Renyi entropy"]
    #print(mean)
    sigmas = complete_table['sigma']
    dist_sigmas = set()
    for sigma in sigmas:
        dist_sigmas.add(sigma)

    for sigma in sorted(list(dist_sigmas)):
        sigma_value = complete_table.loc[complete_table['sigma'] == sigma]

        figures3d(sigma_value, [1, 2, 3, 5, 10, 15, 20, 30, 40, 50], r"\Huge Dependence of Renyi entropy on sample size $\sigma={}$".format(sigma), r"\Large $\frac{H_{\alpha}}{D}$", "mean Renyi entropy", [12, 12, 10, 10, 9, 9, 7, 7, 7, 7], "entropy_mean_{}".format(sigma), "eps", dpi=600, scale=lambda N: N)
        figures3d(sigma_value, [1, 2, 3, 5, 10, 15, 20, 30, 40, 50], r"\Huge Dependence of std of Renyi entropy on sample size $\sigma={}$".format(sigma), r"\Large $\frac{H_{\alpha}}{\sqrt{D}}$", "std Renyi entropy", [12, 12, 10, 10, 9, 9, 7, 7, 7, 7], "entropy_std_{}".format(sigma), "eps", view=(30, 120), dpi=600, scale=lambda N: np.sqrt(N))
        figures3d(sigma_value, [1, 2, 3, 5, 10, 15, 20, 30, 40, 50], r"\Huge Dependence of mean difference on sample size $\sigma={}$".format(sigma), r"$\Large \frac{H_{\alpha}}{D}$", "mean difference", [12, 12, 10, 10, 9, 9, 7, 7, 7, 7], "difference_mean_{}".format(sigma), "eps", dpi=600, scale=lambda N: N)
        figures3d(sigma_value, [1, 2, 3, 5, 10, 15, 20, 30, 40, 50], r"\Huge Dependence of std of difference on sample size $\sigma={}$".format(sigma), r"$\Large \frac{H_{\alpha}}{\sqrt{D}}$", "std of difference", [12, 12, 10, 10, 9, 9, 7, 7, 7, 7], "difference_std_{}".format(sigma), "eps", view=(30, 120), dpi=600, scale=lambda N: np.sqrt(N))

    sigma_value = sigma_value.loc[sigma_value["dimension"] == 2][['alpha', 'sample size', 'mean Renyi entropy']]
    print(sigma_value)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # For each set of style and range settings, plot n random points in
    # the box
    # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
    colors = ["r", "o"]
    markers = ['b', '^']
    dimension = [1, 2]

    ax.set_title(r"Dependence of Renyi entropy on sample size")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$N$")
    ax.set_zlabel(r"$H_{\alpha}$")

    for color, markers, table in zip(colors, markers, dimension):
        xs = sigma_value['alpha']
        ys = sigma_value['sample size']
        zs = sigma_value['mean Renyi entropy']
        #ax.scatter(xs, np.log10(ys), zs, c="r", marker='o')
        #ax.scatter(xs.values, np.log10(ys.values), zs.values, c="r", marker='o')

        #ax.plot_wireframe(np.reshape(xs.values, (-1, 12)), np.reshape(np.log10(ys.values), (-1, 12)), np.reshape(zs.values, (-1, 12)), rstride=1, cstride=1)
        surf = ax.plot_surface(np.reshape(xs.values, (-1, 12)), np.reshape(np.log10(ys.values), (-1, 12)),
                          np.reshape(zs.values, (-1, 12)), rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
        #ax.plot(xs, ys, zs)
        # , logy=True

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    # Plot a basic wireframe.
    #ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)
    #ax.yaxis._set_scale('log')
    #ax.set_yscale("log")
    #ax.set_xlabel(r'$\alpha$')
    #ax.set_ylabel('N')
    #ax.set_zlabel(r'$S_alpha$')

    #plt.savefig('books_read.png', dpi=600)
    #plt.draw()
    #plt.show()
    print("After")
