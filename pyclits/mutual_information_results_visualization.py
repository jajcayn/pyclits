'''
==============
3D scatterplot
==============

Demonstration of a basic scatterplot in 3D.
'''

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from matplotlib import rc

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

if __name__ == "__main__":
    list_of_tables = []
    for root, dirs, files in os.walk("."):
        for filename in files:
            if "statistics" in filename:
                print(filename)
                colnames = ['alpha', 'sample size', 'sigma', 'mean Renyi entropy', 'variance Renyi entropy', 'mean computer time', 'variance computer time', 'mean difference', 'variance of difference']
                table = pd.read_csv(filepath_or_buffer=filename, sep=" ", names=colnames, header=None)
                number = int(filename.split("_")[1].split(".")[0])
                table["dimension"] = number
                list_of_tables.append(table)
                #print(table)
                #print(table.reindex(columns=['alpha', 'sample size', 'sigma', 'dimension']))

    complete_table = pd.concat(list_of_tables, join="inner")
    complete_table_indexed = complete_table.set_index(['dimension', 'sigma', 'alpha', 'sample size'])

    print(complete_table)
    mean = complete_table_indexed.loc[(2,), "mean Renyi entropy"]
    print(mean)

    sigma_value = complete_table.loc[complete_table['sigma'] == 0.1]
    sigma_value = sigma_value.loc[sigma_value["dimension"] == 2][['alpha', 'sample size', 'mean Renyi entropy']]

    print(sigma_value)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # For each set of style and range settings, plot n random points in the box
    # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
    colors = ["r", "o"]
    markers = ['b', '^']
    dimension = [1, 2]

    for color, markers, table in zip(colors, markers, dimension):
        xs = sigma_value[['alpha']]
        ys = sigma_value[['sample size']]
        zs = sigma_value[['mean Renyi entropy']]
        ax.scatter(xs, ys, zs, c="r", marker='o', logy=True)

    ax.yaxis.set_scale('log')
    #ax.set_xlabel(r'$\alpha$')
    #ax.set_ylabel('N')
    #ax.set_zlabel(r'$S_alpha$')

    plt.show()
