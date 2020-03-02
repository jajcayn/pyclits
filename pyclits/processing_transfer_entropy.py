#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

matplotlib.rcParams['text.usetex'] = True

plt.rc('text', usetex=True)
plt.rc('font', family='serif')


def figures3d_TE(dataset, selector, title, zlabel, filename, suffix, view=(70, 120), dpi=300):
    fig = plt.figure(figsize=(13, 8))
    ax = Axes3D(fig)

    # For each set of style and range settings, plot n random points in
    # the box
    # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
    colors = ["r", "g", "b", "c", "m", "y", "k", "orange", "pink"]
    markers = ['b', '^']

    ax.set_title(title)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\varepsilon$")
    ax.set_zlabel(zlabel)
    # ax.set_yticks([1, 2, 3, 4, 5], ["10", "100", "1000", "10000", "100000"])
    # plt.yticks((1.0, 2.0, 3.0, 4.0, 5.0), ("10", "100", "1000", "10000", "100000"))

    row_size = dataset['alpha'].unique()
    xs = dataset[['alpha']]
    ys = dataset[['epsilon']]
    zs = dataset[[selector]]

    try:
        ax.plot_wireframe(np.reshape(xs.values, (-1, row_size)), np.reshape(ys.values, (-1, row_size)), np.reshape(zs.values, (-1, row_size)),
                          rstride=1, cstride=1, color=colors[0], linewidth=1)
    except Exception as exc:
        print(f"{exc}: Problem D=")

    # Add a color bar which maps values to colors.
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.legend(loc=1)
    ax.view_init(view[0], view[1])

    plt.savefig(filename + "." + suffix, dpi=dpi)
    # plt.draw()
    # plt.show()
    plt.close()


def figures2d_TE(dataset, selector, title, zlabel, filename, suffix, view=(70, 120), dpi=300):
    fig = plt.figure(figsize=(13, 8))
    ax = Axes3D(fig)

    # For each set of style and range settings, plot n random points in
    # the box
    # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
    colors = ["r", "g", "b", "c", "m", "y", "k", "orange", "pink"]
    markers = ['b', '^']

    ax.set_title(title)
    ax.set_xlabel(r"$\varepsilon$")
    ax.set_ylabel(zlabel)
    # ax.set_yticks([1, 2, 3, 4, 5], ["10", "100", "1000", "10000", "100000"])
    # plt.yticks((1.0, 2.0, 3.0, 4.0, 5.0), ("10", "100", "1000", "10000", "100000"))

    xs = dataset['alpha'].unique()

    alpha = 0.1
    subselection = dataset["alpha" == alpha]
    ys = dataset["alpha" == alpha][['epsilon']]
    zs = dataset[[selector]]

    row_size = 100
    try:
        ax.plot(np.reshape(ys.values, (-1, row_size)), np.reshape(zs.values, (-1, row_size)),
                rstride=1, cstride=1, color=colors[0], linewidth=1)
    except Exception as exc:
        print(f"{exc}: Problem D=")

    # Add a color bar which maps values to colors.
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.legend(loc=1)
    ax.view_init(view[0], view[1])

    plt.savefig(filename + "." + suffix, dpi=dpi)
    # plt.draw()
    # plt.show()
    plt.close()


def process_datasets(processed_datasets, result_dataset):
    files = glob.glob(processed_datasets)
    print(files)
    frames = []
    for file in files:
        epsilon = float(file.split("-")[1].split(".b")[0])
        path = Path(file)

        table = pd.read_pickle(path)

        frame = pd.DataFrame(table)
        frame["epsilon"] = epsilon

        old_columns = frame.columns
        frame = frame.reset_index()
        # print(old_columns)

        # give names to the columns
        new_columns = [f"transfer_entropy_{item[1]}" for item in old_columns[:-1]]
        column_names = ["alpha"]
        column_names.extend(new_columns)
        column_names.append("epsilon")
        # .append(["epsilon"])
        frame.columns = column_names

        # reorder columns
        columns = frame.columns.tolist()
        columns = columns[:1] + columns[-1:] + columns[1:-1]
        frame = frame[columns]
        frames.append(frame)

    join_table = pd.concat(frames, ignore_index=True)
    pivot_table = pd.pivot_table(join_table, index=['alpha', 'epsilon'])
    # print(pivot_table)

    print(pivot_table[["transfer_entropy_30"]])
    TE = pivot_table.reset_index()

    pivot_table.to_pickle(result_dataset)

    return TE, new_columns


def processed_dataset(dataset):
    TE = pd.read_pickle(dataset)

    return TE


if __name__ == "__main__":
    processed_dataset = "transfer_entropy/pivot_dataset.bin"
    files = glob.glob(processed_dataset)
    if files:
        TE, TE_column_names = process_datasets("transfer_entropy/Transfer_entropy_dataset-*.bin", processed_dataset)
    else:
        TE = processed_dataset(processed_dataset)

    for item in TE_column_names:
        figures2d_TE(TE, item, "Transfer entropy", "TE", item + "_2d", "png")
        figures3d_TE(TE, item, "Transfer entropy", "TE", item, "png")
