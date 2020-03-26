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


def figures3d_TE(dataset, selector, title, zlabel, filename, suffix, view=(70, 280), dpi=300):
    fig = plt.figure(figsize=(13, 8))
    ax = Axes3D(fig)

    colors = ["r", "g", "b", "c", "m", "y", "k", "orange", "pink"]
    markers = ['b', '^']

    ax.set_title(title)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\varepsilon$")
    ax.set_zlabel(zlabel)
    # ax.set_yticks([1, 2, 3, 4, 5], ["10", "100", "1000", "10000", "100000"])
    # plt.yticks((1.0, 2.0, 3.0, 4.0, 5.0), ("10", "100", "1000", "10000", "100000"))

    row_size = len(dataset['epsilon'].unique())
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


def figures2d_TE(dataset, selector, title, ylabel, filename, suffix, view=(70, 120), dpi=300):
    matplotlib.style.use("seaborn")

    color_map = matplotlib.cm.get_cmap("summer")

    fig = plt.figure(figsize=(13, 8))
    ax = fig.add_subplot(1, 1, 1)

    markers = ['b', '^']

    ax.set_title(title)
    ax.set_xlabel(r"$\varepsilon$")
    ax.set_ylabel(ylabel)
    # ax.set_yticks([1, 2, 3, 4, 5], ["10", "100", "1000", "10000", "100000"])
    # plt.yticks((1.0, 2.0, 3.0, 4.0, 5.0), ("10", "100", "1000", "10000", "100000"))

    alphas = dataset['alpha'].unique()
    mean = int(len(alphas) / 2)
    neghborhood = 5
    subselected_alphas = alphas[mean - neghborhood:  mean + neghborhood]

    for alpha in subselected_alphas:
        subselection = dataset.loc[dataset["alpha"] == alpha]
        ys = subselection[['epsilon']]
        zs = subselection[[selector]]

        trasform = lambda alpha: (alpha - min(subselected_alphas)) / (max(subselected_alphas) - min(subselected_alphas))
        color = color_map(trasform(alpha))
        row_size = 100
        try:
            ax.plot(ys.values, zs.values, color=color, linewidth=3, label=r'$\alpha={}$'.format(round(alpha, 3)))
        except Exception as exc:
            print(f"{exc}: Problem D=")

    # Add a color bar which maps values to colors.
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.legend(loc=4)

    plt.savefig(filename + "." + suffix, dpi=dpi)
    # plt.draw()
    # plt.show()
    plt.close()


def process_datasets(processed_datasets, result_dataset, new_columns_base_name="transfer_entropy"):
    files = glob.glob(processed_datasets)
    print(files)
    frames = []
    for file in files:
        epsilon = float(file.split("-")[1].split(".b")[0])
        path = Path(file)

        # with open(path, "rb") as fh:
        #    table = pickle.load(fh)
        table = pd.read_pickle(path)

        frame = pd.DataFrame(table)
        frame["epsilon"] = epsilon

        # print(frame)

        old_columns = frame.columns
        for item in old_columns[:-1]:
            mean_column_name = f"{new_columns_base_name}_{item[1]}_{item[2]}"
            std_column_name = f"{new_columns_base_name}_{item[1]}_{item[2]}"

            # add mean of entropy
            frame[mean_column_name, "mean", "", item[3]] = frame.apply(lambda row: np.mean(row[item]), axis=1, raw=True)

            # add std of entropy
            frame[std_column_name, "std", "", item[3]] = frame.apply(lambda row: np.std(row[item]), axis=1, raw=True)

        # effective transfer entropy
        for item in [item for item in frame.columns.tolist() if item[3] == False and "entropy" not in str(item[0])]:
            mean_column_name = f"effective_{new_columns_base_name}_{item[1]}_{item[2]}"
            std_column_name = f"effective_{new_columns_base_name}_{item[1]}_{item[2]}"

            frame[mean_column_name, "mean", "", ""] = frame.apply(lambda row: np.mean(row[item]) - np.mean(row[item[0], item[1], item[2], True]), axis=1,
                                                                  raw=True)
            frame[mean_column_name, "std", "", ""] = frame.apply(lambda row: np.std(row[item]) + np.std(row[item[0], item[1], item[2], True]), axis=1, raw=True)

        # dropping the index
        frame = frame.reset_index()

        # print(frame.columns.tolist())
        column = [("alpha", "", "") if "index" == item[0] else item for item in frame.columns.tolist()]
        new_columns = pd.MultiIndex.from_tuples([("alpha", "", "", "") if "index" == item[0] else item for item in frame.columns])
        frame.columns = new_columns

        # give names to the columns
        # new_columns = [f"{new_columns_base_name}_{item[1]}_{item[2]}" for item in old_columns[:-1]]
        # column_names = ["alpha"]
        # column_names.extend(new_columns)
        # column_names.append("epsilon")
        # .append(["epsilon"])
        # frame.columns = column_names

        # selection of columns
        columns = [item for item in frame.columns.tolist() if
                   "mean" in str(item[1]) or "std" in str(item[1]) or "alpha" in str(item[0]) or "epsilon" in str(item[0])]
        frame = frame[columns]
        # print(frame)

        # append frame for processing
        frames.append(frame)

    # join the table
    join_table = pd.concat(frames, ignore_index=True)
    #print(join_table)
    pivot_table = pd.pivot_table(join_table, index=['alpha', 'epsilon'])
    print(pivot_table, join_table.columns.tolist())

    #print(pivot_table[["transfer_entropy_15_5_mean"]])
    TE = pivot_table.reset_index()

    TE.to_pickle(result_dataset)

    return TE, [item for item in join_table.columns.tolist() if "mean" in str(item[1])]


def load_processed_dataset(dataset, new_columns_base_name="transfer_entropy_"):
    TE = pd.read_pickle(dataset)
    columns = TE.columns

    return TE, [item for item in TE.columns.tolist() if "mean" in str(item[1])]


if __name__ == "__main__":
    processed_dataset = "transfer_entropy/pivot_dataset.bin"
    files = glob.glob(processed_dataset)
    if len(files) == 0:
        TE, TE_column_names = process_datasets("transfer_entropy/Transfer_entropy_dataset-*.bin", processed_dataset)
    else:
        TE, TE_column_names = load_processed_dataset(processed_dataset)

    for item in TE_column_names:
        if "effective" in item[0]:
            m = item[0].split("_")[3]
            l = item[0].split("_")[4]
        else:
            m = item[0].split("_")[2]
            l = item[0].split("_")[3]

        label = "$T^{}_{} ({},{})$".format("{(R, eff)}" if "effective" in item[0] else "{(R)}",
                                           r"{\alpha: Y_{shuffled}\rightarrow X}" if item[3] else r"{\alpha: Y\rightarrow X}", m, l)
        figures2d_TE(TE, item, r"$\large\rm{Transfer\ entropy}$", label, item[0] + ("_shuffled" if item[3] else "") + "_2d", "pdf")
        figures3d_TE(TE, item, r"$\large\rm{Transfer\ entropy}$", label, item[0] + ("_shuffled" if item[3] else ""), "pdf")
