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


def figures3d_TE(dataset, selector, title, zlabel, filename, suffix, view=(50, -20), dpi=300):
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


def figures2d_TE(dataset, selector, title, ylabel, filename, suffix, view=(40, 40), dpi=300):
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


def figures2d_TE_errorbar(dataset, selector, error_selector, title, ylabel, filename, suffix, view=(70, 120), dpi=300):
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
        error_bar = subselection[[error_selector]]

        print(error_bar)
        error_bar["-std"] = error_bar.apply(lambda x: -x, axis=1, raw=True)
        errors = error_bar.values[:].T

        trasform = lambda alpha: (alpha - min(subselected_alphas)) / (max(subselected_alphas) - min(subselected_alphas))
        color = color_map(trasform(alpha))
        row_size = 100
        try:
            ax.errorbar(ys.values, zs.values, yerr=errors, color=color, linewidth=3,
                        label=r'$\alpha={}$'.format(round(alpha, 3)))
        except Exception as exc:
            print(f"{exc}: Problem D=")

    # Add a color bar which maps values to colors.
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.legend(loc=4)

    plt.savefig(filename + "." + suffix, dpi=dpi)
    plt.close()


def process_datasets(processed_datasets, result_dataset, result_raw_dataset, new_columns_base_name="transfer_entropy"):
    files = glob.glob(processed_datasets)
    print(files)
    frames = []
    frames_raw = []
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

            if isinstance(item[3], bool):
                bool_column = 3
            else:
                bool_column = 4

            # add mean of entropy
            calculation = frame.apply(lambda row: float(np.mean(row[item])), axis=1, raw=True)
            if bool_column == 3:
                frame[mean_column_name, "mean", "", item[bool_column]] = calculation
            else:
                frame[mean_column_name, "mean", "", "", item[bool_column]] = calculation

            # add std of entropy
            calculation = frame.apply(lambda row: float(np.std(row[item])), axis=1, raw=True)
            if bool_column == 3:
                frame[std_column_name, "std", "", item[bool_column]] = calculation
            else:
                frame[mean_column_name, "mean", "", "", item[bool_column]] = calculation

        # effective transfer entropy
        for item in [item for item in frame.columns.tolist() if item[bool_column] is False and "entropy" not in str(item[0])]:
            mean_column_name = f"effective_{new_columns_base_name}_{item[1]}_{item[2]}"
            std_column_name = f"effective_{new_columns_base_name}_{item[1]}_{item[2]}"

            if bool_column == 3:
                frame[mean_column_name, "mean", "", ""] = frame.apply(lambda row: float(np.mean(row[item]) - np.mean(row[item[0], item[1], item[2], True])),
                                                                      axis=1,
                                                                      raw=True)
                frame[mean_column_name, "std", "", ""] = frame.apply(lambda row: float(np.std(row[item]) + np.std(row[item[0], item[1], item[2], True])),
                                                                     axis=1,
                                                                     raw=True)
            else:
                frame[mean_column_name, "mean", "", "", ""] = frame.apply(
                    lambda row: float(np.mean(row[item]) - np.mean(row[item[0], item[1], item[2], item[3], True])), axis=1, raw=True)
                frame[mean_column_name, "std", "", "", ""] = frame.apply(
                    lambda row: float(np.std(row[item]) + np.std(row[item[0], item[1], item[2], item[3], True])),
                    axis=1, raw=True)

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
        frame_with_processed_results = frame[columns]

        columns = [item for item in frame.columns.tolist() if
                   isinstance(item[0], float) or "alpha" in str(item[0]) or "epsilon" in str(item[0])]
        frame_with_raw_results = frame[columns]
        # print(frame)
        # if item[0] not in ["alpha", "epsilon"] else item[0:3]
        columns = [str(item[1]) + "_" + str(item[2]) + "_" + str(item[3]) if isinstance(item[0], float) else item[0] for item in
                   frame_with_raw_results.columns.tolist()]
        frame_with_raw_results.columns = columns

        # append frame for processing
        frames.append(frame_with_processed_results)
        frames_raw.append(frame_with_raw_results)

    # join the table
    join_table = pd.concat(frames, ignore_index=True)
    join_table_raw = pd.concat(frames_raw, ignore_index=True)

    # print(join_table)
    index_alpha = join_table.columns.tolist()
    pivot_table = pd.pivot_table(join_table, index=[index_alpha[0], index_alpha[1]])
    print(pivot_table, join_table.columns.tolist())

    print(join_table_raw)
    index_alpha = join_table_raw.columns.tolist()
    pivot_table_raw = join_table_raw.set_index([index_alpha[0], index_alpha[-1]])
    # pd.pivot_table(join_table_raw, index=[index_alpha[0], index_alpha[1]])
    print(pivot_table_raw)

    # print(pivot_table[["transfer_entropy_15_5_mean"]])
    TE = pivot_table.reset_index()
    TE_raw = pivot_table_raw.reset_index()

    TE.to_pickle(result_dataset)
    TE_raw.to_pickle(result_raw_dataset)

    return TE, [item for item in join_table.columns.tolist() if "mean" in str(item[1])], TE_raw


def figures2d_samples_TE(dataset, selector, title, ylabel, filename, suffix, view=(70, 120), dpi=300):
    matplotlib.style.use("seaborn")

    color_map = matplotlib.cm.get_cmap("summer")

    # ax.set_yticks([1, 2, 3, 4, 5], ["10", "100", "1000", "10000", "100000"])
    # plt.yticks((1.0, 2.0, 3.0, 4.0, 5.0), ("10", "100", "1000", "10000", "100000"))

    alphas = dataset['alpha'].unique()
    epsilons = dataset['epsilon'].unique()
    subselection = dataset.loc[dataset["alpha"] == alphas[0]]
    subselection = subselection.loc[subselection["epsilon"] == epsilons[0]]

    one_subselection = subselection[[selector]]
    number_of_samples = len(subselection[[selector]].values[0, 0])
    mean = int(len(alphas) / 2)
    neghborhood = 5
    subselected_alphas = alphas[mean - neghborhood:  mean + neghborhood]

    for sample in range(number_of_samples):
        fig = plt.figure(figsize=(13, 8))
        ax = fig.add_subplot(1, 1, 1)

        markers = ['b', '^']

        ax.set_title(title)
        ax.set_xlabel(r"$\varepsilon$")
        ax.set_ylabel(ylabel)

        for alpha in subselected_alphas:
            subselection = dataset.loc[dataset["alpha"] == alpha]
            subselection.sort_values(by=['epsilon'], inplace=True)
            # print(subselection)
            ys = subselection[['epsilon']]
            zs = subselection[[selector]]

            trasform = lambda alpha: (alpha - min(subselected_alphas)) / (max(subselected_alphas) - min(subselected_alphas))
            color = color_map(trasform(alpha))
            row_size = 100
            try:
                ax.plot(ys.values, [float(item[0][sample]) for item in zs.values], color=color, linewidth=3, label=r'$\alpha={}$'.format(round(alpha, 3)))
            except Exception as exc:
                print(f"{exc}: Problem D=")

        # Add a color bar which maps values to colors.
        # fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.legend(loc=4)

        plt.savefig(filename.format(sample) + "." + suffix, dpi=dpi)
        # plt.draw()
        # plt.show()
        plt.close()


def load_processed_dataset(dataset, dataset_raw, new_columns_base_name="transfer_entropy_"):
    TE = pd.read_pickle(dataset)
    columns = TE.columns

    TE_raw = pd.read_pickle(dataset_raw)

    return TE, [item for item in TE.columns.tolist() if "mean" in str(item[1])], TE_raw


if __name__ == "__main__":
    directory = "transfer_entropy"
    processed_dataset = directory + "/pivot_dataset.bin"
    processed_raw_dataset = directory + "/pivot_dataset_raw.bin"
    files = glob.glob(processed_dataset)
    if len(files) == 0:
        TE, TE_column_names, TE_raw = process_datasets(directory + "/Transfer_entropy_dataset-*.bin", processed_dataset, processed_raw_dataset)
    else:
        TE, TE_column_names, TE_raw = load_processed_dataset(processed_dataset, processed_raw_dataset)

    # figures2d_samples_TE(TE_raw, "0,5,10_4_0_False", r"$\large\rm{Transfer\ entropy - samples}$", "", "TE_sample_0,5,10_1_0_{}", "pdf")
    # figures2d_samples_TE(TE_raw, "0,5,10_4_0_True", r"$\large\rm{Transfer\ entropy\ shuffled - samples}$", "", "TE_sample_shuffled_0,5,10_1_0_{}", "pdf")

    for item in TE_column_names:
        item_error = list(item)
        item_error[1] = "std"
        if "effective" in item[0]:
            m = item[0].split("_")[3]
            l = item[0].split("_")[4]
        else:
            m = item[0].split("_")[2]
            l = item[0].split("_")[3]

        label = "$T^{}_{} ({},{})$".format("{(R, eff)}" if "effective" in item[0] else "{(R)}",
                                           r"{\alpha: Y_{shuffled}\rightarrow X}" if item[3] else r"{\alpha: Y\rightarrow X}", m, l)
        figures2d_TE(TE, item, r"$\large\rm{Transfer\ entropy}$", label, item[0] + ("_shuffled" if item[3] else "") + "_2d", "pdf")
        figures2d_TE_errorbar(TE, item, tuple(item_error), r"$\large\rm{Transfer\ entropy}$", label, item[0] + ("_shuffled" if item[3] else "") + "_2d_bars",
                              "pdf")
        figures3d_TE(TE, item, r"$\large\rm{Transfer\ entropy}$", label, item[0] + ("_shuffled" if item[3] else ""), "pdf")

        figures2d_TE(TE, tuple(item_error), r"$\large\rm{Transfer\ entropy - std}$", label, item[0] + ("_shuffled" if item[3] else "") + "_2d_std", "pdf")
