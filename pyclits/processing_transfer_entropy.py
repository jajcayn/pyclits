#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import sys
import traceback
from collections import Counter
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
})

def figures3d_TE(dataset, selector, title, zlabel, filename, suffix, view=(50, -20), dpi=300):
    fig = plt.figure(figsize=(13, 8))
    #ax = Axes3D(fig)
    ax = fig.add_subplot(1, 1, 1, projection='3d')

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
    #plt.show()
    plt.close()
    del fig


def figures3d_surface_TE(dataset, selector, title, zlabel, filename, suffix, cmap="magma", view=(50, -20), dpi=300):
    fig = plt.figure(figsize=(13, 8))
    #ax = Axes3D(fig)
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    colors = ["r", "g", "b", "c", "m", "y", "k", "orange", "pink"]
    markers = ['b', '^']

    ax.set_title(title)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\varepsilon$")
    ax.set_zlabel(zlabel)

    row_size = len(dataset['epsilon'].unique())
    xs = dataset[['alpha']]
    ys = dataset[['epsilon']]
    zs = dataset[[selector]]

    try:
        ax.plot_surface(np.reshape(xs.values, (-1, row_size)),
                        np.reshape(ys.values, (-1, row_size)),
                        np.reshape(zs.values, (-1, row_size)),
                        rstride=1,
                        cstride=1,
                        cmap=cmap,
                        linewidth=0,
                        antialiased=False)
    except Exception as exc:
        print(f"{exc}: Problem D=")

    # Add a color bar which maps values to colors.
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.legend(loc=1)
    ax.view_init(view[0], view[1])

    plt.savefig(filename + "." + suffix, dpi=dpi)
    # plt.draw()
    #plt.show()
    plt.close()
    del fig


def minimal_difference(target):
    epsilon_differences = []
    for item in range(0, len(target) - 1):
        epsilon_differences.append(round(target[item + 1] - target[item], 4))
    return min(epsilon_differences)


def figures2d_imshow(dataset, selector, title, ylabel, filename, suffix, cmap="magma",  dpi=300):
    color_map = matplotlib.cm.get_cmap(cmap)

    fig, ax = plt.subplots(1, 1, figsize=(13, 8))

    ax.set(title=title)
    ax.set(xlabel=r"$\varepsilon$")
    ax.set(ylabel=r"$\alpha$")
    ax.grid(True)

    epsilons = dataset['epsilon'].unique()
    alphas = dataset['alpha'].unique()
    epsilon = dataset[['epsilon']]
    alpha = dataset[['alpha']]
    data = dataset[[('epsilon', "", "", "", ""), ('alpha', "", "", "", ""), selector]]
    xs = dataset[['epsilon']].values.reshape((len(alphas), len(epsilons)))
    ys = dataset[['alpha']].values.reshape((len(alphas), len(epsilons)))
    zs = dataset[[selector]].values.reshape((len(alphas), len(epsilons)))
    coords = np.array(list(zip(xs.flatten(), ys.flatten())))

    minimal_epsilon_difference = minimal_difference(epsilons)
    changed_epsilons = np.arange(epsilons[0], epsilons[-1], minimal_epsilon_difference)
    flatten_zs = zs.flatten()

    grid = np.dstack(np.meshgrid(changed_epsilons, alphas)).reshape(-1, 2)
    sampled_data = griddata(coords, flatten_zs, grid, method='nearest')

    number_epsilons = len(epsilons)
    number_alphas = len(alphas)
    epsilon_margin = (max(epsilons) - min(epsilons)) / (number_epsilons * 2.0)
    alpha_margin = (max(alphas) - min(alphas)) / (number_alphas * 2.0)
    extent = [min(epsilons)-epsilon_margin, max(epsilons)+epsilon_margin, min(alphas)-alpha_margin, max(alphas)+alpha_margin]
    ims = ax.imshow(sampled_data.reshape((len(alphas), len(changed_epsilons))), origin="lower", interpolation='nearest', extent=extent, cmap=color_map, aspect='auto')

    fig.colorbar(ims)
    plt.savefig(filename + "." + suffix, dpi=dpi)
    #plt.show()
    plt.close()
    del fig


def figures2d_TE(dataset, selector, title, ylabel, filename, suffix, cmap="rainbow", dpi=300):
    matplotlib.style.use("seaborn")

    color_map = matplotlib.cm.get_cmap(cmap)

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
    #subselected_alphas = alphas[mean - neghborhood:  mean + neghborhood]
    subselected_alphas = [alpha for number, alpha in enumerate(alphas) if (0.70 <= alpha <= 2 and number % 2 == 0)]

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
    plt.legend(loc=0, ncol=3)

    plt.savefig(filename + "." + suffix, dpi=dpi)
    plt.close()
    del fig


def figures2d_TE_errorbar(dataset, selector, error_selector, title, ylabel, filename, suffix, view=(70, 120), cmap="rainbow", dpi=300):
    matplotlib.style.use("seaborn")

    color_map = matplotlib.cm.get_cmap(cmap)

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
    #subselected_alphas = alphas[mean - neghborhood:  mean + neghborhood]
    subselected_alphas = [alpha for number, alpha in enumerate(alphas) if (0.70 <= alpha <= 2 and number % 2 == 0)]

    for alpha in subselected_alphas:
        subselection = dataset.loc[dataset["alpha"] == alpha]
        ys = subselection[['epsilon']]
        zs = subselection[[selector]]
        error_bar = subselection[[error_selector]].copy()

        error_selector_negative_std = list(error_selector)
        error_selector_negative_std[1] = "-std"
        # error_bar[tuple(error_selector_negative_std)] = error_bar.apply(lambda x: -x, axis=1, raw=True)
        errors = error_bar.copy().T.to_numpy()

        trasform = lambda alpha: (alpha - min(subselected_alphas)) / (max(subselected_alphas) - min(subselected_alphas))
        color = color_map(trasform(alpha))
        row_size = 100
        try:
            label = r"${\tiny" + rf"\alpha={round(alpha, 3)}" + r"}$"
            ax.errorbar(ys.values.flatten(), zs.values.flatten(), yerr=errors.flatten(), color=color, linewidth=3,
                        label=label)
        except Exception as exc:
            print(f"{exc}: {errors.shape}")

    # Add a color bar which maps values to colors.
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.legend(loc=0, ncol=3)

    plt.savefig(filename + "." + suffix, dpi=dpi)
    plt.close()
    del fig


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
            reversed_order = item[4]
            mean_column_name = f"{new_columns_base_name}_{item[1]}_{item[2]}"
            std_column_name = f"{new_columns_base_name}_{item[1]}_{item[2]}"

            if isinstance(item[3], bool):
                bool_column = 3
            else:
                bool_column = 4

            # add mean of entropy
            calculation = frame.apply(lambda row: float(np.mean(row[item])), axis=1, raw=True)
            if bool_column == 3:
                frame[mean_column_name, "mean", "", item[bool_column], reversed_order] = calculation
            else:
                frame[mean_column_name, "mean", "", "", item[bool_column], reversed_order] = calculation

            # add std of entropy
            calculation = frame.apply(lambda row: float(np.std(row[item])), axis=1, raw=True)
            if bool_column == 3:
                frame[std_column_name, "std", "", item[bool_column], reversed_order] = calculation
            else:
                frame[mean_column_name, "std", "", "", item[bool_column], reversed_order] = calculation

        # effective transfer entropy
        column_to_use = [item for item in frame.columns.tolist() if
                         item[bool_column] is False and not ("entropy" in str(item[0]) or "information" in str(item[0]))]
        for item in column_to_use:
            mean_column_name = f"effective_{new_columns_base_name}_{item[1]}_{item[2]}"
            std_column_name = f"effective_{new_columns_base_name}_{item[1]}_{item[2]}"

            if bool_column == 3:
                frame[mean_column_name, "mean", "", False, item[4]] = frame.apply(
                    lambda row: float(np.mean(row[item]) - np.mean(row[item[0], item[1], item[2], not item[3], item[4]])),
                    axis=1,
                    raw=True)
                frame[std_column_name, "std", "", False, item[4]] = frame.apply(
                    lambda row: float(np.std(row[item]) + np.std(row[item[0], item[1], item[2], not item[3], item[4]])),
                    axis=1,
                    raw=True)
            else:
                frame[mean_column_name, "mean", "", False, item[4]] = frame.apply(
                    lambda row: float(np.mean(row[item]) - np.mean(row[item[0], item[1], item[2], item[3], not item[4]])), axis=1, raw=True)
                frame[std_column_name, "std", "", False, item[4]] = frame.apply(
                    lambda row: float(np.std(row[item]) + np.std(row[item[0], item[1], item[2], item[3], not item[4]])),
                    axis=1, raw=True)

        # balance of entropy
        balance_names = [item for item in frame.columns.tolist() if not bool(item[4]) and "information" not in str(item[0]) and "epsilon" not in str(item[0])]
        for item in balance_names:
            mean_column_name = f"balance_{new_columns_base_name}_{item[1]}_{item[2]}"
            std_column_name = f"balance_{new_columns_base_name}_{item[1]}_{item[2]}"

            frame[mean_column_name, "mean", "", item[3], False] = frame.apply(
                lambda row: float(np.mean(row[item]) - np.mean(row[item[0], item[1], item[2], item[3], not item[4]])), axis=1, raw=True)
            frame[std_column_name, "std", "", item[3], False] = frame.apply(
                lambda row: float(np.std(row[item]) + np.std(row[item[0], item[1], item[2], item[3], not item[4]])),
                axis=1, raw=True)

        # balance of effective entropy
        balance_names = [item for item in frame.columns.tolist() if not bool(item[4]) and not bool(item[3]) and "information" not in str(item[0]) and "epsilon" not in str(item[0])]
        for item in balance_names:
            mean_column_name = f"balance_effective_{new_columns_base_name}_{item[1]}_{item[2]}"
            std_column_name = f"balance_effective_{new_columns_base_name}_{item[1]}_{item[2]}"

            frame[mean_column_name, "mean", "", item[3], False] = frame.apply(
                lambda row: float(np.mean(row[item]) - np.mean(row[item[0], item[1], item[2], not item[3], item[4]]) - np.mean(row[item[0], item[1], item[2], item[3], not item[4]]) + np.mean(row[item[0], item[1], item[2], not item[3], not item[4]])), axis=1, raw=True)
            frame[std_column_name, "std", "", item[3], False] = frame.apply(
                lambda row: float(np.std(row[item]) + np.std(row[item[0], item[1], item[2], not item[3], item[4]]) + np.std(row[item[0], item[1], item[2], item[3], not item[4]]) + np.std(row[item[0], item[1], item[2], not item[3], not item[4]])),
                axis=1, raw=True)

        # dropping the index
        frame = frame.reset_index()

        # print(frame.columns.tolist())
        column = [("alpha", "", "") if "index" == item[0] else item for item in frame.columns.tolist()]
        new_columns = pd.MultiIndex.from_tuples([("alpha", "", "", "", "") if "index" == item[0] else item for item in frame.columns])
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
    try:
        join_table_raw = pd.concat(frames_raw, ignore_index=True)
    except:
        print("Problem with columns")
        first_frame = frames_raw[0]
        for file, frame in zip(files, frames_raw):
            comparison = len(frame.columns.tolist()) == len(first_frame.columns.tolist())
            if not comparison:
                print(file, frame.columns, comparison)
        sys.exit(1)

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


def figures2d_samples_TE(dataset, selector, title, ylabel, filename, suffix, view=(70, 120), cmap="rainbow", dpi=300):
    matplotlib.style.use("seaborn")

    color_map = matplotlib.cm.get_cmap(cmap)

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
    dpi = 150
    output = "png"
    # "conditional_information_transfer", "conditional_information_transfer_Dh=1", "conditional_information_transfer_Dh=2", "conditional_information_transfer_Dh=n",
    #                       "conditional_information_transfer_full_Dh=1", "conditional_information_transfer_full_Dh=2", "conditional_information_transfer_full_Dh=n",
    #                    "conditional_information_transfer_GARCH_single", "roessler_oscilator/conditional_information_transfer_X_3_Y_3", "roessler_oscilator/conditional_information_transfer_X_3_Y_1", "roessler_oscilator/conditional_information_transfer_X_1_Y_3", "roessler_oscilator/conditional_information_transfer_X_2_Y_2"
    directories = ["conditional_information_transfer"]
    #directory = "transfer_entropy"

    for directory in directories:
        name_of_title = "conditional_information_transfer"
        processed_dataset = directory + "/pivot_dataset.bin"
        processed_raw_dataset = directory + "/pivot_dataset_raw.bin"
        files = glob.glob(processed_dataset)
        if len(files) == 0:
            TE, TE_column_names, TE_raw = process_datasets(processed_datasets=directory + "/Conditional_information_transfer-*.bin",
                                                           result_dataset=processed_dataset, result_raw_dataset=processed_raw_dataset,
                                                           new_columns_base_name=name_of_title)
        else:
            TE, TE_column_names, TE_raw = load_processed_dataset(processed_dataset, processed_raw_dataset)

        names = {"balance_effective_conditional_information_transfer": 5, "balance_conditional_information_transfer": 4, "effective_conditional_information_transfer": 4, "conditional_information_transfer": 3}
        for item in TE_column_names:
            try:
                shift = 0
                for key, value in names.items():
                    if key in item[0]:
                        shift = value
                        break

                item_error = list(item)
                column_name = item[0]
                shuffled_calculation = item[3]
                swapped_datasets = item_error[4]
                item_error[1] = "std"

                history_first_TS = column_name.split("_")[shift]
                history_second_TS = column_name.split("_")[shift+1]
                try:
                    future_first_TS = column_name.split("_")[shift+2]
                except IndexError as err:
                    future_first_TS = None

                name_of_title = column_name.split("r_")[0]+"r"
                balance = "balance" in name_of_title
                latex_title = r"{\Large{" + name_of_title.capitalize().replace("_", " ") + r"}}"
                latex_title_std = latex_title + r"$\large\rm{\ -\ std}$"
                latex_title = None
                latex_title_std = None
                title_graph = {"transfer_entropy": r"$\Large\rm{Transfer\ entropy}$",
                               "conditional_information_transfer": r"$\Large\rm{Conditional\ information\ transfer}$", }
                filename_direction = {True: "Y->X", False: "X->Y"}
                title_map = {(False, False): r"{\alpha: X\rightarrow Y}", (True, False): r"{\alpha: X_{shuffled}\rightarrow Y}",
                             (False, True): r"{\alpha: Y\rightarrow X}", (True, True): r"{\alpha: Y_{shuffled}\rightarrow X}"}

                if future_first_TS is not None:
                    if balance:
                        label = "$T^{}_{} ([{}],[{}],[{}])$".format("{(R, eff)}" if "effective" in column_name else "{(R)}",
                                                                    title_map[(shuffled_calculation, swapped_datasets)], history_first_TS,
                                                                    history_second_TS, future_first_TS) + "-" + \
                                "$T^{}_{} ([{}],[{}],[{}])$".format("{(R, eff)}" if "effective" in column_name else "{(R)}",
                                                                    title_map[(shuffled_calculation, not swapped_datasets)], history_first_TS,
                                                                    history_second_TS, future_first_TS)
                    else:
                        label = "$T^{}_{} ([{}],[{}],[{}])$".format("{(R, eff)}" if "effective" in column_name else "{(R)}",
                                                                    title_map[(shuffled_calculation, swapped_datasets)],
                                                                    history_first_TS, history_second_TS, future_first_TS)
                else:
                    label = "$T^{}_{} ([{}],[{}])$".format("{(R, eff)}" if "effective" in column_name else "{(R)}",
                                                           title_map[(shuffled_calculation, swapped_datasets)], history_first_TS, history_second_TS)

                print(column_name, label)

                errorbar_filename = directory + "/" + column_name + "_" + filename_direction[swapped_datasets] + ("_shuffled" if shuffled_calculation else "") + "_2d_bars"
                standard_filename = directory + "/" + column_name + "_" + filename_direction[swapped_datasets] + ("_shuffled" if shuffled_calculation else "") + "_2d"
                plot_3D_filename = directory + "/" + column_name + "_" + filename_direction[swapped_datasets] + ("_shuffled" if shuffled_calculation else "")
                plot_3D_surf_filename = directory + "/" + column_name + "_" + filename_direction[swapped_datasets] + ("_shuffled_surf" if shuffled_calculation else "_surf")
                plot_2D_filename_implot = directory + "/" + column_name + "_" + filename_direction[swapped_datasets] + ("_shuffled" if shuffled_calculation else "") + "_implot"
                plot_2D_filename_implot_std = directory + "/" + column_name + "_" + filename_direction[swapped_datasets] + ("_shuffled" if shuffled_calculation else "") + "_implot_std"
                std_filename = directory + "/" + column_name + "_" + filename_direction[swapped_datasets] + ("_shuffled" if shuffled_calculation else "") + "_2d_std"

                figures2d_imshow(TE, item, latex_title, label, plot_2D_filename_implot, output, cmap="rainbow", dpi=dpi)
                figures2d_imshow(TE, tuple(item_error), latex_title, label, plot_2D_filename_implot_std, output, cmap="rainbow", dpi=dpi)
                figures3d_surface_TE(TE, item, latex_title, label, plot_3D_surf_filename, output, cmap="rainbow", dpi=dpi)
                figures2d_TE_errorbar(TE, item, tuple(item_error), latex_title, label, errorbar_filename, output, dpi=dpi)
                figures2d_TE(TE, item, latex_title, label, standard_filename, output, dpi=dpi)
                figures3d_TE(TE, item, latex_title, label, plot_3D_filename, output, dpi=dpi)
                figures2d_TE(TE, tuple(item_error), latex_title_std, label, std_filename, output, dpi=dpi)
            except Exception as exc:
                print(f"Problem {exc} {item}")
                traceback.print_exc()

        del TE, TE_column_names, TE_raw

    print("Finished")
