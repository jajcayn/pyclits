#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pathlib
import pickle
from itertools import product

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams['text.usetex'] = True

plt.rc('text', usetex=True)
plt.rc('font', family='serif')


def process_base_datafiles(data_directory, output_file="join_dataset.pickle", show=False):
    datasets = []
    for item_of_folder in pathlib.Path(data_directory).iterdir():
        if item_of_folder.is_file():
            print(item_of_folder)
            parameter1 = float(item_of_folder.name.split("_")[-2])
            parameter2 = int(item_of_folder.name.split("_")[-1].split(".")[0])

            full_filepath = item_of_folder
            loaded_dataset = pd.read_csv(full_filepath, sep='\t', index_col=False)
            loaded_dataset['correlation'] = parameter1

            # print(loaded_dataset)

            datasets.append(loaded_dataset)

    concatenated_dataframe = pd.concat(datasets, ignore_index=True)
    if show:
        print(concatenated_dataframe)
        print(concatenated_dataframe.dtypes)

    with open(output_file, "wb") as fb:
        pickle.dump(concatenated_dataframe, fb)

    return concatenated_dataframe


def load_dataset(output_file):
    with open(output_file, "rb") as fb:
        return pickle.load(fb)


def figure(dataset, x_column, comparison_columns, columns, title, xlabel, ylabel, filename, suffix="eps", dpi=300):
    matplotlib.style.use("seaborn")

    color_map = matplotlib.cm.get_cmap("summer")

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    for index_column, column in enumerate(columns):
        xs = dataset[[x_column]]
        ys = dataset[column]

        color = color_map(index_column / len(columns))
        try:
            if index_column % 20 == 0:
                ax.plot(xs.values, ys.values, color=color, linewidth=1, linestyle="dotted", label=f"Order {index_column}")
            else:
                ax.plot(xs.values, ys.values, color=color, linewidth=1, linestyle="dotted")
        except Exception as exc:
            print(f"{exc}: Problem column {column}")

    for comparison_column in comparison_columns:
        xs = dataset[[x_column]]
        ys = dataset[comparison_column["column"]]

        color = comparison_column["color"]
        label = comparison_column["label"]
        try:
            if "error_column" in comparison_column:
                errorbar = dataset[comparison_column["error_column"]] * 1.96
                ax.errorbar(xs.values, ys.values, yerr=errorbar.values, color=color, label=label)
            else:
                ax.plot(xs.values, ys.values, color=color, linewidth=2, label=label)
        except Exception as exc:
            print(f"{exc}: Problem column {comparison_column}")

    ax.legend(loc=1)

    plt.savefig(filename + "." + suffix, dpi=dpi)
    plt.close()


def filter_by(dataset, column_names):
    sets_of_values = []
    for column_name in column_names:
        values = set(dataset[column_name].to_numpy())
        sets_of_values.append(values)

    selected_dataset = {}

    for multiindex in product(*sets_of_values):
        sliced_dataset = dataset
        for mi, column in zip(multiindex, column_names):
            sliced_dataset = sliced_dataset.loc[sliced_dataset[column] == mi]

        selected_dataset[multiindex] = sliced_dataset

    return selected_dataset


def scan_columns(dataset, result_column_basename):
    columns = []
    for column in dataset.columns.array:
        if result_column_basename in column:
            columns.append(column)
    return columns


def mean_calculation(results, columns, column_name):
    temp_results = results[columns]
    results[column_name] = temp_results.apply(lambda x: np.mean(x), axis=1, raw=True)


def std_calculation(results, columns, column_name):
    temp_results = results[columns]
    results[column_name] = temp_results.apply(lambda x: np.std(x), axis=1, raw=True)


if __name__ == "__main__":
    data_directory = "data"
    output_filename = "join_dataset.pickle"

    if os.path.isfile(output_filename):
        results = load_dataset(output_filename)
    else:
        results = process_base_datafiles(data_directory, output_filename)

    set_of_theoretical_values = [{"column": "theoretical value", "color": "b", "label": "Theoretical value"},
                                 {"column": "mean", "color": "r", "label": r"Mean of order 95\% bars", "error_column": "std"}]
    sets_of_mean_results = scan_columns(results, "mean Renyi entropy ")
    sets_of_std_results = scan_columns(results, "std Renyi entropy ")
    sets_of_mean_difference_results = scan_columns(results, "mean difference ")
    sets_of_std_difference_results = scan_columns(results, "std of difference ")

    # calculate mean
    mean_calculation(results, sets_of_mean_results, "mean")
    # calculate std
    std_calculation(results, sets_of_mean_results, "std")

    # filter dataset
    filtered_results = filter_by(results, ["correlation", "sigma", "dimension"])

    # show dataset
    for key, filtered_results in filtered_results.items():
        correlation = key[0]
        sigma = key[1]
        dimension = key[2]
        figure(filtered_results, "alpha", set_of_theoretical_values, sets_of_mean_results,
               r"\Huge Comparison of calculated Renyi entropy and theoretical value", r"$\alpha$", r"$H_{\alpha}(X)$",
               f"Entropy_comparison_correlation_{correlation}_sigma_{sigma}_dimension_{dimension}", suffix="eps")
