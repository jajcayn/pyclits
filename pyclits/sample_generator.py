#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random

import numpy as np

from roessler_system import roessler_oscillator


def shuffle_sample(data):
    shape = data.shape
    items = list(range(shape[0]))
    i = len(items)
    while i > 1:
        i = i - 1
        j = random.randrange(i)
        items[j], items[i] = items[i], items[j]

    # print(f"New shuffle arrangement {items}")
    return data[items, :]


def samples_from_arrays(data, **kwargs):
    if "history" in kwargs:
        history = kwargs["history"]
    else:
        allocated_space = 5
        history = 5

    if "skip_last" in kwargs:
        skip_last = kwargs["skip_last"]
    else:
        skip_last = 0

    if "skip_first" in kwargs:
        skip_first = kwargs["skip_first"]
    else:
        skip_first = 0

    if "transpose" in kwargs and kwargs["transpose"]:
        data = data.T

    if "select_indices" in kwargs:
        select_indices = kwargs["select_indices"]
        allocated_space = len(select_indices)
        history = max(select_indices)
        # skip_first += allocated_space

    shape_of_array = data.shape
    length_of_timeserie = shape_of_array[1] - skip_first - skip_last
    sampled_dataset = np.zeros((shape_of_array[0] * allocated_space, length_of_timeserie))  # history

    range_of_history = select_indices if "select_indices" in kwargs else range(history)

    for position_of_item, item in enumerate(range(skip_first, skip_first + length_of_timeserie)):
        for position_of_history, hist in enumerate(range_of_history):
            for dim_iter in range(shape_of_array[0]):
                inserted_data = data[dim_iter, item - hist]
                big_coordinate = position_of_history * shape_of_array[0] + dim_iter
                sampled_dataset[big_coordinate, position_of_item] = inserted_data

    # print(sampled_dataset)
    if "transpose" in kwargs and kwargs["transpose"]:
        sampled_dataset = sampled_dataset.T

    return sampled_dataset


def preparation_dataset_for_transfer_entropy(marginal_solution_1, marginal_solution_2, **kwargs):
    if "history_x" in kwargs:
        history_x = kwargs["history_x"]
    else:
        history_x = 5

    if "history_y" in kwargs:
        history_y = kwargs["history_y"]
    else:
        history_y = 5

    if "skip_last" in kwargs:
        skip_last = kwargs["skip_last"]
    else:
        skip_last = 0

    if "skip_first" in kwargs:
        skip_first = kwargs["skip_first"]
    else:
        skip_first = 0

    if "history_index_x" in kwargs:
        history_index_x = kwargs["history_index_x"]
        history_x = max(history_index_x)

    if "history_index_y" in kwargs:
        history_index_y = kwargs["history_index_y"]
        history_y = max(history_index_y)

    # time lag between X and Y
    if "time_shift_between_X_Y" in kwargs:
        time_shift_between_X_Y = kwargs["time_shift_between_X_Y"]
    else:
        time_shift_between_X_Y = 1

    if "future_index_x" in kwargs:
        future_index_x = kwargs["future_index_x"]
        time_shift_between_X_Y = max(future_index_x)

    if "transpose" in kwargs and kwargs["transpose"]:
        marginal_solution_1 = marginal_solution_1.T
        marginal_solution_2 = marginal_solution_2.T

    shape = marginal_solution_1.shape
    marginal_solution_1_selected = marginal_solution_1[:, skip_last:] if skip_last == 0 else marginal_solution_1[:,
                                                                                             skip_last: -skip_last]
    marginal_solution_2_selected = marginal_solution_2[:, skip_last:] if skip_last == 0 else marginal_solution_2[:,
                                                                                             skip_last: -skip_last]

    kwargs["transpose"] = False
    # additional move in history is there because then actual timeserie is separated
    kwargs["history"] = history_x
    kwargs["skip_first"] = time_shift_between_X_Y if history_x > history_y else history_y - history_x + time_shift_between_X_Y
    if "history_index_x" in kwargs:
        if "future_index_x" in kwargs:
            indices = [max(future_index_x) - item for item in future_index_x]

            indices += [max(future_index_x) + item for item in history_index_x]
        else:
            indices = [1 + item for item in history_index_x]

        kwargs["select_indices"] = indices
        kwargs["skip_first"] = max(history_index_y + history_index_x) + max(future_index_x)
        kwargs["history"] = history_x

    samples_marginal_1 = samples_from_arrays(marginal_solution_1_selected, **kwargs)

    kwargs["history"] = history_y
    kwargs["skip_first"] = max(history_index_y + history_index_x) + max(future_index_x)
    kwargs["skip_last"] = 0
    if "history_index_y" in kwargs:
        indices = [max(future_index_x) + item for item in history_index_y]

        kwargs["select_indices"] = indices
    else:
        del kwargs["select_indices"]

    samples_marginal_2 = samples_from_arrays(marginal_solution_2_selected, **kwargs)

    if "future_index_x" in kwargs:
        y_fut = samples_marginal_1[:shape[0] * len(future_index_x), :]
        y_history = samples_marginal_1[shape[0] * len(future_index_x):, :]
    else:
        y_fut = samples_marginal_1[:shape[0], :]
        y_history = samples_marginal_1[shape[0]:, :]

    z = samples_marginal_2

    return (y_fut, y_history, z)


def check_timesteps(data):
    for item in range(data.shape[0] - 1):
        print(data[item + 1] - data[item])


if __name__ == "__main__":
    test_sample = "transfer_entropy"
    if test_sample in ["sample_array"]:
        kwargs = {"method": "RK45"}
        kwargs["tStop"] = 30
        sol = roessler_oscillator(**kwargs)
        print(sol)
        check_timesteps(sol.t)

        samples = samples_from_arrays(sol.y)
        print(samples.shape, sol.y.shape)
        print(sol.y[0:3, :].shape)
    elif test_sample in ["transfer_entropy"]:
        kwargs = {"history_index_x": [0, 1, 2], "history_index_y": [1], "future_index_x": [2, 3]}

        pattern = [list(range(0, 20, 1))]

        X_solution = np.array(pattern)
        Y_solution = np.array(pattern)

        solution = preparation_dataset_for_transfer_entropy(X_solution, Y_solution, **kwargs)

        print(solution[0])
        print(solution[1])
        print(solution[2])
