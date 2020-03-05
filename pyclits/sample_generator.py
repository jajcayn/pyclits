#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from roessler_system import roessler_oscillator


def samples_from_arrays(data, **kwargs):
    if "history" in kwargs:
        history = kwargs["history"]
    else:
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

    shape_of_array = data.shape
    sampled_dataset = np.zeros((shape_of_array[0] * history, shape_of_array[1] - history - skip_first - skip_last))

    for position_of_item, item in enumerate(range(skip_first, shape_of_array[1] - history - skip_last)):
        for hist in range(history):
            for dim_iter in range(shape_of_array[0]):
                inserted_data = data[dim_iter, item - hist]
                big_coordinate = hist * shape_of_array[0] + dim_iter
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

    # time lag between X and Y
    if "time_shift_between_X_Y" in kwargs:
        time_shift_between_X_Y = kwargs["time_shift_between_X_Y"]
    else:
        time_shift_between_X_Y = 1

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
    kwargs["skip_first"] = 1 if history_x > history_y else history_y - history_x + time_shift_between_X_Y
    samples_marginal_1 = samples_from_arrays(marginal_solution_1_selected, **kwargs)

    kwargs["history"] = history_y
    kwargs["skip_first"] = 0 if history_y > history_x else history_x - history_y
    kwargs["skip_last"] = time_shift_between_X_Y
    samples_marginal_2 = samples_from_arrays(marginal_solution_2_selected, **kwargs)

    y = samples_marginal_1[:shape[0], :]
    y_history = samples_marginal_1[shape[0]:, :]
    z = samples_marginal_2

    return (y, y_history, z)


def check_timesteps(data):
    for item in range(data.shape[0] - 1):
        print(data[item + 1] - data[item])


if __name__ == "__main__":
    kwargs = {"method": "RK45"}
    kwargs["tStop"] = 100
    sol = roessler_oscillator(**kwargs)
    print(sol)
    check_timesteps(sol.t)

    samples = samples_from_arrays(sol.y)
    print(samples.shape, sol.y.shape)
    print(sol.y[0:3, :].shape)
