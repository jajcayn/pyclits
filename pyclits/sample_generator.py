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
    sampled_dataset = np.empty((shape_of_array[0] * history, shape_of_array[1] - history))

    for item in range(skip_first, shape_of_array[1] - history - skip_last):
        for hist in range(history):
            for dim_iter in range(shape_of_array[0]):
                big_coordinate = hist * shape_of_array[0] + dim_iter
                sampled_dataset[big_coordinate, item] = data[dim_iter, item + hist]

    if "transpose" in kwargs and kwargs["transpose"]:
        sampled_dataset = sampled_dataset.T

    return sampled_dataset


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
