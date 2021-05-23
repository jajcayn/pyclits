#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os.path
import pickle
from argparse import Namespace
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

matplotlib.rcParams['text.usetex'] = True

plt.rc('text', usetex=True)
plt.rc('font', family='serif')


def right_side(y, t, params=[0, 0, 0, 0, 0, 0, 0, 0]):
    x1, x2, x3, y1, y2, y3 = y  # unpack current values of y
    a1, a2, b1, b2, c1, c2, omega1, omega2, epsilon = params  # unpack parameters
    derivs = [-omega1 * x2 - x3, omega1 * x1 + a1 * x2, b1 + x3 * (x1 - c1), - omega2 * y2 - y3 + epsilon * (x1 - y1),
              omega2 * y1 + a2 * y2, b2 + y3 * (y1 - c2)]

    return derivs


def right_side_ivp(t, y, params):
    x1, x2, x3, y1, y2, y3 = y      # unpack current values of y
    a1, a2, b1, b2, c1, c2, omega1, omega2, epsilon = params  # unpack parameters
    derivs = [-omega1 * x2 - x3, omega1 * x1 + a1 * x2, b1 + x3 * (x1 - c1), - omega2 * y2 - y3 + epsilon * (x1 - y1), omega2 * y1 + a2 * y2, b2 + y3 * (y1 - c2)]

    return derivs


def roessler_oscillator(**kwargs):
    # Parameters
    if "a1" in kwargs:
        a1 = kwargs["a1"]
    else:
        a1 = 0.15

    if "a2" in kwargs:
        a2 = kwargs["a2"]
    else:
        a2 = 0.15

    if "b1" in kwargs:
        b1 = kwargs["b1"]
    else:
        b1 = 0.2

    if "b2" in kwargs:
        b2 = kwargs["b2"]
    else:
        b2 = 0.2

    if "c1" in kwargs:
        c1 = kwargs["c1"]
    else:
        c1 = 10.0

    if "c2" in kwargs:
        c2 = kwargs["c2"]
    else:
        c2 = 10.0

    if "omega1" in kwargs:
        omega1 = kwargs["omega1"]
    else:
        omega1 = 1.015

    if "omega2" in kwargs:
        omega2 = kwargs["omega2"]
    else:
        omega2 = 0.985

    if "epsilon" in kwargs:
        epsilon = kwargs["epsilon"]
    else:
        epsilon = 0.001

    # Initial values
    if "x1" in kwargs:
        x1 = kwargs["x1"]
    else:
        x1 = 0.0

    if "x2" in kwargs:
        x2 = kwargs["x2"]
    else:
        x2 = 0.0

    if "x3" in kwargs:
        x3 = kwargs["x3"]
    else:
        x3 = 0.0

    if "y1" in kwargs:
        y1 = kwargs["y1"]
    else:
        y1 = 0.0

    if "y2" in kwargs:
        y2 = kwargs["y2"]
    else:
        y2 = 0.0

    if "y3" in kwargs:
        y3 = kwargs["y3"]
    else:
        y3 = 1.0

    if "method" in kwargs:
        method = kwargs["method"]
    else:
        method = "LSODA"

    # Bundle parameters for ODE solver
    params = [a1, a2, b1, b2, c1, c2, omega1, omega2, epsilon]

    # Bundle initial conditions for ODE solver
    y0 = [x1, x2, x3, y1, y2, y3]

    # Make time array for solution
    if "tStart" in kwargs:
        tStart = kwargs["tStart"]
    else:
        tStart = 0.0

    if "tStop" in kwargs:
        tStop = kwargs["tStop"]
    else:
        tStop = 20000.

    if "tInc" in kwargs:
        tInc = kwargs["tInc"]
    else:
        tInc = 0.001

    if "cache" in kwargs:
        cache = kwargs["cache"]
    else:
        cache = False

    if "file_template" in kwargs:
        file_template = kwargs["file"]
    else:
        file_template = "roessler_system/cache-{}.bin"

    file = file_template.format(epsilon)
    path = Path(file)

    if not cache or not os.path.isfile(path):
        timepoints = np.arange(tStart, tStop, tInc)

        print("Calculating numerical solution")
        # Call the ODE solver
        # old version
        # psoln = odeint(right_side, y0, t, args=(params,))
        solution = solve_ivp(lambda t, y: right_side_ivp(t, y, params), [tStart, tStop], y0, method=method)

        print("Saving the solution to the cache")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fb:
            pickle.dump((params, y0, solution), fb)

        return solution
    else:
        print("Load dataset from cache")
        with open(path, "rb") as fb:
            (params, y0, solution) = pickle.load(fb)

        return solution


def visualization(sol, **kwargs):
    if hasattr(kwargs, "tStart"):
        tStart = kwargs["tStart"]
    else:
        tStart = 0.0

    if hasattr(kwargs, "tStop"):
        tStop = kwargs["tStop"]
    else:
        tStop = 20000.

    fig, axes = plt.subplots(6, 1, sharex=True, sharey=False, figsize=(8, 8))
    plt.tight_layout()
    plt.xlim(tStart, tStop)

    for num, ax in enumerate(axes):
        ax.plot(sol.t, sol.y[num, :])

    plt.xlabel("Time")
    plt.ylabel("Coordinate")

    plt.show()


def roessler_plot(dataset, title, filename, suffix, dpi=300):
    fig = plt.figure(figsize=(13, 8))
    ax = fig.add_subplot(1, 1, 1)

    markers = ['b', '^']

    ax.set_title(title)
    ax.set_xlabel(r"$X$")
    ax.set_ylabel(r"$Y$")

    ys = dataset[:, 0]
    zs = dataset[:, 1]

    try:
        ax.plot(ys, zs, linewidth=1)
    except Exception as exc:
        print(f"{exc}")

    plt.savefig(filename + "." + suffix, dpi=dpi)
    plt.close()


def roessler_3d_plot(dataset, configurations, title, filename, suffix, dpi=300):
    columns = 3
    rows = 3
    fig, ax = plt.subplots(columns, rows, sharey=False, sharex=False, figsize=(13, 8))
    fig.suptitle(title)
    plt.subplots_adjust(hspace=0.24, wspace=0.24, left=0.05, right=0.95, top=0.95, bottom=0.03)
    for row in range(rows):
        for column in range(columns):
            configuration = configurations[(row, column)]
            plot_configuration = {key: value for key, value in configuration.items() if "data_index" not in key}
            xdataset = dataset[configuration["xdata_index"]]
            ydataset = dataset[configuration["ydata_index"]]
            xlim = [min(xdataset), max(xdataset)]
            ylim = [min(ydataset), max(ydataset)]
            ax[row, column].set(**plot_configuration)
            ax[row, column].set(xlim=xlim)
            ax[row, column].set(ylim=ylim)
            ax[row, column].plot(xdataset, ydataset)

    #plt.subplot_tool()
    if not isinstance(suffix, tuple or list):
        suffix = [suffix]
    for item_suffix in suffix:
        plt.savefig(filename + "." + item_suffix, dpi=dpi)
    plt.close()

def roessler_3d_multocolor_plot(dataset, configurations, title, filename, suffix, dpi=300):
    columns = 3
    rows = 3
    fig, ax = plt.subplots(columns, rows, sharey=False, sharex=False, figsize=(13, 8))
    fig.suptitle(title)
    plt.subplots_adjust(hspace=0.24, wspace=0.24, left=0.05, right=0.95)
    for row in range(rows):
        for column in range(columns):
            configuration = configurations[(row, column)]
            plot_configuration = {key: value for key, value in configuration.items() if "data_index" not in key}
            xdataset = dataset[configuration["xdata_index"]]
            ydataset = dataset[configuration["ydata_index"]]
            xlim = [min(xdataset), max(xdataset)]
            ylim = [min(ydataset), max(ydataset)]
            ax[row, column].set(**plot_configuration)
            ax[row, column].set(xlim=xlim)
            ax[row, column].set(ylim=ylim)
            ax[row, column].plot(xdataset, ydataset)

    #plt.subplot_tool()
    if not isinstance(suffix, tuple or list):
        suffix = [suffix]
    for item_suffix in suffix:
        plt.savefig(filename + "." + item_suffix, dpi=dpi)
    plt.close()


if __name__ == "__main__":
    dataset = False
    if (dataset):
        from data_plugin import load_static_dataset

        args = Namespace(dataset_range="0-100")
        datasets, epsilons = load_static_dataset(args)

        for dataset_item in datasets:
            metadata = dataset_item[0]
            dataset = dataset_item[1]
            roessler_plot(dataset, r"Evolution of Rössler oscilator $\varepsilon=$", "roessler-" + str(metadata["eps1"]), "pdf")
    else:
        configuration = {(0, 0): {"xlabel": r"$x_1$", "ylabel": r"$x_1$", "xdata_index": 0, "ydata_index": 1},
                         (1, 0): {"xlabel": r"$x_2$", "ylabel": r"$x_3$", "xdata_index": 1, "ydata_index": 2},
                         (2, 0): {"xlabel": r"$x_3$", "ylabel": r"$x_1$", "xdata_index": 2, "ydata_index": 0},
                         (0, 1): {"xlabel": r"$x_1$", "ylabel": r"$y_1$", "xdata_index": 0, "ydata_index": 3},
                         (1, 1): {"xlabel": r"$x_2$", "ylabel": r"$y_2$", "xdata_index": 1, "ydata_index": 4},
                         (2, 1): {"xlabel": r"$x_3$", "ylabel": r"$y_3$", "xdata_index": 2, "ydata_index": 5},
                         (0, 2): {"xlabel": r"$y_1$", "ylabel": r"$y_2$", "xdata_index": 3, "ydata_index": 4},
                         (1, 2): {"xlabel": r"$y_2$", "ylabel": r"$y_3$", "xdata_index": 4, "ydata_index": 5},
                         (2, 2): {"xlabel": r"$y_3$", "ylabel": r"$y_1$", "xdata_index": 5, "ydata_index": 3},
                         }
        for epsilon in np.arange(0.0, 0.25, 0.005):
            print(f"Calculation of epsilon {epsilon}")
            configuration_of_integration = {"method": "LSODA", "tInc": 0.01, "tStop": 10000, "cache": True, "epsilon": epsilon, "cache": False}
            sol = roessler_oscillator(**configuration_of_integration)
            roessler_3d_plot(sol.y, configuration, fr"\Large Evolution of Rössler oscilator $\varepsilon= {epsilon}$", f"roessler-{epsilon}", "png")

        print(sol)
        # print(sol.y)
        print(sol.y.shape)

        visualization(sol)
