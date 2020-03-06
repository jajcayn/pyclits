#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import pickle
import time
from pathlib import Path

import numpy as np
import scipy.interpolate

import data_plugin
from mutual_inf import renyi_transfer_entropy
from roessler_system import roessler_oscillator
from sample_generator import preparation_dataset_for_transfer_entropy, shuffle_sample


def prepare_dataset(args, index_epsilon, datasets=None, shuffle_dataset=False):
    if not args.dataset:
        # calculate Rössler coupled oscilators
        t0 = time.process_time()
        sol = roessler_oscillator(**configuration)
        t1 = time.process_time()
        duration = t1 - t0
        print(f"Solution duration [s]: {duration}", flush=True)

        if args.interpolate:
            number = int((args.t_stop - args.skip) * args.interpolate_samples_per_unit_time)

            new_t = np.linspace(args.skip, args.t_stop, num=number, endpoint=True)
            solution = []
            for dimension in range(sol.y.shape[0]):
                function = scipy.interpolate.interp1d(sol.t, sol.y[dimension], kind='cubic')

                solution.append(function(new_t))

            filtrated_solution = np.vstack(solution)
        else:
            # preparation of sources
            if args.skip_real_t:
                indices = np.where(sol.t >= args.skip)
                if len(indices) > 0:
                    filtrated_solution = sol.y[:, indices[0]:]
                else:
                    logging.error("Skipping is too large and no data were selected for processing")
                    raise AssertionError("No data selected")
            else:
                filtrated_solution = sol.y[:, args.skip:]

        if shuffle_dataset:
            filtrated_solution = shuffle_sample(filtrated_solution)

        print(f"Shape of solution: {filtrated_solution.shape}", flush=True)
        joint_solution = filtrated_solution
        marginal_solution_1 = filtrated_solution[0:3, :].T
        marginal_solution_2 = filtrated_solution[3:6, :].T
    else:
        filtrated_solution = datasets[index_epsilon][1].T

        if shuffle_dataset:
            filtrated_solution = shuffle_sample(filtrated_solution)

        print(f"Shape of solution: {filtrated_solution.shape}", flush=True)
        joint_solution = filtrated_solution
        marginal_solution_1 = filtrated_solution[0:1, :].T
        marginal_solution_2 = filtrated_solution[1:2, :].T

    return marginal_solution_1, marginal_solution_2


def load_static_dataset(args):
    print("Load dataset", flush=True)
    datasets = data_plugin.load_datasets()

    if args.dataset_range:
        dataset_start = int(args.dataset_range.split("-")[0])
        dataset_end = int(args.dataset_range.split("-")[1])
        datasets = datasets[dataset_start:dataset_end]

    epsilons = []
    for dataset in datasets:
        epsilons.append(dataset[0]["eps1"])
    print(f"Epsilons: {epsilons}", flush=True)

    return datasets, epsilons


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculates transfer entropy for coupled Rössler systems with strength of coupling epsilon.')
    parser.add_argument('--epsilon', metavar='XXX', type=float, nargs='+', help='Epsilons')
    parser.add_argument('--t_stop', metavar='XXX', type=float, default=10000.0, help='T stop')
    parser.add_argument('--t_inc', metavar='XXX', type=float, default=0.01, help='T increment')
    parser.add_argument('--no_cache', action='store_true', help='Skips cached results of the Rössler system')
    parser.add_argument('--skip', metavar='XXX', type=int, default=2000, help='Skipped results of integration')
    parser.add_argument('--blockwise', metavar='XXX', type=int, default=0, help='Blockwise calculation of distances to prevent excessive memory usage')
    parser.add_argument('--skip_real_t', action='store_true', help='Indicates skip in time')
    parser.add_argument('--history_first', metavar='XXX', type=int, nargs='+', help='History to take into account')
    parser.add_argument('--history_second', metavar='XXX', type=int, nargs='+', help='History to take into account')
    parser.add_argument('--method', metavar='XXX', type=str, default="LSODA", help='Method of integration')
    parser.add_argument('--maximal_neighborhood', metavar='XXX', type=int, default=2, help='Maximal neighborhood')
    parser.add_argument('--arbitrary_precision', action='store_true', help='Calculates the main part in arbitrary precision')
    parser.add_argument('--arbitrary_precision_decimal_places', metavar='XXX', type=int, default=100,
                        help='Sets number saved in arbitrary precision arithmetic')
    parser.add_argument('--interpolate', action='store_true', help='Switch on intepolation')
    parser.add_argument('--interpolate_samples_per_unit_time', metavar='XXX', type=int, default=10, help='Number of samples generated per unit time')
    parser.add_argument('--dataset', action='store_true', help='Use dataset provided by dr. Paluš')
    parser.add_argument('--dataset_range', metavar='XXX-YYY', type=str, help='Dataset with range')
    args = parser.parse_args()
    # print(args.epsilon, flush=True)

    if args.epsilon:
        epsilons = args.epsilon
    else:
        epsilons = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13]

    if args.history_first:
        histories_first = args.history_first
    else:
        histories_first = range(2, 25)

    if args.history_second:
        histories_second = args.history_second
    else:
        histories_second = range(2, 25)

    # load static dataset
    if args.dataset:
        datasets, epsilons = load_static_dataset(args)

    # loop over different realizations for various epsilon
    for index_epsilon, epsilon in enumerate(epsilons):
        configuration = {"method": args.method, "tInc": args.t_inc, "tStop": args.t_stop, "cache": True, "epsilon": epsilon,
                         "arbitrary_precision": args.arbitrary_precision, "arbitrary_precision_decimal_numbers": args.arbitrary_precision_decimal_places}

        # create structure for results
        results = {}

        # loop over shuffling
        for shuffle_dataset in [False, True]:
            # prepare dataset that is been processed
            marginal_solution_1, marginal_solution_2 = prepare_dataset(args, index_epsilon=index_epsilon, datasets=datasets, shuffle_dataset=shuffle_dataset)

            # create alphas that are been calculated
            alphas = np.round(np.linspace(0.1, 1.9, 54), 3)

            # looping history of X timeserie
            for history_first in histories_first:

                # looping history of Y timeserie
                for history_second in histories_second:
                    print(f"History first: {history_first}, history second: {history_second} and epsilon: {epsilon} is processed", flush=True)

                    # preparation of the configuration dictionary
                    # additional +1 is there for separation
                    configuration = {"transpose": True, "history_x": history_first + 1, "history_y": history_second, "blockwise": args.blockwise}

                    # prepare samples to be used to calculate transfer entropy
                    t0 = time.process_time()
                    y, y_hist, z = preparation_dataset_for_transfer_entropy(marginal_solution_2, marginal_solution_1, **configuration)
                    t1 = time.process_time()
                    duration = t1 - t0
                    print(f" * Preparation of datasets [s]: {duration}", flush=True)

                    # create range of indices that will be used for calculation
                    indices_to_use = list(range(1, args.maximal_neighborhood + 1))
                    configuration = {"transpose": True, "axis_to_join": 0, "method": "LeonenkoProzanto", "alphas": alphas,
                                     "enhanced_calculation": True, "indices_to_use": indices_to_use, "arbitrary_precision": args.arbitrary_precision,
                                     "arbitrary_precision_decimal_numbers": args.arbitrary_precision_decimal_places}

                    # calculation of transfer entropy
                    print(
                        f" * Transfer entropy for history first: {history_first}, history second: {history_second} and epsilon: {epsilon} shuffling; {shuffle_dataset} is calculated",
                        flush=True)
                    t0 = time.process_time()
                    transfer_entropy = renyi_transfer_entropy(y, y_hist, z, **configuration)
                    t1 = time.process_time()
                    duration = t1 - t0
                    print(f" * Duration of calculation of transfer entropy [s]: {duration}", flush=True)
                    # print(f" * Transfer Renyi entropy with {history} {epsilon}: {transfer_entropy}", flush=True)

                    # store transfer entropy to the result structure
                    results[(shuffle_dataset, epsilon, history_first, history_second)] = transfer_entropy
                    print(
                        f" * Transfer entropy calculation for history first: {history_first}, history second: {history_second} and epsilon: {epsilon}, shuffling; {shuffle_dataset} is finished",
                        flush=True)

        # save result structure to the file
        path = Path(f"transfer_entropy/Transfer_entropy_dataset-{epsilon}.bin")
        print(f"Save to file {path}", flush=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fb:
            pickle.dump(results, fb)
