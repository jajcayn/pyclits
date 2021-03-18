import datetime
import logging
import os
import pickle
import time
from pathlib import Path

import numpy as np
import scipy.interpolate

from roessler_system import roessler_oscillator
from sample_generator import shuffle_sample

file = Path(__file__).parents[0] / "roessler_system_reference" / "arosf11n00eps100raw.dat"
file_pickled = Path(__file__).parents[0] / "roessler_system_reference" / "dataset.bin"


def read_header(fh):
    parameters = {}
    for linenumber in range(6):
        line = fh.readline()
        if "Realization number" in line:
            parameters["Realization number"] = int(line.split("number")[1])
        elif "Ros RAW" in line:
            parameters[line.split(",")[0].replace("#", "")] = line.split(",")[1]
        elif "=" in line:
            if "count" in line:
                value = int(line.split("=")[1])
            else:
                value = float(line.split("=")[1])

            parameters[line.split("=")[0].replace(" ", "").replace("#", "")] = value
        else:
            print(f"Incompatible line detected: {line}, position {fh.tell()}")

    return parameters


def read_dataset(fh, parameter):
    dataset = []
    if "count" in parameter:
        for linenumber in range(parameter["count"]):
            line = fh.readline()
            first = float(line[:15])
            second = float(line[15:])
            dataset.append([first, second])

        frame = np.array(dataset)
        return frame
    else:
        return None


def load_datasets():
    dataset = []
    try:
        with open(file, "rt") as fh:
            while True:
                parameters = read_header(fh)
                frame = read_dataset(fh, parameters)
                if frame is not None:
                    dataset.append([parameters, frame])
                else:
                    break
    except EOFError as exc:
        pass

    return dataset


def prepare_dataset(args, index_epsilon, datasets=None, swap_datasets=False, shuffle_dataset=False, configuration_of_integration=None):
    if not args.dataset:
        # calculate Rössler coupled oscilators
        t0 = time.process_time()
        sol = roessler_oscillator(**configuration_of_integration)
        t1 = time.process_time()
        duration = t1 - t0
        print(f"PID:{os.getpid()} {datetime.datetime.now().isoformat()} Solution duration [s]: {duration}", flush=True)

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

        print(f"PID:{os.getpid()} {datetime.datetime.now().isoformat()} Shape of solution: {filtrated_solution.shape}", flush=True)
        if args.full_system:
            marginal_solution_1 = filtrated_solution[0:3, :].T
            marginal_solution_2 = filtrated_solution[3:6, :].T
        else:
            marginal_solution_1 = filtrated_solution[0:1, :].T
            marginal_solution_2 = filtrated_solution[3:4, :].T
    else:
        filtrated_solution = datasets[index_epsilon][1].T

        print(f"PID:{os.getpid()} {datetime.datetime.now().isoformat()} Shape of solution: {filtrated_solution.shape}", flush=True)
        if swap_datasets:
            marginal_solution_1 = filtrated_solution[0:1, :].T
            marginal_solution_2 = filtrated_solution[1:2, :].T
        else:
            marginal_solution_2 = filtrated_solution[0:1, :].T
            marginal_solution_1 = filtrated_solution[1:2, :].T

    if shuffle_dataset:
        marginal_solution_1 = shuffle_sample(marginal_solution_1)

    return marginal_solution_1, marginal_solution_2


def load_static_dataset(args):
    print(f"PID:{os.getpid()} {datetime.datetime.now().isoformat()} Load dataset", flush=True)
    datasets = load_datasets()

    if args.dataset_range:
        dataset_start = int(args.dataset_range.split("-")[0])
        dataset_end = int(args.dataset_range.split("-")[1])
        datasets = datasets[dataset_start:dataset_end]

    epsilons = []
    for dataset in datasets:
        epsilons.append(dataset[0]["eps1"])
    print(f"PID:{os.getpid()} {datetime.datetime.now().isoformat()} Epsilons: {epsilons}", flush=True)

    return datasets, epsilons


if __name__ == "__main__":
    dataset = load_datasets()
    print(f"We aggregated {len(dataset)} records")
    with open(file_pickled, "wb") as fh:
        pickle.dump(dataset, fh)
