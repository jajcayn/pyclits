#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sample_generator import shuffle_sample
import numpy as np
import time
import os
import datetime
import logging
import scipy.interpolate

import random_samples


def prepare_dataset(
    args,
    swap_datasets=False,
    shuffle_dataset=False,
):
    # provide random numbers
    t0 = time.process_time()
    sol = random_samples.sample_normal_distribution(np.identity(2), args.size)
    t1 = time.process_time()
    duration = t1 - t0
    print(
        f"PID:{os.getpid()} {datetime.datetime.now().isoformat()} Solution duration [s]: {duration}",
        flush=True,
    )

    filtrated_solution = sol.T

    print(
        f"PID:{os.getpid()} {datetime.datetime.now().isoformat()} Shape of solution: {filtrated_solution.shape}",
        flush=True,
    )
    marginal_solution_1 = filtrated_solution[0:1, :].T
    marginal_solution_2 = filtrated_solution[1:2, :].T

    if swap_datasets:
        marginal_solution_1, marginal_solution_2 = (
            marginal_solution_2,
            marginal_solution_1,
        )

    if shuffle_dataset:
        marginal_solution_1 = shuffle_sample(marginal_solution_1)

    return marginal_solution_1, marginal_solution_2
