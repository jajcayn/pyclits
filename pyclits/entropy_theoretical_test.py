#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

import entropy_calculation_test

if __name__ == "__main__":
    sigma = 1.0
    Sigma = np.identity(5) * np.power(sigma, 2)

    for alpha in [0.1, 0.9, 1, 1.2, 1.9]:
        renyi = entropy_calculation_test.Renyi_normal_distribution(np.asmatrix(Sigma), alpha)
        print(f"{alpha} {renyi}")
