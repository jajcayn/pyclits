#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

import entropy_calculation_test

if __name__ == "__main__":
    for sigma in [0.1, 1.0, 10]:
        Sigma = np.identity(5) * np.power(sigma, 2)

        for alpha in [0.1, 0.9, 0.99, 1, 1.01, 1.2, 1.9]:
            renyi = entropy_calculation_test.Renyi_normal_distribution(np.asmatrix(Sigma), alpha)
            print(f"{alpha} {sigma} {renyi}")
