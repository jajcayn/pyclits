#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stat


def sample_normal_distribution(sigma, size_sample):
    if isinstance(sigma, np.ndarray) and len(sigma.shape) == 2 and (sigma.shape[0] == sigma.shape[1]):
        dimension = sigma.shape[0]
        uncorrelated_sample = np.random.normal(0, 1.0, (dimension, size_sample))
        eigenvalues, eigenvectors = np.linalg.eig(sigma)
        standard_deviations = np.sqrt(eigenvalues)
        identity_sqrt = np.diag(standard_deviations)
        scaled_sample = identity_sqrt.dot(uncorrelated_sample)
        correlated_sample = eigenvectors.dot(scaled_sample)

        return correlated_sample.T
    else:
        raise ArithmeticError("sigma parameter has wrong type")


def sample_elliptical_contour_stable(sigma, alpha, gamma, delta, size_sample):
    recalculated_gamma = 2 * np.power(gamma, 2) * np.power(np.cos(np.pi * alpha / 4), (2 / alpha))
    random_stable = stat.levy_stable.rvs(alpha=alpha / 2.0, beta=1.0, loc=0, scale=recalculated_gamma, size=size_sample)
    sqrt_random_stable = np.sqrt(random_stable)

    random_normal = sample_normal_distribution(sigma, size_sample)
    stable_samples = np.array([multiplicator * vector - delta for multiplicator, vector in zip(sqrt_random_stable, random_normal)])

    return stable_samples


def sample_asymmetric_laplace_distribution(sigma, mean, size_sample):
    exponentially_distributed = stat.expon.rvs(size=size_sample)
    sqrt_exponentially_distributed = np.sqrt(exponentially_distributed)
    random_normal = sample_normal_distribution(sigma, size_sample)

    stable_samples = np.array([mean * multiplicator + multiplicator_sqrt * vector for multiplicator, multiplicator_sqrt, vector in
                               zip(exponentially_distributed, sqrt_exponentially_distributed, random_normal)])

    return stable_samples


def sample_student_t_distribution(sigma, mean, degrees_of_freedom, size_sample):
    chi_squared_distributed = stat.chi2.rvs(df=degrees_of_freedom, size=size_sample)
    random_normal = sample_normal_distribution(sigma, size_sample)

    stable_samples = np.array([mean + vector * (np.sqrt(degrees_of_freedom / multiplicator)) for multiplicator, vector in
                               zip(chi_squared_distributed, random_normal)])

    return stable_samples


if __name__ == "__main__":
    sigma = np.array([[1, 0.9], [0.9, 1]])

    normal_samples = sample_normal_distribution(sigma, 10000)

    plt.scatter(normal_samples[:, 0], normal_samples[:, 1], marker=".")
    plt.show()

    samples_stable = sample_elliptical_contour_stable(sigma, 1.6, 1.0, np.array([0, 0]), 10000)

    plt.scatter(samples_stable[:, 0], samples_stable[:, 1], marker=".")
    plt.show()

    samples_laplace = sample_asymmetric_laplace_distribution(sigma, np.array([0, 0]), 10000)

    plt.scatter(samples_laplace[:, 0], samples_laplace[:, 1], marker=".")
    plt.show()

    samples_student = sample_student_t_distribution(sigma, np.array([0, 0]), 2, 10000)

    plt.scatter(samples_student[:, 0], samples_student[:, 1], marker=".")
    plt.show()
