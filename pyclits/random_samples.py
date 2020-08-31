#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stat


def sample_normal_distribution(sigma, size_sample=10):
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


def sample_elliptical_contour_stable(sigma, alpha, gamma=1.0, delta=np.array([0]), size_sample=10):
    recalculated_gamma = 2 * np.power(gamma, 2) * np.power(np.cos(np.pi * alpha / 4), (2 / alpha))
    random_stable = stat.levy_stable.rvs(alpha=alpha / 2.0, beta=1.0, loc=0, scale=recalculated_gamma, size=size_sample)
    sqrt_random_stable = np.sqrt(random_stable)

    random_normal = sample_normal_distribution(sigma, size_sample)
    stable_samples = np.array([multiplicator * vector - delta for multiplicator, vector in zip(sqrt_random_stable, random_normal)])

    return stable_samples


def sample_linear_depended_stable(sigma, alpha, beta=np.array([0]), delta=np.array([0]), size_sample=10):
    dimension = sigma.shape[0]
    if (dimension == beta.shape[0]) and (dimension == delta.shape[0]):
        random_stable_sample = np.ndarray(shape=(0, size_sample))
        for index in range(dimension):
            random_stable = stat.levy_stable.rvs(alpha=alpha, beta=beta[index], loc=delta[index], scale=1.0, size=size_sample)
            random_stable_sample = np.vstack([random_stable_sample, random_stable])

        random_stable_sample = np.swapaxes(random_stable_sample, 0, 1)
        stable_samples = np.array([np.dot(sigma, vector) for vector in random_stable_sample])

        return stable_samples
    else:
        raise Exception("Invalid input detected")


def sample_asymmetric_laplace_distribution(sigma=np.array([1]), mean=np.array([0]), size_sample=10):
    exponentially_distributed = stat.expon.rvs(size=size_sample)
    sqrt_exponentially_distributed = np.sqrt(exponentially_distributed)
    random_normal = sample_normal_distribution(sigma, size_sample)

    stable_samples = np.array([mean * multiplicator + multiplicator_sqrt * vector for multiplicator, multiplicator_sqrt, vector in
                               zip(exponentially_distributed, sqrt_exponentially_distributed, random_normal)])

    return stable_samples


def sample_student_t_distribution(degrees_of_freedom, sigma=np.array([1]), mean=np.array([0]), size_sample=10):
    chi_squared_distributed = stat.chi2.rvs(df=degrees_of_freedom, size=size_sample)
    random_normal = sample_normal_distribution(sigma, size_sample)

    stable_samples = np.array([mean + vector * (np.sqrt(degrees_of_freedom / multiplicator)) for multiplicator, vector in
                               zip(chi_squared_distributed, random_normal)])

    return stable_samples


if __name__ == "__main__":
    sigma = np.array([[1, -0.2], [-0.2, 1]])
    samples = 100000

    normal_samples = sample_normal_distribution(sigma=sigma, size_sample=samples)

    # plt.scatter(normal_samples[:, 0], normal_samples[:, 1], marker=".")
    # plt.show()

    stable_linear_depended_samples = sample_linear_depended_stable(sigma=sigma, alpha=1.9, beta=np.array([-0.3, 0.2]), delta=np.array([0, 0]),
                                                                   size_sample=samples)

    # plt.scatter(stable_linear_depended_samples[:, 0], stable_linear_depended_samples[:, 1], marker=".")
    # plt.show()

    samples_stable = sample_elliptical_contour_stable(sigma, 1.6, 1.0, np.array([0, 0]), size_sample=samples)

    # plt.scatter(samples_stable[:, 0], samples_stable[:, 1], marker=".")
    # plt.show()

    samples_laplace = sample_asymmetric_laplace_distribution(sigma=sigma, mean=np.array([1, -1]), size_sample=samples)

    # plt.scatter(samples_laplace[:, 0], samples_laplace[:, 1], marker=".")
    # plt.show()

    samples_student = sample_student_t_distribution(sigma=sigma, mean=np.array([0, 0]), degrees_of_freedom=2, size_sample=samples)

    # plt.scatter(samples_student[:, 0], samples_student[:, 1], marker=".")
    # plt.show()

    datasamples = [[normal_samples, "Normal distribution"], [stable_linear_depended_samples, "Stable linear dependent"],
                   [samples_stable, "Stable sub-Gaussian"],
                   [samples_laplace, "Laplace asymmetric"], [samples_student, "Student t-distribution"]]

    gamma = 0.1

    for index, (sample, title) in enumerate(datasamples):
        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(8, 10))
        ax1.set_title(title)
        ax1.scatter(sample[:, 0], sample[:, 1], marker=".")
        #  mcolors.PowerNorm(gamma)
        ax2.hist2d(sample[:, 0], sample[:, 1], bins=100, norm=mcolors.LogNorm(), cmap='Reds')
        plt.show()
        plt.close()
