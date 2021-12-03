#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import mpmath
import numpy as np
import scipy.special as spec
import scipy.stats as stat
from numpy.linalg import inv


def full_correlation_matrix(dimension: int, q: float):
    return pow(1 - q, dimension - 1) * (1 + (dimension - 1) * q)


def tridiagonal_matrix_determinant(dimension: int, q: float):
    if dimension == 1:
        return 1
    elif dimension == 2:
        return (1 - q ** 2)
    else:
        sample = [1, 1 - q ** 2]
        for n in range(2, dimension):
            result = sample[n - 1] - q ** 2 * sample[n - 2]
            sample.append(result)
        return sample[dimension - 1]


def Renyi_laplace_assymetric_distribution(sigma, mean, alpha, sigma_inversion=None):
    dimension = sigma.shape[0]
    if sigma_inversion is None:
        sigma_inversion = inv(sigma)

    m_sigma_inv = (mean * sigma_inversion)
    k = alpha - dimension / 2.0 - 1
    M = mean * sigma_inversion * mean - m_sigma_inv.T * sigma * m_sigma_inv
    one_over_2M = 1 / (2 * M)

    prefactor = pow(2 * np.pi, dimension) / (2 * pow(alpha, dimension)) * gamma(k + 1) / gamma(alpha) * math.exp(one_over_2M) * pow(2 / M, k + 1) * Fox - H


def sample_normal_distribution(sigma, size_sample=10):
    if isinstance(sigma, np.ndarray) and len(sigma.shape) == 2 and (sigma.shape[0] == sigma.shape[1]):
        dimension = sigma.shape[0]
        uncorrelated_sample = np.random.normal(0, 1.0, (dimension, size_sample))
        eigenvalues, eigenvectors = np.linalg.eig(sigma)

        # check whether eigenvalues are all positive
        for eigenvalue in eigenvalues:
            if eigenvalue <= 0:
                raise ArithmeticError(f"All eigenvalues have to be positive. Eigenvalue {eigenvalue} breaks the rule. Sigma: {sigma}")

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


def sample_sub_linear_levy_stable(sigma, alpha, alpha_orig, gamma=1.0, delta=np.array([0]), size_sample=10):
    pass


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


def Renyi_student_t_distribution(degrees_of_freedom: float, sigma, q: float, determinant=None):
    dimension = sigma.shape[0]
    if (q * (degrees_of_freedom + dimension) / 2 - dimension / 2 <= 0):
        raise Exception(f"Wrong parameters")

    if determinant:
        arg = pow(determinant, (1. - q) / 2)
    else:
        arg = pow(np.linalg.det(sigma), (1. - q) / 2)

    arg2 = pow(degrees_of_freedom * np.pi, dimension / 2 * (1. - q))
    arg31 = pow(spec.gamma((degrees_of_freedom + dimension) / 2), q)
    arg32 = spec.gamma(q * (degrees_of_freedom + dimension) / 2 - dimension / 2)
    arg33 = spec.gamma(q * (degrees_of_freedom + dimension) / 2)
    arg34 = pow(spec.gamma(degrees_of_freedom / 2), q)
    arg3 = (arg31 * arg32) / (arg33 * arg34)

    if isinstance(arg3, mpmath.mpf) or isinstance(arg, mpmath.mpf) or isinstance(arg2, mpmath.mpf):
        return 1 / (1 - q) * mpmath.log(arg3 * arg * arg2)
    else:
        return 1 / (1 - q) * np.log2(arg3 * arg * arg2)


def Renyi_normal_distribution(sigma, alpha):
    if isinstance(sigma, float):
        return Renyi_normal_distribution_1D(sigma, alpha)
    elif isinstance(sigma, (np.matrix)):
        return Renyi_normal_distribution_ND(sigma, alpha)
    else:
        raise ArithmeticError("sigma parameter has wrong type")


def Renyi_normal_distribution_1D(sigma_number, alpha):
    if alpha == 1:
        return math.log(2 * math.pi * math.exp(1) * np.power(sigma_number, 2)) / 2
    else:
        return math.log(2 * math.pi) / 2 + math.log(sigma_number) + math.log(alpha) / (alpha - 1) / 2


def Renyi_normal_distribution_ND(sigma_matrix: np.matrix, alpha, determinant=None):
    dimension = sigma_matrix.shape[0]
    if alpha == 1:
        return math.log(2 * math.pi * math.exp(1)) * dimension / 2.0 + np.log(np.sqrt(np.linalg.det(sigma_matrix)))
    else:
        if determinant:
            return math.log(2 * math.pi) * dimension / 2.0 + math.log(determinant) / 2.0 + dimension * math.log(alpha) / (alpha - 1) / 2
        else:
            return math.log(2 * math.pi) * dimension / 2.0 + math.log(np.linalg.det(sigma_matrix)) / 2.0 + dimension * math.log(alpha) / (alpha - 1) / 2


def Renyi_student_t_distribution_1D(sigma, degrees_of_freedom, alpha):
    if isinstance(sigma, float) or isinstance(sigma, int):
        dimension = 1
        determinant = sigma
    elif isinstance(sigma, np.matrix):
        if len(sigma.shape) == 2 and (sigma.shape[0] == sigma.shape[1]):
            dimension = sigma.shape[0]
            determinant = np.linalg.det(sigma)
        else:
            raise ArithmeticError("sigma parameter has wrong type")
    else:
        raise ArithmeticError("sigma parameter has wrong type")

    if alpha == 1:
        return (degrees_of_freedom + 1.0) / 2 * (spec.digamma((degrees_of_freedom + 1.0) / 2) - spec.digamma((degrees_of_freedom) / 2)) + np.log2(
            np.sqrt(degrees_of_freedom) * spec.beta(0.5, degrees_of_freedom / 2.0))
    else:
        nominator = spec.beta(dimension / 2.0, alpha * (dimension + degrees_of_freedom) / 2.0 - dimension / 2.0)
        denominator = math.pow(spec.beta(degrees_of_freedom / 2.0, dimension / 2.0), alpha)
        beta_factor = math.log2(nominator / denominator)
        return 1 / (1 - alpha) * beta_factor * math.log2(math.pow(np.pi * degrees_of_freedom, dimension) * determinant) - math.log2(spec.gamma(dimension / 2.0))


def Renyi_beta_distribution(a, b, alpha):
    return 1 / (1 - alpha) * math.log2(spec.beta(alpha * a + alpha - 1, alpha * b + alpha - 1) / math.pow(spec.beta(a, b), alpha))


if __name__ == "__main__":
    sigma = np.array([[1, -0.1], [-0.1, 1]])
    samples = 2000000
    dimension = 4
    correlation = 0.2
    save = True

    sigma_skeleton = np.identity(dimension) + correlation * np.eye(dimension, k=1) + correlation * np.eye(dimension, k=-1)
    normal_samples = sample_normal_distribution(sigma=sigma_skeleton, size_sample=samples)
    if save:
        with open("sample_normal.txt", "wt") as fd:
            for item in normal_samples:
                for item_element in item:
                    print(item_element, end=" ", file=fd)
                print("", file=fd)

    stable_linear_depended_samples = sample_linear_depended_stable(sigma=sigma, alpha=1.999, beta=np.array([0.5, -0.5]), delta=np.array([0, 0]),
                                                                   size_sample=samples)
    samples_stable = sample_elliptical_contour_stable(sigma, 1.95, 1.0, np.array([0, 0]), size_sample=samples)
    samples_laplace = sample_asymmetric_laplace_distribution(sigma=sigma, mean=np.array([1, -1]), size_sample=samples)
    samples_student = sample_student_t_distribution(sigma=sigma, mean=np.array([0, 0]), degrees_of_freedom=2, size_sample=samples)

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
        plt.ion()
        plt.show()
        plt.close()
