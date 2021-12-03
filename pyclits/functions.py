"""
Various helper functions.

(c) Nikola Jajcay
"""

import numpy as np
from scipy import linalg, stats
from sklearn.neighbors import KernelDensity


def cross_correlation(a, b, max_lag):
    """
    Compute cross-correlation with lag. The first argument is with positive lag,
    the second with negative.

    :param a: first 1D argument to correlate
    :type a: np.ndarray
    :param b: second 1D argument to correlate
    :type b: np.ndarray
    :param max_lag: maximum lag to consider
    :type max_lag: int
    :return: lagged cross-correlation between two vectors
    """
    assert a.ndim == 1
    assert b.ndim == 1
    a = (a - np.mean(a)) / (np.std(a, ddof=1) * (len(a) - 1))
    b = (b - np.mean(b)) / np.std(b, ddof=1)
    cor = np.correlate(a, b, "full")

    return cor[len(cor) // 2 - max_lag : len(cor) // 2 + max_lag + 1]


def kdensity_estimate(a, kernel="gaussian", bandwidth=1.0, n_points=100):
    """
    Perform kernel density estimation.

    :param a: 1D time series to estimate its kernel density
    :type a: np.ndarray
    :param kernel: kernel to use, use on of the: 'gaussian', 'tophat',
        'epanechnikov', 'exponential', 'linear', 'cosine'
    :type kernel: str
    :param bandwidth: bandwith for the estimation
    :type bandwidth: float
    :param n_points: how many points on the x-axis
    :type n_points: int
    :return: x values and estimated kernel density
    :rtype: np.ndarray, np.ndarray
    """

    a = a[:, np.newaxis]
    x = np.linspace(a.min(), a.max(), n_points)[:, np.newaxis]
    kde = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(a)
    logkde = kde.score_samples(x)

    return np.squeeze(x), np.exp(logkde)


def partial_corr(a):
    """
    Computes partial correlation of the array `a`. Partial correlation is
    between the first two dimensions, conditioned on others.

    :param a: input array, (dim x time)
    :type a: np.ndarray
    :return: partial correlation between a[0, :] and a[1, :], conditioned on
        a[2:, :], and its p-value
    :rtype: float, float
    """
    array = a.copy()
    D, T = array.shape
    if np.isnan(array).any():
        raise ValueError("NaNs in the array!")
    # standardise
    array -= array.mean(axis=1).reshape(D, 1)
    array /= array.std(axis=1).reshape(D, 1)
    if np.isnan(array).any():
        raise ValueError("NaNs after standardising")
    x = array[0, :]
    y = array[1, :]
    if len(array) > 2:
        confounds = array[2:, :]
        ortho_confounds = linalg.qr(
            np.fastCopyAndTranspose(confounds), mode="economic"
        )[0].T
        x -= np.dot(np.dot(ortho_confounds, x), ortho_confounds)
        y -= np.dot(np.dot(ortho_confounds, y), ortho_confounds)

    val, _ = stats.pearsonr(x, y)
    df = float(T - D)
    if df < 1:
        pval = np.nan
        raise ValueError("D > T: Not enough degrees of freedom!")
    else:
        # Two-sided p-value accouting for degrees of freedom
        trafo_val = val * np.sqrt(df / (1.0 - np.array([val]) ** 2))
        pval = stats.t.sf(np.abs(trafo_val), df) * 2

    return val, float(pval)
