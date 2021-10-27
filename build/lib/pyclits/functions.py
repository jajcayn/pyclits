"""
created on Sep 22, 2017

@author: Nikola Jajcay, jajcay(at)cs.cas.cz
"""

import numpy as np


def cross_correlation(a, b, max_lag):
    """
    Cross correlation with lag.
    When computing cross-correlation, the first parameter, a, is
    in 'future' with positive lag and in 'past' with negative lag.
    """

    a = (a - np.mean(a)) / (np.std(a, ddof=1) * (len(a) - 1))
    b = (b - np.mean(b)) / np.std(b, ddof=1)
    cor = np.correlate(a, b, "full")

    return cor[len(cor) // 2 - max_lag : len(cor) // 2 + max_lag + 1]


def kdensity_estimate(a, kernel="gaussian", bandwidth=1.0):
    """
    Estimates kernel density. Uses sklearn.
    kernels: 'gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine'
    """

    from sklearn.neighbors import KernelDensity

    a = a[:, None]
    x = np.linspace(a.min(), a.max(), 100)[:, None]
    kde = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(a)
    logkde = kde.score_samples(x)

    return np.squeeze(x), np.exp(logkde)


def partial_corr(a):
    """
    Computes partial correlation of array a.
    Array as dim x time; partial correlation is between first two dimensions, conditioned on others.
    """

    from scipy import linalg, stats

    array = a.copy()
    D, T = array.shape
    if np.isnan(array).sum() != 0:
        raise ValueError("nans in the array!")
    # Standardize
    array -= array.mean(axis=1).reshape(D, 1)
    array /= array.std(axis=1).reshape(D, 1)
    if np.isnan(array).sum() != 0:
        raise ValueError(
            "nans after standardizing, " "possibly constant array!"
        )
    x = array[0, :]
    y = array[1, :]
    if len(array) > 2:
        confounds = array[2:, :]
        ortho_confounds = linalg.qr(
            np.fastCopyAndTranspose(confounds), mode="economic"
        )[0].T
        x -= np.dot(np.dot(ortho_confounds, x), ortho_confounds)
        y -= np.dot(np.dot(ortho_confounds, y), ortho_confounds)

    val, pvalwrong = stats.pearsonr(x, y)
    df = float(T - D)
    if df < 1:
        pval = np.nan
        raise ValueError("D > T: Not enough degrees of freedom!")
    else:
        # Two-sided p-value accouting for degrees of freedom
        trafo_val = val * np.sqrt(df / (1.0 - np.array([val]) ** 2))
        pval = stats.t.sf(np.abs(trafo_val), df) * 2

    return val, pval
