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

    a = (a - np.mean(a)) / (np.std(a, ddof = 1) * (len(a) - 1))
    b = (b - np.mean(b)) / np.std(b, ddof = 1)
    cor = np.correlate(a, b, 'full')

    return cor[len(cor)//2 - max_lag : len(cor)//2 + max_lag+1]



def kdensity_estimate(a, kernel = 'gaussian', bandwidth = 1.0):
    """
    Estimates kernel density. Uses sklearn.
    kernels: 'gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine'
    """

    from sklearn.neighbors import KernelDensity
    a = a[:, None]
    x = np.linspace(a.min(), a.max(), 100)[:, None]
    kde = KernelDensity(kernel = kernel, bandwidth = bandwidth).fit(a)
    logkde = kde.score_samples(x)

    return np.squeeze(x), np.exp(logkde)



def detrend_with_return(arr, axis = 0):
    """
    Removes the linear trend along the axis, ignoring Nans.
    """
    a = arr.copy()
    rnk = len(a.shape)
    # determine axis
    if axis < 0:
        axis += rnk # axis -1 means along last dimension
    
    # reshape that axis is 1. dimension and other dimensions are enrolled into 2. dimensions
    newdims = np.r_[axis, 0:axis, axis + 1:rnk]
    newdata = np.reshape(np.transpose(a, tuple(newdims)), (a.shape[axis], np.prod(a.shape, axis = 0) // a.shape[axis]))
    newdata = newdata.copy()
    
    # compute linear fit as least squared residuals
    x = np.arange(0, a.shape[axis], 1)
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, newdata)[0]
    
    # remove the trend from the data along 1. axis
    for i in range(a.shape[axis]):
        newdata[i, ...] = newdata[i, ...] - (m*x[i] + c)
    
    # reshape back to original shape
    tdshape = np.take(a.shape, newdims, 0)
    ret = np.reshape(newdata, tuple(tdshape))
    vals = list(range(1,rnk))
    olddims = vals[:axis] + [0] + vals[axis:]
    ret = np.transpose(ret, tuple(olddims))
    
    # return detrended data and linear coefficient
    
    return ret, m, c


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
        raise ValueError("nans after standardizing, "
                         "possibly constant array!")
    x = array[0, :]
    y = array[1, :]
    if len(array) > 2:
        confounds = array[2:, :]
        ortho_confounds = linalg.qr(
            np.fastCopyAndTranspose(confounds), mode='economic')[0].T
        x -= np.dot(np.dot(ortho_confounds, x), ortho_confounds)
        y -= np.dot(np.dot(ortho_confounds, y), ortho_confounds)

    val, pvalwrong = stats.pearsonr(x, y)
    df = float(T - D)
    if df < 1:
        pval = np.nan
        raise ValueError("D > T: Not enough degrees of freedom!")
    else:
        # Two-sided p-value accouting for degrees of freedom
        trafo_val = val*np.sqrt(df/(1. - np.array([val])**2))
        pval = stats.t.sf(np.abs(trafo_val), df)*2

    return val, pval


def get_haar_flucs(ts, min_dt = 2, run_backwards = True):
    """
    Computes Haar fluctuations of the data -- scaling.
    if run_backwards is True, the function runs twice, the second time with reversed time seres,
      this is used for better statistics
    """
    min_dt = min_dt
    max_dt = ts.shape[0]
    dts = np.arange(min_dt, max_dt, 2) # only even as we are dividing the interval into two
    runs = 2 if run_backwards else 1
    haar = np.zeros((dts.shape[0], runs), dtype = np.float32)
    for run in range(runs):
        if run == 1:
            ts = ts[::-1]
        for i in range(dts.shape[0]):
            # split index every dt
            split_ndx = list(np.arange(dts[i], max_dt, dts[i]))
            # split array, result is array with shape [x,dt]
            if ts.shape[0] % dts[i] == 0:
                splitted = np.array(np.split(ts, split_ndx))
            else:
                # if last window is shorter, just omit it
                splitted = np.array(np.split(ts, split_ndx)[:-1])
            # split into two equal parts for averaging -- dt/2, shape is [x, dt/2, 2]
            splitted = splitted.reshape((splitted.shape[0], dts[i]//2, 2), order = "F")
            # average parts over second axis [the dt/2 one]
            means = np.mean(splitted, axis = 1)
            # compute Haar squared with C = 2
            haars = (2*means[:, 1] - 2*means[:, 0])**2
            haar[i, run] = np.mean(haars)
        
    return dts, np.mean(np.sqrt(haar), axis = 1)