#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
created on May 6, 2015

@author: Nikola Jajcay, jajcay(at)cs.cas.cz

last update on Sep 22, 2017
"""

import collections
import datetime
import logging
import os
import time

import mpmath
import numpy as np
import numpy.random as random
import scipy.special as scipyspecial
from sklearn.neighbors import KDTree


def get_time_series_condition(ts, tau=1, reversed=False, dim_of_condition=1, eta=0,
                              close_condition=False, phase_diff=False, add_cond=None):
    """
    Returns time series for CMI as list in the sense
        I(x; y | z), where x = x(t); y = y(t+tau) | z = [y(t), y(t-eta), y(t-2eta), ...] up to dim_of_condition
    so x -- master time series, y -- slave time series, z -- list of condtions (slave time series in the past)
    tau and eta are forward and backward time lags, respectively.
    If reversed is True, the master (ts[0, :]) and the slave (ts[1, :]) are reversed (for CMI in other direction).
    If close_condition is True, the conditions are returned as
        z = [y(t+tau-1), y(t+tau-1-eta), y(t+tau-1-2eta), ...] up to dim_of_condition,
    so the conditions are closer in temporal sense to the slave time series.
    If phase_diff is True, as y, the phase differences (future - first cond.) will be used (use only with phase data, not raw).
    add_cond if None, is time series of shape [x, len] where x is number of additional time series to be put into the
        condition as of 'now', thus without any backward lag. These additional conditions are NOT counted in the dimension
        of condition parameter.
    """

    if isinstance(ts, list) and len(ts) > 1:
        if len(ts) != 2:
            raise Exception("Input must be a list of 1D arrays (or a 2 x length array).")
        if ts[0].shape != ts[1].shape:
            raise Exception("Both time series must be the same length.") 
        master = ts[1].copy() if reversed else ts[0].copy()
        slave = ts[0].copy() if reversed else ts[1].copy()
    elif isinstance(ts, np.ndarray):
        if np.squeeze(ts).ndim != 2:
            raise Exception("Input must be 2 x length array (or a list of 1D arrays).")
        master = ts[1, :].copy() if reversed else ts[0, :].copy()
        slave = ts[0, :].copy() if reversed else ts[1, :].copy()
    else:
        raise Exception("Input not understood. Use either list of 1D arrays or 2 x length array.")

    if add_cond is not None:
        add_cond = np.atleast_2d(add_cond)
        assert add_cond.shape[1] == master.shape[0]

    if dim_of_condition > 4:
        print("** WARNING -- for %d dimensional condition the estimation might be biased." % (dim_of_condition))
    if (dim_of_condition > 1 and eta == 0):
        raise Exception("For multidimensional condition the backward lag eta must be chosen.")

    n_eta = dim_of_condition - 1
    n = master.shape[0] - tau - n_eta*eta
    if eta is None:
        eta = 0
    else:
        eta = int(eta)

    x = master[n_eta*eta : -tau] # "now"
    if x.shape[0] != n:
        raise Exception("Something went wrong! Check input data.")

    y = slave[n_eta*eta+tau :] # "tau future"
    if y.shape[0] != n:
        raise Exception("Something went wrong! Check input data.")

    z = []
    for i in range(dim_of_condition):
        if close_condition:
            cond = slave[(n_eta-i)*eta+tau-1 : -1-i*eta] # "almost future" until ...
        else:    
            cond = slave[(n_eta-i)*eta : -tau-i*eta] # "now" until "n_eta eta past"
        if cond.shape[0] != n:
            raise Exception("Something went wrong! Check input data.")
        z.append(cond)

    if add_cond is not None:
        for condextra in range(add_cond.shape[0]):
            if close_condition:
                extra_cond = add_cond[condextra, n_eta*eta+tau-1 : -1]
            else:
                extra_cond = add_cond[condextra, n_eta*eta : -tau] # "now"
            if extra_cond.shape[0] != n:
                raise Exception("Something went wrong! Check input data.")
            z.append(extra_cond) 

    if phase_diff:
        y = y - z[0]
        y[y < -np.pi] += 2*np.pi

    return (x, y, z)


def mutual_information(x, y, algorithm='EQQ', bins=8, log2=True):
    """
    Computes mutual information between two time series x and y as
        I(x; y) = sum( p(x,y) * log( p(x,y) / p(x)p(y) ),
        where p(x), p(y) and p(x, y) are probability distributions.
    The probability distributions could be estimated using these algorithms:
        equiquantal binning - algorithm keyword 'EQQ' or 'EQQ2'
            EQQ - equiquantality is forced (even if many samples have the same value 
                at and near the bin edge), can happen that samples with same value fall
                into different bin
            EQQ2 - if more than one sample has the same value at the bin edge, the edge is shifted,
                so that all samples with the same value fall into the same bin, can happen that bins
                do not necessarily contain the same amount of samples
        
        equidistant binning - algorithm keyword 'EQD'
        
        (preparing more...)
    If log2 is True (default), the units of mutual information are bits, if False
      the mutual information is be estimated using natural logarithm and therefore
      units are nats.
    """

    log_f = np.log2 if log2 else np.log

    if algorithm == 'EQD':
        x_bins = bins
        y_bins = bins
        xy_bins = bins

    elif algorithm == 'EQQ':
        # create EQQ bins
        x_sorted = np.sort(x)
        x_bins = [x.min()]
        [x_bins.append(x_sorted[i*x.shape[0]/bins]) for i in range(1, bins)]
        x_bins.append(x.max())

        y_sorted = np.sort(y)
        y_bins = [y.min()]
        [y_bins.append(y_sorted[i*y.shape[0]/bins]) for i in range(1, bins)]
        y_bins.append(y.max())
        
        xy_bins = [x_bins, y_bins]

    elif algorithm == 'EQQ2':
        x_sorted = np.sort(x)
        x_bins = [x.min()]
        one_bin_count = x.shape[0] / bins
        for i in range(1, bins):
            idx = i * one_bin_count
            if np.all(np.diff(x_sorted[idx-1:idx+2]) != 0):
                x_bins.append(x_sorted[idx])
            elif np.any(np.diff(x_sorted[idx-1:idx+2]) == 0):
                where = np.where(np.diff(x_sorted[idx-1:idx+2]) != 0)[0]
                expand_idx = 1
                while where.size == 0:
                    where = np.where(np.diff(x_sorted[idx-expand_idx:idx+1+expand_idx]) != 0)[0]
                    expand_idx += 1
                if where[0] == 0:
                    x_bins.append(x_sorted[idx-expand_idx])
                else:
                    x_bins.append(x_sorted[idx+expand_idx])
        x_bins.append(x.max())

        y_sorted = np.sort(y)
        y_bins = [y.min()]
        one_bin_count = y.shape[0] / bins
        for i in range(1, bins):
            idx = i * one_bin_count
            if np.all(np.diff(y_sorted[idx-1:idx+2]) != 0):
                y_bins.append(y_sorted[idx])
            elif np.any(np.diff(y_sorted[idx-1:idx+2]) == 0):
                where = np.where(np.diff(y_sorted[idx-1:idx+2]) != 0)[0]
                expand_idx = 1
                while where.size == 0:
                    where = np.where(np.diff(y_sorted[idx-expand_idx:idx+1+expand_idx]) != 0)[0]
                    expand_idx += 1
                if where[0] == 0:
                    y_bins.append(y_sorted[idx-expand_idx])
                else:
                    y_bins.append(y_sorted[idx+expand_idx])
        y_bins.append(y.max())

        xy_bins = [x_bins, y_bins]

    # histo
    count_x = np.histogramdd([x], bins = [x_bins])[0]
    count_y = np.histogramdd([y], bins = [y_bins])[0]
    count_xy = np.histogramdd([x, y], bins = xy_bins)[0]

    # normalise
    count_xy /= np.float(np.sum(count_xy))
    count_x /= np.float(np.sum(count_x))
    count_y /= np.float(np.sum(count_y))

    # sum
    mi = 0
    for i in range(bins):
        for j in range(bins):
            if count_x[i] != 0 and count_y[j] != 0 and count_xy[i, j] != 0:
                mi += count_xy[i, j] * log_f(count_xy[i, j] / (count_x[i] * count_y[j]))

    return mi



def knn_mutual_information(x, y, k, standardize = True, symm_algorithm = True, dualtree = True):
    """
    Computes mutual information between two time series x and y as
        I(x; y) = sum( p(x,y) * log( p(x,y) / p(x)p(y) ),
        where p(x), p(y) and p(x, y) are probability distributions.
    Performs k-nearest neighbours search using k-dimensional tree.
    Uses sklearn.neighbors for KDTree class.
    
    standardize - whether transform data to zero mean and unit variance
    symm_algorithm
      True - use symmetric algorithm with one eps in both dimensions
      False - use different eps_x and eps_y in respective dimensions -- SOME PROBLEMS, DON'T USE NOW
    dualtree - whether to use dualtree formalism in k-d tree for the k-NN search
      could lead to better performance with large N

    According to Kraskov A., Stogbauer H. and Grassberger P., Phys. Rev. E, 69, 2004.
    """

    from sklearn.neighbors import KDTree
    from scipy.special import digamma

    # prepare data
    if standardize:
        x = _center_ts(x)
        y = _center_ts(y)
    data = np.vstack([x, y]).T

    # build k-d tree using the maximum (Chebyshev) norm
    tree = KDTree(data, leaf_size = 15, metric = "chebyshev")
    # find k-nearest neighbour indices for each point
    dist, ind = tree.query(data, k = k + 1, return_distance = True, dualtree = dualtree)

    sum_ = 0
    for n in range(data.shape[0]):
        # find x and y distances between nth point and its k-nearest neighbour
        eps_x = np.abs(data[n, 0] - data[ind[n, -1], 0])
        eps_y = np.abs(data[n, 1] - data[ind[n, -1], 1])
        if symm_algorithm:
            # use symmetric algorithm with one eps - see the paper
            eps = np.max((eps_x, eps_y))
            # find number of points within eps distance
            n_x = np.sum(np.less(np.abs(x - x[n]), eps)) - 1
            n_y = np.sum(np.less(np.abs(y - y[n]), eps)) - 1
            # add to digamma sum
            sum_ += digamma(n_x + 1) + digamma(n_y + 1)
        else:
            # use asymmetric algorithm with eps_x and eps_y - see the paper
            # find number of points within eps distance
            n_x = np.sum(np.less(np.abs(x - x[n]), eps_x)) - 1
            n_y = np.sum(np.less(np.abs(y - y[n]), eps_y)) - 1
            # add to digamma sum
            if n_x != 0:
                sum_ += digamma(n_x) 
            if n_y != 0:
                sum_ += digamma(n_y)

    sum_ /= data.shape[0]

    if symm_algorithm:
        return digamma(k) - sum_ + digamma(data.shape[0])
    else:
        return digamma(k) - 1./k - sum_ + digamma(data.shape[0])



def _center_ts(ts):
    """
    Returns centered time series with zero mean and unit variance.
    """

    if np.squeeze(ts).ndim != 1:
        raise Exception("Only 1D time series can be centered")
    ts -= np.mean(ts)
    ts /= np.std(ts, ddof = 1)

    return ts



def _get_corr_entropy(list_ts, log2 = True):
    """
    Returns modified entropy to use in Gaussian correlation matrix CMI computation.
        H = -0.5 * sum( log(eigvals) )
    where eigvals are eigenvalues of correlation matrix between time series.
    """

    log_f = np.log2 if log2 else np.log

    corr_matrix = np.corrcoef(list_ts)
    eigvals = np.linalg.eigvals(corr_matrix)
    eigvals = eigvals[eigvals > 0.]

    return -0.5 * np.nansum(log_f(eigvals))



def cond_mutual_information(x, y, z, algorithm = 'EQQ', bins = 8, log2 = True):
    """
    Computes conditional mutual information between two time series x and y 
    conditioned on a third z (which can be multi-dimensional) as
        I(x; y | z) = sum( p(x,y,z) * log( p(z)*p(x,y,z) / p(x,z)*p(y,z) ),
        where p(z), p(x,z), p(y,z) and p(x,y,z) are probability distributions.
    The probability distributions could be estimated using these algorithms:
        equiquantal binning - algorithm keyword 'EQQ' or 'EQQ2'
            EQQ - equiquantality is forced (even if many samples have the same value 
                at and near the bin edge), can happen that samples with same value fall
                into different bin
            EQQ2 - if more than one sample has the same value at the bin edge, the edge is shifted,
                so that all samples with the same value fall into the same bin, can happen that bins
                do not necessarily contain the same amount of samples
        
        equidistant binning - algorithm keyword 'EQD'
        
        Gaussian correlation matrix - algorithm keyword 'GCM' (for phase - amplitude dependence)
        
        (preparing more...)
    If log2 is True (default), the units of cond. mutual information are bits, if False
      the mutual information is estimated using natural logarithm and therefore
      units are nats.
    """

    log_f = np.log2 if log2 else np.log

    # binning algorithms
    if 'EQ' in algorithm:
        # for multi-dimensional condition -- create array from list as (dim x length of ts)
        x = np.atleast_2d(x)
        y = np.atleast_2d(y)
        z = np.atleast_2d(z)

        if algorithm == 'EQQ':
            # create EQQ bins -- condition [possibly multidimensional]
            z_bins = [] # arrays of bins for all conditions
            for cond in range(z.shape[0]):
                z_sorted = np.sort(z[cond, :])
                z_bin = [z_sorted.min()]
                [z_bin.append(z_sorted[i*z_sorted.shape[0]/bins]) for i in range(1, bins)]
                z_bin.append(z_sorted.max())
                z_bins.append(np.array(z_bin))

            # create EQQ bins -- variables
            x_bins = []
            for cond in range(x.shape[0]):
                x_sorted = np.sort(x[cond, :])
                x_bin = [x_sorted.min()]
                [x_bin.append(x_sorted[i*x_sorted.shape[0]/bins]) for i in range(1, bins)]
                x_bin.append(x_sorted.max())
                x_bins.append(np.array(x_bin))

            y_bins = []
            for cond in range(y.shape[0]):
                y_sorted = np.sort(y[cond, :])
                y_bin = [y_sorted.min()]
                [y_bin.append(y_sorted[i*y_sorted.shape[0]/bins]) for i in range(1, bins)]
                y_bin.append(y_sorted.max())
                y_bins.append(np.array(y_bin))

            # create multidim bins for histogram
            xyz_bins = x_bins + y_bins + z_bins

            xz_bins = x_bins + z_bins

            yz_bins = y_bins + z_bins


        elif algorithm == 'EQQ2':
            # create EQQ bins -- condition [possibly multidimensional]
            z_bins = [] # arrays of bins for all conditions
            for cond in range(z.shape[0]):
                z_sorted = np.sort(z[cond, :])
                z_bin = [z_sorted.min()]
                one_bin_count = z_sorted.shape[0] / bins
                for i in range(1, bins):
                    idx = i * one_bin_count
                    if np.all(np.diff(z_sorted[idx-1:idx+2]) != 0):
                        z_bin.append(z_sorted[idx])
                    elif np.any(np.diff(z_sorted[idx-1:idx+2]) == 0):
                        where = np.where(np.diff(z_sorted[idx-1:idx+2]) != 0)[0]
                        expand_idx = 1
                        while where.size == 0:
                            where = np.where(np.diff(z_sorted[idx-expand_idx:idx+1+expand_idx]) != 0)[0]
                            expand_idx += 1
                        if where[0] == 0:
                            z_bin.append(z_sorted[idx-expand_idx])
                        else:
                            z_bin.append(z_sorted[idx+expand_idx])
                z_bin.append(z_sorted.max())
                z_bin = np.array(z_bin)
                z_bins.append(np.array(z_bin))

            # create EQQ bins -- variables
            x_bins = [] # arrays of bins for all conditions
            for cond in range(x.shape[0]):
                x_sorted = np.sort(x[cond, :])
                x_bin = [x_sorted.min()]
                one_bin_count = x_sorted.shape[0] / bins
                for i in range(1, bins):
                    idx = i * one_bin_count
                    if np.all(np.diff(x_sorted[idx-1:idx+2]) != 0):
                        x_bin.append(x_sorted[idx])
                    elif np.any(np.diff(x_sorted[idx-1:idx+2]) == 0):
                        where = np.where(np.diff(x_sorted[idx-1:idx+2]) != 0)[0]
                        expand_idx = 1
                        while where.size == 0:
                            where = np.where(np.diff(x_sorted[idx-expand_idx:idx+1+expand_idx]) != 0)[0]
                            expand_idx += 1
                        if where[0] == 0:
                            x_bin.append(x_sorted[idx-expand_idx])
                        else:
                            x_bin.append(x_sorted[idx+expand_idx])
                x_bin.append(x_sorted.max())
                x_bin = np.array(x_bin)
                x_bins.append(np.array(x_bin))

            y_bins = [] # arrays of bins for all conditions
            for cond in range(y.shape[0]):
                y_sorted = np.sort(y[cond, :])
                y_bin = [y_sorted.min()]
                one_bin_count = y_sorted.shape[0] / bins
                for i in range(1, bins):
                    idx = i * one_bin_count
                    if np.all(np.diff(y_sorted[idx-1:idx+2]) != 0):
                        y_bin.append(y_sorted[idx])
                    elif np.any(np.diff(y_sorted[idx-1:idx+2]) == 0):
                        where = np.where(np.diff(y_sorted[idx-1:idx+2]) != 0)[0]
                        expand_idx = 1
                        while where.size == 0:
                            where = np.where(np.diff(y_sorted[idx-expand_idx:idx+1+expand_idx]) != 0)[0]
                            expand_idx += 1
                        if where[0] == 0:
                            y_bin.append(y_sorted[idx-expand_idx])
                        else:
                            y_bin.append(y_sorted[idx+expand_idx])
                y_bin.append(y_sorted.max())
                y_bin = np.array(y_bin)
                y_bins.append(np.array(y_bin))

            # create multidim bins for histogram
            xyz_bins = x_bins + y_bins + z_bins

            xz_bins = x_bins + z_bins

            yz_bins = y_bins + z_bins

        
        elif algorithm == 'EQD':
            # bins are just integer
            xyz_bins = bins
            xz_bins = bins
            yz_bins = bins
            z_bins = bins

        # histo
        count_z = np.histogramdd(z.T, bins = z_bins)[0]

        xyz = np.vstack((x, y, z))
        count_xyz = np.histogramdd(xyz.T, bins = xyz_bins)[0]

        xz = np.vstack((x, z))
        count_xz = np.histogramdd(xz.T, bins = xz_bins)[0]

        yz = np.vstack((y, z))
        count_yz = np.histogramdd(yz.T, bins = yz_bins)[0]

        # normalise
        count_z /= np.float(np.sum(count_z))
        count_xyz /= np.float(np.sum(count_xyz))
        count_xz /= np.float(np.sum(count_xz))
        count_yz /= np.float(np.sum(count_yz))

        # sum
        cmi = 0
        iterator = np.nditer(count_xyz, flags = ['multi_index'])
        while not iterator.finished:
            idx = iterator.multi_index
            xz_idx = tuple([ item for sublist in [idx[:len(x)], idx[-len(z):]] for item in sublist ]) # creates index for xz histo
            yz_idx = idx[-len(z)-len(y):]
            z_idx = idx[-len(z):]
            if count_xyz[idx] == 0 or count_z[z_idx] == 0 or count_xz[xz_idx] == 0 or count_yz[yz_idx] == 0:
                iterator.iternext()
                continue
            else:
                cmi += count_xyz[idx] * log_f(count_z[z_idx] * count_xyz[idx] / (count_xz[xz_idx] * count_yz[yz_idx]))

            iterator.iternext()

    elif algorithm == 'GCM':
        if len(z) <= 1:
            raise Exception("Gaussian correlation matrix method should be used with multidimensional condition.")
        
        # center time series - zero mean, unit variance
        x = _center_ts(x)
        y = _center_ts(y)
        for cond_ts in z:
            cond_ts = _center_ts(cond_ts)

        # get CMI
        Hall = _get_corr_entropy([x, y] + list(z), log2 = log2)
        Hxz = _get_corr_entropy([x] + list(z), log2 = log2)
        Hyz = _get_corr_entropy([y] + list(z), log2 = log2)
        Hz = _get_corr_entropy(z, log2 = log2)

        cmi = Hall - Hxz - Hyz + Hz

    return cmi


def _neg_harmonic(n):
    """
    Returns a negative Nth harmonic number.
    For knn CMI computation
    """

    return -np.sum(1./np.arange(1,n+1))


def knn_cond_mutual_information(x, y, z, k, standardize = True, dualtree = True):
    """
    Computes conditional mutual information between two time series x and y 
    conditioned on a third z (which can be multi-dimensional) as
        I(x; y | z) = sum( p(x,y,z) * log( p(z)*p(x,y,z) / p(x,z)*p(y,z) ),
        where p(z), p(x,z), p(y,z) and p(x,y,z) are probability distributions.
    Performs k-nearest neighbours search using k-dimensional tree.
    Uses sklearn.neighbors for KDTree class.

    standardize - whether transform data to zero mean and unit variance
    dualtree - whether to use dualtree formalism in k-d tree for the k-NN search
      could lead to better performance with large N

    According to Frenzel S. and Pompe B., Phys. Rev. Lett., 99, 2007.
    """

    from sklearn.neighbors import KDTree

    # prepare data
    if standardize:
        x = _center_ts(x)
        y = _center_ts(y)
        if isinstance(z, np.ndarray):
            z = _center_ts(z)
        elif isinstance(z, list):
            for cond_ts in z:
                cond_ts = _center_ts(cond_ts)
    z = np.atleast_2d(z)
    data = np.vstack([x, y, z]).T

    # build k-d tree using the maximum (Chebyshev) norm
    tree = KDTree(data, leaf_size = 15, metric = "chebyshev")
    # find distance to k-nearest neighbour per point
    dist, _ = tree.query(data, k = k + 1, return_distance = True, dualtree = dualtree)

    sum_ = 0
    # prepare marginal vectors xz, yz and z
    n_x_z_data = np.delete(data, 1, axis = 1)
    n_y_z_data = np.delete(data, 0, axis = 1)
    n_z_data = np.delete(data, [0, 1], axis = 1)

    # build and query k-d trees in marginal spaces for number of points in a given dist from a point
    tree_x_z = KDTree(n_x_z_data, leaf_size = 15, metric = "chebyshev")
    n_x_z = tree_x_z.query_radius(n_x_z_data, r = dist[:, -1], count_only = True) - 2
    tree_y_z = KDTree(n_y_z_data, leaf_size = 15, metric = "chebyshev")
    n_y_z = tree_y_z.query_radius(n_y_z_data, r = dist[:, -1], count_only = True) - 2
    tree_z = KDTree(n_z_data, leaf_size = 15, metric = "chebyshev")
    n_z = tree_z.query_radius(n_z_data, r = dist[:, -1], count_only = True) - 2

    # count points
    for n in range(data.shape[0]):
        sum_ += _neg_harmonic(n_x_z[n]) + _neg_harmonic(n_y_z[n]) - _neg_harmonic(n_z[n])

    sum_ /= data.shape[0]

    return sum_ - _neg_harmonic(k-1)
