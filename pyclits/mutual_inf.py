"""
Functions for (conditional) mutual information using Shannon entropy measure.
"""


import logging

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma

# leaf sizes for k-d trees used in knn estimators
LEAF_SIZE = 15


def get_conditioned_timeseries(
    timeseries,
    tau=1,
    reversed=False,
    dim_of_condition=1,
    eta=0,
    close_condition=False,
    phase_diff=False,
    add_cond=None,
):
    """
    Transforms two timeseries to a conditioned format which is used for
    computing causality using CMI along the lines: I(x; y | z), where
        x = x(t); y = y(t+tau) | z = [y(t), y(t-eta), y(t-2eta), ...] up to
        `dim_of_condition`
    so that x is master time series, y is slave time series, z is a list of
    condtions (slave time series in the past).

    :param timeseries: timeseries to transform, either list or 2D array, by
        default first one is `master` timeseries, second is `slave` timeseries
    :type timeseries: np.ndarray|list
    :param tau: forward time lag
    :type tau: int
    :param reversed: switch master and slave timeseries - compute CMI in other
        direction
    :type reversed: bool
    :param dim_of_condition: dimensionality of conditioning in CMI (the length
        of z)
    :type dim_of_condition: int
    :param eta: backward time lag
    :type eta: int
    :param close_condition: if True, the conditions are returned as
        z = [y(t+tau-1), y(t+tau-1-eta), y(t+tau-1-2eta), ...]
    :type close_condition: bool
    :param phase_diff: the phase differences (future - first cond.) will be used
        in y
    :type phase_diff: bool
    :param add_cond: optional additional condition for the CMI (added to z), an
        array of [x, len] where x is number of additional time series to be put
        into the condition as of `now`, thus without any backward lag; these
        additional conditions are NOT counted in the dimension of condition
        parameter
    :type add_cond: np.ndarray|None
    :return: correctly shifted timeseries as per CMI
    :rtype: (np.ndarray, np.ndarray, list[np.ndarray])
    """
    if isinstance(timeseries, list):
        assert len(timeseries) == 2
        assert timeseries[0].shape == timeseries[1].shape
        master = timeseries[1].copy() if reversed else timeseries[0].copy()
        slave = timeseries[0].copy() if reversed else timeseries[1].copy()
    elif isinstance(timeseries, np.ndarray):
        assert np.squeeze(timeseries).ndim == 2
        master = (
            timeseries[1, :].copy() if reversed else timeseries[0, :].copy()
        )
        slave = timeseries[0, :].copy() if reversed else timeseries[1, :].copy()

    if add_cond is not None:
        add_cond = np.atleast_2d(add_cond)
        assert add_cond.shape[1] == master.shape[0]

    if dim_of_condition > 4:
        logging.warning(
            f"For {dim_of_condition} dimensional condition the estimation might"
            " be biased."
        )
    if dim_of_condition > 1 and eta == 0:
        raise ValueError(
            "For multi-D condition the backward lag `eta` must be chosen."
        )

    n_eta = dim_of_condition - 1
    eta = int(eta or 0)
    resulting_length = int(master.shape[0] - tau - n_eta * eta)

    # master as of `now`
    x = master[n_eta * eta : -tau]
    assert x.shape[0] == resulting_length, (
        f"master: {x.shape[0]}, should be: {resulting_length}, tau: {tau}, eta:"
        f" {eta}, dim cond. {dim_of_condition}"
    )

    # slave in the `tau future`
    y = slave[n_eta * eta + tau :]
    assert y.shape[0] == resulting_length, (
        f"slave: {y.shape[0]}, should be: {resulting_length}, tau: {tau}, eta:"
        f" {eta}, dim cond. {dim_of_condition}"
    )

    z = []
    for i in range(dim_of_condition):
        if close_condition:
            # condition in `almost future`
            cond = slave[(n_eta - i) * eta + tau - 1 : -1 - i * eta]
        else:
            # condition in `now` until `n_eta eta past`
            cond = slave[(n_eta - i) * eta : -tau - i * eta]
        assert cond.shape[0] == resulting_length
        z.append(cond)

    if add_cond is not None:
        for condextra in range(add_cond.shape[0]):
            if close_condition:
                extra_cond = add_cond[condextra, n_eta * eta + tau - 1 : -1]
            else:
                extra_cond = add_cond[condextra, n_eta * eta : -tau]
            assert extra_cond.shape[0] == resulting_length
            z.append(extra_cond)

    if phase_diff:
        y = y - z[0]
        y[y < -np.pi] += 2 * np.pi

    return (x, y, z)


def _standardize_ts(ts):
    """
    Returns centered time series with zero mean and unit variance.
    """
    assert np.squeeze(ts).ndim == 1, "Only 1D time series can be centered"
    ts -= np.mean(ts)
    ts /= np.std(ts, ddof=1)

    return ts


def _create_naive_eqq_bins(ts, no_bins):
    """
    Create naive EQQ bins given the timeseries.
    """
    assert ts.ndim == 1, "Only 1D timeseries supported"
    ts_sorted = np.sort(ts)
    # bins start with minimum
    ts_bins = [ts.min()]
    for i in range(1, no_bins):
        # add bin edge according to number of bins
        ts_bins.append(ts_sorted[int(i * ts.shape[0] / no_bins)])
    # add last bin - maximum
    ts_bins.append(ts.max())
    return ts_bins


def _create_shifted_eqq_bins(ts, no_bins):
    """
    Create EQQ bins with possible shift if the same values would fall into
    different bins.
    """
    assert ts.ndim == 1, "Only 1D timeseries supported"
    ts_sorted = np.sort(ts)
    # bins start with minimum
    ts_bins = [ts.min()]
    # ideal case
    one_bin_count = ts.shape[0] / no_bins
    for i in range(1, no_bins):
        idx = int(i * one_bin_count)
        if np.all(np.diff(ts_sorted[idx - 1 : idx + 2]) != 0):
            ts_bins.append(ts_sorted[idx])
        elif np.any(np.diff(ts_sorted[idx - 1 : idx + 2]) == 0):
            where = np.where(np.diff(ts_sorted[idx - 1 : idx + 2]) != 0)[0]
            expand_idx = 1
            while where.size == 0:
                where = np.where(
                    np.diff(ts_sorted[idx - expand_idx : idx + 1 + expand_idx])
                    != 0
                )[0]
                expand_idx += 1
            if where[0] == 0:
                ts_bins.append(ts_sorted[idx - expand_idx])
            else:
                ts_bins.append(ts_sorted[idx + expand_idx])
    ts_bins.append(ts.max())
    return ts_bins


def _neg_harmonic(n):
    """
    Returns a negative Nth harmonic number.
    """
    return -np.sum(1.0 / np.arange(1, n + 1))


def _get_corr_entropy(list_ts, log2=True):
    """
    Returns modified entropy to use in Gaussian correlation matrix CMI
    computation as
        H = -0.5 * sum( log(eigvals) )
    where eigvals are eigenvalues of correlation matrix between time series.
    """

    log_f = np.log2 if log2 else np.log

    corr_matrix = np.corrcoef(list_ts)
    assert corr_matrix.ndim >= 2, corr_matrix
    eigvals = np.linalg.eigvals(corr_matrix)
    eigvals = eigvals[eigvals > 0.0]

    return -0.5 * np.nansum(log_f(eigvals))


def mutual_information(
    x, y, algorithm="EQQ", bins=None, k=None, log2=True, standardize=True
):
    """
    Compute mutual information between two timeseries x and y as
        I(x; y) = sum( p(x,y) * log( p(x,y) / p(x)p(y) )
    where p(x), p(y) and p(x, y) are probability distributions.

    :param x: first timeseries, has to be 1D
    :type x: np.ndarray
    :param y: second timeseries, has to be 1D
    :type y: np.ndarray
    :param algorithm: which algorithm to use for probability density estimation:
        - EQD: equidistant binning [1]
        - EQQ_naive: naive equiquantal binning, can happen that samples with
            same value fall into different bins [2]
        - EQQ: equiquantal binning with edge shifting, if same values happen to
            be at the bin edge, the edge is shifted so that all samples of the
            same value will fall into the same bin, can happen that not all the
            bins have necessarily the same number of samples [2]
        - knn: k-nearest neighbours search using k-dimensional tree [3]
        number of bins, at least for EQQ algorithms, should not exceed 3rd root
            of the number of the data samples, in case of I(x,y), i.e. MI of
            two variables [2]
    :param bins: number of bins for binning algorithms
    :type bins: int|None
    :param k: number of neighbours for knn algorithm
    :type k: int|None
    :param log2: whether to use log base 2 for binning algorithms, then the
        units are bits, if False, will use natural log which makes the units
        nats
    :type log2: bool
    :param standardize: whether to standardize timeseries before computing MI,
        i.e. transformation to zero mean and unit variance
    :type standardize: bool
    :return: estimate of the mutual information between x and y
    :rtype: float

    [1] Butte, A. J., & Kohane, I. S. (1999). Mutual information relevance
        networks: functional genomic clustering using pairwise entropy
        measurements. In Biocomputing 2000 (pp. 418-429).
    [2] Paluš, M. (1995). Testing for nonlinearity using redundancies:
        Quantitative and qualitative aspects. Physica D: Nonlinear Phenomena,
        80(1-2), 186-205.
    [3] Kraskov, A., Stögbauer, H., & Grassberger, P. (2004). Estimating mutual
        information. Physical review E, 69(6), 066138.
    """
    assert x.ndim == 1 and y.ndim == 1, "Only 1D timeseries supported"
    if standardize:
        x = _standardize_ts(x)
        y = _standardize_ts(y)

    if algorithm == "knn":
        assert k is not None, "For knn algorithm, `k` must be provided"
        data = np.vstack([x, y]).T
        # build k-d tree
        tree = cKDTree(data, leafsize=LEAF_SIZE)
        # find k-nearest neighbour indices for each point, use the maximum
        # (Chebyshev) norm, which is also limit p -> infinity in Minkowski
        _, ind = tree.query(data, k=k + 1, p=np.inf)
        sum_ = 0
        for n in range(data.shape[0]):
            # find x and y distances between nth point and its k-nearest
            # neighbour
            eps_x = np.abs(data[n, 0] - data[ind[n, -1], 0])
            eps_y = np.abs(data[n, 1] - data[ind[n, -1], 1])
            # use symmetric algorithm with one eps - see the paper
            eps = np.max((eps_x, eps_y))
            # find number of points within eps distance
            n_x = np.sum(np.less(np.abs(x - x[n]), eps)) - 1
            n_y = np.sum(np.less(np.abs(y - y[n]), eps)) - 1
            # add to digamma sum
            sum_ += digamma(n_x + 1) + digamma(n_y + 1)

        sum_ /= data.shape[0]

        mi = digamma(k) - sum_ + digamma(data.shape[0])

    elif algorithm.startswith("E"):
        assert (
            bins is not None
        ), "For binning algorithms, `bins` must be provided"
        log_f = np.log2 if log2 else np.log

        if algorithm == "EQD":
            # bins are simple number of bins - will be divided equally
            x_bins = bins
            y_bins = bins

        elif algorithm == "EQQ_naive":
            x_bins = _create_naive_eqq_bins(x, no_bins=bins)
            y_bins = _create_naive_eqq_bins(y, no_bins=bins)

        elif algorithm == "EQQ":
            x_bins = _create_shifted_eqq_bins(x, no_bins=bins)
            y_bins = _create_shifted_eqq_bins(y, no_bins=bins)

        else:
            raise ValueError(f"Unknown MI algorithm: {algorithm}")

        xy_bins = [x_bins, y_bins]

        # compute histogram counts
        count_x = np.histogramdd([x], bins=[x_bins])[0]
        count_y = np.histogramdd([y], bins=[y_bins])[0]
        count_xy = np.histogramdd([x, y], bins=xy_bins)[0]

        # normalise
        count_xy /= np.float(np.sum(count_xy))
        count_x /= np.float(np.sum(count_x))
        count_y /= np.float(np.sum(count_y))

        # sum
        mi = 0
        for i in range(bins):
            for j in range(bins):
                if count_x[i] != 0 and count_y[j] != 0 and count_xy[i, j] != 0:
                    mi += count_xy[i, j] * log_f(
                        count_xy[i, j] / (count_x[i] * count_y[j])
                    )

    else:
        raise ValueError(f"Unknown MI algorithm: {algorithm}")

    return mi


def conditional_mutual_information(
    x, y, z, algorithm="EQQ", bins=None, k=None, log2=True, standardize=True
):
    """
    Compute conditional mutual information between two timeseries x and y,
    conditioned on a third z (can be multidimensional) as
        I(x; y | z) = sum( p(x,y,z) * log( p(z)*p(x,y,z) / p(x,z)*p(y,z) ),
    where p(z), p(x,z), p(y,z) and p(x,y,z) are probability distributions.

    :param x: first timeseries, has to be 1D
    :type x: np.ndarray
    :param y: second timeseries, has to be 1D
    :type y: np.ndarray
    :param z: conditional timeseries, can be multidimensional
    :type z: list of 1D timeseries
    :param algorithm: which algorithm to use for probability density estimation:
        - EQD: equidistant binning [1]
        - EQQ_naive: naive equiquantal binning, can happen that samples with
            same value fall into different bins [2]
        - EQQ: equiquantal binning with edge shifting, if same values happen to
            be at the bin edge, the edge is shifted so that all samples of the
            same value will fall into the same bin, can happen that not all the
            bins have necessarily the same number of samples [2]
        - knn: k-nearest neighbours search using k-dimensional tree [3]
        - GCM: Gaussian correlation matrix, uses correlation entropies
        number of bins, at least for EQQ algorithms, should not exceed 3rd root
            of the number of the data samples, in case of I(x,y), i.e. MI of
            two variables [2]
    :param bins: number of bins for binning algorithms
    :type bins: int|None
    :param k: number of neighbours for knn algorithm
    :type k: int|None
    :param log2: whether to use log base 2 for binning algorithms, then the
        units are bits, if False, will use natural log which makes the units
        nats
    :type log2: bool
    :param standardize: whether to standardize timeseries before computing MI,
        i.e. transformation to zero mean and unit variance
    :type standardize: bool
    :return: estimate of the conditional mutual information between x and y,
        conditioned on z
    :rtype: float

    [1] Butte, A. J., & Kohane, I. S. (1999). Mutual information relevance
        networks: functional genomic clustering using pairwise entropy
        measurements. In Biocomputing 2000 (pp. 418-429).
    [2] Paluš, M. (1995). Testing for nonlinearity using redundancies:
        Quantitative and qualitative aspects. Physica D: Nonlinear Phenomena,
        80(1-2), 186-205.
    [3] Frenzel, S., & Pompe, B. (2007). Partial mutual information for
        coupling analysis of multivariate time series. Physical review letters,
        99(20), 204101.
    """
    assert x.ndim == 1 and y.ndim == 1, "Only 1D timeseries supported"
    if isinstance(z, np.ndarray):
        z = [z]
    assert isinstance(z, (list, tuple))
    assert all(zi.ndim == 1 for zi in z), "Only 1D timeseries supported"
    if standardize:
        x = _standardize_ts(x)
        y = _standardize_ts(y)
        z = [_standardize_ts(zi) for zi in z]

    if algorithm == "knn":
        assert k is not None, "For knn algorithm, `k` must be provided"
        z = np.atleast_2d(z)
        data = np.vstack([x, y, z]).T
        # build k-d tree
        tree = cKDTree(data, leafsize=LEAF_SIZE)
        # find k-nearest neighbour indices for each point, use the maximum
        # (Chebyshev) norm, which is also limit p -> infinity in Minkowski
        dist, _ = tree.query(data, k=k + 1, p=np.inf)

        sum_ = 0
        # prepare marginal vectors xz, yz and z
        n_x_z_data = np.delete(data, 1, axis=1)
        n_y_z_data = np.delete(data, 0, axis=1)
        n_z_data = np.delete(data, [0, 1], axis=1)

        def query_radius_count_only(tree, data, radius):
            """
            Count all points within distance `radius` of points in `data` using
            already built `tree`
            """
            assert isinstance(tree, cKDTree)
            return np.array(
                [
                    len(radius_points)
                    # use the maximum (Chebyshev) norm, which is also limit
                    # p -> infinity in Minkowski
                    for radius_points in tree.query_ball_point(
                        data, r=radius, p=np.inf
                    )
                ]
            )

        # build and query k-d trees in marginal spaces for number of points in
        # a given dist from a point
        tree_x_z = cKDTree(n_x_z_data, leafsize=LEAF_SIZE)
        n_x_z = query_radius_count_only(tree_x_z, n_x_z_data, dist[:, -1]) - 2
        tree_y_z = cKDTree(n_y_z_data, leafsize=LEAF_SIZE)
        n_y_z = query_radius_count_only(tree_y_z, n_y_z_data, dist[:, -1]) - 2
        tree_z = cKDTree(n_z_data, leafsize=LEAF_SIZE)
        n_z = query_radius_count_only(tree_z, n_z_data, dist[:, -1]) - 2

        # count points
        for n in range(data.shape[0]):
            sum_ += (
                _neg_harmonic(n_x_z[n])
                + _neg_harmonic(n_y_z[n])
                - _neg_harmonic(n_z[n])
            )

        sum_ /= data.shape[0]
        cmi = sum_ - _neg_harmonic(k - 1)

    elif algorithm == "GCM":
        # get CMI
        Hall = _get_corr_entropy([x, y] + list(z), log2=log2)
        Hxz = _get_corr_entropy([x] + list(z), log2=log2)
        Hyz = _get_corr_entropy([y] + list(z), log2=log2)
        Hz = _get_corr_entropy(z, log2=log2)

        cmi = Hall - Hxz - Hyz + Hz

    elif algorithm.startswith("E"):
        assert (
            bins is not None
        ), "For binning algorithms, `bins` must be provided"
        log_f = np.log2 if log2 else np.log
        x = np.atleast_2d(x)
        y = np.atleast_2d(y)
        z = np.atleast_2d(z)

        if algorithm == "EQD":
            # bins are simple number of bins - will be divided equally
            xyz_bins = bins
            xz_bins = bins
            yz_bins = bins
            z_bins = bins
        elif algorithm == "EQQ_naive":
            x_bins = [
                np.array(_create_naive_eqq_bins(x[i, :], no_bins=bins))
                for i in range(x.shape[0])
            ]
            y_bins = [
                np.array(_create_naive_eqq_bins(y[i, :], no_bins=bins))
                for i in range(y.shape[0])
            ]
            z_bins = [
                np.array(_create_naive_eqq_bins(z[i, :], no_bins=bins))
                for i in range(z.shape[0])
            ]
            # create multidim bins for histogram
            xyz_bins = x_bins + y_bins + z_bins
            xz_bins = x_bins + z_bins
            yz_bins = y_bins + z_bins
        elif algorithm == "EQQ":
            x_bins = [
                np.array(_create_shifted_eqq_bins(x[i, :], no_bins=bins))
                for i in range(x.shape[0])
            ]
            y_bins = [
                np.array(_create_shifted_eqq_bins(y[i, :], no_bins=bins))
                for i in range(y.shape[0])
            ]
            z_bins = [
                np.array(_create_shifted_eqq_bins(z[i, :], no_bins=bins))
                for i in range(z.shape[0])
            ]
            # create multidim bins for histogram
            xyz_bins = x_bins + y_bins + z_bins
            xz_bins = x_bins + z_bins
            yz_bins = y_bins + z_bins

        else:
            raise ValueError(f"Unknown CMI algorithm: {algorithm}")

        # compute histogram counts
        count_z = np.histogramdd(z.T, bins=z_bins)[0]

        xyz = np.vstack((x, y, z))
        count_xyz = np.histogramdd(xyz.T, bins=xyz_bins)[0]

        xz = np.vstack((x, z))
        count_xz = np.histogramdd(xz.T, bins=xz_bins)[0]

        yz = np.vstack((y, z))
        count_yz = np.histogramdd(yz.T, bins=yz_bins)[0]

        # normalise
        count_z /= np.float(np.sum(count_z))
        count_xyz /= np.float(np.sum(count_xyz))
        count_xz /= np.float(np.sum(count_xz))
        count_yz /= np.float(np.sum(count_yz))

        # sum
        cmi = 0
        iterator = np.nditer(count_xyz, flags=["multi_index"])
        while not iterator.finished:
            idx = iterator.multi_index
            xz_idx = tuple(
                [
                    item
                    for sublist in [idx[: len(x)], idx[-len(z) :]]
                    for item in sublist
                ]
            )  # creates index for xz histo
            yz_idx = idx[-len(z) - len(y) :]
            z_idx = idx[-len(z) :]
            if (
                count_xyz[idx] == 0
                or count_z[z_idx] == 0
                or count_xz[xz_idx] == 0
                or count_yz[yz_idx] == 0
            ):
                iterator.iternext()
                continue
            else:
                cmi += count_xyz[idx] * log_f(
                    count_z[z_idx]
                    * count_xyz[idx]
                    / (count_xz[xz_idx] * count_yz[yz_idx])
                )

            iterator.iternext()

    else:
        raise ValueError(f"Unknown CMI algorithm: {algorithm}")

    return cmi
