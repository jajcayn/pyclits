"""
Functions for statistical testing.
"""
import logging

import numpy as np
import pywt
from statsmodels.stats.multitest import multipletests, multitest_methods_names

from .geofield_new import DataField

DEFAULT_SEED = None


def get_p_values(data_value, surrogates_value, tailed="upper"):
    """
    Return one-tailed or two-tailed values of percentiles with respect to
    surrogate testing. Two-tailed test assumes that the distribution of the
    test statistic under H0 is symmetric about 0.

    :param data_value: value(s) from data analyses
    :type data_value: np.ndarray
    :param surrogates_value: value(s) from surrogate data analyses, shape must
        be [num_surrogates, ...] where (...) is the shape of the data
    :type surrogates_value: np.ndarray
    :param tailed: which test statistic to compute: `upper`, `lower`, or `both`
    :type tailed: str
    :return: p-value of surrogate testing
    :rtype: float
    """
    assert data_value.shape == surrogates_value.shape[1:], (
        f"Incompatible shapes: data {data_value.shape}; surrogates "
        f"{surrogates_value.shape}"
    )
    num_surrogates = surrogates_value.shape[0]
    if tailed == "upper":
        significance = 1.0 - np.sum(
            np.greater_equal(data_value, surrogates_value), axis=0
        ) / float(num_surrogates)
    elif tailed == "lower":
        significance = np.sum(
            np.less_equal(data_value, surrogates_value), axis=0
        ) / float(num_surrogates)
    elif tailed == "both":
        significance = 2 * (
            1.0
            - np.sum(
                np.greater_equal(np.abs(data_value), surrogates_value), axis=0
            )
            / float(num_surrogates)
        )
    else:
        raise ValueError(f"Unknown tail for testing: {tailed}")

    return significance


def correct_for_multiple_comparisons(p_values, alpha_level, method="hs"):
    """
    Test p-values and correct for multiple comparison problem. Uses statsmodels'
    implementation with multiple methods. For details, see statsmodels'
    documentation.

    :param p_values: uncorrected p-values from multiple testing, 1D
    :type p_values: np.ndarray
    :param alpha_level: family-wise error rate, usually 0.05
    :type alpha_level: float
    :param method: method used for testing and adjustment of p-values, one of:
        'b': Bonferroni,
        's': Sidak,
        'h': Holm,
        'hs': Holm-Sidak,
        'sh': Simes-Hochberg,
        'ho': Hommel,
        'fdr_bh': FDR Benjamini-Hochberg,
        'fdr_by': FDR Benjamini-Yekutieli,
        'fdr_tsbh': FDR 2-stage Benjamini-Hochberg,
        'fdr_tsbky': FDR 2-stage Benjamini-Krieger-Yekutieli,
        'fdr_gbs': FDR adaptive Gavrilov-Benjamini-Sarkar
    :type method: str
    :return: bool array whether p-value is significant after correction and
        the array of corrected p-values
    :rtype: (np.ndarray, np.ndarray)
    """
    assert method in multitest_methods_names, (
        "Unknown method, use one of the: "
        + "; or ".join(
            [
                f"`{key}` for {value}"
                for key, value in multitest_methods_names.items()
            ]
        )
        + " correction method"
    )
    can_reject, corrected_p_vals, _, _ = multipletests(
        p_values, alpha=alpha_level, method=method
    )
    return can_reject, corrected_p_vals


def get_single_time_shift_surrogate(ts, seed=DEFAULT_SEED):
    """
    Return single 1D time shift surrogate. Timeseries is shifted in time
    (assumed periodic in the sense that is wrapped in the end) by a random
    amount of time. Useful surrogate for testing phase relationships.

    :param ts: timeseries to transform as [time x N]
    :type ts: np.ndarray
    :param seed: seed for random number generator
    :type seed: int|None
    :return: 1D time shift surrogate of timeseries
    :rtype: np.ndarray
    """
    np.random.seed(seed)
    roll = np.random.choice(ts.shape[0], 1)[0]
    # assert roll is not 0 - would give the same timeseries
    while roll == 0:
        roll = np.random.choice(ts.shape[0], 1)[0]
    return np.roll(ts, roll, axis=0)


def get_single_shuffle_surrogate(ts, cut_points=None, seed=DEFAULT_SEED):
    """
    Return single 1D shuffle surrogate. Timeseries is cut into `cut_points`
    pieces at random and then shuffled. If `cut_points` is None, will cut each
    point, hence whole timeseries is shuffled.

    :param ts: timeseries to transform as [time x N]
    :type ts: np.ndarray
    :param cut_points: number of cutting points, timeseries will be partitioned
        into n+1 partitions with n cut_points; if None, each point is its own
        partition, i.e. classical shuffle surrogate
    :type cut_points: int|None
    :param seed: seed for random number generator
    :type seed: int|None
    :return: 1D shuffle surrogate of timeseries
    :rtype: np.ndarray
    """
    np.random.seed(seed)
    if cut_points is None:
        cut_points = ts.shape[0]
    assert (
        cut_points <= ts.shape[0]
    ), "Cannot have more cut points than length of the timeseries"
    # generate random partition points without replacement
    partion_points = np.sort(
        np.random.choice(ts.shape[0], cut_points, replace=False)
    )

    def split_permute_concat(x, split_points):
        """
        Helper that splits, permutes and concats the timeseries.
        """
        return np.concatenate(
            np.random.permutation(np.split(x, split_points, axis=0))
        )

    current_permutation = split_permute_concat(ts, partion_points)
    # assert we actually permute the timeseries, useful when using only one
    # cutting point, i.e. two partitions so they are forced to swap
    while np.all(current_permutation == ts):
        current_permutation = split_permute_concat(ts, partion_points)
    return current_permutation


def get_single_FT_surrogate(ts, seed=DEFAULT_SEED):
    """
    Returns single 1D Fourier transform surrogate.

    Theiler, J., Eubank, S., Longtin, A., Galdrikian, B., & Farmer, J. D.
        (1992). Testing for nonlinearity in time series: the method of
        surrogate data. Physica D: Nonlinear Phenomena, 58(1-4), 77-94.

    :param ts: timeseries to transform as [time x N]
    :type ts: np.ndarray
    :param seed: seed for random number generator
    :type seed: int|None
    :return: 1D FT surrogate of timeseries
    :rtype: np.ndarray
    """
    np.random.seed(seed)
    if ts.ndim == 1:
        ts = ts[:, np.newaxis]
    xf = np.fft.rfft(ts, axis=0)
    angle = np.random.uniform(0, 2 * np.pi, (xf.shape[0],))[:, np.newaxis]
    # set the slowest frequency to zero, i.e. not to be randomised
    angle[0] = 0

    cxf = xf * np.exp(1j * angle)

    return np.fft.irfft(cxf, n=ts.shape[0], axis=0).squeeze()


def get_single_AAFT_surrogate(ts, seed=DEFAULT_SEED):
    """
    Returns single 1D amplitude-adjusted Fourier transform surrogate.

    Schreiber, T., & Schmitz, A. (2000). Surrogate time series. Physica D:
        Nonlinear Phenomena, 142(3-4), 346-382.

    :param ts: timeseries to transform as [time x N]
    :type ts: np.ndarray
    :param seed: seed for random number generator
    :type seed: int|None
    :return: 1D AAFT surrogate of timeseries
    :rtype: np.ndarray
    """
    # create Gaussian data
    if ts.ndim == 1:
        ts = ts[:, np.newaxis]
    gaussian = np.broadcast_to(
        np.random.randn(ts.shape[0])[:, np.newaxis], ts.shape
    )
    gaussian = np.sort(gaussian, axis=0)
    # rescale data to Gaussian distribution
    ranks = ts.argsort(axis=0).argsort(axis=0)
    rescaled_data = np.zeros_like(ts)
    for i in range(ts.shape[1]):
        rescaled_data[:, i] = gaussian[ranks[:, i], i]
    # do phase randomization
    phase_randomized_data = get_single_FT_surrogate(rescaled_data, seed=seed)
    if phase_randomized_data.ndim == 1:
        phase_randomized_data = phase_randomized_data[:, np.newaxis]
    # rescale back to amplitude distribution of original data
    sorted_original = ts.copy()
    sorted_original.sort(axis=0)
    ranks = phase_randomized_data.argsort(axis=0).argsort(axis=0)

    for i in range(ts.shape[1]):
        rescaled_data[:, i] = sorted_original[ranks[:, i], i]

    return rescaled_data.squeeze()


def get_single_IAAFT_surrogate(ts, n_iterations=1000, seed=DEFAULT_SEED):
    """
    Returns single 1D iteratively refined amplitude-adjusted Fourier transform
    surrogate. A set of AAFT surrogates is iteratively refined to produce a
    closer match of both amplitude distribution and power spectrum of surrogate
    and original data.

    Schreiber, T., & Schmitz, A. (2000). Surrogate time series. Physica D:
        Nonlinear Phenomena, 142(3-4), 346-382.

    :param ts: timeseries to transform as [time x N]
    :type ts: np.ndarray
    :param n_iterations: number of iterations of the procedure
    :type n_iterations: int
    :param seed: seed for random number generator
    :type seed: int|None
    :return: 1D IAAFT surrogate of timeseries
    :rtype: np.ndarray
    """
    if ts.ndim == 1:
        ts = ts[:, np.newaxis]
    # FT of original data
    xf = np.fft.rfft(ts, axis=0)
    # FT amplitudes
    xf_amps = np.abs(xf)
    sorted_original = ts.copy()
    sorted_original.sort(axis=0)

    # starting point of iterative procedure
    R = get_single_AAFT_surrogate(ts, seed=seed)
    if R.ndim == 1:
        R = R[:, np.newaxis]
    # iterate: `R` is the surrogate with "true" amplitudes and `s` is the
    # surrogate with "true" spectrum
    for _ in range(n_iterations):
        # get Fourier phases of R surrogate
        r_fft = np.fft.rfft(R, axis=0)
        r_phases = r_fft / np.abs(r_fft)
        # transform back, replacing the actual amplitudes by the desired
        # ones, but keeping the phases exp(i*phase(i))
        s = np.fft.irfft(xf_amps * r_phases, n=ts.shape[0], axis=0)
        #  rescale to desired amplitude distribution
        ranks = s.argsort(axis=0).argsort(axis=0)
        for j in range(R.shape[1]):
            R[:, j] = sorted_original[ranks[:, j], j]

    return R.squeeze()


def get_single_MF_surrogate(ts, randomise_from_scale=2, seed=DEFAULT_SEED):
    """
    Returns single 1D multifractal surrogate: bootstrapped series realization
    from random cascade on wavelet dyadic tree, which preserves multifractal
    properties of the input data. Time series length must be a power of 2.

    Paluš, M. (2008). Bootstrapping multifractals: Surrogate data from random
        cascades on wavelet dyadic trees. Physical review letters, 101(13),
        134101.

    :param ts: timeseries to transform as [time x N]
    :type ts: np.ndarray
    :param randomise_from_scale: from which wavelet scale the coefficient
        should be randomised
    :type randomise_from_scale: int
    :param seed: seed for random number generator
    :type seed: int|None
    :return: 1D multifractal surrogate of timeseries
    :rtype: np.ndarray
    """
    assert ts.ndim == 1, "MF surrogates for 1D timeseries only!"
    np.random.seed(seed)
    n = int(np.log2(ts.shape[0]))  # time series length should be 2^n
    n_real = np.log2(ts.shape[0])

    if n != n_real:
        raise ValueError("Time series length must be power of 2 (2^n).")

    # get coefficient from discrete wavelet transform,
    # it is a list of length n with numpy arrays as every object
    coeffs = pywt.wavedec(ts, "db1", level=n - 1)

    # prepare output lists and append coefficients which will not be shuffled
    coeffs_tilde = []
    for j in range(randomise_from_scale):
        coeffs_tilde.append(coeffs[j])

    shuffled_coeffs = []
    for j in range(randomise_from_scale):
        shuffled_coeffs.append(coeffs[j])

    # run for each desired scale
    for j in range(randomise_from_scale, len(coeffs)):

        # get multiplicators for scale j
        multiplicators = np.zeros_like(coeffs[j])
        for k in range(coeffs[j - 1].shape[0]):
            if coeffs[j - 1][k] == 0:
                logging.warning("Some zero coefficients in DWT transform!")
                coeffs[j - 1][k] = 1
            multiplicators[2 * k] = coeffs[j][2 * k] / coeffs[j - 1][k]
            multiplicators[2 * k + 1] = coeffs[j][2 * k + 1] / coeffs[j - 1][k]

        # shuffle multiplicators in scale j randomly
        coef = np.zeros_like(multiplicators)
        multiplicators = np.random.permutation(multiplicators)

        # get coefficients with tilde according to a cascade
        for k in range(coeffs[j - 1].shape[0]):
            coef[2 * k] = multiplicators[2 * k] * coeffs_tilde[j - 1][k]
            coef[2 * k + 1] = multiplicators[2 * k + 1] * coeffs_tilde[j - 1][k]
        coeffs_tilde.append(coef)

        # sort original coefficients
        coeffs[j] = np.sort(coeffs[j])

        # sort shuffled coefficients
        idx = np.argsort(coeffs_tilde[j])

        # finally, rearange original coefficient according to coefficient with
        # tilde
        temporary = np.zeros_like(coeffs[j])
        temporary[idx] = coeffs[j]
        shuffled_coeffs.append(temporary)

    # return randomised time series as inverse discrete wavelet transform
    mf_surr = pywt.waverec(shuffled_coeffs, "db1")

    return mf_surr


class SurrogateField(DataField):
    """
    Class holds geofield of surrogate data and can construct surrogates.
    """

    @classmethod
    def from_datafield(cls, datafield):
        pass

    def construct_surrogates(self, args, inplace=True):
        pass

    # def __init__(self, data=None):
    #     DataField.__init__(self)
    #     self.data = None
    #     self.model_grid = None
    #     self.original_data = data

    # def copy_field(self, field):
    #     """
    #     Makes a copy of another DataField
    #     """

    #     self.original_data = field.data.copy()
    #     if field.lons is not None:
    #         self.lons = field.lons.copy()
    #     else:
    #         self.lons = None
    #     if field.lats is not None:
    #         self.lats = field.lats.copy()
    #     else:
    #         self.lats = None
    #     self.time = field.time.copy()
    #     self.nans = field.nans
    #     if field.data_mask is not None:
    #         self.data_mask = field.data_mask.copy()

    # def add_seasonality(self, mean, var, trend):
    #     """
    #     Adds seasonality to surrogates if there were constructed from deseasonalised
    #     and optionally detrended data.
    #     """

    #     if self.data is not None:
    #         self.data *= var
    #         self.data += mean
    #         if trend is not None:
    #             self.data += trend
    #     else:
    #         raise Exception("Surrogate data has not been created yet.")

    # def remove_seasonality(self, mean, var, trend):
    #     """
    #     Removes seasonality from surrogates in order to use same field for
    #     multiple surrogate construction.
    #     """

    #     if self.data is not None:
    #         if trend is not None:
    #             self.data -= trend
    #         self.data -= mean
    #         self.data /= var
    #     else:
    #         raise Exception("Surrogate data has not been created yet.")

    # def center_surr(self):
    #     """
    #     Centers the surrogate data to zero mean and unit variance.
    #     """

    #     if self.data is not None:
    #         self.data -= np.nanmean(self.data, axis=0)
    #         self.data /= np.nanstd(self.data, axis=0, ddof=1)
    #     else:
    #         raise Exception("Surrogate data has not been created yet.")

    # def get_surr(self):
    #     """
    #     Returns the surrogate data
    #     """

    #     if self.data is not None:
    #         return self.data.copy()
    #     else:
    #         raise Exception("Surrogate data has not been created yet.")

    # def construct_fourier_surrogates(
    #     self, algorithm="FT", pool=None, preserve_corrs=False, n_iterations=10
    # ):
    #     """
    #     Constructs Fourier Transform (FT) surrogates - shuffles angle in Fourier space of the original data.
    #     algorithm:
    #         FT - basic FT surrogates [1]
    #         AAFT - amplitude adjusted FT surrogates [2]
    #         IAAFT - iterative amplitude adjusted FT surrogates [3]
    #     pool:
    #         instance of multiprocessing's pool in order to exploit multithreading for high-dimensional data
    #     preserve_corrs:
    #         bool, whether to preserve covariance structure in spatially distributed data
    #     n_iterations:
    #         int, only when algorithm = IAAFT, number of iterations
    #     """

    #     if algorithm not in ["FT", "AAFT", "IAAFT"]:
    #         raise Exception(
    #             "Unknown algorithm type, please use 'FT', 'AAFT' or 'IAAFT'."
    #         )

    #     if self.original_data is not None:

    #         np.random.seed()

    #         if pool is None:
    #             map_func = map
    #         else:
    #             map_func = pool.map

    #         if algorithm == "FT":
    #             surr_func = _compute_FT_surrogates
    #         elif algorithm == "AAFT":
    #             surr_func = _compute_AAFT_surrogates
    #         elif algorithm == "IAAFT":
    #             surr_func = _compute_IAAFT_surrogates

    #         if self.original_data.ndim > 1:
    #             orig_shape = self.original_data.shape
    #             self.original_data = np.reshape(
    #                 self.original_data,
    #                 (self.original_data.shape[0], np.prod(orig_shape[1:])),
    #             )
    #         else:
    #             orig_shape = None
    #             self.original_data = self.original_data[:, np.newaxis]

    #         # generate uniformly distributed random angles
    #         a = np.fft.rfft(np.random.rand(self.original_data.shape[0]), axis=0)
    #         if preserve_corrs:
    #             angle = np.random.uniform(0, 2 * np.pi, (a.shape[0],))
    #             # set the slowest frequency to zero, i.e. not to be randomised
    #             angle[0] = 0
    #             del a
    #             if algorithm == "IAAFT":
    #                 job_data = [
    #                     (i, n_iterations, self.original_data[:, i], angle)
    #                     for i in range(self.original_data.shape[1])
    #                 ]
    #             else:
    #                 job_data = [
    #                     (i, self.original_data[:, i], angle)
    #                     for i in range(self.original_data.shape[1])
    #                 ]
    #         else:
    #             angle = np.random.uniform(
    #                 0, 2 * np.pi, (a.shape[0], self.original_data.shape[1])
    #             )
    #             angle[0, ...] = 0
    #             del a
    #             if algorithm == "IAAFT":
    #                 job_data = [
    #                     (i, n_iterations, self.original_data[:, i], angle[:, i])
    #                     for i in range(self.original_data.shape[1])
    #                 ]
    #             else:
    #                 job_data = [
    #                     (i, self.original_data[:, i], angle[:, i])
    #                     for i in range(self.original_data.shape[1])
    #                 ]

    #         job_results = map_func(surr_func, job_data)

    #         self.data = np.zeros_like(self.original_data)

    #         for i, surr in job_results:
    #             self.data[:, i] = surr

    #         # squeeze single-dimensional entries (e.g. station data)
    #         self.data = np.squeeze(self.data)
    #         self.original_data = np.squeeze(self.original_data)

    #         # reshape back to original shape
    #         if orig_shape is not None:
    #             self.data = np.reshape(self.data, orig_shape)
    #             self.original_data = np.reshape(self.original_data, orig_shape)

    #     else:
    #         raise Exception(
    #             "No data to randomise in the field. First you must copy some DataField."
    #         )

    # def construct_multifractal_surrogates(
    #     self, pool=None, randomise_from_scale=2
    # ):
    #     """
    #     Constructs multifractal surrogates (independent shuffling of the scale-specific coefficients,
    #     preserving so-called multifractal structure - hierarchical process exhibiting information flow
    #     from large to small scales)
    #     written according to: Palus, M., Phys. Rev. Letters, 101, 2008.
    #     """

    #     import pywt

    #     if self.original_data is not None:

    #         if pool is None:
    #             map_func = map
    #         else:
    #             map_func = pool.map

    #         if self.original_data.ndim > 1:
    #             orig_shape = self.original_data.shape
    #             self.original_data = np.reshape(
    #                 self.original_data,
    #                 (self.original_data.shape[0], np.prod(orig_shape[1:])),
    #             )
    #         else:
    #             orig_shape = None
    #             self.original_data = self.original_data[:, np.newaxis]

    #         self.data = np.zeros_like(self.original_data)

    #         job_data = [
    #             (i, self.original_data[:, i], randomise_from_scale, None)
    #             for i in range(self.original_data.shape[1])
    #         ]
    #         job_results = map_func(_compute_MF_surrogates, job_data)

    #         for i, surr in job_results:
    #             self.data[:, i] = surr

    #         # squeeze single-dimensional entries (e.g. station data)
    #         self.data = np.squeeze(self.data)
    #         self.original_data = np.squeeze(self.original_data)

    #         # reshape back to original shape
    #         if orig_shape is not None:
    #             self.data = np.reshape(self.data, orig_shape)
    #             self.original_data = np.reshape(self.original_data, orig_shape)

    #     else:
    #         raise Exception(
    #             "No data to randomise in the field. First you must copy some DataField."
    #         )

    # def prepare_AR_surrogates(self, pool=None, order_range=[1, 1], crit="sbc"):
    #     """
    #     Prepare for generating AR(k) surrogates by identifying the AR model and computing
    #     the residuals. Adapted from script by Vejmelka -- https://github.com/vejmelkam/ndw-climate
    #     """

    #     if self.original_data is not None:

    #         if pool is None:
    #             map_func = map
    #         else:
    #             map_func = pool.map

    #         if self.original_data.ndim > 1:
    #             orig_shape = self.original_data.shape
    #             self.original_data = np.reshape(
    #                 self.original_data,
    #                 (self.original_data.shape[0], np.prod(orig_shape[1:])),
    #             )
    #         else:
    #             orig_shape = None
    #             self.original_data = self.original_data[:, np.newaxis]
    #         num_tm = self.time.shape[0]

    #         job_data = [
    #             (i, order_range, crit, self.original_data[:, i])
    #             for i in range(self.original_data.shape[1])
    #         ]
    #         job_results = map_func(_prepare_AR_surrogates, job_data)
    #         max_ord = 0
    #         for r in job_results:
    #             if r[1] is not None and r[1].order() > max_ord:
    #                 max_ord = r[1].order()
    #         num_tm_s = num_tm - max_ord
    #         if orig_shape is None:
    #             self.model_grid = np.zeros((1,), dtype=np.object)
    #             self.residuals = np.zeros((num_tm_s, 1), dtype=np.float64)
    #         else:
    #             self.model_grid = np.zeros(
    #                 (np.prod(orig_shape[1:]),), dtype=np.object
    #             )
    #             self.residuals = np.zeros(
    #                 (num_tm_s, np.prod(orig_shape[1:])), dtype=np.float64
    #             )

    #         for i, v, r in job_results:
    #             self.model_grid[i] = v
    #             if v is not None:
    #                 self.residuals[:, i] = r[:num_tm_s, 0]
    #             else:
    #                 self.residuals[:, i] = np.nan

    #         self.max_ord = max_ord

    #         self.original_data = np.squeeze(self.original_data)
    #         self.residuals = np.squeeze(self.residuals)

    #         # reshape back to original shape
    #         if orig_shape is not None:
    #             self.original_data = np.reshape(self.original_data, orig_shape)
    #             self.model_grid = np.reshape(
    #                 self.model_grid, list(orig_shape[1:])
    #             )
    #             self.residuals = np.reshape(
    #                 self.residuals, [num_tm_s] + list(orig_shape[1:])
    #             )

    #     else:
    #         raise Exception(
    #             "No data to randomise in the field. First you must copy some DataField."
    #         )

    # def construct_surrogates_with_residuals(self, pool=None):
    #     """
    #     Constructs a new surrogate time series from AR(k) model.
    #     Adapted from script by Vejmelka -- https://github.com/vejmelkam/ndw-climate
    #     """

    #     if self.model_grid is not None:

    #         if pool is None:
    #             map_func = map
    #         else:
    #             map_func = pool.map

    #         if self.original_data.ndim > 1:
    #             orig_shape = self.original_data.shape
    #             self.original_data = np.reshape(
    #                 self.original_data,
    #                 (self.original_data.shape[0], np.prod(orig_shape[1:])),
    #             )
    #             self.model_grid = np.reshape(
    #                 self.model_grid, np.prod(self.model_grid.shape)
    #             )
    #             self.residuals = np.reshape(
    #                 self.residuals,
    #                 (self.residuals.shape[0], np.prod(orig_shape[1:])),
    #             )
    #         else:
    #             orig_shape = None
    #             self.original_data = self.original_data[:, np.newaxis]
    #             # self.model_grid = self.model_grid[:, np.newaxis]
    #             self.residuals = self.residuals[:, np.newaxis]
    #         num_tm_s = self.time.shape[0] - self.max_ord

    #         job_data = [
    #             (i, self.residuals[:, i], self.model_grid[i], num_tm_s, None)
    #             for i in range(self.original_data.shape[1])
    #         ]
    #         job_results = map_func(_compute_AR_surrogates, job_data)

    #         self.data = np.zeros((num_tm_s, self.original_data.shape[1]))

    #         for i, surr in job_results:
    #             self.data[:, i] = surr

    #         self.data = np.squeeze(self.data)
    #         self.original_data = np.squeeze(self.original_data)
    #         self.residuals = np.squeeze(self.residuals)

    #         # reshape back to original shape
    #         if orig_shape is not None:
    #             self.original_data = np.reshape(self.original_data, orig_shape)
    #             self.model_grid = np.reshape(
    #                 self.model_grid, list(orig_shape[1:])
    #             )
    #             self.residuals = np.reshape(
    #                 self.residuals, [num_tm_s] + list(orig_shape[1:])
    #             )
    #             self.data = np.reshape(
    #                 self.data, [num_tm_s] + list(orig_shape[1:])
    #             )

    #     else:
    #         raise Exception(
    #             "The AR(k) model is not simulated yet. First, prepare surrogates!"
    #         )

    # def amplitude_adjust_surrogates(self, mean, var, trend, pool=None):
    #     """
    #     Performs amplitude adjustment to already created surrogate data.
    #     """

    #     if self.data is not None and self.original_data is not None:

    #         if pool is None:
    #             map_func = map
    #         else:
    #             map_func = pool.map

    #         if self.original_data.ndim > 1:
    #             orig_shape = self.original_data.shape
    #             self.original_data = np.reshape(
    #                 self.original_data,
    #                 (self.original_data.shape[0], np.prod(orig_shape[1:])),
    #             )
    #             self.data = np.reshape(
    #                 self.data, (self.data.shape[0], np.prod(orig_shape[1:]))
    #             )
    #             mean = np.reshape(
    #                 mean, (mean.shape[0], np.prod(orig_shape[1:]))
    #             )
    #             var = np.reshape(var, (var.shape[0], np.prod(orig_shape[1:])))
    #             trend = np.reshape(
    #                 trend, (trend.shape[0], np.prod(orig_shape[1:]))
    #             )
    #         else:
    #             orig_shape = None
    #             self.original_data = self.original_data[:, np.newaxis]
    #             self.data = self.data[:, np.newaxis]
    #             mean = mean[:, np.newaxis]
    #             var = var[:, np.newaxis]
    #             trend = trend[:, np.newaxis]

    #         old_shape = self.data.shape

    #         job_data = [
    #             (
    #                 i,
    #                 self.original_data[:, i],
    #                 self.data[:, i],
    #                 mean[:, i],
    #                 var[:, i],
    #                 trend[:, i],
    #             )
    #             for i in range(self.original_data.shape[1])
    #         ]
    #         job_results = map_func(
    #             _create_amplitude_adjusted_surrogates, job_data
    #         )

    #         self.data = np.zeros(old_shape)

    #         for i, AAsurr in job_results:
    #             self.data[:, i] = AAsurr

    #         # squeeze single-dimensional entries (e.g. station data)
    #         self.data = np.squeeze(self.data)
    #         self.original_data = np.squeeze(self.original_data)

    #         # reshape back to original shape
    #         if orig_shape is not None:
    #             self.original_data = np.reshape(self.original_data, orig_shape)
    #             self.data = np.reshape(
    #                 self.data, [self.data.shape[0]] + list(orig_shape[1:])
    #             )

    #     else:
    #         raise Exception(
    #             "No surrogate data or/and no data in the field. "
    #             "Amplitude adjustment works on already copied data and created surrogates."
    #         )
