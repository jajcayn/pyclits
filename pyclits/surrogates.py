"""
Functions for statistical testing.
"""
import logging
from copy import deepcopy

import numpy as np
import pandas as pd
import pywt
import xarray as xr
from statsmodels.stats.multitest import multipletests, multitest_methods_names

from .geofield import DataField

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
    :param tailed: which test statistic to compute: `upper` or `lower`
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
        significance = 1.0 - np.sum(
            np.less_equal(data_value, surrogates_value), axis=0
        ) / float(num_surrogates)
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

    orig_data_xr = None

    @classmethod
    def from_datafield(cls, datafield):
        """
        Init SurrogateField from other DataField.
        """
        assert isinstance(datafield, DataField)
        surr_field = cls(datafield.data).__finalize__(datafield)
        surr_field.orig_data_xr = deepcopy(datafield.data)

        return surr_field

    def __finalize__(self, other, add_steps=None):
        """
        Additionally to parent function, also copy `orig_data_xr`.
        """
        if hasattr(other, "orig_data_xr"):
            setattr(
                self, "orig_data_xr", deepcopy(getattr(other, "orig_data_xr"))
            )
        return super().__finalize__(other, add_steps=add_steps)

    def construct_surrogates(
        self, surrogate_type, univariate=True, inplace=True, **kwargs
    ):
        """
        Construct surrogates from data.

        :param surrogate_type: type of surrogate to construct:
            shift - time shift surrogate
            shuffle - shuffle surrogate
            FT - Fourier Transform surrogate
            AAFT - amplitude-adjusted FT surrogate
            IAAFT - iterative amplitude-adjusted FT surrogate
            MF - multifractal surrogate
        :type surrogate_type: str
        :param univariate: whether to create univariate or multivariate
            surrogates
        :type univariate: bool
        :param inplace: whether to make operation in-place or return
        :type inplace: bool
        :param **kwargs: possible keyword arguments for surrogate creation
        """
        self.data = deepcopy(self.orig_data_xr)
        seed = (
            None
            if univariate
            else np.random.randint(low=0, high=np.iinfo(np.uint32).max)
        )

        def get_surr(ts, surr_type, seed, **kwargs):
            if surr_type == "shift":
                return get_single_time_shift_surrogate(ts, seed=seed)
            elif surr_type == "shuffle":
                return get_single_shuffle_surrogate(ts, seed=seed, **kwargs)
            elif surr_type == "FT":
                return get_single_FT_surrogate(ts, seed=seed)
            elif surr_type == "AAFT":
                return get_single_AAFT_surrogate(ts, seed=seed)
            elif surr_type == "IAAFT":
                return get_single_IAAFT_surrogate(ts, seed=seed, **kwargs)
            elif surr_type == "MF":
                return get_single_MF_surrogate(ts, seed=seed, **kwargs)
            else:
                raise ValueError(f"Unknown surrogate type {surr_type}")

        surrogates = xr.apply_ufunc(
            lambda x: get_surr(x, surrogate_type, seed, **kwargs),
            self.data,
            input_core_dims=[["time"]],
            output_core_dims=[["time"]],
            vectorize=True,
        ).transpose(*(["time"] + self.dims_not_time))

        univar = "univariate" if univariate else "multivariate"
        add_steps = [f"{univar} {surrogate_type} surrogates"]
        if inplace:
            self.data = surrogates
            self.process_steps += add_steps
        else:
            return self.__constructor__(surrogates).__finalize__(
                self, add_steps
            )

    def add_seasonality(self, mean, var, trend, inplace=True):
        """
        Adds seasonality to surrogates if there were constructed from
        deseasonalised data.

        :param mean: mean of the data
        :type mean: xr.DataArray
        :param var: std of the data
        :type mean: xr.DataArray|float
        :param trend: linear trend of the data
        :type trend: xr.DataArray|float
        :param inplace: whether to make operation in-place or return
        :type inplace: bool
        """
        inferred_freq = pd.infer_freq(self.time)
        if inferred_freq in ["M", "SM", "BM", "MS", "SMS", "BMS"]:
            # monthly data
            groupby = self.data.time.dt.month
        elif inferred_freq in ["C", "B", "D"]:
            # daily data
            groupby = self.data.time.dt.dayofyear
        else:
            raise ValueError(
                "Anomalise supported only for daily or monthly data"
            )
        if isinstance(var, xr.DataArray):
            added = self.data.groupby(groupby) * var
        else:
            added = self.data * var
        if isinstance(mean, xr.DataArray):
            added = added.groupby(groupby) + mean
        else:
            added = added + mean
        if isinstance(trend, xr.DataArray):
            added = added.groupby(groupby) + trend
        else:
            added = added + trend

        add_steps = ["add seasonality"]
        if inplace:
            self.data = added
            self.process_steps += add_steps
        else:
            return self.__constructor__(added).__finalize__(self, add_steps)
