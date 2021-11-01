"""
Functions for wavelet analysis. Rewritten to Python from "A Practical Guide to
Wavelet Analysis" by Ch. Torrence and G. Compo
http://paos.colorado.edu/research/wavelets/
"""

import numpy as np
from scipy.fftpack import fft, ifft
from scipy.special import gamma


class MotherWavelet:
    """
    Base class for mother wavelets
    """

    name = ""

    def fourier_factor(self, k0):
        raise NotImplementedError

    def coi(self, k0):
        raise NotImplementedError

    def get_wavelet(self, k, scale, k0):
        raise NotImplementedError


class MorletWavelet(MotherWavelet):

    name = "Morlet"

    def fourier_factor(self, k0):
        """
        Compute Fourier factor - multiplier between wavelet scales and periods.

        :param k0: wavenumber
        :type k0: float
        :return: Fourier factor
        :rtype: float
        """
        return (4.0 * np.pi) / (k0 + np.sqrt(2.0 + np.power(k0, 2)))

    def coi(self, k0):
        """
        Compute Cone-of-Influence - a maximum period of useful information at
        particular time

        :param k0: wavenumber
        :type k0: float
        :return: CoI
        :rtype: float
        """
        return self.fourier_factor(k0) / np.sqrt(2.0)

    def get_wavelet(self, k, scale, k0=6.0):
        """
        Returns the Morlet wavelet function as a function of Fourier frequency,
        used for the wavelet transform in Fourier space.

        psi(x) = pi^(-1/4) * exp(i*k0*x) * exp(-x^2 / 2)

        :param k: Fourier frequencies at which to calculate the wavelet
        :type k: np.ndarray
        :param scale: wavelet scale
        :type scale: float
        :param k0: wavenumber
        :type k0: float
        :return: Morlet wavelet
        :rtype: np.ndarray
        """
        exponent = -np.power((scale * k - k0), 2) / 2.0 * (k > 0.0)
        norm = (
            np.sqrt(scale * k[1]) * (np.power(np.pi, -0.25)) * np.sqrt(len(k))
        )
        output = norm * np.exp(exponent)
        output *= k > 0.0

        return output


class PaulWavelet(MotherWavelet):
    name = "Paul"

    def fourier_factor(self, k0):
        """
        Compute Fourier factor - multiplier between wavelet scales and periods.

        :param k0: order
        :type k0: float
        :return: Fourier factor
        :rtype: float
        """
        return (4 * np.pi) / (2 * k0 + 1)

    def coi(self, k0):
        """
        Compute Cone-of-Influence - a maximum period of useful information at
        particular time

        :param k0: order
        :type k0: float
        :return: CoI
        :rtype: float
        """
        return self.fourier_factor(k0) * np.sqrt(2.0)

    def get_wavelet(self, k, scale, k0=4.0):
        """
        Returns the Paul wavelet function as a function of Fourier frequency,
        used for the wavelet transform in Fourier space.

        psi(x) = (2^m * i^m * m!) / sqrt(pi * (2m)!) * (1 - ix)^(-m + 1)

        :param k: Fourier frequencies at which to calculate the wavelet
        :type k: np.ndarray
        :param scale: wavelet scale
        :type scale: float
        :param k0: order
        :type k0: float
        :return: Paul wavelet
        :rtype: np.ndarray
        """
        exponent = -np.power((scale * k), 2) * (k > 0.0)
        norm = (
            np.sqrt(scale * k[1])
            * (np.power(2, k0) / np.sqrt(k0 * np.prod(np.arange(2, 2 * k0))))
            * np.sqrt(len(k))
        )
        output = norm * np.power((scale * k), k0) * np.exp(exponent)
        output *= k > 0.0

        return output


class DoGWavelet(MotherWavelet):

    name = "DOG"

    def fourier_factor(self, k0):
        """
        Compute Fourier factor - multiplier between wavelet scales and periods.

        :param k0: derivative
        :type k0: float
        :return: Fourier factor
        :rtype: float
        """
        return 2 * np.pi * np.sqrt(2 / (2 * k0 + 1))

    def coi(self, k0):
        """
        Compute Cone-of-Influence - a maximum period of useful information at
        particular time

        :param k0: derivative
        :type k0: float
        :return: CoI
        :rtype: float
        """
        return self.fourier_factor(k0) / np.sqrt(2.0)

    def get_wavelet(self, k, scale, k0=2.0):
        """
        Returns the Derivative of Gaussian wavelet function as a function of Fourier
        frequency, used for the wavelet transform in Fourier space. For m = 2 this
        wavelet is the Marr or Mexican hat wavelet.

        psi(x) = (-1)^(m+1) / sqrt (gamma(m+1/2)) * (d^m / dx^m) exp(-x^2 / 2)

        :param k: Fourier frequencies at which to calculate the wavelet
        :type k: np.ndarray
        :param scale: wavelet scale
        :type scale: float
        :param k0: derivative
        :type k0: float
        :return: DOG wavelet
        :rtype: np.ndarray
        """
        exponent = -np.power((scale * k), 2) / 2.0
        norm = np.sqrt(scale * k[1] / gamma(k0 + 0.5)) * np.sqrt(len(k))
        output = (
            -norm
            * np.power(1j, k0)
            * np.power((scale * k), k0)
            * np.exp(exponent)
        )

        return output


def continous_wavelet(X, dt, pad=False, wavelet=MorletWavelet(), **kwargs):
    """
    Computes the wavelet transform of the vector X, with sampling rate dt.

    :param X: timeseries
    :type X: np.ndarray
    :param dt: sampling time of timeseries
    :type dt: float
    :param pad: whether to pad time series to the next power of 2 - speeds up
        the FFT
    :type pad: bool
    :param wavelet: mother wavelet function
    :type wavelet: `MotherWavelet`
    :param *kwargs:
        dj: spacing between discrete scales
        s0: smallest scale of the wavelet
        j1: number of scales minus one; scales range from `s0` up to
            `s0 * 2^(j1+dj)` to give a total of j1+1 scales
        k0: parameter of mother wavelet
    :return: wavelet transform of X as (X.shape[0], j1+1); vector of Fourier
        periods in time units; the vector of scale indices, given by
        s0 * 2^(j*dj); Cone-of-Influence vector that contains a maximum period
        of useful information at particular time
    :rtype: np.ndarray, np.ndarray, np.ndarray, np.ndarray

    """
    dj = kwargs.get("dj", 0.25)
    s0 = kwargs.get("s0", 2 * dt)
    j1 = int(
        kwargs.get("j1", np.fix(np.log(len(X) * dt / s0) / np.log(2)) / dj)
    )
    k0 = kwargs.get("k0", 6.0)

    assert isinstance(wavelet, MotherWavelet)

    n1 = len(X)

    Y = X - np.mean(X)
    # Y = X

    # padding, if needed
    if pad:
        base2 = int(
            np.fix(np.log(n1) / np.log(2) + 0.4999999)
        )  # power of 2 nearest to len(X)
        Y = np.concatenate((Y, np.zeros((np.power(2, (base2 + 1)) - n1))))
    n = len(Y)

    # wavenumber array
    k = np.arange(1, np.fix(n / 2) + 1)
    k *= (2.0 * np.pi) / (n * dt)
    k_minus = -k[int(np.fix(n - 1)) // 2 - 1 :: -1]
    k = np.concatenate((np.array([0.0]), k, k_minus))

    # compute FFT of the (padded) time series
    f = fft(Y)

    # construct scale array and empty period & wave arrays
    scale = np.array([s0 * np.power(2, x * dj) for x in range(0, j1 + 1)])
    period = scale
    wave = np.zeros((j1 + 1, n), dtype=np.complex)

    # loop through scales and compute tranform
    for i in range(j1 + 1):
        daughter = wavelet.get_wavelet(k, scale[i], k0)
        wave[i, :] = ifft(f * daughter)
        coi = wavelet.coi(k0)

    period = wavelet.fourier_factor(k0) * scale
    coi *= dt * np.concatenate(
        (
            np.array([1e-5]),
            np.arange(1, (n1 + 1) / 2),
            np.arange((n1 / 2 - 1), 0, -1),
            np.array([1e-5]),
        )
    )
    wave = wave[:, :n1]

    return wave, period, scale, coi
