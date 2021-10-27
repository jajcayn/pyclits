"""
Integration tests for wavelet analysis functions.
"""

import unittest

import numpy as np
from pyclits.wavelet_analysis import (
    DoGWavelet,
    MorletWavelet,
    MotherWavelet,
    PaulWavelet,
    continous_wavelet,
)

WAVENUMS = np.array([12.0, 14.0, 16.0]) * 10e-3


class TestMotherWavelet(unittest.TestCase):
    def test_init(self):
        mw = MotherWavelet()
        self.assertEqual(mw.name, "")
        self.assertTrue(hasattr(mw, "fourier_factor"))
        self.assertTrue(hasattr(mw, "coi"))
        self.assertTrue(hasattr(mw, "get_wavelet"))


class TestMorletWavelet(unittest.TestCase):
    FF_6 = 1.0330436477492537
    FF_12 = 0.5217932411098392
    FF_3 = 1.9894122306498652

    WVLT = np.array([0.00333906, 0.01300961, 0.04319344])

    def test_init(self):
        mor = MorletWavelet()
        self.assertTrue(isinstance(mor, MotherWavelet))
        self.assertEqual(mor.name, "Morlet")

    def test_fourier_factor(self):
        mor = MorletWavelet()
        self.assertEqual(mor.fourier_factor(k0=6.0), self.FF_6)
        self.assertEqual(mor.fourier_factor(k0=12.0), self.FF_12)
        self.assertEqual(mor.fourier_factor(k0=3.0), self.FF_3)

    def test_coi(self):
        mor = MorletWavelet()
        self.assertEqual(mor.coi(k0=6.0), self.FF_6 / np.sqrt(2.0))
        self.assertEqual(mor.coi(k0=12.0), self.FF_12 / np.sqrt(2.0))
        self.assertEqual(mor.coi(k0=3.0), self.FF_3 / np.sqrt(2.0))

    def test_get_wavelet(self):
        mor = MorletWavelet()
        wvlt = mor.get_wavelet(k=WAVENUMS, scale=20.0, k0=6.0)
        np.testing.assert_almost_equal(wvlt, self.WVLT)


class TestPaulWavelet(unittest.TestCase):
    FF_6 = 0.966643893412244
    FF_12 = 0.5026548245743669
    FF_3 = 1.7951958020513104

    WVLT = np.array([0.0072177, 0.0022738, 0.0004596])

    def test_init(self):
        paul = PaulWavelet()
        self.assertTrue(isinstance(paul, MotherWavelet))
        self.assertEqual(paul.name, "Paul")

    def test_fourier_factor(self):
        paul = PaulWavelet()
        self.assertEqual(paul.fourier_factor(k0=6.0), self.FF_6)
        self.assertEqual(paul.fourier_factor(k0=12.0), self.FF_12)
        self.assertEqual(paul.fourier_factor(k0=3.0), self.FF_3)

    def test_coi(self):
        paul = PaulWavelet()
        self.assertEqual(paul.coi(k0=6.0), self.FF_6 * np.sqrt(2.0))
        self.assertEqual(paul.coi(k0=12.0), self.FF_12 * np.sqrt(2.0))
        self.assertEqual(paul.coi(k0=3.0), self.FF_3 * np.sqrt(2.0))

    def test_get_wavelet(self):
        paul = PaulWavelet()
        wvlt = paul.get_wavelet(k=WAVENUMS, scale=20.0, k0=6.0)
        np.testing.assert_almost_equal(wvlt, self.WVLT)


class TestDoGWavelet(unittest.TestCase):
    FF_6 = 2.464468037602168
    FF_12 = 1.7771531752633465
    FF_3 = 3.358503816725428

    WVLT = np.array([1.83243787 + 0.0j, 1.63321641 + 0.0j, 1.09607946 + 0.0j])

    def test_init(self):
        dog = DoGWavelet()
        self.assertTrue(isinstance(dog, MotherWavelet))
        self.assertEqual(dog.name, "DOG")

    def test_fourier_factor(self):
        dog = DoGWavelet()
        self.assertEqual(dog.fourier_factor(k0=6.0), self.FF_6)
        self.assertEqual(dog.fourier_factor(k0=12.0), self.FF_12)
        self.assertEqual(dog.fourier_factor(k0=3.0), self.FF_3)

    def test_coi(self):
        dog = DoGWavelet()
        self.assertEqual(dog.coi(k0=6.0), self.FF_6 / np.sqrt(2.0))
        self.assertEqual(dog.coi(k0=12.0), self.FF_12 / np.sqrt(2.0))
        self.assertEqual(dog.coi(k0=3.0), self.FF_3 / np.sqrt(2.0))

    def test_get_wavelet(self):
        dog = DoGWavelet()
        wvlt = dog.get_wavelet(k=WAVENUMS, scale=20.0, k0=6.0)
        np.testing.assert_almost_equal(wvlt, self.WVLT)


class TestContinuousWavelet(unittest.TestCase):
    def _generate_signal(self, freq):
        N = 100
        return np.sin(np.arange(N) * 2.0 * np.pi * freq) + np.random.normal(
            loc=0.0, scale=0.1, size=(N,)
        )

    def test_basic(self):
        sig = self._generate_signal(freq=6.0)
        wave, per, sc, coi = continous_wavelet(
            sig,
            dt=1.0,
            pad=True,
            wavelet=MorletWavelet(),
            dj=0,
            s0=6.0,
            j1=0,
            k0=6.0,
        )
        self.assertEqual(wave.dtype.kind, "c")
        self.assertTupleEqual((1, len(sig)), wave.shape)
        self.assertEqual(6.0, sc)
        self.assertEqual(len(per), 1)
        self.assertEqual(len(coi), len(sig) + 1)

    def test_pad(self):
        sig = self._generate_signal(freq=6.0)
        wvlt, per, sc, coi = continous_wavelet(
            sig, dt=1.0, pad=False, wavelet=MorletWavelet(), j1=2
        )
        wvlt_w_pad, per_w_pad, sc_w_pad, coi_w_pad = continous_wavelet(
            sig, dt=1.0, pad=True, wavelet=MorletWavelet(), j1=2
        )
        np.testing.assert_equal(per, per_w_pad)
        np.testing.assert_equal(sc, sc_w_pad)
        np.testing.assert_equal(coi, coi_w_pad)
        self.assertEqual(wvlt.dtype.kind, "c")
        self.assertEqual(wvlt_w_pad.dtype.kind, "c")
        self.assertTupleEqual(wvlt.shape, wvlt_w_pad.shape)


if __name__ == "__main__":
    unittest.main()
