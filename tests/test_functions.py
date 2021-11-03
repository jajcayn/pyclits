"""
Integration tests for helper functions.
"""

import unittest

import numpy as np
import py
import pytest
from pyclits.functions import cross_correlation, kdensity_estimate, partial_corr

DEFAULT_SEED = 42


class TestHelperFunctions(unittest.TestCase):
    def generate_ts(self):
        np.random.seed(DEFAULT_SEED)
        return np.random.multivariate_normal(
            [0.0, 0.4], [[0.1, 0.4], [-0.6, 0.2]], size=(100)
        )

    def test_cross_correlation(self):
        MAX_LAG = 5
        ts = self.generate_ts()
        cross_cor = cross_correlation(ts[:, 0], ts[:, 1], max_lag=MAX_LAG)
        self.assertTrue(isinstance(cross_cor, np.ndarray))
        self.assertEqual(cross_cor.shape, (2 * MAX_LAG + 1,))

        with pytest.raises(AssertionError):
            cross_cor = cross_correlation(ts, ts, max_lag=MAX_LAG)

    def test_kdensity_estimate(self):
        ts = self.generate_ts()
        x, kde = kdensity_estimate(ts[:, 0], kernel="gaussian")
        self.assertTrue(isinstance(x, np.ndarray))
        self.assertTrue(isinstance(kde, np.ndarray))
        self.assertEqual(x.shape, kde.shape)
        self.assertEqual(x.shape, (100,))

    def test_partial_corr(self):
        # no confounds case
        ts = self.generate_ts()
        corr, pval = partial_corr(ts.T)
        self.assertEqual(corr, -0.06701731434192172)
        self.assertEqual(pval, 0.5076554556468957)

        # confounds
        np.random.seed(DEFAULT_SEED)
        ts = np.random.rand(4, 100)
        corr, pval = partial_corr(ts)
        self.assertEqual(corr, -0.05145952721058166)
        self.assertEqual(pval, 0.6148090827114683)

        # NaNs
        ts[3, 13] = np.nan
        with pytest.raises(ValueError):
            _ = partial_corr(ts)

        # NaNs after stand
        ts = np.ones((4, 50))
        with pytest.raises(ValueError):
            _ = partial_corr(ts)

        # not enought df
        ts = np.random.rand(5, 5)
        with pytest.raises(ValueError):
            _ = partial_corr(ts)


if __name__ == "__main__":
    unittest.main()
