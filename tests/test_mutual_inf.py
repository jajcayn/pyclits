"""
Integration tests for mutual information functions.
"""

import unittest

import numpy as np
import pytest
from pyclits.mutual_inf import (
    _create_naive_eqq_bins,
    _create_shifted_eqq_bins,
    _get_corr_entropy,
    _neg_harmonic,
    _standardize_ts,
    conditional_mutual_information,
    get_conditioned_timeseries,
    mutual_information,
)

DEFAULT_SEED = 42


class TestHelperFunctions(unittest.TestCase):
    def test_standardize_ts(self):
        ts = np.random.normal(12.4, 1.32, size=(100,))
        stand = _standardize_ts(ts)
        self.assertAlmostEqual(stand.mean(), 0.0)
        self.assertAlmostEqual(np.std(stand, ddof=1), 1.0)

    def test_neg_harmonic(self):
        self.assertEqual(_neg_harmonic(2), -1.5)
        self.assertEqual(_neg_harmonic(15), -3.3182289932289937)

    def test_get_corr_entropy(self):
        np.random.seed(DEFAULT_SEED)
        a = np.random.multivariate_normal(
            [0.0, 0.1], [[0.1, 0.4], [-0.6, 0.4]], size=(100,)
        )
        corr_en_loge = np.real(_get_corr_entropy(a, log2=False))
        self.assertEqual(corr_en_loge, 1010.4900806576101)
        corr_en_log2 = np.real(_get_corr_entropy(a, log2=True))
        self.assertEqual(corr_en_log2, 1457.8290282322223)

    def test_create_shifted_eqq_bins(self):
        np.random.seed(DEFAULT_SEED)
        ts = np.random.rand(100)
        bins = _create_shifted_eqq_bins(ts, no_bins=2)
        self.assertListEqual(
            bins, [0.005522117123602399, 0.4722149251619493, 0.9868869366005173]
        )

    def test_create_naive_eqq_bins(self):
        np.random.seed(DEFAULT_SEED)
        ts = np.random.rand(100)
        bins = _create_naive_eqq_bins(ts, no_bins=2)
        self.assertListEqual(
            bins, [0.005522117123602399, 0.4722149251619493, 0.9868869366005173]
        )


class TestGetConditionedTimeseries(unittest.TestCase):

    TAU = 10
    ETA = 3
    N = 100

    def generate_ts(self):
        np.random.seed(DEFAULT_SEED)
        return np.random.rand(2, self.N)

    def test_1d(self):
        ts = self.generate_ts()
        x, y, z = get_conditioned_timeseries(
            ts, tau=self.TAU, reversed=False, dim_of_condition=1
        )
        self.assertTrue(isinstance(x, np.ndarray))
        self.assertTrue(isinstance(y, np.ndarray))
        self.assertEqual(len(z), 1)
        self.assertTrue(all(isinstance(zi, np.ndarray) for zi in z))
        self.assertEqual(len(x), self.N - self.TAU)
        self.assertEqual(len(y), self.N - self.TAU)
        self.assertEqual(len(z[0]), self.N - self.TAU)
        np.testing.assert_equal(x, ts[0, : -self.TAU])
        np.testing.assert_equal(y, ts[1, self.TAU :])
        np.testing.assert_equal(z[0], ts[1, : -self.TAU])

    def test_1d_close_cond(self):
        ts = self.generate_ts()
        x, y, z = get_conditioned_timeseries(
            ts,
            tau=self.TAU,
            reversed=False,
            dim_of_condition=1,
            close_condition=True,
        )
        np.testing.assert_equal(x, ts[0, : -self.TAU])
        np.testing.assert_equal(y, ts[1, self.TAU :])
        np.testing.assert_equal(z[0], ts[1, self.TAU - 1 : -1])

    def test_1d_phase_diff(self):
        ts = self.generate_ts()
        x, y, z = get_conditioned_timeseries(
            ts,
            tau=self.TAU,
            reversed=False,
            dim_of_condition=1,
            phase_diff=True,
        )
        np.testing.assert_equal(x, ts[0, : -self.TAU])
        np.testing.assert_equal(y, ts[1, self.TAU :] - z[0])
        np.testing.assert_equal(z[0], ts[1, : -self.TAU])

    def test_1d_add_cond(self):
        ts = self.generate_ts()
        np.random.seed(DEFAULT_SEED)
        add_ts = np.random.rand(2, 100)
        x, y, z = get_conditioned_timeseries(
            ts,
            tau=self.TAU,
            reversed=False,
            dim_of_condition=1,
            add_cond=add_ts,
        )
        self.assertEqual(len(z), 3)
        self.assertTrue(all(isinstance(zi, np.ndarray) for zi in z))
        self.assertTrue(all(len(zi) == self.N - self.TAU for zi in z))
        np.testing.assert_equal(x, ts[0, : -self.TAU])
        np.testing.assert_equal(y, ts[1, self.TAU :])
        np.testing.assert_equal(z[0], ts[1, : -self.TAU])
        np.testing.assert_equal(z[1], add_ts[0, : -self.TAU])
        np.testing.assert_equal(z[2], add_ts[1, : -self.TAU])

    def test_1d_add_cond_close(self):
        ts = self.generate_ts()
        np.random.seed(DEFAULT_SEED)
        add_ts = np.random.rand(2, 100)
        x, y, z = get_conditioned_timeseries(
            ts,
            tau=self.TAU,
            reversed=False,
            dim_of_condition=1,
            add_cond=add_ts,
            close_condition=True,
        )
        np.testing.assert_equal(x, ts[0, : -self.TAU])
        np.testing.assert_equal(y, ts[1, self.TAU :])
        np.testing.assert_equal(z[0], ts[1, self.TAU - 1 : -1])
        np.testing.assert_equal(z[1], add_ts[0, self.TAU - 1 : -1])
        np.testing.assert_equal(z[2], add_ts[1, self.TAU - 1 : -1])

    def test_1d_rev(self):
        ts = self.generate_ts()
        x, y, z = get_conditioned_timeseries(
            ts, tau=self.TAU, reversed=True, dim_of_condition=1
        )
        np.testing.assert_equal(x, ts[1, : -self.TAU])
        np.testing.assert_equal(y, ts[0, self.TAU :])
        np.testing.assert_equal(z[0], ts[0, : -self.TAU])

    def test_3d(self):
        ts = self.generate_ts()
        ND = 3
        x, y, z = get_conditioned_timeseries(
            ts, tau=self.TAU, reversed=False, dim_of_condition=ND, eta=self.ETA
        )
        self.assertTrue(isinstance(x, np.ndarray))
        self.assertTrue(isinstance(y, np.ndarray))
        self.assertEqual(len(z), ND)
        self.assertTrue(all(isinstance(zi, np.ndarray) for zi in z))
        self.assertEqual(len(x), self.N - self.TAU - 2 * self.ETA)
        self.assertEqual(len(y), self.N - self.TAU - 2 * self.ETA)
        self.assertTrue(
            all(len(zi) == self.N - self.TAU - 2 * self.ETA for zi in z)
        )
        np.testing.assert_equal(x, ts[0, 2 * self.ETA : -self.TAU])
        np.testing.assert_equal(y, ts[1, 2 * self.ETA + self.TAU :])
        np.testing.assert_equal(z[0], ts[1, 2 * self.ETA : -self.TAU])
        np.testing.assert_equal(z[1], ts[1, self.ETA : -self.TAU - self.ETA])
        np.testing.assert_equal(z[2], ts[1, : -self.TAU - 2 * self.ETA])


class TestMutualInformation(unittest.TestCase):
    def generate_ts(self):
        np.random.seed(DEFAULT_SEED)
        rand = np.random.rand(50)
        return rand, np.exp(rand / 12.65 + np.sqrt(rand)) + np.random.normal(
            0.0, 0.1, size=(50)
        )

    def test_mutual_inf_eqd(self):
        x, y = self.generate_ts()
        mi = mutual_information(x, y, algorithm="EQD", bins=4, log2=True)
        self.assertEqual(mi, 1.4675603576588012)
        mi = mutual_information(x, y, algorithm="EQD", bins=4, log2=False)
        self.assertEqual(mi, 1.017235324212743)

    def test_mutual_inf_eqq_naive(self):
        x, y = self.generate_ts()
        mi = mutual_information(x, y, algorithm="EQQ_naive", bins=4, log2=True)
        self.assertEqual(mi, 1.2873827393211736)
        mi = mutual_information(x, y, algorithm="EQQ_naive", bins=4, log2=False)
        self.assertEqual(mi, 0.8923457160620105)

    def test_mutual_inf_eqq(self):
        x, y = self.generate_ts()
        mi = mutual_information(x, y, algorithm="EQQ", bins=4, log2=True)
        self.assertEqual(mi, 1.2873827393211736)
        mi = mutual_information(x, y, algorithm="EQQ", bins=4, log2=False)
        self.assertEqual(mi, 0.8923457160620105)

    def test_mutual_inf_knn(self):
        x, y = self.generate_ts()
        mi = mutual_information(x, y, algorithm="knn", k=16, log2=True)
        self.assertEqual(mi, 0.8398477550540533)

    def test_mutual_inf_unknown(self):
        x, y = self.generate_ts()
        with pytest.raises(ValueError):
            mutual_information(x, y, algorithm="BHSVADS")
        with pytest.raises(ValueError):
            mutual_information(x, y, algorithm="EEEHSVADS", bins=12)


class TestConditionalMutualInformation(unittest.TestCase):
    def generate_ts(self):
        np.random.seed(DEFAULT_SEED)
        rand = np.random.rand(50)
        dep = np.exp(rand / 12.65 + np.sqrt(rand)) + np.random.normal(
            0.0, 0.1, size=(50)
        )
        return get_conditioned_timeseries(
            [rand, dep], tau=2, dim_of_condition=2, eta=1
        )

    def test_cond_mutual_inf_eqd(self):
        x, y, z = self.generate_ts()
        cmi = conditional_mutual_information(
            x, y, z, algorithm="EQD", bins=4, log2=True
        )
        self.assertEqual(cmi, 0.37205717910686587)
        cmi = conditional_mutual_information(
            x, y, z, algorithm="EQD", bins=4, log2=False
        )
        self.assertEqual(cmi, 0.23206180465500795)

    def test_cond_mutual_inf_eqq_naive(self):
        x, y, z = self.generate_ts()
        cmi = conditional_mutual_information(
            x, y, z, algorithm="EQQ_naive", bins=4, log2=True
        )
        self.assertEqual(cmi, 0.1437210106843291)
        cmi = conditional_mutual_information(
            x, y, z, algorithm="EQQ_naive", bins=4, log2=False
        )
        self.assertEqual(cmi, 0.07012418863838997)

    def test_cond_mutual_inf_eqq(self):
        x, y, z = self.generate_ts()
        cmi = conditional_mutual_information(
            x, y, z, algorithm="EQQ", bins=4, log2=True
        )
        self.assertEqual(cmi, 0.1437210106843291)
        cmi = conditional_mutual_information(
            x, y, z, algorithm="EQQ", bins=4, log2=False
        )
        self.assertEqual(cmi, 0.07012418863838997)

    def test_cond_mutual_inf_knn(self):
        x, y, z = self.generate_ts()
        cmi = conditional_mutual_information(
            x, y, z, algorithm="knn", k=16, log2=True
        )
        self.assertEqual(cmi, 0.010902506625959596)
        cmi = conditional_mutual_information(
            x, y, z, algorithm="knn", k=16, log2=False
        )
        self.assertEqual(cmi, 0.0071026719472646604)

    def test_cond_mutual_inf_gcm(self):
        x, y, z = self.generate_ts()
        cmi = conditional_mutual_information(
            x, y, z, algorithm="GCM", log2=True
        )
        self.assertEqual(cmi, 0.0006656697681635682)
        cmi = conditional_mutual_information(
            x, y, z, algorithm="GCM", log2=False
        )
        self.assertEqual(cmi, 0.002527512095056351)

    def test_mutual_inf_unknown(self):
        x, y, z = self.generate_ts()
        with pytest.raises(ValueError):
            conditional_mutual_information(x, y, z, algorithm="BHSVADS")
        with pytest.raises(ValueError):
            conditional_mutual_information(
                x, y, z, algorithm="EEEHSVADS", bins=12
            )


if __name__ == "__main__":
    unittest.main()
