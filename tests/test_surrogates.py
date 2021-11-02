"""
Integration tests for surrogate creation.
"""


import os
import unittest
from datetime import datetime

import numpy as np
import pytest
import xarray as xr
from pyclits.surrogates import (
    DataField,
    SurrogateField,
    correct_for_multiple_comparisons,
    get_p_values,
    get_single_shuffle_surrogate,
    get_single_time_shift_surrogate,
)

from . import TestHelperTempSave

DEFAULT_SEED = 42


class TestSurrogateField(TestHelperTempSave):
    def load_df(self):
        """
        Loads testing dataset: temperature at sigma 0.995 level from NCEP/NCAR
        reanalysis.
        """
        return DataField.load_nc(
            os.path.join(self.test_data_path, "air.sig995.nc")
        )

    def test_init(self):
        df = self.load_df()
        df.select_date(None, datetime(1998, 5, 2))
        surrs = SurrogateField.from_datafield(df)
        self.assertTrue(isinstance(surrs, SurrogateField))
        self.assertTrue(isinstance(surrs, DataField))
        self.assertTrue(hasattr(surrs, "orig_data_xr"))
        xr.testing.assert_equal(surrs.orig_data_xr, df.data)
        xr.testing.assert_equal(surrs.data, df.data)
        self.assertListEqual(df.process_steps, surrs.process_steps)

    def test_construct_surrogates(self):
        df = self.load_df()
        df.select_date(None, datetime(1995, 4, 10))
        df.select_lat_lon([30, 40], [17.00001, 28])
        surrs = SurrogateField.from_datafield(df)

        for surr_type in ["shift", "shuffle", "FT", "AAFT", "IAAFT", "MF"]:
            sf = surrs.construct_surrogates(
                surr_type, univariate=True, inplace=False
            )
            self.assertTrue(isinstance(sf, SurrogateField))
            self.assertEqual(len(sf.process_steps), len(df.process_steps) + 1)
            self.assertTupleEqual(sf.shape, df.shape)
            xr.testing.assert_equal(surrs.orig_data_xr, df.data)

        surrs.construct_surrogates("shift", univariate=False, inplace=True)
        self.assertTrue(isinstance(surrs, SurrogateField))
        self.assertEqual(len(surrs.process_steps), len(df.process_steps) + 1)
        xr.testing.assert_equal(surrs.orig_data_xr, df.data)

        with pytest.raises(ValueError):
            sf = surrs.construct_surrogates(
                "asdigays", univariate=True, inplace=False
            )

    def test_add_seasonality(self):
        df = self.load_df()
        df.select_date(None, datetime(1995, 4, 10))
        df.select_lat_lon([30, 40], [17.00001, 28])
        surrs = SurrogateField.from_datafield(df)

        surrs.construct_surrogates("shift", univariate=False, inplace=True)
        np.random.seed(DEFAULT_SEED)
        rand_mean = np.random.rand(*surrs.shape)

        added = surrs.add_seasonality(rand_mean, 1.0, -0.5, inplace=False)
        self.assertTrue(isinstance(added, SurrogateField))
        self.assertEqual(len(added.process_steps), len(surrs.process_steps) + 1)
        xr.testing.assert_equal(added.orig_data_xr, df.data)
        xr.testing.assert_equal(added.orig_data_xr, surrs.orig_data_xr)

        surrs.add_seasonality(rand_mean, 1.0, -0.5, inplace=True)
        xr.testing.assert_equal(surrs.data, added.data)


class TestHelperFunctions(unittest.TestCase):
    def test_correct_for_multiple_comparisons(self):
        np.random.seed(DEFAULT_SEED)
        pvals = np.random.rand(20)
        for method in [
            "b",
            "s",
            "h",
            "hs",
            "sh",
            "ho",
            "fdr_bh",
            "fdr_by",
            "fdr_tsbh",
            "fdr_tsbky",
            "fdr_gbs",
        ]:
            rej, corr_pvals = correct_for_multiple_comparisons(
                pvals, alpha_level=0.05, method=method
            )
            self.assertTrue(isinstance(rej, np.ndarray))
            self.assertTrue(isinstance(corr_pvals, np.ndarray))
            self.assertTrue(np.all(np.less_equal(pvals, corr_pvals)))

    def test_get_p_values(self):
        RESULT = np.array([0.56, 0.05, 0.23])

        np.random.seed(DEFAULT_SEED)
        data = np.random.rand(3) + 0.01
        surrs = np.random.rand(100, 3)
        p_vals = get_p_values(data, surrs, tailed="upper")
        np.testing.assert_almost_equal(p_vals, RESULT)
        p_vals = get_p_values(data, surrs, tailed="lower")
        np.testing.assert_almost_equal(p_vals, 1.0 - RESULT)


class TestSurrogateFunctions(unittest.TestCase):
    def get_ts(self):
        np.random.seed(DEFAULT_SEED)
        return np.random.rand(100)

    def test_get_single_time_shift_surrogate(self):
        ts = self.get_ts()
        surr = get_single_time_shift_surrogate(ts, seed=DEFAULT_SEED)
        with pytest.raises(AssertionError):
            np.testing.assert_equal(ts, surr)
        np.testing.assert_equal(ts.sort(), surr.sort())

    def test_get_single_shuffle_surrogate(self):
        ts = self.get_ts()
        surr = get_single_shuffle_surrogate(
            ts, cut_points=None, seed=DEFAULT_SEED
        )
        with pytest.raises(AssertionError):
            np.testing.assert_equal(ts, surr)
        np.testing.assert_equal(ts.sort(), surr.sort())

        surr = get_single_shuffle_surrogate(
            ts, cut_points=12, seed=DEFAULT_SEED
        )
        with pytest.raises(AssertionError):
            np.testing.assert_equal(ts, surr)
        np.testing.assert_equal(ts.sort(), surr.sort())

    def test_get_single_FT_surrogate(self):
        pass

    def test_get_single_AAFT_surrogate(self):
        pass

    def test_get_single_IAAFT_surrogate(self):
        pass

    def test_get_single_MF_surrogate(self):
        pass


if __name__ == "__main__":
    unittest.main()
