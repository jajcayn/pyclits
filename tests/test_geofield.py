"""
Integration tests for base class - DataField.

(c) Nikola Jajcay
"""

import os
import unittest
from datetime import datetime

import numpy as np

from pyclits.geofield_new import DataField

from . import TestHelperTempSave


class TestLoadSave(TestHelperTempSave):
    """
    Test basic loading of nc files.
    """

    CORRECT_LATS = np.array(
        [
            30.0,
            32.5,
            35.0,
            37.5,
            40.0,
            42.5,
            45.0,
            47.5,
            50.0,
            52.5,
            55.0,
            57.5,
            60.0,
            62.5,
            65.0,
            67.5,
            70.0,
        ]
    )

    CORRECT_LONS = np.array(
        [
            0.0,
            2.5,
            5.0,
            7.5,
            10.0,
            12.5,
            15.0,
            17.5,
            20.0,
            22.5,
            25.0,
            27.5,
            30.0,
            32.5,
            35.0,
            37.5,
            40.0,
            350.0,
            352.5,
            355.0,
            357.5,
        ]
    )

    SELECTED_LATS = np.array([32.5, 35.0, 37.5, 40.0])
    SELECTED_LONS = np.array([17.5, 20.0, 22.5, 25.0, 27.5])
    SELECTED_LONS_PRIME_MERIDIAN = np.array([0.0, 2.5, 357.5])

    COS_WEIGHTS = np.array(
        [
            0.93060486,
            0.91836346,
            0.90507019,
            0.89070385,
            0.87523965,
            0.85864855,
            0.84089642,
            0.82194295,
            0.80174036,
            0.78023165,
            0.75734829,
            0.73300724,
            0.70710678,
            0.67952087,
            0.65009096,
            0.61861412,
            0.58482488,
        ]
    )

    def load_df(self):
        """
        Loads testing dataset: temperature at sigma 0.995 level from NCEP/NCAR
        reanalysis.
        """
        return DataField.load_nc(
            os.path.join(self.test_data_path, "air.sig995.nc")
        )

    def test_load_save(self):
        df = self.load_df()
        filename = os.path.join(self.temp_dir, "test.nc")
        df.save(filename)

        self.compare_nc_files(
            os.path.join(self.test_results_path, "load_save_result.nc"),
            filename,
        )

    def test_properties(self):
        df = self.load_df()
        np.testing.assert_equal(df.lats, self.CORRECT_LATS)
        np.testing.assert_equal(df.lons, self.CORRECT_LONS)
        self.compare_time_range(
            df.time, datetime(1990, 1, 1), datetime(2000, 1, 1)
        )
        self.assertTrue(
            df.spatial_dims
            == (self.CORRECT_LATS.shape[0], self.CORRECT_LONS.shape[0])
        )

    def test_select_date(self):
        df = self.load_df()
        selected_df = df.select_date(
            datetime(1991, 5, 20), datetime(1992, 3, 7), inplace=False
        )
        self.compare_time_range(
            selected_df.time, datetime(1991, 6, 1), datetime(1992, 3, 1)
        )

    def test_select_months(self):
        df = self.load_df()
        # select jan, july and august
        jja_months = [1, 7, 8]
        jja_df = df.select_months(months=jja_months, inplace=False)
        # test that all datetimes have months only in the allowed range
        self.assertTrue(
            all(
                jja_df.time.astype("datetime64[D]")[time].item().month
                in jja_months
                for time in range(jja_df.time.shape[0])
            )
        )

    def test_select_lat_lon(self):
        df = self.load_df()
        lats_cut = [32, 41.32]
        lons_cut = [17.00001, 28]
        selected_df = df.select_lat_lon(lats_cut, lons_cut, inplace=False)
        np.testing.assert_equal(selected_df.lats, self.SELECTED_LATS)
        np.testing.assert_equal(selected_df.lons, self.SELECTED_LONS)

    def test_select_lat_lon_through_prime_meridian(self):
        df = self.load_df()
        lats_cut = [32, 41.32]
        lons_cut = [356, 4.12]
        selected_df = df.select_lat_lon(lats_cut, lons_cut, inplace=False)
        np.testing.assert_equal(selected_df.lats, self.SELECTED_LATS)
        np.testing.assert_equal(
            selected_df.lons, self.SELECTED_LONS_PRIME_MERIDIAN
        )

    def test_cos_weights(self):
        df = self.load_df()
        np.testing.assert_allclose(df.cos_weights[:, 0], self.COS_WEIGHTS)

    def test_temporal_resample(self):
        df = self.load_df()
        resampled_df = df.temporal_resample(
            resample_to="3M", function=np.nanmean, inplace=False
        )
        filename = os.path.join(self.temp_dir, "temp_resample.nc")
        resampled_df.save(filename)

        # assert only time dimension got resample - no changed to spatial
        self.assertTrue(
            resampled_df.spatial_dims
            == (self.CORRECT_LATS.shape[0], self.CORRECT_LONS.shape[0])
        )
        self.compare_nc_files(
            os.path.join(self.test_results_path, "temp_resample_result.nc"),
            filename,
        )

    def test_spatial_resample(self):
        df = self.load_df()
        resampled_df = df.spatial_resample(
            d_lat=5, d_lon=5, method="linear", inplace=False
        )
        # TODO finish test


if __name__ == "__main__":
    unittest.main()
