"""
Integration tests for plotting.
"""
import os
import unittest

import numpy as np
from cartopy.crs import Mercator, Robinson, Geocentric
from pyclits.geofield import DataField
from pyclits.plotting import GeoPlot

from . import TestHelperTempSave


class TestGeoPlot(TestHelperTempSave):
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
        plot = GeoPlot(df.lats, df.lons, whole_world=False, projection=None)
        np.testing.assert_equal(df.lats, plot.lats)
        np.testing.assert_equal(df.lons, plot.lons)
        self.assertFalse(plot.whole_world)
        self.assertEqual(plot.projection, Mercator())

        plot = GeoPlot(df.lats, df.lons, whole_world=True, projection=None)
        self.assertTrue(plot.whole_world)
        self.assertEqual(plot.projection, Robinson())

        plot = GeoPlot(df.lats, df.lons, projection=Geocentric())
        self.assertEqual(plot.projection, Geocentric())

    def test_setup(self):
        df = self.load_df()
        plot = GeoPlot(df.lats, df.lons, whole_world=False)
        plot._set_up_map()
        self.assertTrue(hasattr(plot, "axis"))

        plot = GeoPlot(df.lats, df.lons, whole_world=True)
        plot._set_up_map()
        self.assertTrue(hasattr(plot, "axis"))

    def test_plot(self):
        df = self.load_df()
        plot = GeoPlot(df.lats, df.lons, whole_world=False, colormesh=False)
        plot.plot(df.data.values[0, ...])
        plot.plot(df.data.values[0, ...], symmetric_plot=True)

        plot = GeoPlot(df.lats, df.lons, whole_world=False, colormesh=True)
        self.assertTrue(plot.colormesh)
        plot.plot(df.data.values[0, ...])


if __name__ == "__main__":
    unittest.main()
