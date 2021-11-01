"""
Test helpers.

(c) Nikola Jajcay
"""


import os
import shutil
import tempfile
import unittest

import numpy as np
import xarray as xr


class TestHelper(unittest.TestCase):
    """
    Helper class for single test case.
    """

    PATTERN_NC = ".nc"

    def compare_nc_files(self, file1, file2):
        """
        Compare two netcdf files. Uses xarray for comparison.

        :param file1: file 1 for comparison
        :type file1: str
        :param file2: file 2 for comparison
        :type file2: str
        """
        assert file1.endswith(self.PATTERN_NC), f"Unknown extension: {file1}"
        assert file2.endswith(self.PATTERN_NC), f"Unknown extension: {file2}"

        xr.testing.assert_allclose(
            xr.open_dataset(file1), xr.open_dataset(file2)
        )

    def compare_time_range(self, time_array, start_date, end_date):
        """
        Test time range: compares start and end datetime.

        :param time_array: time array from xr.DataArray
        :type time_array: np.ndarray[np.datetime64]
        :param start_date: start date of the range
        :type start_date: datetime|date
        :param end_date: end date of the range
        :type end_date: datetime|date
        """
        self.assertTrue(time_array[0] == np.datetime64(start_date))
        self.assertTrue(time_array[-1] == np.datetime64(end_date))


class TestHelperTempSave(TestHelper):
    """
    Helper that supports saving nc files in a temporary location and removing
    the temp on tear down.
    """

    @classmethod
    def setUpClass(cls):
        abs_path = os.path.dirname(os.path.realpath(__file__))
        cls.test_data_path = os.path.join(abs_path, "data")
        cls.test_results_path = os.path.join(cls.test_data_path, "test_results")
        cls.temp_dir = tempfile.mkdtemp(dir=cls.test_data_path)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir)
