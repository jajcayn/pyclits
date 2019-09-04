"""
Test helpers.

(c) Nikola Jajcay
"""


import os
import shutil
import tempfile
import unittest

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

        xr.testing.assert_equal(xr.open_dataset(file1), xr.open_dataset(file2))


class TestHelperTempSave(TestHelper):
    """
    Helper that supports saving nc files in a temporary location and removing
    the temp on tear down.
    """

    @classmethod
    def setUpClass(cls):
        abs_path = os.path.dirname(os.path.realpath(__file__))
        cls.test_data_path = os.path.join(abs_path, "data")
        cls.temp_dir = tempfile.mkdtemp(dir=cls.test_data_path)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir)
