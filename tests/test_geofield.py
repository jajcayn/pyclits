"""
Integration tests for base class - DataField.

(c) Nikola Jajcay
"""

import os
import unittest

from pyclits.geofield_new import DataField

from . import TestHelperTempSave


class TestLoadSave(TestHelperTempSave):
    """
    Test basic loading of nc files.
    """

    def test_load_save(self):
        df = DataField.load_nc(
            os.path.join(self.test_data_path, "air.sig995.nc")
        )
        filename = os.path.join(self.temp_dir, "test.nc")
        df.save(filename)

        self.compare_nc_files(
            os.path.join(self.test_data_path, "load_save_result.nc"), filename
        )


if __name__ == "__main__":
    unittest.main()
