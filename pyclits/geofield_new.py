"""
Base class for any geo spatio-temporal field. Represented as xarray.DataArray.

(c) Nikola Jajcay
"""

from datetime import date, datetime

import xarray as xr


class DataField:
    """
    Class holds spatio-temporal geophysical field.
    """

    def __init__(self, data=None):
        """
        :param data: spatio-temporal data
        :type data: xr.DataArray
        """
        assert isinstance(
            data, xr.DataArray
        ), f"Data has to be xr.DataArray, got {type(data)}"

        self.data = data
        # rename coords to unified format
        self._rename_coords("la", "lats")
        self._rename_coords("lo", "lons")
        self._rename_coords("time", "time")
        # transpose - make time dim always first

    def _rename_coords(self, substring, new_name):
        """
        Rename coordinate in xr.DataArray to new_name.

        :param substring: substring for old coordinate name to match
        :type substring: str
        :param new_name: new name for matched coordinate
        :type new_name: str
        """
        for coord in self.data.coords:
            if substring in coord:
                old_name = coord
                break
        self.data = self.data.rename({old_name: new_name})

    @property
    def time(self):
        return self.data.time

    @property
    def lats(self):
        return self.data.lats

    @property
    def lons(self):
        return self.data.lons

    def select_date(self, date_from, date_to, inplace=True):
        """
        Selects date range. Both ends are inclusive.

        :param date_from: date from
        :type date_from: datetime.date|datetime.datetime
        :param date_to: date to
        :type date_to: datetime.date|datetime.datetime
        :param inplace: whether to make operation in-place or return
        :type inplace: bool
        """
        assert isinstance(
            date_from, (date, datetime)
        ), f"Date from must be datetime, got {type(date_from)}"
        assert isinstance(
            date_to, (date, datetime)
        ), f"Date to must be datetime, got {type(date_to)}"

        selected_data = self.data.sel(time=slice(date_from, date_to))

        if inplace:
            self.data = selected_data
        else:
            return selected_data

    def select_months(self, months, inplace=True):
        """
        Subselects only certain months.

        :param months: months to keep
        :type months: list[int]
        :param inplace: whether to make operation in-place or return
        :type inplace: bool
        """
        months_index = filter(
            lambda x: self.data.time[x].dt.month.values in months,
            range(len(self.time)),
        )

        selected_data = self.data.isel(time=list(months_index))

        if inplace:
            self.data = selected_data
        else:
            return selected_data
