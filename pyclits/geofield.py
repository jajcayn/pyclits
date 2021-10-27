"""
Base class for any geo spatio-temporal field. Represented as xarray.DataArray.

(c) Nikola Jajcay
"""

from datetime import date, datetime
from multiprocessing import cpu_count

import numpy as np
import pandas as pd
import xarray as xr
from pathos.multiprocessing import Pool
from scipy.signal import detrend
from sklearn.decomposition import PCA

from .wavelet_analysis import MorletWavelet, continous_wavelet

TIME_UNITS = {
    "seconds": "s",
    "hours": "h",
    "days": "D",
    "months": "M",
    "years": "Y",
}


class DataField:
    """
    Class holds spatio-temporal geophysical field.
    """

    @classmethod
    def load_nc(cls, filename, variable=None):
        """
        Loads NetCDF file using xarray.

        :param filename: filename of nc file
        :type filename: str
        :param variable: which variable to load
        :type variable: str
        """
        try:
            loaded = xr.open_dataarray(filename)
        except ValueError:
            assert variable is not None
            loaded = xr.open_dataset(filename)
            loaded = loaded[variable]
        return cls(loaded)

    def __init__(self, data):
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
        # sort by lats and time, NOT lons
        self.data = self.data.sortby(self.data.lats)
        self.data = self.data.sortby(self.data.time)

        if np.any(self.data.lons.values < 0.0):
            self.data = self.shift_lons_to_all_east_notation(self.data)

        # assert time dimensions is the first one
        self.data = self.data.transpose(*(["time"] + self.dims_not_time))

        # get average dt
        self._dt = self.data.time.diff(dim="time").mean()

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

    @staticmethod
    def _update_coord_attributes(old_coord, new_coord, new_units=None):
        """
        Update coordinate attributes after shifting, resampling, etc.

        :param old_coord: old coordinate with attributes
        :type old_coord: xr.DataArray
        :param new_coord: new coordinate to fill attributes to
        :type new_coord: xr.DataArray
        :param new_units: new unit name if desired
        :type new_units: str|None
        :return: coordinate with attributes
        :rtype: xr.DataArray
        """
        new_attrs = old_coord.attrs
        # change range if present in attributes
        if "actual_range" in new_attrs:
            new_attrs["actual_range"] = [
                new_coord.values.min(),
                new_coord.values.max(),
            ]
        if "units" in new_attrs and new_units is not None:
            new_attrs["units"] = new_units

        new_coord.attrs = new_attrs
        return new_coord

    def shift_lons_to_minus_notation(self, dataarray):
        """
        Shift longitude coordinate to minus notation, i.e. -180E -- 179E.
        Longitudes less than 0 are considered W.

        :param dataarray: geospatial field with longitudes as coordinates
        :type dataarray: xr.DataArray|xr.Dataset
        :return: field with shifted coords to -180 -- 179
        :rtype: xr.DataArray|xr.Dataset
        """
        shifted = dataarray.assign_coords(
            lons=((dataarray.lons + 180) % 360) - 180
        )

        shifted["lons"] = self._update_coord_attributes(
            dataarray.lons, shifted.lons, new_units="degrees_east_west"
        )
        return shifted

    def shift_lons_to_all_east_notation(self, dataarray):
        """
        Shift longitude coordinate to 0-360 notation. Longitudes > 180 are
        considered W.

        :param dataarray: geospatial field with longitudes as coordinates
        :type dataarray: xr.DataArray|xr.Dataset
        :return: field with shifted coords to 0 - 360
        :rtype: xr.DataArray|xr.Dataset
        """
        shifted = dataarray.assign_coords(lons=(dataarray.lons + 360) % 360)

        shifted["lons"] = self._update_coord_attributes(
            dataarray.lons, shifted.lons, new_units="degrees_east"
        )
        return shifted

    @property
    def time(self):
        return self.data.time.values

    @property
    def lats(self):
        return self.data.lats.values

    @property
    def lons(self):
        return self.data.lons.values

    @property
    def spatial_dims(self):
        return (self.lats.shape[0], self.lons.shape[0])

    @property
    def shape(self):
        return self.data.shape

    @property
    def dims_not_time(self):
        """
        Return list of dimensions that are not time.
        """
        return [dim for dim in self.data.dims if dim != "time"]

    @property
    def coords_not_time(self):
        """
        Return dict with all coordinates except time.
        """
        return {k: v.values for k, v in self.data.coords.items() if k != "time"}

    def dt(self, units="seconds"):
        """
        Return average dt in units.

        :param units: units in which to return dt, supported are:
            - seconds
            - hours
            - days
            - months
            - years
        :type units: str
        :return dt in selected unit
        :rtype: float
        """
        if units not in TIME_UNITS:
            raise ValueError(
                f"`{units}` not understood, use one of the "
                f"{TIME_UNITS.keys()}"
            )
        return self._dt.values.astype("timedelta64[s]") / np.timedelta64(
            1, TIME_UNITS[units]
        ).astype("timedelta64[s]")

    def save(self, filename):
        """
        Save DataField as nc file.

        :param filename: filename for nc file
        :type filename: str
        """
        # harmonize missing and fill_values for saving
        try:
            self.data.variable.encoding[
                "missing_value"
            ] = self.data.variable.encoding["_FillValue"]
        except KeyError:
            pass

        if not filename.endswith(".nc"):
            filename += ".nc"

        self.data.to_netcdf(filename)

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
            return DataField(data=selected_data)

    def select_months(self, months, inplace=True):
        """
        Subselects only certain months.

        :param months: months to keep
        :type months: list[int]
        :param inplace: whether to make operation in-place or return
        :type inplace: bool
        """
        assert isinstance(
            months, (list, tuple)
        ), f"Months must be an iterable, got {type(months)}"

        months_index = filter(
            lambda x: self.data.time[x].dt.month.values in months,
            range(len(self.time)),
        )
        selected_data = self.data.isel(time=list(months_index))

        if inplace:
            self.data = selected_data
        else:
            return DataField(data=selected_data)

    def select_lat_lon(self, lats=None, lons=None, inplace=True):
        """
        Subselects region in the data given by lats and lons.

        :param lats: latitude bounds to keep, both are inclusive
        :type lats: list[int|None]|None
        :param lons: longitude bounds to keep, both are inclusive
        :type lons: list[int|None]|None
        :param inplace: whether to make operation in-place or return
        :type inplace: bool
        """
        selected_data = self.data.copy()
        if lats is not None:
            assert len(lats) == 2, f"Need two lats, got {lats}"
            selected_data = selected_data.sel(lats=slice(lats[0], lats[1]))

        if lons is not None:
            assert len(lons) == 2, f"Need two lons, got {lons}"
            # shift to -180 -- 179 (xarray only selects on monotically
            # increasing coordinates)
            shifted = self.shift_lons_to_minus_notation(selected_data)
            # shift user passed lons to -180 -- 179
            lons = [((lon + 180) % 360 - 180) for lon in lons]
            # select
            selected_data = shifted.sel(lons=slice(lons[0], lons[1]))
            # shift back
            selected_data = self.shift_lons_to_all_east_notation(selected_data)

        if inplace:
            self.data = selected_data
        else:
            return DataField(data=selected_data)

    @property
    def cos_weights(self):
        """
        Returns a grid with scaling weights based on cosine of latitude.
        """
        cos_weights = np.zeros(self.spatial_dims)
        for ndx in range(self.lats.shape[0]):
            cos_weights[ndx, :] = np.cos(self.lats[ndx] * np.pi / 180.0) ** 0.5

        return cos_weights

    def temporal_resample(self, resample_to, function=np.nanmean, inplace=True):
        """
        Resamples data in temporal sense.

        :param resample_to: target resampling period for the data
        :type resample_to: pandas time str
        :param function: function to use for reducing
        :type function: callable
        :param inplace: whether to make operation in-place or return
        :type inplace: bool
        """
        resampled = self.data.resample(time=resample_to).reduce(
            function, dim="time"
        )

        if inplace:
            self.data = resampled
        else:
            return DataField(data=resampled)

    def spatial_resample(self, d_lat, d_lon, method="linear", inplace=True):
        """
        Resamples data in spatial sense. Uses scipy's internal `interp` method.

        :param d_lat: target difference in latitude
        :param d_lat: float|int
        :param d_lon: target difference in longitude
        :param d_lon: float|int
        :param method: interpolation method to use
        :type method: str['nearest' or 'linear']
        :param inplace: whether to make operation in-place or return
        :type inplace: bool
        """
        # shift to -180 -- 179 for resampling
        shifted = self.shift_lons_to_minus_notation(self.data)

        resampled = shifted.interp(
            lats=np.arange(shifted.lats.min(), shifted.lats.max() + 1, d_lat),
            lons=np.arange(shifted.lons.min(), shifted.lons.max() + 1, d_lon),
            method=method,
        )
        # create new coordinates attributes
        resampled["lons"] = self._update_coord_attributes(
            shifted.lons, resampled.lons
        )
        resampled["lats"] = self._update_coord_attributes(
            shifted.lats, resampled.lats
        )

        resampled_shifted_back = self.shift_lons_to_all_east_notation(resampled)

        if inplace:
            self.data = resampled_shifted_back
        else:
            return DataField(data=resampled_shifted_back)

    def deseasonalise(
        self,
        base_period=None,
        standardise=False,
        detrend_data=False,
        inplace=True,
    ):
        """
        Removes seasonality in mean and optionally in std.

        :param base_period: period for computing cycle, if None, take all data
        :type base_period: list[datetimes]|None
        :param standardise: whether to also remove seasonality in STD
        :type standardise: bool
        :param detrend_data: whether to remove linear trend from the data
        :type detrend_data: bool
        :param inplace: whether to make operation in-place or return
        :type inplace: bool
        :return: seasonal mean, seasonal std, trend
        :rtype: xr.DataArray, xr.DataArray, xr.DataArray
        """
        inferred_freq = pd.infer_freq(self.time)
        if base_period is None:
            base_period = [None, None]

        base_data = self.data.sel(time=slice(base_period[0], base_period[1]))

        if inferred_freq in ["M", "SM", "BM", "MS", "SMS", "BMS"]:
            # monthly data
            groupby = base_data.time.dt.month
        elif inferred_freq in ["C", "B", "D"]:
            # daily data
            groupby = base_data.time.dayofyear
        else:
            raise ValueError(
                "Anomalise supported only for daily or monthly data"
            )

        # compute climatologies
        climatology_mean = base_data.groupby(groupby).mean("time")
        if standardise:
            climatology_std = base_data.groupby(groupby).std("time")
        else:
            climatology_std = 1.0

        def _detrend(x):
            if np.any(np.isnan(x)):
                return x
            else:
                return detrend(x, axis=0, type="linear")

        if detrend_data:
            detrended = xr.apply_ufunc(
                _detrend,
                self.data,
                input_core_dims=[["time"]],
                output_core_dims=[["time"]],
            ).transpose(*(["time"] + self.dims_not_time))
            trend = self.data - detrended
        else:
            detrended = self.data
            trend = 0.0

        stand_anomalies = xr.apply_ufunc(
            lambda x, mean, std: (x - mean) / std,
            detrended.groupby(groupby),
            climatology_mean,
            climatology_std,
        )

        if inplace:
            self.data = stand_anomalies
            return climatology_mean, climatology_std, trend
        else:
            return (
                DataField(data=stand_anomalies),
                climatology_mean,
                climatology_std,
                trend,
            )

    def anomalise(self, base_period=None, inplace=True):
        """
        Removes seasonal/yearly cycle from the data.

        :param base_period: period for computing cycle, if None, take all data
        :type base_period: list[datetimes]|None
        :param inplace: whether to make operation in-place or return
        :type inplace: bool
        :return: seasonal mean
        :rtype: xr.DataArray
        """
        return self.deseasonalise(
            base_period=base_period, standardise=False, inplace=inplace
        )[:-1]

    def pca(self, n_comps, return_nans=False):
        """
        Perform PCA decomposition. NaNs are removed beforehand, and optionally
        added back to the EOFs.

        :param n_comps: number of PCA components [if int], or ratio of variance
            to retain [if float < 1]
        :type n_comps: int|float
        :param return_nans: whether to return NaNs after the PCA
        :rtype return_nans: bool
        :return: principal components (the timeseries from PCA), empirical
            orthogonal functions (the components from PCA, variance explained by
            the components
        :rtype: xr.DataArray, xr.DataArray, np.ndarray
        """
        flat_data = self.data.stack(space=self.dims_not_time).dropna(
            dim="space", how="any"
        )
        self.pca_mean = flat_data.mean(dim="time")
        self.pca = PCA(n_components=n_comps)
        pcs = self.pca.fit_transform((flat_data - self.pca_mean).values)
        pcs = xr.DataArray(
            data=pcs.copy(),
            dims=["time", "component"],
            coords={"time": self.time, "component": np.arange(1, n_comps + 1)},
        )
        eofs = xr.DataArray(
            data=self.pca.components_.copy(),
            dims=["component", "space"],
            coords={
                "component": np.arange(1, n_comps + 1),
                "space": flat_data.coords["space"],
            },
        ).unstack()
        if return_nans:
            eofs_full = np.empty((eofs.shape[0],) + self.data.shape[1:])
            eofs_full[:] = np.nan
            _, _, idx_lats = np.intersect1d(
                eofs.lats.values, self.lats, return_indices=True
            )
            _, _, idx_lons = np.intersect1d(
                eofs.lons.values, self.lons, return_indices=True
            )
            eofs_full[:, idx_lats.reshape((-1, 1)), idx_lons] = eofs.values
            eofs = xr.DataArray(
                data=eofs_full,
                dims=["component"] + self.dims_not_time,
                coords={
                    "component": np.arange(1, n_comps + 1),
                    **self.coords_not_time,
                },
            )

        var = self.pca.explained_variance_ratio_.copy()

        return pcs, eofs, var

    def invert_pca(self, eofs, pcs):
        pass

    @staticmethod
    def _get_parametric_phase(args):
        """
        Helper function for parallel computation of parametric phase.
        """
        i, half_length, upper_bound, freq, window, data = args
        if np.any(np.isnan(data)):
            return i, np.nan

        c = np.cos(np.arange(-half_length, upper_bound, 1) * freq)
        s = np.sin(np.arange(-half_length, upper_bound, 1) * freq)
        cx = np.dot(c, data) / data.shape[0]
        sx = np.dot(s, data) / data.shape[0]
        mx = np.sqrt(cx ** 2 + sx ** 2)
        phi = np.angle(cx - 1j * sx)
        z = mx * np.cos(np.arange(-half_length, upper_bound, 1) * freq + phi)

        # iterate with window
        iphase = np.zeros_like(data)
        half_window = int(np.floor(window / 2))
        upper_bound_window = half_window + 1 if window & 0x1 else half_window
        co = np.cos(np.arange(-half_window, upper_bound_window, 1) * freq)
        so = np.sin(np.arange(-half_window, upper_bound_window, 1) * freq)

        for shift in range(0, data.shape[0] - window + 1):
            y = data[shift : shift + window].copy()
            y -= np.mean(y)
            cxo = np.dot(co, y) / window
            sxo = np.dot(so, y) / window
            phio = np.angle(cxo - 1j * sxo)
            iphase[shift + half_window] = phio

        iphase[shift + half_window + 1 :] = np.angle(
            np.exp(1j * (np.arange(1, upper_bound_window) * freq + phio))
        )
        y = data[:window].copy()
        y -= np.mean(y)
        cxo = np.dot(co, y) / window
        sxo = np.dot(so, y) / window
        phio = np.angle(cxo - 1j * sxo)
        iphase[:half_window] = np.angle(
            np.exp(1j * (np.arange(-half_window, 0, 1) * freq + phio))
        )

        return i, iphase

    def parametric_phase(
        self,
        central_period,
        window,
        units="years",
        return_wrapped=True,
        inplace=True,
    ):
        """
        Computes phase of the analytic signal using parametric method. Data are
        padded in the temporal dimension by the 1/4 of the central period at
        both sides.

        :param central_period: central period of the analytic signal, in units
            given in `units`
        :type central_period: float
        :param window: length of the estimation window, in units given in
            `units`
        :type window: float
        :param units: units for the central period and window
        :type units: str
        :param return_wrapped: return wrapped phase (bounded in -pi, pi)
        :type return_wrapped: bool
        :param inplace: whether to make operation in-place or return
        :type inplace: bool
        """
        demeaned_data = self.data - self.data.mean(dim="time")
        # pad before computation
        how_much = int((central_period * 2) * (1.0 / self.dt(units)))
        padded_data = np.pad(
            demeaned_data.values,
            pad_width=[(how_much, how_much)]
            + [(0, 0)] * len(self.dims_not_time),
            mode="symmetric",
        )
        half_length = int(np.floor(padded_data.shape[0] / 2))
        upper_bound = (
            half_length + 1 if padded_data.shape[0] & 0x1 else half_length
        )
        orig_shape = padded_data.shape[1:]
        data_reshape = padded_data.reshape((padded_data.shape[0], -1))
        freq = 2 * np.pi / (central_period * (1.0 / self.dt(units)))
        args = [
            (
                i,
                half_length,
                upper_bound,
                freq,
                int(window * (1.0 / self.dt(units))),
                data_reshape[:, i],
            )
            for i in range(data_reshape.shape[1])
        ]
        pool = Pool(cpu_count())
        results = pool.imap_unordered(self._get_parametric_phase, args)
        pool.close()
        pool.join()
        parametric_phase = np.zeros((len(self.time), data_reshape.shape[1]))
        for result in results:
            i, phase_ = result
            parametric_phase[:, i] = phase_[how_much:-how_much]
        parametric_phase = parametric_phase.reshape(
            (len(self.time),) + orig_shape
        )
        parametric_phase = (
            parametric_phase
            if return_wrapped
            else np.unwrap(parametric_phase, axis=0)
        )

        parametric_phase = xr.DataArray(
            data=parametric_phase,
            dims=self.data.dims,
            coords=self.data.coords,
            attrs={
                "parametric_phase_period": central_period,
                "parametric_phase_window": window,
                "parametric_phase_units": units,
                **self.data.attrs,
            },
        )

        if inplace:
            self.data = parametric_phase
        else:
            return DataField(data=parametric_phase)

    @staticmethod
    def _regress_amps(args):
        """
        Helper function for parallel regression of amplitudes to the data range.
        """
        i, amp, recon, data = args
        fit_x = np.vstack([recon, np.ones((recon.shape[0]))]).T
        try:
            m, c = np.linalg.lstsq(fit_x, data)[0]
            fit_amp = m * amp + c
        except np.linalg.LinAlgError:
            fit_amp = np.zeros_like(amp)
            amp[:] = np.nan
        return i, fit_amp

    @staticmethod
    def _get_wvlt_coefficients(args):
        """
        Helper function for parallel wavelet computing
        """
        i, s0, wavelet, k0, data = args
        wave, _, _, _ = continous_wavelet(
            data, dt=1.0, pad=True, wavelet=wavelet, dj=0, s0=s0, j1=0, k0=k0
        )
        return i, wave

    def ccwt(
        self,
        central_period,
        units="years",
        wavelet=MorletWavelet(),
        k0=6.0,
        return_as="complex",
        inplace=True,
    ):
        """
        Perform continuous complex wavelet transform of the data. Data are
        padded in the temporal dimension by the 1/4 of the longest period at
        both sides.

        :param central_periods: central period to use for the wavelet, in units
            given in `units`
        :type central_period: float
        :param units: units for the wavelet
        :type units: str
        :param wavelet: mother wavelet to use
        :type wavelet: `wvlt.MotherWavelet`
        :param k0: mother wavelet parameter - wavenumber for Morlet, order for
            Paul, derivative for DoG
        :type k0: float
        :param return_as: what type of result to return
            `raw`: return raw wavelet coefficients
            `amplitude` will compute amplitude, hence abs(H(x))
            `amplitude_regressed` will compute amplitude and regress it onto the
                data, so the ranges are equal
            `phase_wrapped` will compute phase, hence angle(H(x)), in -pi,pi
            `phase_unwrapped` will compute phase in a continuous sense, hence
                monotonic
            `reconstruction` will compute reconstruction of the signal as
                A*cos(phi)
            `reconstruction_regressed` will compute reconstruction of the signal
                with amplitdes regressed to the data range
        :type return_as: str
        :param inplace: whether to make operation in-place or return
        :type inplace: bool
        """
        if units not in TIME_UNITS:
            raise ValueError(
                f"`{units}` not understood, use one of the "
                f"{TIME_UNITS.keys()}"
            )
        s0 = (central_period * (1.0 / self.dt(units))) / wavelet.fourier_factor(
            k0
        )
        # pad before wavelet
        how_much = int((central_period * 2) * (1.0 / self.dt(units)))
        padded_data = np.pad(
            self.data.values,
            pad_width=[(how_much, how_much)]
            + [(0, 0)] * len(self.dims_not_time),
            mode="symmetric",
        )
        orig_shape = padded_data.shape[1:]
        data_reshape = padded_data.reshape((padded_data.shape[0], -1))
        args = [
            (i, s0, wavelet, k0, data_reshape[:, i])
            for i in range(data_reshape.shape[1])
        ]
        pool = Pool(cpu_count())
        results = pool.imap_unordered(self._get_wvlt_coefficients, args)
        pool.close()
        pool.join()
        coeffs = np.zeros(
            (len(self.time), data_reshape.shape[1]), dtype=np.complex128
        )
        for result in results:
            i, wave = result
            coeffs[:, i] = wave.squeeze()[how_much:-how_much]
        coeffs = coeffs.reshape((len(self.time),) + orig_shape)
        amplitude = np.abs(coeffs)
        phase = np.angle(coeffs)
        reconstruction = amplitude * np.cos(phase)

        if return_as in ["amplitude_regressed", "reconstruction_regressed"]:
            reconstruction_reshaped = reconstruction.reshape(
                (len(self.time), -1)
            )
            amplitude_reshape = amplitude.reshape((len(self.time), -1))
            data_reshape = self.data.values.reshape((len(self.time), -1))
            args = [
                (
                    i,
                    amplitude_reshape[:, i],
                    reconstruction_reshaped[:, i],
                    data_reshape[:, i],
                )
                for i in range(data_reshape.shape[1])
            ]
            pool = Pool(cpu_count())
            results = pool.imap_unordered(self._regress_amps, args)
            pool.close()
            pool.join()
            amps_regressed = np.zeros_like(reconstruction_reshaped)
            for result in list(results):
                i, amps_r = result
                amps_regressed[:, i] = amps_r
            amplitude = amps_regressed.reshape((len(self.time),) + orig_shape)

        if return_as == "phase_wrapped":
            wvlt_result = phase
        elif return_as == "phase_unwrapped":
            wvlt_result = np.unwrap(phase, axis=0)
        elif return_as in ["amplitude", "amplitude_regressed"]:
            wvlt_result = amplitude
        elif return_as in ["reconstruction", "reconstruction_regressed"]:
            wvlt_result = amplitude * np.cos(phase)
        elif return_as == "raw":
            wvlt_result = coeffs

        wvlt = xr.DataArray(
            data=wvlt_result,
            dims=self.data.dims,
            coords=self.data.coords,
            attrs={
                "CCWT_period": central_period,
                "CCWT_units": units,
                **self.data.attrs,
            },
        )

        if inplace:
            self.data = wvlt
        else:
            return DataField(data=wvlt)


class StationDataField(DataField):
    """
    Class holds station data, hence 1D in space.
    """

    data_name = ""

    @classmethod
    def init_with_numpy(cls, data, time, data_name="", lat=None, lon=None):
        assert data.ndim == 1
        assert len(data) == len(time)
        initd = cls(
            xr.DataArray(
                data[:, np.newaxis, np.newaxis],
                dims=["time", "lats", "lons"],
                coords={
                    "time": time,
                    "lats": [lat or np.nan],
                    "lons": [lon or np.nan],
                },
            )
        )
        initd.data_name = data_name

        return initd

    @classmethod
    def load_ECAD_station(
        cls,
        filename,
        station_name,
        skiprows=19,
        missing=-9999,
        multiplier=0.1,
        lat=None,
        lon=None,
    ):
        data = pd.read_csv(filename, skiprows=skiprows, delimiter=",")
        data.columns = data.columns.str.strip()
        data["DATE"] = pd.to_datetime(data["DATE"], format="%Y%m%d")
        col_name = data.columns[2]
        data[col_name] = data[col_name].astype(float).replace({missing: np.nan})
        data = data.drop("SOUID", axis=1)
        data = data.rename({"DATE": "time"}, axis=1)
        data = data.set_index("time")
        data[col_name] = data[col_name] * multiplier

        initd = cls(
            xr.DataArray.from_series(data[col_name])
            .expand_dims(["lats", "lons"])
            .assign_coords({"lats": [lat or np.nan], "lons": [lon or np.nan]})
        )
        initd.data_name = f"ECAD-{col_name}-{station_name}"

        return initd
