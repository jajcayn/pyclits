"""
created on Jan 29, 2014

@author: Nikola Jajcay, jajcay(at)cs.cas.cz
based on class by Martin Vejmelka -- https://github.com/vejmelkam/ndw-climate --

last update on Sep 26, 2017
"""

import numpy as np
from datetime import date, timedelta, datetime
from dateutil.relativedelta import relativedelta
from functions import detrend_with_return
import csv
from os.path import split
import os

        

class DataField:
    """
    Class holds the time series of a geophysical field. The fields for reanalysis data are
    3-dimensional - two spatial and one temporal dimension. The fields for station data contains
    temporal dimension and location specification.
    """
    
    def __init__(self, data_folder = '', data = None, lons = None, lats = None, time = None, verbose = False):
        """
        Initializes either an empty data set or with given values.
        """
        
        self.data_folder = data_folder
        self.data = data
        self.lons = lons
        self.lats = lats
        self.time = time
        self.location = None # for station data
        self.missing = None # for station data where could be some missing values
        self.station_id = None # for station data
        self.station_elev = None # in metres, for station data
        self.var_name = None
        self.nans = False
        self.cos_weights = None
        self.data_mask = None
        self.verbose = verbose



    def __str__(self):
        """
        String representation.
        """

        if self.data is not None:
            return ("Geo data of shape %s as time x lat x lon." % str(self.data.shape))
        else:
            return("Empty DataField instance.")



    def shape(self):
        """
        Prints shape of data field.
        """

        if self.data is not None:
            return self.data.shape
        else:
            raise Exception("DataField is empty.")
        
        
        
    def __getitem__(self, key):
        """
        getitem representation.
        """        

        if self.data is not None:
            return self.data[key]
        else:
            raise Exception("DataField is empty.")



    def load(self, filename = None, variable_name = None, dataset = 'ECA-reanalysis', print_prog = True):
        """
        Loads geophysical data from netCDF file for reanalysis or from text file for station data.
        Now supports following datasets: (dataset - keyword passed to function)
            ECA&D E-OBS gridded dataset reanalysis - 'ECA-reanalysis'
            ECMWF gridded reanalysis - 'ERA'
            NCEP/NCAR Reanalysis 1 - 'NCEP'
        """

        from netCDF4 import Dataset
        
        if dataset == 'ECA-reanalysis':
            d = Dataset(self.data_folder + filename, 'r')
            v = d.variables[variable_name]
            
            data = v[:] # masked array - only land data, not ocean/sea
            self.data = data.data.copy() # get only data, not mask
            self.data[data.mask] = np.nan # filled masked values with NaNs
            self.lons = d.variables['longitude'][:]
            self.lats = d.variables['latitude'][:]
            self.time = d.variables['time'][:] # days since 1950-01-01 00:00
            self.time += date.toordinal(date(1950, 1, 1))
            self.var_name = variable_name
            if np.any(np.isnan(self.data)):
                self.nans = True
            if print_prog:
                print("Data saved to structure. Shape of the data is %s" % (str(self.data.shape)))
                print("Lats x lons saved to structure. Shape is %s x %s" % (str(self.lats.shape[0]), str(self.lons.shape[0])))
                print("Time stamp saved to structure as ordinal values where Jan 1 of year 1 is 1")
                print("The first data value is from %s and the last is from %s" % (str(self.get_date_from_ndx(0)), str(self.get_date_from_ndx(-1))))
                print("Default temporal sampling in the data is %.2f day(s)" % (np.nanmean(np.diff(self.time))))
                if np.any(np.isnan(self.data)):
                    print("The data contains NaNs! All methods are compatible with NaNs, just to let you know!")
            
            d.close()     
                    
        elif dataset == 'ERA':
            d = Dataset(self.data_folder + filename, 'r')
            v = d.variables[variable_name]

            data = v[:]
            if isinstance(data, np.ma.masked_array):             
                self.data = data.data.copy() # get only data, not mask
                self.data[data.mask] = np.nan # filled masked values with NaNs
            else:
                self.data = data
            self.lons = d.variables['longitude'][:]
            self.lats = d.variables['latitude'][:]
            if 'level' in d.variables.keys():
                self.level = d.variables['level'][:]
            self.time = d.variables['time'][:] # hours since 1900-01-01 00:00
            self.time = self.time / 24.0 + date.toordinal(date(1900, 1, 1))
            self.var_name = variable_name
            if np.any(np.isnan(self.data)):
                self.nans = True
            if print_prog:
                print("Data saved to structure. Shape of the data is %s" % (str(self.data.shape)))
                print("Lats x lons saved to structure. Shape is %s x %s" % (str(self.lats.shape[0]), str(self.lons.shape[0])))
                print("Time stamp saved to structure as ordinal values where Jan 1 of year 1 is 1")
                print("The first data value is from %s and the last is from %s" % (str(self.get_date_from_ndx(0)), str(self.get_date_from_ndx(-1))))
                print("Default temporal sampling in the data is %.2f day(s)" % (np.nanmean(np.diff(self.time))))
                if np.any(np.isnan(self.data)):
                    print("The data contains NaNs! All methods are compatible with NaNs, just to let you know!")
            
            d.close()
            
        elif dataset == 'NCEP':
            d = Dataset(self.data_folder + filename, 'r')
            v = d.variables[variable_name]
            
            data = v[:] # masked array - only land data, not ocean/sea
            if isinstance(data, np.ma.masked_array):             
                self.data = data.data.copy() # get only data, not mask
                self.data[data.mask] = np.nan # filled masked values with NaNs
            else:
                self.data = data
            self.lons = d.variables['lon'][:]
            if np.any(self.lons < 0):
                self._shift_lons_to_360()
            self.lats = d.variables['lat'][:]
            if 'level' in d.variables.keys():
                self.level = d.variables['level'][:]
            self.time = d.variables['time'][:] # hours or days since some date
            date_since = self._parse_time_units(d.variables['time'].units)
            if "hours" in d.variables['time'].units:
                self.time = self.time / 24.0 + date.toordinal(date_since)
            elif "days" in d.variables['time'].units:
                self.time += date.toordinal(date_since)
            elif "months" in d.variables['time'].units:
                from dateutil.relativedelta import relativedelta
                for t in range(self.time.shape[0]):
                    self.time[t] = date.toordinal(date_since + relativedelta(months = +int(self.time[t])))
            self.var_name = variable_name
            if np.any(np.isnan(self.data)):
                self.nans = True
            if print_prog:
                print("Data saved to structure. Shape of the data is %s" % (str(self.data.shape)))
                print("Lats x lons saved to structure. Shape is %s x %s" % (str(self.lats.shape[0]), str(self.lons.shape[0])))
                print("Time stamp saved to structure as ordinal values where Jan 1 of year 1 is 1")
                print("The first data value is from %s and the last is from %s" % (str(self.get_date_from_ndx(0)), str(self.get_date_from_ndx(-1))))
                print("Default temporal sampling in the data is %.2f day(s)" % (np.nanmean(np.diff(self.time))))
                if np.any(np.isnan(self.data)):
                    print("The data contains NaNs! All methods are compatible with NaNs, just to let you know!")
            
            d.close()

        elif dataset == 'arbitrary':
            d = Dataset(self.data_folder + filename, 'r')
            v = d.variables[variable_name]

            data = v[:] # masked array - only land data, not ocean/sea
            if isinstance(data, np.ma.masked_array):             
                self.data = data.data.copy() # get only data, not mask
                self.data[data.mask] = np.nan # filled masked values with NaNs
                self.data_mask = data.mask.copy()
            else:
                self.data = data.copy()

            self.data = np.squeeze(self.data)

            for key in d.variables.keys():
                if key == variable_name:
                    continue
                if 'lat' in str(d.variables[key].name):
                    self.lats = d.variables[key][:]
                if 'lon' in str(d.variables[key].name):
                    self.lons = d.variables[key][:]
                    if np.any(self.lons < 0):
                        self._shift_lons_to_360()
                try: # handling when some netCDF variable hasn't assigned units
                    if 'since' in d.variables[key].units:
                        self.time = d.variables[key][:]
                        date_since = self._parse_time_units(d.variables[key].units)
                        if "hours" in d.variables[key].units:
                            self.time = self.time / 24.0 + date.toordinal(date_since)
                        elif "seconds" in d.variables[key].units:
                            self.time = self.time / 86400. + date.toordinal(date_since)
                        elif "days" in d.variables[key].units:
                            self.time += date.toordinal(date_since)
                        elif "months" in d.variables[key].units:
                            from dateutil.relativedelta import relativedelta
                            for t in range(self.time.shape[0]):
                                self.time[t] = date.toordinal(date_since + relativedelta(months = +int(self.time[t])))
                except AttributeError:
                    pass

            self.var_name = variable_name
            if np.any(np.isnan(self.data)):
                self.nans = True
            if print_prog:
                print("Data saved to structure. Shape of the data is %s" % (str(self.data.shape)))
                print("Lats x lons saved to structure. Shape is %s x %s" % (str(self.lats.shape[0]), str(self.lons.shape[0])))
                print("Time stamp saved to structure as ordinal values where Jan 1 of year 1 is 1")
                print("The first data value is from %s and the last is from %s" % (str(self.get_date_from_ndx(0)), str(self.get_date_from_ndx(-1))))
                print("Default temporal sampling in the data is %.2f day(s)" % (np.nanmean(np.diff(self.time))))
                if np.any(np.isnan(self.data)):
                    print("The data contains NaNs! All methods are compatible with NaNs, just to let you know!")
            
            d.close()

        else:
            raise Exception("Unknown or unsupported dataset!")



    def _shift_lons_to_360(self):
        """
        Shifts lons to 0-360 degree east.
        """

        self.lons[self.lons < 0] += 360
        ndx = np.argsort(self.lons)
        self.lons = self.lons[ndx]
        self.data = self.data[..., ndx]



    @staticmethod
    def _parse_time_units(time_string):
        """
        Parses time units from netCDF file, returns date since the record.
        """

        date_split = time_string.split('-')
        y = ("%04d" % int(date_split[0][-4:]))
        m = ("%02d" % int(date_split[1]))
        d = ("%02d" % int(date_split[2][:2]))

        return datetime.strptime("%s-%s-%s" % (y, m, d), '%Y-%m-%d')
            
            
            
    def load_station_data(self, filename, dataset = 'ECA-station', print_prog = True, offset_in_file = 0):
        """
        Loads station data, usually from text file. Uses numpy.loadtxt reader.
        """
        
        if dataset == 'Klem_day':
            raw_data = np.loadtxt(self.data_folder + filename) # first column is continous year and second is actual data
            self.data = np.array(raw_data[:, 1])
            time = []
            
            # use time iterator to go through the dates
            y = int(np.modf(raw_data[0, 0])[1]) 
            if np.modf(raw_data[0, 0])[0] == 0:
                start_date = date(y, 1, 1)
            delta = timedelta(days = 1)
            d = start_date
            while len(time) < raw_data.shape[0]:
                time.append(d.toordinal())
                d += delta
            self.time = np.array(time)
            self.location = 'Praha-Klementinum, Czech Republic'
            print("Station data from %s saved to structure. Shape of the data is %s" % (self.location, str(self.data.shape)))
            print("Time stamp saved to structure as ordinal values where Jan 1 of year 1 is 1")
            
        if dataset == 'ECA-station':
            with open(self.data_folder + filename, 'rb') as f:
                time = []
                data = []
                missing = []
                i = 0 # line-counter
                reader = csv.reader(f)
                for row in reader:
                    i += 1
                    if i == 16 + offset_in_file: # line with location
                        c_list = filter(None, row[1].split(" "))
                        del c_list[-2:]
                        country = ' '.join(c_list).lower()
                        station = ' '.join(row[0].split(" ")[7:]).lower()
                        self.location = station.title() + ', ' + country.title()
                    if i > 20 + offset_in_file: # actual data - len(row) = 5 as STAID, SOUID, DATE, TG, Q_TG
                        staid = int(row[0])
                        value = float(row[3])
                        year = int(row[2][:4])
                        month = int(row[2][4:6])
                        day = int(row[2][6:])
                        time.append(date(year, month, day).toordinal())
                        if value == -9999.:
                            missing.append(date(year, month, day).toordinal())
                            data.append(np.nan)
                        else:
                            data.append(value/10.)
            self.station_id = staid
            self.data = np.array(data)
            self.time = np.array(time)
            self.missing = np.array(missing)
            if print_prog:
                print("Station data from %s saved to structure. Shape of the data is %s" % (self.location, str(self.data.shape)))
                print("Time stamp saved to structure as ordinal values where Jan 1 of year 1 is 1")
            if self.missing.shape[0] != 0 and self.verbose:
                print("** WARNING: There were some missing values! To be precise, %d missing values were found!" % (self.missing.shape[0]))
                  
                  
                  
    def copy_data(self):
        """
        Returns the copy of data.
        """              
        
        return self.data.copy()



    def copy(self, temporal_ndx = None):
        """
        Returns a copy of DataField with data, lats, lons and time fields.
        If temporal_ndx is not None, copies only selected temporal part of data.
        """

        copied = DataField()
        copied.data = self.data.copy()
        copied.time = self.time.copy()
        if temporal_ndx is not None:
            copied.data = copied.data[temporal_ndx]
            copied.time = copied.time[temporal_ndx]

        if self.lats is not None:
            copied.lats = self.lats.copy()
        if self.lons is not None:
            copied.lons = self.lons.copy()
        if self.location is not None:
            copied.location = self.location
        if self.missing is not None:
            copied.missing = self.missing.copy()
        if self.station_id is not None:
            copied.station_id = self.station_id
        if self.station_elev is not None:
            copied.station_elev = self.station_elev
        if self.var_name is not None:
            copied.var_name = self.var_name
        if self.cos_weights is not None:
            copied.cos_weights = self.cos_weights
        if self.data_mask is not None:
            copied.data_mask = self.data_mask
        
        copied.nans = self.nans

        return copied   
                                            
                    
                    
    def select_date(self, date_from, date_to, apply_to_data = True, exclusive = True):
        """
        Selects the date range - date_from is inclusive, date_to is exclusive. Input is date(year, month, day).
        """
        
        d_start = date_from.toordinal()
        d_to = date_to.toordinal()
        
        if exclusive:
            ndx = np.logical_and(self.time >= d_start, self.time < d_to)
        else:
            ndx = np.logical_and(self.time >= d_start, self.time <= d_to)
        if apply_to_data:
            self.time = self.time[ndx] # slice time stamp
            self.data = self.data[ndx, ...] # slice data
            if self.data_mask is not None and self.data_mask.ndim > 2:
                self.data_mask = self.data_mask[ndx, ...] # slice missing if exists
        if self.missing is not None:
            missing_ndx = np.logical_and(self.missing >= d_start, self.missing < d_to)
            self.missing = self.missing[missing_ndx] # slice missing if exists
            
        return ndx



    def get_sliding_window_indexes(self, window_length, window_shift, unit = 'm', return_half_dates = False):
        """
        Returns list of indices for sliding window analysis.
        If return_half_dates is True, also returns dates in the middle of the interval for reference.
        """

        from dateutil.relativedelta import relativedelta

        if unit == 'm':
            length = relativedelta(months = +window_length)
            shift = relativedelta(months = +window_shift)
        elif unit == 'd':
            length = relativedelta(days = +window_length)
            shift = relativedelta(days = +window_shift)
        elif unit == 'y':
            length = relativedelta(years = +window_length)
            shift = relativedelta(years = +window_shift)
        else:
            raise Exception("Unknown time unit! Please, use one of the 'd', 'm', 'y'!")

        ndxs = []
        if return_half_dates:
            half_dates = []
        window_start = self.get_date_from_ndx(0)
        window_end = window_start + length
        while window_end <= self.get_date_from_ndx(-1):
            ndx = self.select_date(window_start, window_end, apply_to_data = False)
            ndxs.append(ndx)
            if return_half_dates:
                half_dates.append(window_start + (window_end - window_start) / 2)
            window_start += shift
            window_end = window_start + length

        # add last
        ndxs.append(self.select_date(window_start, window_end, apply_to_data = False))
        if return_half_dates:
            half_dates.append(window_start + (self.get_date_from_ndx(-1) - window_start) / 2)

        if np.sum(ndxs[-1]) != np.sum(ndxs[-2]) and self.verbose:
            print("**WARNING: last sliding window is shorter than others! (%d vs. %d in others)" 
                % (np.sum(ndxs[-1]), np.sum(ndxs[-2])))

        if return_half_dates:
            return ndxs, half_dates
        else:
            return ndxs



    def create_time_array(self, date_from, sampling = 'm'):
        """
        Creates time array for already saved data in 'self.data'.
        From date_from to date_from + data length. date_from is inclusive.
        Sampling: 
            'm' for monthly, could be just 'm' or '3m' as three-monthly
            'd' for daily
            'xh' where x = {1, 6, 12} for sub-daily.
        """

        if 'm' in sampling:
            if 'm' != sampling:
                n_months = int(sampling[:-1])
                timedelta = relativedelta(months = +n_months)
            elif 'm' == sampling:
                timedelta = relativedelta(months = +1)
        elif sampling == 'd':
            timedelta = relativedelta(days = +1)
        elif sampling in ['1h', '6h', '12h']:
            hourly_data = int(sampling[:-1])
            timedelta = relativedelta(hours = +hourly_data)
        elif sampling == 'y':
            timedelta = relativedelta(years = +1)
        else:
            raise Exception("Unknown sampling.")

        d_now = date_from
        self.time = np.zeros((self.data.shape[0],))
        for t in range(self.data.shape[0]):
            self.time[t] = d_now.toordinal()
            d_now += timedelta
        
    
    
    def get_date_from_ndx(self, ndx):
        """
        Returns the date of the variable from given index.
        """
        
        return date.fromordinal(np.int(self.time[ndx]))
        
        
        
    def get_spatial_dims(self):
        """
        Returns the spatial dimensions of the data as list.
        """
        
        return list(self.data.shape[-2:])
        
    
        
    def find_date_ndx(self, date):
        """
        Returns index which corresponds to the date. Returns None if the date is not contained in the data.
        """
        
        d = date.toordinal()
        pos = np.nonzero(self.time == d)
        if not np.all(np.isnan(pos)):
            return int(pos[0])
        else:
            return None



    def get_closest_lat_lon(self, lat, lon):
        """
        Returns closest lat, lon index in the data.
        """

        return [np.abs(self.lats - lat).argmin(), np.abs(self.lons - lon).argmin()]
            
            
            
    def select_months(self, months, apply_to_data = True):
        """
        Subselects only certain months. Input as a list of months number.
        """
        
        ndx = filter(lambda i: date.fromordinal(int(self.time[i])).month in months, range(len(self.time)))
        
        if apply_to_data:
            self.time = self.time[ndx]
            self.data = self.data[ndx, ...]
        
        return ndx
        
        
        
    def select_lat_lon(self, lats, lons, apply_to_data = True):
        """
        Selects region in lat/lon. Input is for both [from, to], both are inclusive. If None, the dimension is not modified.
        """
        
        if self.lats is not None and self.lons is not None:
            if lats is not None:
                lat_ndx = np.nonzero(np.logical_and(self.lats >= lats[0], self.lats <= lats[1]))[0]
            else:
                lat_ndx = np.arange(len(self.lats))
                
            if lons is not None:
                if lons[0] < lons[1]:
                    lon_ndx = np.nonzero(np.logical_and(self.lons >= lons[0], self.lons <= lons[1]))[0]
                elif lons[0] > lons[1]:
                    l1 = list(np.nonzero(np.logical_and(self.lons >= lons[0], self.lons <= 360))[0])
                    l2 = list(np.nonzero(np.logical_and(self.lons >= 0, self.lons <= lons[1]))[0])
                    lon_ndx = np.array(l1 + l2)
            else:
                lon_ndx = np.arange(len(self.lons))
            
            if apply_to_data:
                if self.data.ndim >= 3:
                    d = self.data.copy()
                    d = d[..., lat_ndx, :]
                    self.data = d[..., lon_ndx].copy()
                    self.lats = self.lats[lat_ndx]
                    self.lons = self.lons[lon_ndx]
                    if self.data_mask is not None:
                        d = self.data_mask
                        d = d[..., lat_ndx, :]
                        self.data_mask = d[..., lon_ndx]
                elif self.data.ndim == 2: # multiple stations data
                    d = self.data.copy()
                    d = d[:, lat_ndx]
                    self.lons = self.lons[lat_ndx]
                    self.lats = self.lats[lat_ndx]
                    if lons is not None:
                        if lons[0] < lons[1]:
                            lon_ndx = np.nonzero(np.logical_and(self.lons >= lons[0], self.lons <= lons[1]))[0]
                        elif lons[0] > lons[1]:
                            l1 = list(np.nonzero(np.logical_and(self.lons >= lons[0], self.lons <= 360))[0])
                            l2 = list(np.nonzero(np.logical_and(self.lons >= 0, self.lons <= lons[1]))[0])
                            lon_ndx = np.array(l1 + l2)
                    else:
                        lon_ndx = np.arange(len(self.lons))
                    self.data = d[:, lon_ndx].copy()
                    self.lons = self.lons[lon_ndx]
                    self.lats = self.lats[lon_ndx]

                if np.any(np.isnan(self.data)):
                    self.nans = True
                else:
                    self.nans = False
            
            return lat_ndx, lon_ndx

        else:
            raise Exception('Slicing data with no spatial dimensions, probably station data.')



    def cut_lat_lon(self, lats_to_cut, lons_to_cut):
        """
        Cuts region in lats/lons (puts NaNs in the selected regions). 
        Input is for both [from, to], both are inclusive. If None, the dimension is not modified.
        """

        if self.lats is not None and self.lons is not None:
            if lats_to_cut is not None:
                lat_ndx = np.nonzero(np.logical_and(self.lats >= lats_to_cut[0], self.lats <= lats_to_cut[1]))[0]
                if lons_to_cut is None:
                    self.data[..., lat_ndx, :] = np.nan
                
            if lons_to_cut is not None:
                if lons_to_cut[0] < lons_to_cut[1]:
                    lon_ndx = np.nonzero(np.logical_and(self.lons >= lons_to_cut[0], self.lons <= lons_to_cut[1]))[0]
                elif lons_to_cut[0] > lons_to_cut[1]:
                    l1 = list(np.nonzero(np.logical_and(self.lons >= lons_to_cut[0], self.lons <= 360))[0])
                    l2 = list(np.nonzero(np.logical_and(self.lons_to_cut >= 0, self.lons <= lons_to_cut[1]))[0])
                    lon_ndx = np.array(l1 + l2)
                if lats_to_cut is None:
                    self.data[..., lon_ndx] = np.nan   

            if lats_to_cut is not None and lons_to_cut is not None:
                
                for lat in lat_ndx:
                    for lon in lon_ndx: 
                        self.data[..., lat, lon] = np.nan

        else:
            raise Exception('Slicing data with no spatial dimensions, probably station data.')
            
            
            
    def select_level(self, level):
        """
        Selects the proper level from the data. Input should be integer >= 0.
        """
        
        if self.data.ndim > 3:
            self.data = self.data[:, level, ...]
        else:
            raise Exception('Slicing level in single-level data.')
        
        
        
    def extract_day_month_year(self):
        """
        Extracts the self.time field into three fields containg days, months and years.
        """
        
        n_days = len(self.time)
        days = np.zeros((n_days,), dtype = np.int)
        months = np.zeros((n_days,), dtype = np.int)
        years = np.zeros((n_days,), dtype = np.int)
        
        for i,d in zip(range(n_days), self.time):
            dt = date.fromordinal(int(d))
            days[i] = dt.day
            months[i] = dt.month
            years[i] = dt.year
            
        return days, months, years



    def latitude_cos_weights(self):
        """
        Returns a grid with scaling weights based on cosine of latitude.
        """
        
        if (np.all(self.cos_weights) is not None) and (self.cos_weights.shape == self.get_spatial_dims()):
            return self.cos_weights

        cos_weights = np.zeros(self.get_spatial_dims())
        for ndx in range(self.lats.shape[0]):
            cos_weights[ndx, :] = np.cos(self.lats[ndx] * np.pi/180.) ** 0.5

        self.cos_weights = cos_weights
        return cos_weights

        
        
    def missing_day_month_year(self):
        """
        Extracts the self.missing field (if exists and is non-empty) into three fields containing days, months and years.
        """
        
        if (self.missing is not None) and (self.missing.shape[0] != 0):
            n_days = len(self.missing)
            days = np.zeros((n_days,), dtype = np.int)
            months = np.zeros((n_days,), dtype = np.int)
            years = np.zeros((n_days,), dtype = np.int)
            
            for i,d in zip(range(n_days), self.missing):
                dt = date.fromordinal(int(d))
                days[i] = dt.day
                months[i] = dt.month
                years[i] = dt.year
                
            return days, months, years
            
        else:
            raise Exception('Luckily for you, there is no missing values!')
            

            
    def flatten_field(self, f = None):
        """
        Reshape the field to 2dimensions such that axis 0 is temporal and axis 1 is spatial.
        If f is None, reshape the self.data field, else reshape the f field.
        Should only be used with single-level data.
        """        

        if f is None:
            if self.data.ndim == 3:
                self.data = np.reshape(self.data, (self.data.shape[0], np.prod(self.data.shape[1:])))
            else:
                raise Exception('Data field is already flattened, multi-level or only temporal (e.g. station)!')

        elif f is not None:
            if f.ndim == 3:
                f = np.reshape(f, (f.shape[0], np.prod(f.shape[1:])))

                return f
            else:
                raise Exception('The field f is already flattened, multi-level or only temporal (e.g. station)!')



    def reshape_flat_field(self, f = None):
        """
        Reshape flattened field to original time x lat x lon shape.
        If f is None, reshape the self.data field, else reshape the f field.
        Supposes single-level data.
        """

        if f is None:
            if self.data.ndim == 2:
                new_shape = [self.data.shape[0]] + list((self.lats.shape[0], self.lons.shape[0]))
                self.data = np.reshape(self.data, new_shape)
            else:
                raise Exception('Data field is not flattened, is multi-level or is only temporal (e.g. station)!')

        elif f is not None:
            if f.ndim == 2:
                new_shape = [f.shape[0]] + list((self.lats.shape[0], self.lons.shape[0]))
                f = np.reshape(f, new_shape)

                return f
            else:
                raise Exception('The field f is not flattened, is multi-level or is only temporal (e.g. station)!')
                
                
                
    def get_data_of_precise_length(self, length = '16k', start_date = None, end_date = None, apply_to_data = False):
        """
        Selects the data such that the length of the time series is exactly length.
        If apply_to_data is True, it will replace the data and time, if False it will return them.
        If end_date is defined, it is exclusive.
        """
        
        if isinstance(length, int):
            ln = length
        elif 'k' in length:
            order = int(length[:-1])
            pow2list = np.array([np.power(2,n) for n in range(10,22)])
            ln = pow2list[np.where(order == pow2list/1000)[0][0]]
        else:
            raise Exception('Could not understand the length! Please type length as integer or as string like "16k".')
        
        if start_date is not None and self.find_date_ndx(start_date) is None:
            start_date = self.get_date_from_ndx(0)
        if end_date is not None and self.find_date_ndx(end_date) is None:
            end_date = self.get_date_from_ndx(-1)
        
        if end_date is None and start_date is not None:
            # from start date until length
            idx = self.find_date_ndx(start_date)
            data_temp = self.data[idx : idx + ln, ...].copy()
            time_temp = self.time[idx : idx + ln, ...].copy()
            idx_tuple = (idx, idx+ln)
            
        elif start_date is None and end_date is not None:
            idx = self.find_date_ndx(end_date)
            data_temp = self.data[idx - ln + 1 : idx + 1, ...].copy()
            time_temp = self.time[idx - ln + 1 : idx + 1, ...].copy()
            idx_tuple = (idx - ln, idx)
            
        else:
            raise Exception('You messed start / end date selection! Pick only one!')
            
        if apply_to_data:
            self.data = data_temp.copy()
            self.time = time_temp.copy()
            return idx_tuple
            
        else:
            return data_temp, time_temp, idx_tuple



    def _shift_index_by_month(self, current_idx):
        """
        Returns the index in data shifted by month.
        """
        
        dt = date.fromordinal(np.int(self.time[current_idx]))
        if dt.month < 12:
            mi = dt.month + 1
            y = dt.year
        else:
            mi = 1
            y = dt.year + 1
            
        return self.find_date_ndx(date(y, mi, dt.day))



    def get_annual_data(self, means = True, ts = None):
        """
        Converts the data to annual means or sums.
        If ts is None, uses self.data.
        if means is True, computes annual means, otherwise computes sums.
        """

        yearly_data = []
        yearly_time = []

        day, mon, year = self.extract_day_month_year()

        for y in range(year[0], year[-1]+1, 1):
            year_ndx = np.where(year == y)[0]
            if ts is None:
                if means:
                    yearly_data.append(np.squeeze(np.nanmean(self.data[year_ndx, ...], axis = 0)))
                else:
                    yearly_data.append(np.squeeze(np.nansum(self.data[year_ndx, ...], axis = 0)))
            else:
                if means:
                    yearly_data.append(np.squeeze(np.nanmean(ts[year_ndx, ...], axis = 0)))
                else:
                    yearly_data.append(np.squeeze(np.nansum(ts[year_ndx, ...], axis = 0)))
            yearly_time.append(date(y, 1, 1).toordinal())

        if ts is None:
            self.data = np.array(yearly_data)
            self.time = np.array(yearly_time) 
        else:
            return np.array(yearly_data)
        
            
            
    def get_monthly_data(self, means = True):
        """
        Converts the daily data to monthly means or sums.
        """
        
        delta = self.time[1] - self.time[0]
        if delta == 1:
            # daily data
            day, mon, year = self.extract_day_month_year()
            monthly_data = []
            monthly_time = []
            # if first day of the data is not the first day of month - shift month
            # by one to start with the full month
            if day[0] != 1:
                mi = mon[0]+1 if mon[0] < 12 else 1
                y = year[0] if mon[0] < 12 else year[0] + 1
            else:
                mi = mon[0]
                y = year[0]
            start_idx = self.find_date_ndx(date(y, mi, 1))
            end_idx = self._shift_index_by_month(start_idx)
            while end_idx <= self.data.shape[0] and end_idx is not None:
                if means:
                    monthly_data.append(np.nanmean(self.data[start_idx : end_idx, ...], axis = 0))
                else:
                    monthly_data.append(np.nansum(self.data[start_idx : end_idx, ...], axis = 0))
                monthly_time.append(self.time[start_idx])
                start_idx = end_idx
                end_idx = self._shift_index_by_month(start_idx)
                if end_idx is None: # last piece, then exit the loop
                    if means:
                        monthly_data.append(np.nanmean(self.data[start_idx : , ...], axis = 0))
                    else:
                        monthly_data.append(np.nansum(self.data[start_idx : , ...], axis = 0))
                    monthly_time.append(self.time[start_idx])
            self.data = np.array(monthly_data)
            self.time = np.array(monthly_time)                
        elif abs(delta - 30) < 3.0:
            # monhtly data
            print('The data are already monthly values. Nothing happend.')
        else:
            raise Exception('Unknown temporal sampling in the field.')
            
            
        
    def average_to_daily(self):
        """
        Averages the sub-daily values (e.g. ERA-40 basic sampling is 6 hours) into daily.
        """        
        
        delta = self.time[1] - self.time[0]
        if delta < 1:
            n_times = int(1 / delta)
            d = np.zeros_like(self.data)
            d = np.delete(d, slice(0, (n_times-1) * d.shape[0]/n_times), axis = 0)
            t = np.zeros(self.time.shape[0] / n_times)
            for i in range(d.shape[0]):
                d[i, ...] = np.nanmean(self.data[n_times*i : n_times*i+(n_times-1), ...], axis = 0)
                t[i] = self.time[n_times*i]
                
            self.data = d
            self.time = t.astype(np.int)
        
        else:
            raise Exception('No sub-daily values, you can average to daily only values with finer time sampling.')



    @staticmethod
    def _interp_temporal(a):
        """
        Helper function for temporal interpolation
        """

        import scipy.interpolate as si

        i, j, old_time, data, new_time, kind = a
        f = si.interp1d(old_time, data, kind = kind)
        new_data = f(new_time)

        return i, j, new_data



    def interpolate_to_finer_temporal_resolution(self, to_resolution = 'm', kind = 'linear', use_to_data = False,
                                                    pool = None):
        """
        Interpolates data to finer temporal resolution, e.g. yearly to monthly.
        Uses scipy's interp1d, for 'kind' keyword see the scipy's documentation.
        If use_to_data is True, rewrites data in the class, else returns data.
        """


        if self.data.ndim > 2:
            num_lats = self.lats.shape[0]
            num_lons = self.lons.shape[0]
        elif self.data.ndim == 2: # lot of station data
            num_lats = self.lats.shape[0]
            num_lons = 1
            self.data = self.data[:, :, np.newaxis]
        else:
            num_lats = 1
            num_lons = 1
            self.data = self.data[:, np.newaxis, np.newaxis]

        if 'm' in to_resolution:
            if 'm' != to_resolution:
                n_months = int(to_resolution[:-1])
                timedelta = relativedelta(months = +n_months)
            elif 'm' == to_resolution:
                timedelta = relativedelta(months = +1)
        elif to_resolution == 'd':
            timedelta = relativedelta(days = +1)
        elif to_resolution in ['1h', '6h', '12h']:
            hourly_data = int(to_resolution[:-1])
            timedelta = relativedelta(hours = +hourly_data)
        elif to_resolution == 'y':
            timedelta = relativedelta(years = +1)
        else:
            raise Exception("Unknown to_resolution.")
        
        new_time = []
        first_date = self.get_date_from_ndx(0)
        last_day = self.get_date_from_ndx(-1)
        current_date = first_date
        while current_date <= last_day:
            new_time.append(current_date.toordinal())
            current_date += timedelta
        new_time = np.array(new_time)

        job_args = [ (i, j, self.time, self.data[:, i, j], new_time, kind) for i in range(num_lats) for j in range(num_lons) ]

        interp_data = np.zeros([new_time.shape[0]] + list(self.get_spatial_dims()))
        
        if pool is None:
            job_result = map(self._interp_temporal, job_args)
        elif pool is not None:
            job_result = pool.map(self._interp_temporal, job_args)
        del job_args
        
        for i, j, res in job_result:
            interp_data[:, i, j] = res

        interp_data = np.squeeze(interp_data)
        self.data = np.squeeze(self.data)
        if use_to_data:
            self.time = new_time.copy()
            self.data = interp_data.copy()
        else:
            return interp_data, new_time



    def _ascending_descending_lat_lons(self, lats = True, lons = False, direction = 'asc'):
        """
        Transforms the data (and lats and lons) so that they have strictly ascending (direction = 'asc')
        or descending (direction = 'des') order. (Needed for interpolation).
        Returns True if manipulation took place.
        """

        lat_flg, lon_flg = False, False
        if np.all(np.diff(self.lats) < 0) and lats and direction == 'asc':
            self.lats = self.lats[::-1]
            self.data = self.data[..., ::-1, :]
            lat_flg = True
        elif np.all(np.diff(self.lats) > 0) and lats and direction == 'des':
            self.lats = self.lats[::-1]
            self.data = self.data[..., ::-1, :]
            lat_flg = True

        if np.all(np.diff(self.lons) < 0) and lons and direction == 'asc':
            self.lons = self.lons[::-1]
            self.data = self.data[..., ::-1]
            lon_flg = True
        elif np.all(np.diff(self.lons) > 0) and lons and direction == 'des':
            self.lons = self.lons[::-1]
            self.data = self.data[..., ::-1]
            lon_flg = True

        return lat_flg, lon_flg



    def subsample_spatial(self, lat_to, lon_to, start, average = False):
        """
        Subsamples the data in the spatial sense to grid "lat_to" x "lon_to" in degress.
        Start is starting point for subsampling in degrees as [lat, lon]
        If average is True, the subsampling is due to averaging the data -- using SciPy's spline
        interpolation on the rectangle. The interpolation is done for each time step and level 
        independently.
        If average is False, the subsampling is just subsampling certain values.
        """

        if self.lats is not None and self.lons is not None:
            delta_lats = np.abs(self.lats[1] - self.lats[0])
            delta_lons = np.abs(self.lons[1] - self.lons[0])
            if lat_to % delta_lats == 0 and lon_to % delta_lons == 0:
                lat_ndx = int(lat_to // delta_lats)
                lon_ndx = int(lon_to // delta_lons)

                lat_flg, lon_flg = self._ascending_descending_lat_lons(lats = True, lons = True, direction = 'asc')

                start_lat_ndx = np.where(self.lats == start[0])[0]
                start_lon_ndx = np.where(self.lons == start[1])[0]
                if start_lon_ndx.size == 1 and start_lat_ndx.size == 1:
                    start_lat_ndx = start_lat_ndx[0]
                    start_lon_ndx = start_lon_ndx[0]

                    if not average:
                        self.lats = self.lats[start_lat_ndx::lat_ndx]
                        self.lons = self.lons[start_lon_ndx::lon_ndx]
                        d = self.data
                        d = d[..., start_lat_ndx::lat_ndx, :]
                        self.data = d[..., start_lon_ndx::lon_ndx]

                    else:

                        nan_flag = False
                        if self.nans:
                            if self.check_NaNs_only_spatial():
                                # for interpolation purposes, fill NaNs with 0.
                                msk = np.isnan(self.data)
                                self.data[msk] = 0.
                                msk = msk[0, ...]
                                nan_flag = True
                            else:
                                raise Exception("NaNs in the data are not only spatial, cannot interpolate!")

                        from scipy.interpolate import RectBivariateSpline
                        # if data is single-level - create additional dummy dimension
                        if self.data.ndim == 3:
                            self.data = self.data[:, np.newaxis, :, :]

                        # fields for new lats / lons
                        new_lats = np.arange(start[0], self.lats[-1]+lat_to, lat_to)
                        new_lons = np.arange(start[1], self.lons[-1], lon_to)
                        d = np.zeros((list(self.data.shape[:2]) + [new_lats.shape[0], new_lons.shape[0]]))
                        # interpolate using Bivariate spline
                        for t in range(self.time.shape[0]):
                            for lvl in range(self.data.shape[1]):
                                int_scheme = RectBivariateSpline(self.lats, self.lons, self.data[t, lvl, ...])
                                d[t, lvl, ...] = int_scheme(new_lats, new_lons)
                        
                        if nan_flag:
                            # subsample mask to new grid
                            msk_temp = msk[start_lat_ndx::lat_ndx, :]
                            msk = msk_temp[..., start_lon_ndx::lon_ndx]
                            # return back NaNs
                            for t in range(self.time.shape[0]):
                                for lvl in range(self.data.shape[1]):
                                    d[t, lvl, msk] = np.nan

                        self.lats = new_lats
                        self.lons = new_lons
                        self.data = np.squeeze(d)

                    if np.any(np.isnan(self.data)):
                        self.nans = True
                    else:
                        self.nans = False

                else:
                    raise Exception("Start lat and / or lon for subsampling does not exist in the data!")

                self._ascending_descending_lat_lons(lats = lat_flg, lons = lon_flg, direction = 'des')

            else:
                raise Exception("Subsampling lats only to multiples of %.2f and lons of %.2f" % (delta_lats, delta_lons))

        else:
            raise Exception("Cannot subsample station data, or data from one grid point!")



    def smoothing_running_avg(self, points, cut_edges = False, use_to_data = False, ts = None):
        """
        Smoothing of time series using running average over points.
        If use_to_data is False, returns the data, otherwise rewrites the data in class.
        """

        if ts is None:
            ts = self.data.copy()

        if cut_edges:
            d = np.zeros(([ts.shape[0] - points + 1] + list(ts.shape[1:])))
        else:
            d = np.zeros_like(ts)
            window = points//2
        
        for i in range(d.shape[0]):
            if cut_edges:
                d[i, ...] = np.nanmean(ts[i : i+points, ...], axis = 0)
            else:
                d[i, ...] = np.nanmean(ts[max(i-window,1) : min(i+window,d.shape[0]), ...], axis = 0)

        if use_to_data and ts is None:
            self.data = d.copy()
            if cut_edges:
                if points % 2 == 1:
                # time slicing when points is odd -- cut points//2 from the beginning and from the end
                    self.time = self.time[points//2 : -points//2 + 1]
                else:
                # time slicing when points is even -- not sure where to cut
                    pass
        else:
            return d



    def plot_FFT_spectrum(self, ts = None, log = True, vlines = np.arange(1,11), fname = None):
        """
        Estimates power spectrum using Welch method.
        if ts is None, plots spectrum of the data.
        ts should have same sampling frequency as data!
        y axis is log by default, if log is True, also x axis is log.
        """

        import matplotlib.pyplot as plt

        delta = self.time[1] - self.time[0]
        if delta == 1:
            # daily time series
            fs = 1./86400 # Hz
        elif abs(delta - 30) < 3.0:
            # monthly time series
            fs = 1./2.628e+6
        elif abs(delta - 365) < 2.0:
            # yearly time series
            fs = 1./3.154e+7

        plt.figure(figsize = (15,7))
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)
        ts = ts if ts is not None else self.data.copy()
        if isinstance(ts, list):
            ts = np.array(ts).T
        if ts.ndim > 2:
            ts = ts.reshape([ts.shape[0], np.prod(ts.shape[1:])])
        fft = np.abs(np.fft.rfft(ts, axis = 0))
        freqs = np.fft.rfftfreq(ts.shape[0], d = 1./fs)
        freqs *= 3.154e+7
        if log:
            plt.semilogx(freqs, 20*np.log10(fft), linewidth = 0.8) # in dB hopefully...
            plt.xlabel('FREQUENCY [log 1/year]', size = 25)
        else:
            plt.plot(freqs, 20*np.log10(fft), linewidth = 0.8)
            plt.xlabel('FREQUENCY [1/year]', size = 25)
        for vline in vlines:
            plt.axvline(1./vline, 0, 1, linestyle = ':',linewidth = 0.6, color = "#333333")
        plt.xlim([freqs[0], freqs[-1]])
        plt.ylabel('FFT SPECTRUM [dB]', size = 25)
        if fname is None:
            plt.show()
        else:
            plt.savefig(fname, bbox_inches = 'tight')



    def temporal_filter(self, cutoff, btype, ftype = 'butter', order = 2, cut = 1, pool = None, cut_time = False,
        rp = None, rs = None, cut_data = False):
        """
        Filters data in temporal sense.
        Uses Butterworth filter of order order.
        btype:
            lowpass
            highpass
            bandpass
            bandstop
        cutoff:
            for low/high pass one frequency in months
            for band* list of frequencies in months
        ftype:
            butter - for Butterworth filter
            cheby1 - for Chebyshev type I filter
            cheby2 - for Chebyshev type II filter
            ellip - for Cauer/elliptic filter
            bessel - for Bessel/Thomson filter
        cut in years
        """

        from scipy.signal import iirfilter

        delta = self.time[1] - self.time[0]
        if delta == 1:
            # daily time series
            fs = 1./86400 # Hz
            y = 365.25
        elif abs(delta - 30) < 3.0:
            # monthly time series
            fs = 1./2.628e+6 # Hz
            y = 12

        nyq = 0.5 * fs # Nyquist frequency
        if 'cheby' in ftype or 'ellip' == ftype:
            rp = rp if rp is not None else 60

        if type(cutoff) == list and btype in ['bandpass', 'bandstop']:
            low = cutoff[0] if cutoff[0] > cutoff[1] else cutoff[1]
            high = cutoff[1] if cutoff[0] > cutoff[1] else cutoff[0]
            low = 1./(low*2.628e+6) # in months
            high = 1./(high*2.628e+6)
            # get coefficients
            b, a = iirfilter(order, [low/nyq, high/nyq], rp = rp, rs = rs, btype = btype, analog = False, ftype = ftype)
        elif btype in ['lowpass', 'highpass']:
            cutoff = 1./(cutoff*2.628e+6)
            b, a = iirfilter(order, cutoff/nyq, rp = rp, rs = rs, btype = btype, analog = False, ftype = ftype)
        else:
            raise Exception("For band filter cutoff must be a list of [low,high] for low/high-pass cutoff must be a integer!")

        if pool is None:
            map_func = map
        elif pool is not None:
            map_func = pool.map

        if self.data.ndim > 1:
            num_lats = self.lats.shape[0]
            num_lons = self.lons.shape[0]
        else:
            num_lats = 1
            num_lons = 1
            self.data = self.data[:, np.newaxis, np.newaxis]

        self.filtered_data = np.zeros_like(self.data)

        job_args = [ (i, j, self.data[:, i, j], b, a) for i in range(num_lats) for j in range(num_lons) ]
        job_result = map_func(self._get_filtered_data, job_args)
        del job_args
        for i, j, res in job_result:
            self.filtered_data[:, i, j] = res

        del job_result

        if cut is not None:
            to_cut = int(y*cut)
            if cut_time:
                self.time = self.time[to_cut:-to_cut]
            if cut_data:
                self.data = self.data[to_cut:-to_cut]

        self.data = np.squeeze(self.data)
        self.filtered_data = np.squeeze(self.filtered_data) if cut is None else np.squeeze(self.filtered_data[to_cut:-to_cut, ...])
        


    def spatial_filter(self, filter_weights = [1, 2, 1], use_to_data = False):
        """
        Filters the data in spatial sense with weights filter_weights.
        If use_to_data is False, returns the data, otherwise rewrites the data in class.
        """

        if self.data.ndim == 3: 
            self.data = self.data[:, np.newaxis, :, :]

        mask = np.zeros(self.data.shape[-2:])
        filt = np.outer(filter_weights, filter_weights)

        mask[:filt.shape[0], :filt.shape[1]] = filt

        d = np.zeros((list(self.data.shape[:-2]) + [self.lats.shape[0] - len(filter_weights) + 1, self.lons.shape[0] - len(filter_weights) + 1]))

        for i in range(d.shape[-2]):
            for j in range(d.shape[-1]):
                avg_mask = np.array([[mask for kk in range(d.shape[1])] for ll in range(d.shape[0])])
                d[:, :, i, j] = np.average(self.data, axis = (2, 3), weights = avg_mask)
                mask = np.roll(mask, 1, axis = 1)
            # return mask to correct y position
            mask = np.roll(mask, len(filter_weights)-1, axis = 1)
            mask = np.roll(mask, 1, axis = 0)

        if use_to_data:
            self.data = np.squeeze(d).copy()
            # space slicing when length of filter is odd -- cut length//2 from the beginning and from the end
            if len(filter_weights) % 2 == 1:
                self.lats = self.lats[len(filter_weights)//2 : -len(filter_weights)//2 + 1]
                self.lons = self.lons[len(filter_weights)//2 : -len(filter_weights)//2 + 1]
            else:
            # space slicing when length of filter is even -- not sure where to cut
                pass
        else:
            return np.squeeze(d)


    @staticmethod
    def _interp_spatial(a):
        """
        Helper function for spatial interpolation.
        """

        import scipy.interpolate as si

        t, d, points, msk, grid_lat, grid_lon, method = a
        new_data = si.griddata(points, d[~msk], (grid_lat, grid_lon), method = method)

        return t, new_data


    def interpolate_spatial_nans(self, method = 'cubic', apply_to_data = True, pool = None):
        """
        Interpolates data with spatial NaNs in them.
        Method is one of the following:
          nearest, linear, cubic
        If apply to data, interpolation is done in-place, if False, data field is returned.
        Uses scipy's griddata.
        """

        if self.nans:
            if self.check_NaNs_only_spatial():
                import scipy.interpolate as si

                if self.data.ndim < 4:
                    self.data = self.data[:, np.newaxis, ...]

                new_data = np.zeros_like(self.data)
                for lvl in range(self.data.shape[1]):
                    msk = np.isnan(self.data[0, lvl, ...]) # nan mask
                    grid_lat, grid_lon = np.meshgrid(self.lats, self.lons, indexing = 'ij') # final grids
                    points = np.zeros((grid_lat[~msk].shape[0], 2))
                    points[:, 0] = grid_lat[~msk]
                    points[:, 1] = grid_lon[~msk]
                    args = [(t, self.data[t, lvl, ...], points, msk, grid_lat, grid_lon, method) for t in range(self.time.shape[0])]
                    if pool is None:
                        job_res = map(self._interp_spatial, args)
                    else:
                        job_res = pool.map(self._interp_spatial, args)
                    
                    for t, i_data in job_res:
                        new_data[t, lvl, ...] = i_data

                new_data = np.squeeze(new_data)

                if apply_to_data:
                    self.data = new_data.copy()
                else:
                    self.data = np.squeeze(self.data)
                    return new_data
            else:
                raise Exception("NaNs are also temporal, no way to filter them out!")
        else:
            print("No NaNs in the data, nothing happened!")



    def check_NaNs_only_spatial(self, field = None):
        """
        Returns True if the NaNs contained in the data are of spatial nature, e.g.
        masked land from sea dataset and so on.
        returns False if also there are some NaNs in the temporal sense.
        E.g. with spatial NaNs, the PCA could be still done, when filtering out the NaNs.
        """

        if self.nans or field is not None:
            field = self.data.copy() if field is None else field
            cnt = 0
            nangrid0 = np.isnan(field[0, ...])
            for t in range(1, field.shape[0]):
                if np.all(nangrid0 == np.isnan(field[t, ...])):
                    cnt += 1

            if field.shape[0] - cnt == 1:
                return True
            else:
                return False

        else:
            pass
            # print("No NaNs in the data, nothing happened!")



    def filter_out_NaNs(self, field = None):
        """
        Returns flattened version of 3D data field without NaNs (e.g. for computational purposes).
        The data is just returned, self.data is still full 3D version. Returned data has first axis
        temporal and second combined spatial.
        Mask is saved for internal purposes (e.g. PCA) but also returned.
        """

        if (field is None and self.nans) or (field is not None and np.any(np.isnan(field))):
            if self.check_NaNs_only_spatial(field = field):
                d = self.data.copy() if field is None else field
                d = self.flatten_field(f = d)
                mask = np.isnan(d)
                spatial_mask = mask[0, :]
                d_out_shape = (d.shape[0], d.shape[1] - np.sum(spatial_mask))
                d_out = d[~mask].reshape(d_out_shape)
                self.spatial_mask = spatial_mask

                return d_out, spatial_mask

            else:
                raise Exception("NaNs are also temporal, no way to filter them out!")

        else:
            print("No NaNs in the data, nothing happened!")



    def return_NaNs_to_data(self, field, mask = None):
        """
        Returns NaNs to the data and reshapes it to the original shape.
        Field has first axis temporal and second combined spatial.
        """

        if self.nans:
            if mask is not None or self.spatial_mask is not None:
                mask = mask if mask is not None else self.spatial_mask
                d_out = np.zeros((field.shape[0], mask.shape[0]))
                ndx = np.where(mask == False)[0]
                d_out[:, ndx] = field
                d_out[:, mask] = np.nan

                return self.reshape_flat_field(f = d_out)

            else:
                raise Exception("No mask given!")
        
        else:
            print("No NaNs in the data, nothing happened!")



    def pca_components(self, n_comps, field = None):
        """
        Estimate the PCA (EOF) components of geo-data.
        Shoud be used on single-level data.
        Returns eofs as (n_comps x lats x lons), pcs as (n_comps x time) and var as (n_comps)
        """

        if self.data.ndim == 3:
            from scipy.linalg import svd

            # reshape field so the first axis is temporal and second is combined spatial
            # if nans, filter-out
            if (self.nans and field is None) or (field is not None and np.any(np.isnan(field))):
                d = self.filter_out_NaNs(field)[0]
            else:
                if field is None:
                    d = self.data.copy()
                else:
                    d = field.copy()
                d = self.flatten_field(f = d)

            # remove mean of each time series
            pca_mean = np.mean(d, axis = 0)
            if field is None:
                self.pca_mean = pca_mean
            d -= pca_mean  

            U, s, V = svd(d, False, True, True)
            exp_var = (s ** 2) / (self.time.shape[0] - 1)
            exp_var /= np.sum(exp_var)
            eofs = V[:n_comps]
            pcs = U[:, :n_comps]
            var = exp_var[:n_comps]
            pcs *= s[:n_comps]

            if self.nans:
                eofs = self.return_NaNs_to_data(field = eofs)
            else:
                eofs = self.reshape_flat_field(f = eofs)

            if field is not None:
                return eofs, pcs.T, var, pca_mean
            elif field is None:
                return eofs, pcs.T, var

        else:
            raise Exception("PCA analysis cannot be used on multi-level data or only temporal (e.g. station) data!")



    def invert_pca(self, eofs, pcs, pca_mean = None):
        """
        Inverts the PCA and returns the original data.
        Suitable for modelling, pcs could be different than obtained from PCA.
        """

        if self.nans:
            e = self.filter_out_NaNs(field = eofs)[0]
        else:
            e = eofs.copy()
            e = self.flatten_field(f = e)
        e = e.transpose()

        pca_mean = pca_mean if pca_mean is not None else self.pca_mean

        recons = np.dot(e, pcs).T
        recons += pca_mean.T

        if self.nans:
            recons = self.return_NaNs_to_data(field = recons)
        else:
            recons = self.reshape_flat_field(f = recons)

        return recons


        
    def anomalise(self, base_period = None, ts = None):
        """
        Removes the seasonal/yearly cycle from the data.
        If base_period is None, the seasonal cycle is relative to whole period,
        else base_period = (date, date) for climatology within period. Both dates are inclusive.
        """
        
        delta = self.time[1] - self.time[0]
        seasonal_mean = np.zeros_like(self.data) if ts is None else np.zeros_like(ts)
        
        if base_period is None:
            ndx = np.arange(self.time.shape[0])
        else:
            ndx = np.logical_and(self.time >= base_period[0].toordinal(), self.time <= base_period[1].toordinal())
        d = self.data.copy() if ts is None else ts
        t = self.time.copy()
        self.time = self.time[ndx]

        if delta == 1:
            # daily data
            day_avg, mon_avg, _ = self.extract_day_month_year()
            self.time = t.copy()
            day_data, mon_data, _ = self.extract_day_month_year()
            d = d[ndx, ...]
            for mi in range(1,13):
                mon_mask_avg = (mon_avg == mi)
                mon_mask_data = (mon_data == mi)
                for di in range(1,32):
                    sel_avg = np.logical_and(mon_mask_avg, day_avg == di)
                    sel_data = np.logical_and(mon_mask_data, day_data == di)
                    if np.sum(sel_avg) == 0:
                        continue
                    seasonal_mean[sel_data, ...] = np.nanmean(d[sel_avg, ...], axis = 0)
                    if ts is None:
                        self.data[sel_data, ...] -= seasonal_mean[sel_data, ...]
                    else:
                        ts[sel_data, ...] -= seasonal_mean[sel_data, ...]
        elif abs(delta - 30) < 3.0:
            # monthly data
            _, mon_avg, _ = self.extract_day_month_year()
            self.time = t.copy()
            _, mon_data, _ = self.extract_day_month_year()
            d = d[ndx, ...]
            for mi in range(1,13):
                sel_avg = (mon_avg == mi)
                sel_data = (mon_data == mi)
                if np.sum(sel_avg) == 0:
                    continue
                seasonal_mean[sel_data, ...] = np.nanmean(d[sel_avg, ...], axis = 0)
                if ts is None:
                    self.data[sel_data, ...] -= seasonal_mean[sel_data, ...]
                else:
                    ts[sel_data, ...] -= seasonal_mean[sel_data, ...]
        else:
            raise Exception('Unknown temporal sampling in the field.')

        return seasonal_mean
            
            
            
    def get_seasonality(self, detrend = False, base_period = None):
        """
        Removes the seasonality in both mean and std (detrending is optional) and 
        returns the seasonal mean and std arrays.
        If base_period is None, the seasonal cycle is relative to whole period,
        else base_period = (date, date) for climatology within period. Both dates are inclusive.
        """
        
        delta = self.time[1] - self.time[0]
        seasonal_mean = np.zeros_like(self.data)
        seasonal_var = np.zeros_like(self.data)

        if base_period is None:
            ndx = np.arange(self.time.shape[0])
        else:
            ndx = np.logical_and(self.time >= base_period[0].toordinal(), self.time <= base_period[1].toordinal())
        d = self.data.copy()
        t = self.time.copy()
        self.time = self.time[ndx]

        if delta == 1:
            # daily data
            day_avg, mon_avg, _ = self.extract_day_month_year()
            self.time = t.copy()
            day_data, mon_data, _ = self.extract_day_month_year()
            d = d[ndx, ...]
            for mi in range(1,13):
                mon_mask_avg = (mon_avg == mi)
                mon_mask_data = (mon_data == mi)
                for di in range(1,32):
                    sel_avg = np.logical_and(mon_mask_avg, day_avg == di)
                    sel_data = np.logical_and(mon_mask_data, day_data == di)
                    if np.sum(sel_avg) == 0:
                        continue
                    seasonal_mean[sel_data, ...] = np.nanmean(d[sel_avg, ...], axis = 0)
                    self.data[sel_data, ...] -= seasonal_mean[sel_data, ...]
                    seasonal_var[sel_data, ...] = np.nanstd(d[sel_avg, ...], axis = 0, ddof = 1)
                    if np.any(seasonal_var[sel_data, ...] == 0.0) and self.verbose:
                        print('**WARNING: some zero standard deviations found for date %d.%d' % (di, mi))
                        seasonal_var[seasonal_var == 0.0] = 1.0
                    self.data[sel_data, ...] /= seasonal_var[sel_data, ...]
            if detrend:
                data_copy = self.data.copy()
                self.data, _, _ = detrend_with_return(self.data, axis = 0)
                trend = data_copy - self.data
            else:
                trend = None
        elif abs(delta - 30) < 3.0:
            # monthly data
            _, mon_avg, _ = self.extract_day_month_year()
            self.time = t.copy()
            _, mon_data, _ = self.extract_day_month_year()
            d = d[ndx, ...]
            for mi in range(1,13):
                sel_avg = (mon_avg == mi)
                sel_data = (mon_data == mi)
                if np.sum(sel_avg) == 0:
                    continue
                seasonal_mean[sel_data, ...] = np.nanmean(d[sel_avg, ...], axis = 0)
                self.data[sel_data, ...] -= seasonal_mean[sel_data, ...]
                seasonal_var[sel_data, ...] = np.nanstd(d[sel_avg, ...], axis = 0, ddof = 1)
                self.data[sel_data, ...] /= seasonal_var[sel_data, ...]
            if detrend:
                data_copy = self.data.copy()
                self.data, _, _ = detrend_with_return(self.data, axis = 0)
                trend = data_copy - self.data
            else:
                trend = None
        else:
            raise Exception('Unknown temporal sampling in the field.')
            
        return seasonal_mean, seasonal_var, trend
        
        
        
    def return_seasonality(self, mean, var, trend):
        """
        Return the seasonality to the data.
        """
        
        if trend is not None:
            self.data += trend
        self.data *= var
        self.data += mean



    def center_data(self, var = False, return_fields = False):
        """
        Centers data time series to zero mean and unit variance (without respect for the seasons or temporal sampling). 
        """

        mean = np.nanmean(self.data, axis = 0)
        self.data -= mean
        if var:
            var = np.nanstd(self.data, axis = 0, ddof = 1)
            self.data /= var 

        if return_fields:
            return mean if var is False else (mean, var)



    def save_field(self, fname):
        """
        Saves entire Data Field to cPickle format.
        """

        import cPickle

        with open(fname, "wb") as f:
            cPickle.dump(self.__dict__, f, protocol = cPickle.HIGHEST_PROTOCOL)



    def load_field(self, fname):
        """
        Loads entire Data Field from pickled file.
        """

        import cPickle

        with open(fname, "rb") as f:
            data = cPickle.load(f)

        self.__dict__ = data



    @staticmethod
    def _get_oscillatory_modes(a):
        """
        Helper function for wavelet.
        """

        import wavelet_analysis as wvlt

        i, j, s0, data, flag, amp_to_data, k0, cont_ph, cut = a
        if not np.any(np.isnan(data)):
            wave, _, _, _ = wvlt.continous_wavelet(data, 1, True, wvlt.morlet, dj = 0, s0 = s0, j1 = 0, k0 = k0)
            phase = np.arctan2(np.imag(wave), np.real(wave))[0, :]
            amplitude = np.sqrt(np.power(np.real(wave),2) + np.power(np.imag(wave),2))[0, :]
            if amp_to_data:
                reconstruction = amplitude * np.cos(phase)
                fit_x = np.vstack([reconstruction, np.ones(reconstruction.shape[0])]).T
                m, c = np.linalg.lstsq(fit_x, data)[0]
                amplitude = m * amplitude + c
            if cut is not None:
                phase = phase[cut:-cut]
                amplitude = amplitude[cut:-cut]
                wave = wave[0, cut:-cut]
            if cont_ph:
                for t in range(phase.shape[0] - 1):
                    if np.abs(phase[t+1] - phase[t]) > 1:
                        phase[t+1: ] += 2 * np.pi
            
            ret = [phase, amplitude]
            if flag:
                ret.append(wave)

            return i, j, ret
        else:
            if flag:
                return i, j, [np.nan, np.nan, np.nan]
            else:
                return i, j, [np.nan, np.nan]



    @staticmethod
    def _get_parametric_phase(a):
        """
        Helper function for parametric phase.
        """

        i, j, freq, data, window, flag, save_wave, cont_ph, cut = a
        
        if not np.any(np.isnan(data)):
            half_length = int(np.floor(data.shape[0]/2))
            upper_bound = half_length + 1 if data.shape[0] & 0x1 else half_length
            # center data to zero mean (NOT climatologically)
            data -= np.mean(data, axis = 0)
            # compute smoothing wave from signal
            c = np.cos(np.arange(-half_length, upper_bound, 1) * freq)
            s = np.sin(np.arange(-half_length, upper_bound, 1) * freq)
            cx = np.dot(c, data) / data.shape[0]
            sx = np.dot(s, data) / data.shape[0]
            mx = np.sqrt(cx**2 + sx**2)
            phi = np.angle(cx - 1j*sx)
            z = mx * np.cos(np.arange(-half_length, upper_bound, 1) * freq + phi)

            # iterate with window
            iphase = np.zeros_like(data)
            half_window = int(np.floor(window/2))
            upper_bound_window = half_window + 1 if window & 0x1 else half_window
            co = np.cos(np.arange(-half_window, upper_bound_window, 1) *freq)
            so = np.sin(np.arange(-half_window, upper_bound_window, 1) *freq)
            
            for shift in range(0, data.shape[0] - window + 1):
                y = data[shift:shift + window].copy()
                y -= np.mean(y)
                cxo = np.dot(co, y) / window
                sxo = np.dot(so, y) / window
                phio = np.angle(cxo - 1j*sxo)
                iphase[shift+half_window] = phio

            iphase[shift+half_window+1:] = np.angle(np.exp(1j*(np.arange(1, upper_bound_window) * freq + phio)))
            y = data[:window].copy()
            y -= np.mean(y)
            cxo = np.dot(co, y) / window
            sxo = np.dot(so, y) / window
            phio = np.angle(cxo - 1j*sxo)
            iphase[:half_window] = np.angle(np.exp(1j*(np.arange(-half_window, 0, 1)*freq + phio)))
            if cut is not None:
                iphase = iphase[cut:-cut]
                z = z[cut:-cut]
            if cont_ph:
                for t in range(iphase.shape[0] - 1):
                    if np.abs(iphase[t+1] - iphase[t]) > 1:
                        iphase[t+1: ] += 2 * np.pi
            if flag:
                sinusoid = np.arange(-half_length, upper_bound)*freq + phi
                sinusoid = np.angle(np.exp(1j*sinusoid))
                if cut is not None:
                    sinusoid = sinusoid[cut:-cut]
                iphase = np.angle(np.exp(1j*(iphase - sinusoid)))
                iphase -= iphase[0]

            ret = [iphase]
            if save_wave:
                ret.append(z)

            return i, j, ret
        
        else:
            if save_wave:
                return i, j, [np.nan, np.nan]
            else:
                return i, j, [np.nan]



    @staticmethod
    def _get_filtered_data(arg):
        """
        Helper function for temporal filtering.
        """

        from scipy.signal import filtfilt

        i, j, data, b, a = arg

        return i, j, filtfilt(b, a, data)



    def get_parametric_phase(self, period, window, period_unit = 'y', cut = 1, ts = None, pool = None, 
                                    phase_fluct = False, save_wave = False, cut_time = False, 
                                    continuous_phase = False, cut_data = False):
        """
        Computes phase of analytic signal using parametric method.
        Period is frequency in years, or days.
        if ts is None, use self.data as input time series.
        cut is either None or number period to be cut from beginning and end of the time series in years
        if phase_fluct if False, computes only phase, otherwise also phase fluctuations from stationary 
            sinusoid and returns this instead of phase - used for phase fluctuations
        """

        delta = self.time[1] - self.time[0]
        if delta == 1:
            # daily data
            if period_unit == 'y':
                y = 365.25
            elif period_unit == 'd':
                y = 1.
            elif period_unit == 'm':
                raise Exception("For daily data is hard to enter wavelet period in months...")
            else:
                raise Exception("Unknown type.")
        elif abs(delta - 30) < 3.0:
            # monthly data
            if period_unit == 'y':
                y = 12.
            elif period_unit == 'm':
                y = 1.
            elif period_unit == 'd':
                raise Exception("For monthly data doesn't make sense to enter wavelet period in days.")
            else:
                raise Exception("Unknown type.")
        elif delta == 365 or delta == 366:
            # annual data
            if period_unit == 'y':
                y = 1.
            elif period_unit == 'm':
                raise Exception("For monthly data doesn't make sense to enter wavelet period in days.")
            elif period_unit == 'd':
                raise Exception("For monthly data doesn't make sense to enter wavelet period in days.")
            else:
                raise Exception("Unknown type.")
        else:
            raise Exception('Unknown temporal sampling in the field.')

        self.frequency = 2*np.pi / (y*period) # frequency of interest
        window = int(y*window)

        if cut is not None:
            to_cut = int(y*cut)
        else:
            to_cut = None

        if ts is None:

            if self.data.ndim > 2:
                num_lats = self.lats.shape[0]
                num_lons = self.lons.shape[0]
            elif self.data.ndim == 2: # lot of station data
                num_lats = self.lats.shape[0]
                num_lons = 1
                self.data = self.data[:, :, np.newaxis]
            else:
                num_lats = 1
                num_lons = 1
                self.data = self.data[:, np.newaxis, np.newaxis]

            self.phase = np.zeros_like(self.data) if cut is None else np.zeros([self.data.shape[0] - 2*to_cut] + self.get_spatial_dims())
            if save_wave:
                self.wave = np.zeros_like(self.data, dtype = np.complex64) if cut is None else np.zeros([self.data.shape[0] - 2*to_cut] + self.get_spatial_dims(), dtype = np.complex64)


            job_args = [ (i, j, self.frequency, self.data[:, i, j].copy(), window, phase_fluct, save_wave, continuous_phase, to_cut) for i in range(num_lats) for j in range(num_lons) ]
            
            if pool is None:
                job_result = map(self._get_parametric_phase, job_args)
            elif pool is not None:
                job_result = pool.map(self._get_parametric_phase, job_args)
            del job_args
            
            for i, j, res in job_result:
                self.phase[:, i, j] = res[0]
                if save_wave:
                    self.wave[:, i, j] = res[1]

            del job_result

            if cut_time and cut is not None:
                self.time = self.time[to_cut:-to_cut]
            if cut is not None and cut_data:
                self.data = self.data[to_cut:-to_cut, ...]

            self.data = np.squeeze(self.data)
            self.phase = np.squeeze(self.phase)# if cut is None else np.squeeze(self.phase[to_cut:-to_cut, ...])
            if save_wave:
                self.wave = np.squeeze(self.wave)# if cut is None else np.squeeze(self.wave[to_cut:-to_cut, ...])
            
        else:
            res = self._get_parametric_phase([0, 0, self.frequency, ts.copy(), window, phase_fluct, save_wave, continuous_phase, to_cut])[-1]
            return res



    def wavelet(self, period, period_unit = 'y', cut = 1, ts = None, pool = None, save_wave = False, 
                    regress_amp_to_data = False, k0 = 6., cut_time = False, continuous_phase = False, 
                    phase_fluct = False, cut_data = False):
        """
        Permforms wavelet transformation on data.
        Period is central wavelet period in years, or days.
        if ts is None, use self.data as input time series.
        cut is either None or number period to be cut from beginning and end of the time series in years
        """

        delta = self.time[1] - self.time[0]
        if delta == 1:
            # daily data
            if period_unit == 'y':
                y = 365.25
            elif period_unit == 'd':
                y = 1.
            elif period_unit == 'm':
                raise Exception("For daily data is hard to enter wavelet period in months...")
            else:
                raise Exception("Unknown type.")
        elif abs(delta - 30) < 3.0:
            # monthly data
            if period_unit == 'y':
                y = 12.
            elif period_unit == 'm':
                y = 1.
            elif period_unit == 'd':
                raise Exception("For monthly data doesn't make sense to enter wavelet period in days.")
            else:
                raise Exception("Unknown type.")
        elif delta == 365 or delta == 366:
            # annual data
            if period_unit == 'y':
                y = 1.
            elif period_unit == 'm':
                raise Exception("For monthly data doesn't make sense to enter wavelet period in days.")
            elif period_unit == 'd':
                raise Exception("For monthly data doesn't make sense to enter wavelet period in days.")
            else:
                raise Exception("Unknown type.")
        else:
            raise Exception('Unknown temporal sampling in the field.')

        fourier_factor = (4 * np.pi) / (k0 + np.sqrt(2 + np.power(k0,2)))
        per = period * y # frequency of interest
        self.frequency = per
        self.omega = 2*np.pi / (y*period)
        s0 = per / fourier_factor # get scale

        if phase_fluct:
            continuous_phase = True

        if cut is not None:
            to_cut = int(y*cut)
        else:
            to_cut = None

        if ts is None:

            if self.data.ndim > 2:
                num_lats = self.lats.shape[0]
                num_lons = self.lons.shape[0]
            elif self.data.ndim == 2: # lot of station data
                num_lats = self.lats.shape[0]
                num_lons = 1
                self.data = self.data[:, :, np.newaxis]
            else:
                num_lats = 1
                num_lons = 1
                self.data = self.data[:, np.newaxis, np.newaxis]

            self.phase = np.zeros_like(self.data) if cut is None else np.zeros([self.data.shape[0] - 2*to_cut] + self.get_spatial_dims())
            self.amplitude = np.zeros_like(self.data) if cut is None else np.zeros([self.data.shape[0] - 2*to_cut] + self.get_spatial_dims())
            if save_wave:
                self.wave = np.zeros_like(self.data, dtype = np.complex64) if cut is None else np.zeros([self.data.shape[0] - 2*to_cut] + self.get_spatial_dims(), dtype = np.complex64)

            job_args = [ (i, j, s0, self.data[:, i, j], save_wave, regress_amp_to_data, k0, continuous_phase, to_cut) for i in range(num_lats) for j in range(num_lons) ]
            
            if pool is None:
                job_result = map(self._get_oscillatory_modes, job_args)
            elif pool is not None:
                job_result = pool.map(self._get_oscillatory_modes, job_args)
            del job_args
            
            for i, j, res in job_result:
                self.phase[:, i, j] = res[0]
                self.amplitude[:, i, j] = res[1]
                if save_wave:
                    self.wave[:, i, j] = res[2]

            del job_result

            if cut is not None and cut_time:
                self.time = self.time[to_cut:-to_cut]

            if cut is not None and cut_data:
                self.data = self.data[to_cut:-to_cut, ...]

            self.data = np.squeeze(self.data)
            if phase_fluct:
                ph0 = self.phase[0, ...]
                sin = np.arange(0, self.phase.shape[0])[:, np.newaxis, np.newaxis] * self.omega + ph0
                self.phase -= sin
                self.phase = np.squeeze(self.phase)
            else:
                self.phase = np.squeeze(self.phase)
            self.amplitude = np.squeeze(self.amplitude)
            if save_wave:
                self.wave = np.squeeze(self.wave)
        
        else:
            res = self._get_oscillatory_modes([0, 0, s0, ts, save_wave, regress_amp_to_data, k0, continuous_phase, to_cut])[-1]
            # add phase fluct!!!
            return res



    def quick_render(self, t = 0, lvl = 0, mean = False, field_to_plot = None, station_data = False, tit = None, 
                        symm = False, whole_world = True, log = None, fname = None, plot_station_points = False, 
                        colormesh = False, cmap = None, vminmax = None, levels = 40, cbar_label = None, 
                        subplot = False, extend = 'neither'):
        """
        Simple plot of the geo data using the Robinson projection for whole world
        or Mercator projection for local plots.
        By default, plots first temporal field in the data.
        t is temporal point (< self.time.shape[0])
        if mean is True, plots the temporal mean.
        to render different field than self.data, enter 2d field of the same shape.
        log is either None or base.
        cmap defaults to 'viridis' if None.
        if fname is None, shows the plot, otherwise saves it to the given filename.
        vminmax is either tuple of (min, max) to plot, or if None, it is determined
        from the data
        """

        import matplotlib.pyplot as plt
        from mpl_toolkits.basemap import Basemap, shiftgrid
        from matplotlib import colors

        # set up field to plot
        if self.var_name is None:
            self.var_name = 'unknown'
        if self.data.ndim == 3:
            field = self.data[t, ...]
            title = ("%s: %d. point -- %s" % (self.var_name.upper(), t, self.get_date_from_ndx(t)))
            if mean:
                field = np.mean(self.data, axis = 0)
                title = ("%s: temporal mean" % (self.var_name.upper()))
        
        elif self.data.ndim == 4:
            field = self.data[t, lvl, ...]
            title = ("%s at %d level: %d. point -- %s" % (self.var_name.upper(), lvl, t, self.get_date_from_ndx(t)))
            if mean:
                field = np.mean(self.data[:, lvl, ...], axis = 0)
                title = ("%s at %d level: temporal mean" % (self.var_name.upper(), lvl))

        if field_to_plot is not None and not station_data:
            if field_to_plot.ndim == 2 and field_to_plot.shape[0] == self.lats.shape[0] and field_to_plot.shape[1] == self.lons.shape[0]:
                field = field_to_plot
                title = ("Some field you should know")
            else:
                raise Exception("field_to_plot has to have shape as lats x lons saved in the data class!")

        if station_data:
            if field_to_plot.ndim != 1:
                raise Exception("Station data must be passed as time x station!")

            import scipy.interpolate as si
            # 0.1 by 0.1 grid
            lats_stations = np.arange(self.lats.min(), self.lats.max()+0.1, 0.1)
            lons_stations = np.arange(self.lons.min(), self.lons.max()+0.3, 0.3)
            grid_lat, grid_lon = np.meshgrid(lats_stations, lons_stations, indexing = 'ij') # final grids
            points = np.zeros((self.lons.shape[0], 2))
            points[:, 0] = self.lats
            points[:, 1] = self.lons
            field = si.griddata(points, field_to_plot, (grid_lat, grid_lon), method = 'nearest')

            title = ("Some interpolated field you should know from station data")


        # set up figure
        if not subplot:
            plt.figure(figsize=(20,10))
            size_parallels = 20
            size_cbarlabel = 27
            size_title = 30
        else:
            size_parallels = 12
            size_cbarlabel = 16
            size_title = 19

        if not station_data:
            lat_ndx = np.argsort(self.lats)
            lats = self.lats[lat_ndx]
        else:
            lat_ndx = np.argsort(lats_stations)
            lats = lats_stations[lat_ndx]
        field = field[lat_ndx, :]

        # set up projections
        if whole_world:
            data = np.zeros((field.shape[0], field.shape[1] + 1))
            data[:, :-1] = field
            data[:, -1] = data[:, 0]
            if not station_data:
                llons = self.lons.tolist()
            else:
                llons = lons_stations.tolist()
            llons.append(360)
            lons = np.array(llons)
            m = Basemap(projection = 'robin', lon_0 = 0, resolution = 'c')
            
            end_lon_shift = np.sort(lons - 180.)
            end_lon_shift = end_lon_shift[end_lon_shift >= 0.]
            data, lons = shiftgrid(end_lon_shift[0] + 180., data, lons, start = False)

            m.drawparallels(np.arange(-90, 90, 30), linewidth = 1.2, labels = [1,0,0,0], color = "#222222", size = size_parallels)
            m.drawmeridians(np.arange(-180, 180, 60), linewidth = 1.2, labels = [0,0,0,1], color = "#222222", size = size_parallels)

        else:
            if not station_data:
                lons = self.lons.copy()
            else:
                lons = lons_stations.copy()
            data = field.copy()

            # if not monotonic
            if not np.all([x < y for x, y in zip(lons, lons[1:])]):
                lons[lons > lons[-1]] -= 360

            m = Basemap(projection = 'merc',
                    llcrnrlat = lats[0], urcrnrlat = lats[-1],
                    llcrnrlon = lons[0], urcrnrlon = lons[-1],
                    resolution = 'i')

            # parallels and meridians to plot
            draw_lats = np.arange(np.around(lats[0]/5, decimals = 0)*5, np.around(lats[-1]/5, decimals = 0)*5, 10)
            draw_lons = np.arange(np.around(lons[0]/5, decimals = 0)*5, np.around(lons[-1]/5, decimals = 0)*5, 20)
            m.drawparallels(draw_lats, linewidth = 1.2, labels = [1,0,0,0], color = "#222222", size = size_parallels)
            m.drawmeridians(draw_lons, linewidth = 1.2, labels = [0,0,0,1], color = "#222222", size = size_parallels)
            

        m.drawcoastlines(linewidth = 2, color = "#333333")
        m.drawcountries(linewidth = 1.5, color = "#333333")
    
        x, y = m(*np.meshgrid(lons, lats))
        
        max = np.nanmax(data) if vminmax is None else vminmax[1]
        min = np.nanmin(data) if vminmax is None else vminmax[0]
        if symm:
            if np.abs(max) > np.abs(min):
                min = -max
            else:
                max = -min

        # draw contours
        cmap = plt.get_cmap(cmap) if cmap is not None else plt.get_cmap('viridis')
        if log is not None:
            levels = np.logspace(np.log10(min)/np.log10(log), np.log10(max)/np.log10(log), levels+1)
            cs = m.contourf(x, y, data, norm = colors.LogNorm(vmin = min, vmax = max), levels = levels, cmap = cmap, 
                extend = extend)
        else:
            levels = np.linspace(min, max, levels+1)
            if colormesh:
                data = np.ma.array(data, mask = np.isnan(data))
                cs = m.pcolormesh(x, y, data, vmin = levels[0], vmax = levels[-1], cmap = cmap)
            else:
                cs = m.contourf(x, y, data, levels = levels, cmap = cmap, extend = extend)

        # draw stations if station data
        if station_data and plot_station_points:
            for lat, lon in zip(self.lats, self.lons):
                x, y = m(lon, lat)
                m.plot(x, y, 'ko', markersize = 3)
        
        # colorbar
        cbar = plt.colorbar(cs, ticks = levels[::4], pad = 0.07, shrink = 0.8, fraction = 0.05)
        cbar.ax.set_yticklabels(np.around(levels[::4], decimals = 2), size = size_parallels)
        if cbar_label is not None:
            cbar.set_label(cbar_label, rotation = 90, size = size_cbarlabel)

        if tit is None:
            plt.title(title, size = size_title)
        else:
            plt.title(tit, size = size_title)

        if not subplot:
            if fname is None:
                plt.show()
            else:
                plt.savefig(fname, bbox_inches = 'tight')
