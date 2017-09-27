"""
Examples for pyCliTS -- https://github.com/jajcayn/pyclits
"""

# import modules
import pyclits as clt
from datetime import date
import matplotlib
# change for your favourite backend
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sts

# load NCEP/NCAR reanlysis data
temp = clt.data_loaders.load_NCEP_data_monthly("example_data/air.mon.mean.sig995.nc", "air", start_date = date(1948, 1, 1), 
    end_date = date(2017, 1, 1), lats = None, lons = None, level = None, anom = False)
# start and end dates are clear, lats and lons are either tuples of which latitudes and longitudes we want to load,
#   None means do not cut anything. This is single level data, so level None and anom is whether we want to anomalise.

# The structure of geofield is as follows:
print temp.data.shape # this is of shape (time x lats x lons); numpy array
print temp.lats.shape # this is array of latitudes
print temp.lons.shape # this is array of longitudes
print temp.time.shape # this of array of ordinal dates (1 is 1 Jan of year 1)

# for temporal mean per grid point do
print np.mean(temp.data, axis = 0) # shape will be (lats, lons)
# for spatial mean do
print np.mean(temp.data, axis = (1,2)) # shape will be (time,)
# for weighted mean [weighted by cosine of latitude] do
print np.mean(temp.data * temp.latitude_cos_weights(), axis = (1,2)) # shape will be (time,)

# cut the data only 1950 - 2010, exclusive end date
temp.select_date(date(1950, 1, 1), date(2011, 1, 1), exclusive = True)

# plot to see whether is everything OK -- hopefully, you've got basemap installed
temp.quick_render(t = 5)

# now lets focus on Europe and lets see its anomalies
#   anomalies are residuals when we subtract long-term climatology -- lets do 1981 - 2010 climatology
temp.select_lat_lon(lats = [20, 70], lons = [340, 60])
temp.anomalise(base_period = [date(1981, 1, 1), date(2010, 12, 31)])
temp.quick_render(t = 5, whole_world = False, cbar_label = r"[$^\circ$C]", tit = "anomalies")

# lets find grid point closest to Prague and compare it with station data
prg_la, prg_lo = temp.get_closest_lat_lon(50.08, 14.44)
prague_from_ncep = temp.data[:, prg_la, prg_lo].copy()

# load Prague station data
prg = clt.data_loaders.load_station_data("example_data/TG_STAID000027.txt", date(1950, 1, 1), date(2011, 1, 1),
    anom = False, to_monthly = False)
print prg.data.shape # station data has shape of (time,); lats and lons fields are empty
print prg.time.shape # same as before -- array of ordinal dates
# this is daily data so make in monthly to match resolution of NCEP, when means is True, makes monthly means, 
#   False would make monthly sums, e.g. for precipitation
prg.get_monthly_data(means = True)

# anomalise with respect to the same period
prg.anomalise(base_period = [date(1981, 1, 1), date(2010, 12, 31)])

# before plotting lets check the date ranges
print temp.get_date_from_ndx(0), temp.get_date_from_ndx(-1)
print prg.get_date_from_ndx(0), prg.get_date_from_ndx(-1)

# now plot
plt.plot(prague_from_ncep, label = "NCEP/NCAR: %.2fN x %.2fE" % (temp.lats[prg_la], temp.lons[prg_lo]))
plt.plot(prg.data, label = "PRG station")
plt.legend()
plt.xticks(np.arange(0, prg.time.shape[0], 10*12), np.arange(prg.get_date_from_ndx(0).year, 
    prg.get_date_from_ndx(-1).year+1, 10), rotation = 30)
plt.title("correlation: %.3f" % sts.pearsonr(prg.data, prague_from_ncep)[0])
plt.show()

# lets compare winter variances
#   select_months is ideal for this, if apply_to_data is False, it returns the indices,
#   which we use in our prague_from_ncep array
DJF_ndx = temp.select_months(months = [12, 1, 2], apply_to_data = False)
prg.select_months(months = [12, 1, 2], apply_to_data = True)
print "NCEP variance: ", np.var(prague_from_ncep[DJF_ndx], ddof = 1)
print "PRG station variance: ", np.var(prg.data, ddof = 1)


# lets try some interpolation -- load CO2 dataset and remove its trend from global anomalies
# load NOAA global temperature anomaly dataset
noaa = clt.geofield.DataField()
# load raw data using numpy
raw = np.loadtxt("example_data/NOAA-NCDC.csv", delimiter = ',', skiprows = 5)
# this should be time x 2, where in the first column we got date as e.g. 188001 for Jan 1880 and then data itself
first_date = date(int(str(raw[0,0])[:4]), int(float(str(raw[0,0])[4:])), 1)
noaa.data = raw[:, 1].copy()
# this creates time array from 'first_date' with monthly sampling
noaa.create_time_array(date_from = first_date, sampling = 'm')
# lets check dates
print noaa.data.shape, noaa.get_date_from_ndx(0), noaa.get_date_from_ndx(-1)

# now load yearly CO2 data
co2 = clt.geofield.DataField()
# load raw data, again using numpy
raw = np.loadtxt("example_data/RCP3PD_MIDYR_CONC.DAT", skiprows = 39)
# co2 concetrations is in first column
co2.data = raw[:, 1]
co2.create_time_array(date(int(raw[0, 0]), 1, 1), sampling = 'y')
# lets select same dates as NOAA global dataset -- BUT CO2 is annual values, so lets pick one year more, so we
#   can interpolate
# lets use relative time delta for it!
from dateutil.relativedelta import relativedelta
co2.select_date(noaa.get_date_from_ndx(0), noaa.get_date_from_ndx(-1) + relativedelta(years = +1), exclusive = False)
# lets check
print co2.data.shape, co2.get_date_from_ndx(0), co2.get_date_from_ndx(-1)

# thats fine, now lets interpolate to monthly data
# in this case, linear and cubic gives almost the same time series, you should try different schemes for e.g. interpolating
#   monthly to daily etc.
co2.interpolate_to_finer_temporal_resolution(to_resolution = 'm', kind = 'linear', use_to_data = True)
print co2.data.shape, co2.get_date_from_ndx(0), co2.get_date_from_ndx(-1)
# perfect! now lets do final cut to match time series length
co2.select_date(noaa.get_date_from_ndx(0), noaa.get_date_from_ndx(-1), exclusive = False)
print co2.data.shape, co2.get_date_from_ndx(0), co2.get_date_from_ndx(-1)

# nice! now plot the global anomalies, so we now what's in them
plt.plot(noaa.data)
plt.xticks(np.arange(0, noaa.time.shape[0], 10*12), np.arange(noaa.get_date_from_ndx(0).year, 
    noaa.get_date_from_ndx(-1).year+1, 10), rotation = 30)
plt.show()

# we definitely see some trend in the second half! lets check whether we can make this time series stationary by 
#   subtracting the CO2 trend, so estimating coefs a and b
# lets use sklearn linear model for this!
import sklearn.linear_model as slm
model = slm.LinearRegression(fit_intercept = True)
model.fit(np.log(co2.data[:, np.newaxis]), noaa.data)
# get the coefficients from the model
a = model.coef_[0]
b = model.intercept_
residuals = noaa.data - (a*np.log(co2.data) + b)
# plot
plt.plot(noaa.data, label = "NOAA global")
plt.plot(residuals, label = "NOAA without CO2 trend")
plt.plot(a*np.log(co2.data) + b, label = "CO2 trend", color = 'k', linewidth = 0.7)
plt.xticks(np.arange(0, noaa.time.shape[0], 10*12), np.arange(noaa.get_date_from_ndx(0).year, 
    noaa.get_date_from_ndx(-1).year+1, 10), rotation = 30)
plt.legend()
plt.show()
