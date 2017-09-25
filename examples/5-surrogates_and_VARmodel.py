"""
Examples for pyCliTS -- https://github.com/jajcayn/pyclits
"""

# import modules
import pyclits as clt
from datetime import date
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import numpy as np

# lets start with 1D series
prg = clt.data_loaders.load_station_data("example_data/TG_STAID000027.txt", date(1770, 1, 1), date(2016, 1, 1),
    anom = False, to_monthly = False)
# lets use 65536 data points, because MF surrogates needs 2^n data points
prg.get_data_of_precise_length(length = '65k', end_date = date(2016, 1, 1), apply_to_data = True)
# print shape
print prg.shape()

# lets create some surrogates
# firstly, lets init empty SurrogateField
prg_surrs = clt.surrogates.SurrogateField()
# SurrogateField class inherits from DataField, so ALL of the methods are the same, plus some new ones -- for creating surrs
# when using surrogates, it is better to create surrogates from anomalised, varnormalised data and after generation of 
#   surrogates add them back
prg_mean, prg_var, prg_trend = prg.get_seasonality(detrend = True, base_period = [date(1981,1,1), date(2010,12,31)])
# copy field to SurrogateField
prg_surrs.copy_field(prg)
# now we can check 
print prg.data.shape, prg_surrs.original_data.shape
print np.all(prg.data == prg_surrs.original_data)

# now we want to work with raw data [no anom, no varnorm, ...] so lets add back these fields
prg.return_seasonality(prg_mean, prg_var, prg_trend)

# lets construct FT surrogate
prg_surrs.construct_fourier_surrogates(algorithm = 'FT')
prg_surrs.add_seasonality(prg_mean, prg_var, prg_trend)
prg_FT = prg_surrs.data.copy()

# lets construct AAFT surrogate
prg_surrs.construct_fourier_surrogates(algorithm = 'AAFT')
prg_surrs.add_seasonality(prg_mean, prg_var, prg_trend)
prg_AAFT = prg_surrs.data.copy()

# lets construct IAAFT surrogate
prg_surrs.construct_fourier_surrogates(algorithm = 'IAAFT', n_iterations = 100)
prg_surrs.add_seasonality(prg_mean, prg_var, prg_trend)
prg_IAAFT = prg_surrs.data.copy()

# lets construct AR(p) surrogate
prg_surrs.prepare_AR_surrogates(order_range = [1,30])
prg_surrs.construct_surrogates_with_residuals()
# time series of AR(p) surrogates are shorter by p [which makes sense, right?]
#   so we need to cut 'order' time points from the end
order = prg_surrs.model_grid[0].order()
prg_surrs.add_seasonality(prg_mean[:-order], prg_var[:-order], prg_trend[:-order])
prg_AR = prg_surrs.data.copy()

# lets construct MF surrogate
prg_surrs.construct_multifractal_surrogates(randomise_from_scale = 2)
prg_surrs.add_seasonality(prg_mean, prg_var, prg_trend)
prg_MF = prg_surrs.data.copy()

# lets plot time-series -- but only some cut as we got a lot of data
plot_ndx = prg.select_date(date(1990, 1, 1), date(2016,1,1), apply_to_data = True)
plt.figure()    
plt.subplot(221)
plt.title("time series")
plt.plot(prg.data, linewidth = 0.3, label = "original Prague SAT")
plt.plot(prg_FT[plot_ndx], linewidth = 0.3, label = "Prague FT surr")
plt.plot(prg_AAFT[plot_ndx], linewidth = 0.3, label = "Prague AAFT surr")
plt.plot(prg_IAAFT[plot_ndx], linewidth = 0.3, label = "Prague IAAFT surr")
plt.plot(prg_AR[plot_ndx[:-order]], linewidth = 0.3, label = "Prague AR(%d) surr" % (order))
plt.plot(prg_MF[plot_ndx], linewidth = 0.3, label = "Prague MF surr")
plt.xticks(np.arange(prg.time.shape[0], 4*365.25), np.arange(prg.get_date_from_ndx(0).year, 
    prg.get_date_from_ndx(-1).year+1, 4), rotation = 30)
plt.legend()

# lets plot FFT spectrum -- this should be same!
plt.subplot(223)
plt.title("Welch spectra")
import scipy.signal as ss
fs = 1./86400 # frequency of data -- this is 1/day in seconds so in Hz
f, pxx = ss.welch(prg.data, fs, 'flattop', 1024, scaling = 'spectrum')
f *= 3.154e+7 # we want frequency in 1/year
plt.semilogy(f, np.sqrt(pxx), label = "original Prague SAT")
f, pxx = ss.welch(prg_FT[plot_ndx], fs, 'flattop', 1024, scaling = 'spectrum')
f *= 3.154e+7 # we want frequency in 1/year
plt.semilogy(f, np.sqrt(pxx), label = "Prague FT surr")
f, pxx = ss.welch(prg_AAFT[plot_ndx], fs, 'flattop', 1024, scaling = 'spectrum')
f *= 3.154e+7 # we want frequency in 1/year
plt.semilogy(f, np.sqrt(pxx), label = "Prague AAFT surr")
f, pxx = ss.welch(prg_IAAFT[plot_ndx], fs, 'flattop', 1024, scaling = 'spectrum')
f *= 3.154e+7 # we want frequency in 1/year
plt.semilogy(f, np.sqrt(pxx), label = "Prague IAAFT surr")
f, pxx = ss.welch(prg_AR[plot_ndx[:-order]], fs, 'flattop', 1024, scaling = 'spectrum')
f *= 3.154e+7 # we want frequency in 1/year
plt.semilogy(f, np.sqrt(pxx), label = "Prague AR(%d) surr" % (order))
f, pxx = ss.welch(prg_MF[plot_ndx], fs, 'flattop', 1024, scaling = 'spectrum')
f *= 3.154e+7 # we want frequency in 1/year
plt.semilogy(f, np.sqrt(pxx), label = "Prague MF surr")
plt.legend()
plt.xlim([0, 12])
plt.xlabel('frequency [1/year]')

# lets plot auto-correlation function! [with max lag of 100 days]
plt.subplot(222)
plt.title("Autocorrelations")
cross_corr = clt.functions.cross_correlation(prg.data, prg.data, max_lag = 100)
cross_corr = cross_corr[cross_corr.shape[0]//2:]
plt.plot(cross_corr, label = "original Prague SAT")
cross_corr = clt.functions.cross_correlation(prg_FT[plot_ndx], prg_FT[plot_ndx], max_lag = 100)
cross_corr = cross_corr[cross_corr.shape[0]//2:]
plt.plot(cross_corr, label = "Prague FT surr")
cross_corr = clt.functions.cross_correlation(prg_AAFT[plot_ndx], prg_AAFT[plot_ndx], max_lag = 100)
cross_corr = cross_corr[cross_corr.shape[0]//2:]
plt.plot(cross_corr, label = "Prague AAFT surr")
cross_corr = clt.functions.cross_correlation(prg_IAAFT[plot_ndx], prg_IAAFT[plot_ndx], max_lag = 100)
cross_corr = cross_corr[cross_corr.shape[0]//2:]
plt.plot(cross_corr, label = "Prague IAAFT surr")
cross_corr = clt.functions.cross_correlation(prg_AR[plot_ndx[:-order]], prg_AR[plot_ndx[:-order]], max_lag = 100)
cross_corr = cross_corr[cross_corr.shape[0]//2:]
plt.plot(cross_corr, label = "Prague AR(%d) surr" % (order))
cross_corr = clt.functions.cross_correlation(prg_MF[plot_ndx], prg_MF[plot_ndx], max_lag = 100)
cross_corr = cross_corr[cross_corr.shape[0]//2:]
plt.plot(cross_corr, label = "Prague MF surr")
plt.xlabel("lag [days]")
plt.legend()

# finaly, lets estimate and plot PDF
plt.subplot(224)
plt.title("Histograms and PDF estimates")
plt.hist(prg.data, bins = 20, fc = 'C0', ec = 'C0', alpha = 0.3, normed = True)
# estimate KDE -- assume Gaussian distribution with bandwidth as SD of the data
x, kde = clt.functions.kdensity_estimate(prg.data, kernel = 'gaussian', bandwidth = np.std(prg.data, ddof = 1))
plt.plot(x, kde, color = 'C0', label = "original Prague SAT")
plt.hist(prg_FT[plot_ndx], bins = 20, fc = 'C1', ec = 'C1', alpha = 0.3, normed = True)
x, kde = clt.functions.kdensity_estimate(prg_FT[plot_ndx], kernel = 'gaussian', bandwidth = np.std(prg_FT[plot_ndx], ddof = 1))
plt.plot(x, kde, color = 'C1', label = "Prague FT surr")
plt.hist(prg_AAFT[plot_ndx], bins = 20, fc = 'C2', ec = 'C2', alpha = 0.3, normed = True)
x, kde = clt.functions.kdensity_estimate(prg_AAFT[plot_ndx], kernel = 'gaussian', bandwidth = np.std(prg_AAFT[plot_ndx], ddof = 1))
plt.plot(x, kde, color = 'C2', label = "Prague AAFT surr")
plt.hist(prg_IAAFT[plot_ndx], bins = 20, fc = 'C3', ec = 'C3', alpha = 0.3, normed = True)
x, kde = clt.functions.kdensity_estimate(prg_IAAFT[plot_ndx], kernel = 'gaussian', bandwidth = np.std(prg_IAAFT[plot_ndx], ddof = 1))
plt.plot(x, kde, color = 'C3', label = "Prague IAAFT surr")
plt.hist(prg_AR[plot_ndx[:-order]], bins = 20, fc = 'C4', ec = 'C4', alpha = 0.3, normed = True)
x, kde = clt.functions.kdensity_estimate(prg_AR[plot_ndx[:-order]], kernel = 'gaussian', bandwidth = np.std(prg_AR[plot_ndx[:-order]], ddof = 1))
plt.plot(x, kde, color = 'C4', label = "Prague AR(%d) surr" % (order))
plt.hist(prg_MF[plot_ndx], bins = 20, fc = 'C5', ec = 'C5', alpha = 0.3, normed = True)
x, kde = clt.functions.kdensity_estimate(prg_MF[plot_ndx], kernel = 'gaussian', bandwidth = np.std(prg_MF[plot_ndx], ddof = 1))
plt.plot(x, kde, color = 'C5', label = "Prague MF surr")
plt.legend()

plt.show()


# # # lets load some spatio-temporal dataset, e.g. SSTs
# # sst = clt.geofield.DataField()
# # sst.load(filename = "example_data/sst.mnmean.nc", variable_name = "sst", dataset = "NCEP")

