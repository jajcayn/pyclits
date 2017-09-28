"""
Examples for pyCliTS -- https://github.com/jajcayn/pyclits
"""

# now, we'll build similar model as in Kondrashov et al., J. Climate, 18, 2005. that is multi-level model based on idea od
#   LIM - linear inverse model, so it will be data-based

# import modules
import pyclits as clt
from datetime import date
import matplotlib
# change for your favourite backend
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import numpy as np

# empirical model inherits from DataField, so all the methods are available
enso_model = clt.empirical_model.EmpiricalModel(no_levels = 3, verbose = True) # we'll use three levels [as per paper]
enso_model.load_geo_data(fname = "example_data/sst.mnmean.nc", varname = "sst", start_date = date(1900, 1, 1), 
    end_date = date(2017, 1, 1), lats = [-30, 60], lons = None, dataset = "NCEP", anom = False)
# plot for quick check
# enso_model.quick_render(t = 653, cbar_label = "SST anomalies")

# first, lets get rid of low frequency variabiliy, which would drift EOFs..
# this is done by filtering time series with 50-year moving average and then subtracting first 5 EOFs of this low-frequency
#   time series; no_comps is number of low-frequency EOFs you wish to subtract, this can be None, in that case
#   the number of subtracted EOFs would be such they would explain 99% of low-frequency variability
# enso_model.remove_low_freq_variability(mean_over = 50, cos_weights = True, no_comps = 5)

# this function prepares the input -- this model is trained in the EOF space.. we will use first 30 EOFs [no_input_ts],
#   then cosweight the data and do the PCA.. sel is None, but you can use it to select some PCA, e.g. you wish to
#   build your model on PCs 1, 3, 5, 6, 7 then the sel would be [0,2,4,5,6], in that case, you could omit no_input_ts
enso_model.prepare_input(anom = True, no_input_ts = 30, cos_weights = True, sel = None)

# now we train model -- we want only linear model, but with harmonic predictor [that is our model will be dx = Ax + b]
#   and matrix A will be trained seasonally, so 12 different matrices for 12 months, as a regressor we'll use partial
#   least squares, other choices are documented in the empirical_model.py source; partialLSQ is house function, other
#   regressor are adapted from sklearn
enso_model.train_model(harmonic_pred = 'first', quad = False, delay_model = False, regressor = 'partialLSQ')

# now we can integrate model forward in time! lets do 30 realizations, of the same length as original data [int_length
#   is None -- you can do shorter or longer, int_length is in months], sigma is noise variance, we'll use default 1, 
#   we'll use 5 workers [threads] to integrate, we want to see diagnostic plots and for the noise type, we can choose:
#   either basic white noise [uncorrelated in time, correlated in space by cov. matrix of the last level residuals],
#   or conditional noise [this will find noise samples from last level residuals, which are similar to the state of our
#   model and use their cov. matrix] or finally, seasonal noise [this will fit 5 harmonics on annual cycle and use
#   seasonally changing covariance matrices] -- lets try seasonal also with conditioning
enso_model.integrate_model(30, int_length = None, sigma = 1., n_workers = 5, diagnostics = True, 
    noise_type = ['cond', 'seasonal'])
# in the diagnostic plots are always plotted data [black] and 2.5 and 97.5 percentile of distribution from realizations

# finally lets reconstruct our 3D field... if lats and/or lons are not None, will cut only this area of original data one
#   [5S-5N and 190-240E is NINO3.4 area, and yeah, you guessed it -- we'll compare with actual NINO3.4 index] and if mean, 
#   it will also do spatial mean of the field -- so we'll get NINO3.4 modelled index directly... to be more precise,
#   we'll get 50 stochastic realizations of NINO3.4 index
enso_model.reconstruct_simulated_field(lats = [-5, 5], lons = [190, 240], mean = True)

print enso_model.reconstructions.shape
# in our case, the reconstructions are of shape no_recons x time

# if needed, we can save all model, or just the results [save_all boolean] and either as Matlab *.mat file or pickle
#   with cPickle and save as *.bin [mat boolean]
# enso_model.save(fname = "null", save_all = False, mat = False)

# anyway, lets compare with actual NINO3.4!
nino34 = clt.data_loaders.load_enso_index("example_data/nino34raw.txt", "3.4", start_date = date(1900, 1, 1), 
    end_date = date(2017, 1, 1), anom = False)
print nino34.shape()

# lets plot time series and spectrum!
import scipy.signal as ss
plt.subplot(211)
plt.plot(nino34.data, color = "k", linewidth = 2.5)
for i in range(enso_model.reconstructions.shape[0]):
    plt.plot(enso_model.reconstructions[i, :], color = 'gray', linewidth = 0.2)
plt.xticks(np.arange(0, nino34.time.shape[0], 10*12), np.arange(nino34.get_date_from_ndx(0).year, 
    nino34.get_date_from_ndx(-1).year+1, 10), rotation = 30)
plt.ylabel("SST [$^\circ$C]")

plt.subplot(212)
f, pxx = ss.welch(nino34.data, 1./2.628e+6, 'flattop', 1024, scaling = 'spectrum')
f *= 3.154e+7
plt.semilogy(f, np.sqrt(pxx), color = "k", linewidth = 2.5)
for i in range(enso_model.reconstructions.shape[0]):
    f, pxx = ss.welch(enso_model.reconstructions[i, :], 1./2.628e+6, 'flattop', 1024, scaling = 'spectrum')
    f *= 3.154e+7
    plt.semilogy(f, np.sqrt(pxx), color = 'gray', linewidth = 0.2)
plt.xlabel("frequency [1/year]")
plt.ylabel("linear spectrum")
plt.xlim([0, 2])
plt.show()

# highly neat!