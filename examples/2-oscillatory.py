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

# load Prague station data
prg = clt.data_loaders.load_station_data("example_data/TG_STAID000027.txt", date(1770, 1, 1), date(2016, 1, 1),
    anom = False, to_monthly = False)

# that is a lot od data [88023] data points.. lets focus on last ~80years
prg.select_date(date(1940, 1, 1), date(2016, 1, 1))

# lets extract annual cycle using CCWT
#   because of cone-of-influence of intrinsic to CCWT, I'd like to cut from the beginning and the end of the time seres
#   cut is in the same units as period, hence years, because 'y', I could also do 'm' and then period would be 12
#   and cut would be 48; cut_date and cut_time asks whether we want to apply our cut also to data and time
#   since CCWT amplitude is in arbitrary units, lets regress it to data, hence we'll have it in degrees C
prg.wavelet(period = 1, period_unit = 'y', cut = 4, cut_data = False, cut_time = False, regress_amp_to_data = True)
wvlt_phase = prg.phase.copy() # lets copy the result
amplitude = prg.amplitude.copy()

# lets extract phase using other technique -- sine model fitting
prg.get_parametric_phase(period = 1, window = 5, period_unit = 'y', cut = 4)
smf_phase = prg.phase.copy()

# finally, lets do bandpass between 11 and 13 months and compare to other methods
prg.temporal_filter(cutoff = [11, 13], btype = 'bandpass', cut = 4)

# plot
plt.plot(amplitude*np.cos(wvlt_phase), label = "CCWT annual")
plt.plot(amplitude*np.cos(smf_phase), label = "SMF phase -- annual")
plt.plot(prg.filtered_data, label = "11-13m bandpass")
plt.legend()
plt.xticks(np.arange(0, prg.time.shape[0], 8*365.25), np.arange(prg.get_date_from_ndx(0).year, 
    prg.get_date_from_ndx(-1).year+1, 8), rotation = 30)
plt.show()

# OK, temporally filtered data are of lower amplitude, CCWT and SMF phases are virtually the same... or are they?
# lets compare phase fluctuations [deviations from the ideal sinusoidal annual cycle]
prg.wavelet(period = 1, period_unit = 'y', cut = 4, cut_data = False, cut_time = False, phase_fluct = True)
wvlt_fluc = prg.phase.copy() # lets copy the result
prg.get_parametric_phase(period = 1, window = 5, period_unit = 'y', cut = 4, phase_fluct = True)
smf_fluc = prg.phase.copy()

plt.plot(wvlt_fluc, label = "CCWT phase fluctuations")
plt.plot(smf_fluc, label = "SMF phase fluctuations")
plt.legend()
plt.xticks(np.arange(0, prg.time.shape[0], 8*365.25), np.arange(prg.get_date_from_ndx(0).year, 
    prg.get_date_from_ndx(-1).year+1, 8), rotation = 30)
plt.show()

# now we see the differences! CCWT phase estimate is smooth, where SMF have some fast variability in it!

# we can of course do wavelet on spatio-temporal data!
# lets import pool and allow for some multi-threading
NUM_WORKERS = 5 # how many threads we want to initialize
import pathos.multiprocessing as mp

# load NCEP/NCAR reanlysis data for NH
temp = clt.data_loaders.load_NCEP_data_monthly("example_data/air.mon.mean.sig995.nc", "air", start_date = date(1948, 1, 1), 
    end_date = date(2017, 1, 1), lats = [0, 80], lons = None, level = None, anom = False)

# lets compare annual amplitudes around the globe!
pool = mp.ProcessingPool(NUM_WORKERS) # compute on 5 workers
temp.wavelet(period = 1, period_unit = 'y', cut = 4, cut_data = True, cut_time = True, 
    regress_amp_to_data = True, pool = pool)
pool.close() # lets be fair and close and join the pool after we're done
pool.join()

# lets plot the temporal mean [datafield is always of shape time x lats x lons]
temp.quick_render(field_to_plot = np.mean(temp.amplitude, axis = 0), tit = "amplitude average %d - %d" \
    % (temp.get_date_from_ndx(0).year, temp.get_date_from_ndx(-1).year))