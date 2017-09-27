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
from mpl_toolkits.axes_grid1 import make_axes_locatable

# with wavelet module, in addition to just obtaining one oscillatory component [as in example 2-oscillatory.py],
#   we can do the classical wavelet analysis...
# lets analyse the periods in NINO34 data

nino34 = clt.data_loaders.load_enso_index("example_data/nino34raw.txt", "3.4", start_date = date(1870, 1, 1), 
    end_date = date(2017, 7, 1), anom = False)


# for this we need to use wavelet module in its raw form
dt = 1./12 # in years -- its monthly data, and we want the result to be in years
pad = True # recommended -- will pad the time series with 0 up to the length of power of 2
mother = clt.wavelet_analysis.morlet # mother wavelet
dj = 0.25 # this will do 4 sub-octaves per octave
s0 = 6 * dt # this will set first period at 6 months
j1 = 7 / dj # this says do 7 powers-of-two with dj sub-octaves each
k0 = 6. # default for Morlet mother

wave, period, scale, coi = clt.wavelet_analysis.continous_wavelet(nino34.data, dt = dt, pad = pad, wavelet = mother,
    dj = dj, s0 = s0, j1 = j1, k0 = k0)
power = (np.abs(wave)) ** 2  # compute wavelet power spectrum

# set time array and levels to plot
time = np.arange(nino34.time.shape[0]) * dt + nino34.get_date_from_ndx(0).year
levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16]
# plot wavelet
cs = plt.contourf(time, period, np.log2(power), len(levels))
im = plt.contourf(cs, levels = np.log2(levels))
# plot cone-of-influence
plt.plot(time, coi, 'k') 
# set log scale and revert y-axis
plt.gca().set_yscale('log', basey=2, subsy=None)
plt.ylim([np.min(period), np.max(period)])
ax = plt.gca().yaxis
ax.set_major_formatter(matplotlib.ticker.ScalarFormatter())
plt.ticklabel_format(axis='y', style='plain')
plt.gca().invert_yaxis()
# set labels and title
plt.xlabel('time [year]')
plt.ylabel('period [years]')
plt.title('NINO3.4 wavelet power spectrum')
# colorbar
divider = make_axes_locatable(plt.gca())
cax = divider.append_axes("bottom", size="5%", pad=0.5)
plt.colorbar(im, cax=cax, orientation='horizontal')
plt.show()