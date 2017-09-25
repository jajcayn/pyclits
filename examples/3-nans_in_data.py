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

# lets import some data from scratch without data loader
sst = clt.geofield.DataField()
sst.load(filename = "example_data/sst.mnmean.nc", variable_name = "sst", dataset = "NCEP")

# lets check
print np.any(np.isnan(sst.data)) # print True, so we've got NaNs in the data..
# there are two options -- 1. either NaNs are spatial (this is the example, since SST is defined only in oceans, so its
#   NaN over land)
#   -- 2. the NaN are temporal
#   with spatial NaNs it's easy, all the methods are compatible, they just omit the grid [atlernatively, you can 
#   interpolate spatial NaNs.. with SST dataset that makes no sense, but sometimes it might help..]
#     over the land, with 

print sst.check_NaNs_only_spatial() # this prints True, so we are lucky, NaNs are only spatial.. 
# lets cut the field so we dont spend too much time on interpolation
sst.select_date(date(1950,1,1), date(2010,1,1))
# lets check how the field looks like and lets try interpolate spatial nans
import pathos.multiprocessing as mp
pool = mp.ProcessingPool(5)
interp = sst.interpolate_spatial_nans(method = 'cubic', apply_to_data = False, pool = pool)
pool.close()
pool.join()

plt.figure()
plt.subplot(121)
sst.quick_render(t = 550, subplot = True, tit = "spatial NaN [orig. field]")
plt.subplot(122)
sst.quick_render(field_to_plot = interp[550, ...], subplot = True, tit = "interp. field")
plt.show()

# looks good! but we know it's not correct to interpolate SST over the land... we'll carry on with NaN

# all the functions as anomalise, wavelet etc are compatible with spatial nans, since they either use
#   numpy's nanmeans etc, own nandetrend function, or just omit the NaN grid points [as wavelet]

# geofield is able to do PCA analysis [sometime refered to as EOF] and it works also with spatial NaN, lets see
# lets do EOFs on anomalised data -- after all, that's what we are interested in.. and lets focus on -60,60
sst.select_lat_lon(lats = [-60,60], lons = None)
sst.anomalise(base_period = [date(1981,1,1), date(2010,12,31)])
eofs, pcs, var = sst.pca_components(n_comps = 5)
print "first 5 EOF components explains: ", np.sum(var)
print eofs.shape, pcs.shape
# lets plot
plt.figure()
for i in range(eofs.shape[0]):
    plt.subplot(2,3,i+1)
    sst.quick_render(field_to_plot = eofs[i, ...], subplot = True, tit = "%s: EOF %d" % (sst.var_name.upper(), i+1))
plt.suptitle("EOF analysis of %s: %d - %d" % (sst.var_name.upper(), sst.get_date_from_ndx(0).year, sst.get_date_from_ndx(-1).year))
plt.show()

