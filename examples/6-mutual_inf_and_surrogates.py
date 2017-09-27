"""
Examples for pyCliTS -- https://github.com/jajcayn/pyclits
"""

# import modules
import pyclits as clt
import pyclits.mutual_inf as MI
from datetime import date
import matplotlib
# change for your favourite backend
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import numpy as np

## WARNING runs approximately 45min, depending on the machine [mine: 5threads on quadcore 2.2GHz i7]

# lets compute mutual information between NINO3.4 index and global SST
# load SSTs
sst = clt.geofield.DataField()
sst.load(filename = "example_data/sst.mnmean.nc", variable_name = "sst", dataset = "NCEP")
sst.select_lat_lon(lats = [-30,30], lons = None)
sst.select_date(date(1900, 1, 1), date(2017, 1, 1), exclusive = False)
# sst.anomalise(base_period = [date(1981,1,1), date(2010,12,31)])
print sst.shape()

# load NINO34
nino34 = clt.data_loaders.load_enso_index("example_data/nino34raw.txt", "3.4", start_date = date(1900, 1, 1), 
    end_date = date(2017, 8, 1), anom = False)
nino34.select_date(date(1900, 1, 1), date(2017, 1, 1), exclusive = False)
print nino34.shape()

# since computations per grid point are independent, we can split between threads
import pathos.multiprocessing as mp

def _compute_MI(a):
    i, ts1, ts2 = a
    return (i, MI.mutual_information(ts1, ts2, algorithm = 'EQD', bins = 16))
    # return (i, MI.knn_mutual_information(ts1, ts2, k = 32))

# a quick note -- its actually recommended to use knn algorithm over binning ones [function knn_mutual_information]
#   as it returns unbiased result.. unfortunately, knn runs much longer than the EQD, so for illustration purposes
#   I am using EQD.. if you got some time to spend, uncomment the knn line and comment the EQD one

# beforehand we need to filter out NaNs... this function will return field as time x space [enrolled in one dimension]
#   and without NaNs, and mask with nans which we will use later to reconstruct former shape of the field [with NaNs]
sst_flat, nan_mask = sst.filter_out_NaNs()
mutual_inf = np.zeros((sst_flat.shape[1],))
pool = mp.ProcessingPool(5)
# lets build our argumnet to the function -- we need to keep track of grid point number
args = [(i, sst_flat[:, i], nino34.data) for i in range(mutual_inf.shape[0])]
result = pool.map(_compute_MI, args)
for i, mi in result:
    mutual_inf[i] = mi

# with this we return NaNs to data and get field as time x lats x lons.. since it is designed to deal with spatio-temporal data
#   we need to use this little trick, to create artifical temporal dimension so it will return 1 x lats x lons
#   and then we want to squeeze it, so in fact we got lats x lons field [as wanted and needed for plotting]
mutual_inf_plot = np.squeeze(sst.return_NaNs_to_data(field = mutual_inf[np.newaxis, :], mask = nan_mask))
# plot
sst.quick_render(field_to_plot = mutual_inf_plot, tit = "Mutual information with NINO3.4")

# since computing just MI we have no information what value is actually high and what not, lets do proper stastical testing,
#   so surrogates -- it's OK to make it easier for us and create surrogate from NINO34 time series rather than SSTs..

nino_surr = clt.surrogates.SurrogateField()
nino_mean, nino_var, _ = nino34.get_seasonality()
nino_surr.copy_field(nino34)

# lets both paralellize with respect to grid point AND number of surrogates
def _compute_MI_surrs(a):
    i, sg, ts2, mean, var = a
    sg.construct_fourier_surrogates(algorithm = 'FT')
    sg.add_seasonality(mean, var, None)
    return (i, MI.mutual_information(sg.data, ts2, algorithm = 'EQD', bins = 16))

NUM_SURR = 100
args = [(i, nino_surr, sst_flat[:, i], nino_mean, nino_var) for i in range(sst_flat.shape[1]) for _ in range(NUM_SURR)]
surr_result = pool.map(_compute_MI_surrs, args)
temp = {}
for i, mi in surr_result:
    if not (i in temp):
        temp[i] = [mi]
    else:
        temp[i].append(mi)

pool.close()
pool.join()

mutual_inf_surrs = np.zeros((NUM_SURR, sst_flat.shape[1]))
for i in temp:
    mutual_inf_surrs[:, i] = temp[i]

p_vals = clt.surrogates.get_p_vals(mutual_inf, mutual_inf_surrs, one_tailed = True)
# print mutual_inf[32], mutual_inf_surrs[:, 32]

# plot -- we are using viridis_r [that is default but reverse], because the lower the p-value is, the better for use
#   so, we are still looking to see some yellow..
p_vals = np.squeeze(sst.return_NaNs_to_data(field = p_vals[np.newaxis, :], mask = nan_mask))
print p_vals
sst.quick_render(field_to_plot = p_vals, tit = "p-values of MI", cmap = "viridis_r", colormesh = False)

# just a little note -- here we've done only 100 surrogates and it took forever... now, if you want your results to be
#   really tested, you need to 1. generate more surrogates, and 2. since we are testing more than 6000 hypotheses at the
#   same time [every grid point is one hypothesis], we are definitely having some false positives [to be more precise, 
#   we expect around 350 false positives].. so, ideally, you would generate a lot surrogates, so that Bonferroni 
