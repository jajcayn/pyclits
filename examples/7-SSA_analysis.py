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

# load Prague station data -- monthly for comparison with NCEP
prg = clt.data_loaders.load_station_data("example_data/TG_STAID000027.txt", date(1948, 1, 1), date(2016, 1, 1),
    anom = False, to_monthly = True)
print prg.shape()

# load NCEP and lets focus on small area around Prague -- our station data
temp = clt.data_loaders.load_NCEP_data_monthly("example_data/air.mon.mean.sig995.nc", "air", start_date = date(1948, 1, 1), 
    end_date = date(2016, 1, 1), lats = [40,60], lons = [0,30], level = None, anom = False)
temp.subsample_spatial(lat_to = 5, lon_to = 5, start = [temp.lats.min(), temp.lons.min()], average = False)
print temp.shape()

# plot, just to see the area we'll work with
temp.quick_render(mean = True, whole_world = False)

# lets init SSA class with embedding winodow of 2years
prg_ssa = clt.ssa.ssa_class(prg.data, M = 2*12) # since station data is 1-D, this reduces to classic SSA
temp.flatten_field()
temp_ssa = clt.ssa.ssa_class(temp.data, M = 2*12) # here we've got spatio-temporal data, so this is 
#   multi-channel SSA.. the input to SSA class needs to be 2D as time x space, so we are flattening our field

# lets run SSA
eigvals_prg, eigvecs_prg, pcs_prg, rcs_prg = prg_ssa.run_ssa() 
eigvals_temp, eigvecs_temp, pcs_temp, rcs_temp = temp_ssa.run_ssa() 


# firstly, lets plot first 20 eigenvalues 
plt.plot(eigvals_prg[:20], 'o', label = "Prague station eigvals")
plt.plot(eigvals_temp[:20], 's', label = "NCEP EU eigvals")
plt.legend()
plt.ylabel("$\lambda$")
plt.show()

# we see that eigenvalues are almost the same
# lets check first 5 eigenvectors and PCs
for i in range(5):
    plt.subplot(5, 2, 2*i+1)
    plt.plot(eigvecs_prg[:24, i], label = "Prague station eigenvector %d" % (i+1))
    plt.plot(eigvecs_temp[:24, i], label = "NCEP EU eigenvector %d" % (i+1))
    plt.legend()

    plt.subplot(5, 2, 2*i+2)
    plt.plot(pcs_prg[:, i], linewidth = 0.3, label = "Prague station PC %d" % (i+1))
    plt.plot(pcs_temp[:, i], linewidth = 0.3, label = "NCEP EU PC %d" % (i+1))
    plt.legend()
plt.show()

# from the eigenvectors [first two in both have period of 12] we clearly see, that the signal is totally dominated by 
#   the annual cycle [no surprise] -- eigenvectors have period of 12 and the first two eigenvalues are much larger
#   than other ones

# we can also use rotated SSA using either VARIMAX or ORTHOMAX
eigvals_prg_rot, eigvecs_prg_rot, pcs_prg_rot, rcs_prg_rot = prg_ssa.apply_varimax(S = 24, structured = True, sort_lam = True)
eigvals_temp_rot, eigvecs_temp_rot, pcs_temp_rot, rcs_temp_rot = temp_ssa.apply_varimax(S = 24, structured = True, sort_lam = True)

# firstly, lets plot first 20 eigenvalues 
plt.plot(eigvals_prg[:20], 'o', label = "Prague station eigvals")
plt.plot(eigvals_temp[:20], 's', label = "NCEP EU eigvals")
plt.plot(eigvals_prg_rot[:20], '^', label = "rotated Prague station eigvals")
plt.plot(eigvals_temp_rot[:20], 'v', label = "rotated NCEP EU eigvals")
plt.legend()
plt.ylabel("$\lambda$")
plt.show()

# we see that eigenvalues are almost the same
# lets check first 5 eigenvectors and PCs
for i in range(5):
    plt.subplot(5, 2, 2*i+1)
    plt.plot(eigvecs_prg[:24, i], label = "Prague station eigenvector %d" % (i+1))
    plt.plot(eigvecs_temp[:24, i], label = "NCEP EU eigenvector %d" % (i+1))
    plt.plot(eigvecs_prg_rot[:24, i], label = "rotated Prague station eigenvector %d" % (i+1))
    plt.plot(eigvecs_temp_rot[:24, i], label = "rotated NCEP EU eigenvector %d" % (i+1))
    plt.legend()

    plt.subplot(5, 2, 2*i+2)
    plt.plot(pcs_prg[:, i], linewidth = 0.3, label = "Prague station PC %d" % (i+1))
    plt.plot(pcs_temp[:, i], linewidth = 0.3, label = "NCEP EU PC %d" % (i+1))
    plt.plot(pcs_prg_rot[:, i], linewidth = 0.3, label = "rotated Prague station PC %d" % (i+1))
    plt.plot(pcs_temp_rot[:, i], linewidth = 0.3, label = "rotated NCEP EU PC %d" % (i+1))
    plt.legend()
plt.show()

# finally, lets do Monte-Carlo SSA, that is lets check the significance of eigenvalues using AR(1) surrogate
prg_ssa.run_Monte_Carlo(n_realizations = 100, p_value = 0.05, plot = True) # for other option, read the code