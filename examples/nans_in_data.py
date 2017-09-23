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
sst.load(filename = "example_data/sst.mnmean.v4.nc", varname = "sst", dataset = "NCEP")