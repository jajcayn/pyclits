pyCliTS
==========

What is pyCliTS?
--------------------
Python Climate Time Series package is open-source python package for easy manipulation with climatic geo-spatial time series such as the reanalysis or CMIP5 outputs, which are usually distributed as netCDF4 files. The package includes functions for:  

* manipulating the data [temporal and spatial slicing, interpolating, subtracting the climatological cycle = anomalising, normalising, filtering, subsampling, etc.] 
* computing continuous complex wavelet transform [CCWT]
* constructing spatio-temporal surrogate data using Monte-Carlo approach [Fourier transform surrogates, amplitude adjusted FT, iterative amplitude adjusted FT, autoregressive surrogates using the VAR(p) model, multifractal surrogates] 
* computing Singular Spectrum Analysis
* computing mutual information and conditional mutual information [using equidistant, equiquantal binning and k-nearest neighbour algorithms] 
* constructing an empirical model from spatio-temporal data based on idea of LIMs [linear inverse modelling].

Uses fast numpy, scipy and scikit-learn libraries and offers multi-thread computations when possible [e.g. computing wavelet transform per grid point].


Documentation
-------------
Instead of proper documentation [I plan to add it later though!], I created a few examples on how to work with basic class ``DataField`` and other functions. These can be found in examples folder. This folder includes some climate data, which are publicly available for download, just for your convenience, I included them here. 


Dependencies
------------
``pyclits`` relies on the following open source packages    

**Required**:

* `numpy <https://github.com/numpy/numpy>`_
* `scipy <https://github.com/scipy/scipy>`_

**Recommended**:

* `sklearn <https://github.com/scikit-learn/scikit-learn>`_  
* `cython <https://github.com/cython/cython>`_  
* `matplotlib <https://github.com/matplotlib/matplotlib>`_  
* `netCDF4 <https://github.com/Unidata/netcdf4-python>`_  
* `basemap toolkit <https://github.com/matplotlib/basemap>`_ 
* `pywavelets <https://github.com/PyWavelets/pywt>`_
* `pathos multiprocessing <https://github.com/uqfoundation/pathos>`_ (for improved multiprocessing capabilities) 

(All of them are installed via pip automatically when installing this package, except basemap, since it is not on PyPI. Basemap still can be installed via pip using ``pip install git+https://github.com/matplotlib/basemap.git``)


Contributing
------------
All contributions are welcome! Just drop me an email or pull request.


License information
-------------------
``pyclits`` is MIT-licensed, for more information see the file LICENSE.txt

