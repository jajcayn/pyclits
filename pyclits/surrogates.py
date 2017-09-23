"""
created on Mar 4, 2014

@author: Nikola Jajcay, jajcay(at)cs.cas.cz

last update on Sep 22, 2017
"""

import numpy as np
from geofield import DataField



def get_p_vals(field, surr_field, one_tailed = True):
    """
    Returns one-tailed or two-tailed values of percentiles with respect to 
    surrogate testing.
    """

    num_surrs = surr_field.shape[0]
    if field.shape != surr_field.shape[1:]:
        raise Exception("Wrong input fields. surr_field has to have shape as num_surr x field.shape!")

    # get significance - p-values
    sig = 1. - np.sum(np.greater_equal(field, surr_field), axis = 0) / float(num_surrs)
    if one_tailed:
        return sig
    else:
        return 2 * np.minimum(sig, 1. - sig)



def bonferroni_test(p_vals, sig_level, Nsurr, Nhyp = None, Sidak = False):
    """
    Run a Bonferroni multiple testing procedure on p-values <p_vals> with significance
    level <sig_level> given that <Nsurr> surrogates were used to compute the p-values.
    Optionally the number of simultaneously tested hypotheses can be set with <Nhyp>,
    if left at None, len(p_vals) will be used.
    If Sidak is True, will run Bonferroni-Sidak test.

    Written by Martin Vejmelka -- https://github.com/vejmelkam/ndw-climate/blob/master/src/multi_stats.py
    """

    if Nhyp is None:
        Nhyp = len(p_vals)

    # if the number of hypotheses is higher than the Nhyp (can happen for robustness testing)
    # remove the tail elements of p_vals until p_vals has the length Nhyp
    if Nhyp < len(p_vals):
        p_vals = p_vals[:Nhyp]

    if not Sidak:
        bonf_level = sig_level / Nhyp
    else:
        bonf_level = 1.0 - (1.0 - sig_level) ** (1.0 / Nhyp)
    
    if bonf_level < 1.0 / Nsurr:
        raise Exception("Will not run Bonferroni, not enough surrogates available for the test!")
    
    return p_vals < bonf_level



def fdr_test(p_vals, sig_level, Nsurr, Nhyp = None):
    """
    Run an FDR multiple testing procedure on p-values <p_vals> with significance
    level <sig_level> given that <Nsurr> surrogate were used to compute the p-values.
    Optionally the number of simultaneously tested hypotheses can be set with <Nhyp>,
    if left at None, len(p_vals) will be used.
    NOTE: if Nhyp < len(p_vals), the p_vals tail will be chopped off so that len(p_val) = Nhyp.
          Then only will the test be run.

    Written by Martin Vejmelka -- https://github.com/vejmelkam/ndw-climate/blob/master/src/multi_stats.py
    """
    
    if Nhyp is None:
        Nhyp = len(p_vals)

    # if the number of hypotheses is higher than the Nhyp (can happen for robustness testing)
    # remove the tail elements of p_vals until p_vals has the length Nhyp
    if Nhyp < len(p_vals):
        p_vals = p_vals[:Nhyp]

    # the sorting is done only after the number of p_vals is fixed, see comments.
    sndx = np.argsort(p_vals)
    
    bonf_level = sig_level / Nhyp
    
    if bonf_level < 1.0 / Nsurr:
        raise Exception("Will not run FDR, not enough surrogates used for the test!")
    
    Npvals = len(p_vals)
    h = np.zeros((Npvals,), dtype = np.bool)
    
    # test the p-values in order of p-values (smallest first)
    for i in range(Npvals - 1, 0, -1):
        
        # select the hypothesis with the i-th lowest p-value
        hndx = sndx[i]
        
        # check if we satisfy the FDR condition
        if p_vals[hndx] <= (i+1)*bonf_level:
            h[sndx[:i+1]] = True
            break
        
    return h



def holm_test(p_vals, sig_level, Nsurr, Nhyp = None):
    """
    Run a Bonferroni-Holm multiple testing procedure on p-values <p_vals> with significance
    level <sig_level> given that <Nsurr> surrogate were used to compute the p-values.
    Optionally the number of simultaneously tested hypotheses can be set with <Nhyp>,
    if left at None, len(p_vals) will be used.
    NOTE: if Nhyp < len(p_vals), the p_vals tail will be chopped off so that len(p_val) = Nhyp.
          Then only will the test be run.
    Written by Martin Vejmelka -- https://github.com/vejmelkam/ndw-climate/blob/master/src/multi_stats.py 
    """
    
    if Nhyp is None:
        Nhyp = len(p_vals)

    # if the number of hypotheses is higher than the Nhyp (can happen for robustness testing)
    # remove the tail elements of p_vals until p_vals has the length Nhyp
    if Nhyp < len(p_vals):
        p_vals = p_vals[:Nhyp]

    # the sorting is done only after the number of p_vals is fixed, see comments.
    sndx = np.argsort(p_vals)
    
    bonf_level = sig_level / Nhyp
    
    if bonf_level < 1.0 / Nsurr:
        raise Exception("Will not run Bonferroni-Holm test, not enough surrogates used for the test!")

    h = np.zeros((Nhyp,), dtype = np.bool)

    # test the p-values in order of p-values (smallest first)
    for i in range(Nhyp):
        
        # select the hypothesis with the i-th lowest p-value
        hndx = sndx[i]
        
        # check if we have violated the Bonf-Holm condition
        if p_vals[hndx] > sig_level / (Nhyp - i):
            break
        
        # the hypothesis is true
        h[hndx] = True
            
    return h



def get_single_FT_surrogate(ts, seed = None):
    """
    Returns single 1D Fourier transform surrogate.
    Seed / integer : when None, random seed, else fixed seed (e.g. for multivariate FT surrogates).
    """

    if seed is None:
        np.random.seed()
    else:
        np.random.seed(seed)
    xf = np.fft.rfft(ts, axis = 0)
    angle = np.random.uniform(0, 2 * np.pi, (xf.shape[0],))
    # set the slowest frequency to zero, i.e. not to be randomised
    angle[0] = 0

    cxf = xf * np.exp(1j * angle)

    return np.fft.irfft(cxf, n = ts.shape[0], axis = 0)



def get_single_AAFT_surrogate(ts, seed = None):
    """
    Returns single amplitude adjusted FT surrogate.
    Seed / integer : when None, random seed, else fixed seed (e.g. for multivariate AAFT surrogates).
    """

    if seed is None:
        np.random.seed()
    else:
        np.random.seed(seed)

    xf = np.fft.rfft(ts, axis = 0)
    angle = np.random.uniform(0, 2 * np.pi, (xf.shape[0],))
    del xf

    return _compute_AAFT_surrogates([None, ts, angle])[-1]



def get_single_IAAFT_surrogate(ts, n_iterations = 10, seed = None):
    """
    Returns single iterative amplitude adjusted FT surrogate.
    n_iterations - number of iterations of the algorithm.
    Seed / integer : when None, random seed, else fixed seed (e.g. for multivariate IAAFT surrogates).
    """

    if seed is None:
        np.random.seed()
    else:
        np.random.seed(seed)

    xf = np.fft.rfft(ts, axis = 0)
    angle = np.random.uniform(0, 2 * np.pi, (xf.shape[0],))
    del xf

    return _compute_IAAFT_surrogates([ None, n_iterations, ts, angle])[-1]



def get_single_MF_surrogate(ts, randomise_from_scale = 2, seed = None):
    """
    Returns single 1D multifractal surrogate.
    Seed / integer : when None, random seed, else fixed seed (e.g. for multivariate MF surrogates).
    """

    return _compute_MF_surrogates([None, ts, randomise_from_scale, seed])[-1]



def get_single_AR_surrogate(ts, order_range = [1,1], seed = None):
    """
    Returns single 1D autoregressive surrogate of some order.
    Order could be found numerically by setting order_range, or
    entered manually by selecting min and max order range to the
    desired order - e.g. [1,1].
    If the order was supposed to estimate, it is also returned.
    Seed / integer : when None, random seed, else fixed seed (e.g. for multivariate AR surrogates).
    """

    _, order, res = _prepare_AR_surrogates([None, order_range, 'sbc', ts])
    num_ts = ts.shape[0] - order.order()
    res = res[:num_ts, 0]

    surr = _compute_AR_surrogates([None, res, order, num_ts, seed])[-1]

    if np.diff(order_range) == 0:
        return surr
    else:
        return surr, order.order()



def amplitude_adjust_single_surrogate(ts, surr, mean = 0, var = 1, trend = None):
    """
    Returns amplitude adjusted surrogate.
    """

    return _create_amplitude_adjusted_surrogates([None, ts, surr, mean, var, trend])[-1]



def _prepare_AR_surrogates(a):
    from var_model import VARModel
    i, order_range, crit, ts = a
    if not np.any(np.isnan(ts)):
        v = VARModel()
        v.estimate(ts, order_range, True, crit, None)
        r = v.compute_residuals(ts)
    else:
        v = None
        r = np.nan
    return (i, v, r) 
    
    
    
def _compute_AR_surrogates(a):
    i, res, model, num_tm_s, seed = a
    r = np.zeros((num_tm_s, 1), dtype = np.float64)       
    if not np.all(np.isnan(res)):
        ndx = np.argsort(np.random.uniform(size = (num_tm_s,)))
        r[ndx, 0] = res

        ar_surr = model.simulate_with_residuals(r, seed)[:, 0]
    else:
        ar_surr = np.nan
        
    return (i, ar_surr)
    
    
    
def _compute_FT_surrogates(a):
    i, data, angle = a
            
    # transform the time series to Fourier domain
    xf = np.fft.rfft(data, axis = 0)
     
    # randomise the time series with random phases     
    cxf = xf * np.exp(1j * angle)
    
    # return randomised time series in time domain
    ft_surr = np.fft.irfft(cxf, n = data.shape[0], axis = 0)
    
    return (i, ft_surr)



def _compute_AAFT_surrogates(a):
    i, data, angle = a

    # create Gaussian data
    gaussian = np.random.randn(data.shape[0])
    gaussian.sort(axis = 0)

    # rescale data
    ranks = data.argsort(axis = 0).argsort(axis = 0)
    rescaled_data = gaussian[ranks]

    # transform the time series to Fourier domain
    xf = np.fft.rfft(rescaled_data, axis = 0)
     
    # randomise the time series with random phases     
    cxf = xf * np.exp(1j * angle)
    
    # return randomised time series in time domain
    ft_surr = np.fft.irfft(cxf, n = data.shape[0], axis = 0)

    # rescale back to amplitude distribution of original data
    sorted_original = data.copy()
    sorted_original.sort(axis = 0)
    ranks = ft_surr.argsort(axis = 0).argsort(axis = 0)

    rescaled_data = sorted_original[ranks]
    
    return (i, rescaled_data)



def _compute_IAAFT_surrogates(a):
    i, n_iters, data, angle = a

    xf = np.fft.rfft(data, axis = 0)
    xf_amps = np.abs(xf)
    sorted_original = data.copy()
    sorted_original.sort(axis = 0)

    # starting point
    R = _compute_AAFT_surrogates([None, data, angle])[-1]

    # iterate
    for _ in range(n_iters):
        r_fft = np.fft.rfft(R, axis = 0)
        r_phases = r_fft / np.abs(r_fft)

        s = np.fft.irfft(xf_amps * r_phases, n = data.shape[0], axis = 0)

        ranks = s.argsort(axis = 0).argsort(axis = 0)
        R = sorted_original[ranks]

    return (i, R)


    
def _compute_MF_surrogates(a):
    import pywt
    
    i, ts, randomise_from_scale, seed = a

    if seed is None:
        np.random.seed()
    else:
        np.random.seed(seed)
    
    if not np.all(np.isnan(ts)):
        n = int(np.log2(ts.shape[0])) # time series length should be 2^n
        n_real = np.log2(ts.shape[0])
        
        if n != n_real:
            # if time series length is not 2^n
            raise Exception("Time series length must be power of 2 (2^n).")
        
        # get coefficient from discrete wavelet transform, 
        # it is a list of length n with numpy arrays as every object
        coeffs = pywt.wavedec(ts, 'db1', level = n-1)
        
        # prepare output lists and append coefficients which will not be shuffled
        coeffs_tilde = []
        for j in range(randomise_from_scale):
            coeffs_tilde.append(coeffs[j])
    
        shuffled_coeffs = []
        for j in range(randomise_from_scale):
            shuffled_coeffs.append(coeffs[j])
        
        # run for each desired scale
        for j in range(randomise_from_scale, len(coeffs)):
            
            # get multiplicators for scale j
            multiplicators = np.zeros_like(coeffs[j])
            for k in range(coeffs[j-1].shape[0]):
                if coeffs[j-1][k] == 0:
                    print("**WARNING: some zero coefficients in DWT transform!")
                    coeffs[j-1][k] = 1
                multiplicators[2*k] = coeffs[j][2*k] / coeffs[j-1][k]
                multiplicators[2*k+1] = coeffs[j][2*k+1] / coeffs[j-1][k]
           
            # shuffle multiplicators in scale j randomly
            coef = np.zeros_like(multiplicators)
            multiplicators = np.random.permutation(multiplicators)
            
            # get coefficients with tilde according to a cascade
            for k in range(coeffs[j-1].shape[0]):
                coef[2*k] = multiplicators[2*k] * coeffs_tilde[j-1][k]
                coef[2*k+1] = multiplicators[2*k+1] * coeffs_tilde[j-1][k]
            coeffs_tilde.append(coef)
            
            # sort original coefficients
            coeffs[j] = np.sort(coeffs[j])
            
            # sort shuffled coefficients
            idx = np.argsort(coeffs_tilde[j])
            
            # finally, rearange original coefficient according to coefficient with tilde
            temporary = np.zeros_like(coeffs[j])
            temporary[idx] = coeffs[j]
            shuffled_coeffs.append(temporary)
        
        # return randomised time series as inverse discrete wavelet transform
        mf_surr = pywt.waverec(shuffled_coeffs, 'db1')

    else:
        mf_surr = np.nan
        
    return (i, mf_surr)



def _create_amplitude_adjusted_surrogates(a):
    i, d, surr, m, v, t = a
    data = d.copy()
    
    if not np.all(np.isnan(data)):
        # sort surrogates
        idx = np.argsort(surr)
        
        # return seasonality back to the data
        if t is not None:
            data += t
        data *= v
        data += m
        
        # amplitude adjustment are original data sorted according to the surrogates
        data = np.sort(data)
        aa_surr = np.zeros_like(data)
        aa_surr[idx] = data

    else:
        aa_surr = np.nan

    return (i, aa_surr)




class SurrogateField(DataField):
    """
    Class holds geofield of surrogate data and can construct surrogates.
    """
    
    def __init__(self, data = None):
        DataField.__init__(self)
        self.data = None
        self.model_grid = None
        self.original_data = data
        

        
    def copy_field(self, field):
        """
        Makes a copy of another DataField
        """
        
        self.original_data = field.data.copy()
        if field.lons is not None:
            self.lons = field.lons.copy()
        else:
            self.lons = None
        if field.lats is not None:
            self.lats = field.lats.copy()
        else:
            self.lats = None
        self.time = field.time.copy()
        
        
        
    def add_seasonality(self, mean, var, trend):
        """
        Adds seasonality to surrogates if there were constructed from deseasonalised
        and optionally detrended data.
        """
        
        if self.data is not None:
            if trend is not None:
                self.data += trend
            self.data *= var
            self.data += mean
        else:
            raise Exception("Surrogate data has not been created yet.")
            
            
        
    def remove_seasonality(self, mean, var, trend):
        """
        Removes seasonality from surrogates in order to use same field for 
        multiple surrogate construction.
        """        
        
        if self.data is not None:
            self.data -= mean
            self.data /= var
            if trend is not None:
                self.data -= trend
        else:
            raise Exception("Surrogate data has not been created yet.")



    def center_surr(self):
        """
        Centers the surrogate data to zero mean and unit variance.
        """

        if self.data is not None:
            self.data -= np.nanmean(self.data, axis = 0)
            self.data /= np.nanstd(self.data, axis = 0, ddof = 1)
        else:
            raise Exception("Surrogate data has not been created yet.")
        
        
        
    def get_surr(self):
        """
        Returns the surrogate data
        """
        
        if self.data is not None:
            return self.data.copy()
        else:
            raise Exception("Surrogate data has not been created yet.")
        


    def construct_fourier_surrogates(self, algorithm = 'FT', pool = None, preserve_corrs = False, n_iterations = 10):
        """
        Constructs Fourier Transform (FT) surrogates - shuffles angle in Fourier space of the original data.
        algorithm:
            FT - basic FT surrogates [1]
            AAFT - amplitude adjusted FT surrogates [2]
            IAAFT - iterative amplitude adjusted FT surrogates [3]
        pool:
            instance of multiprocessing's pool in order to exploit multithreading for high-dimensional data
        preserve_corrs:
            bool, whether to preserve covariance structure in spatially distributed data
        n_iterations:
            int, only when algorithm = IAAFT, number of iterations
        """

        if algorithm not in ['FT', 'AAFT', 'IAAFT']:
            raise Exception("Unknown algorithm type, please use 'FT', 'AAFT' or 'IAAFT'.")
        
        if self.original_data is not None:

            np.random.seed()
            
            if pool is None:
                map_func = map
            else:
                map_func = pool.map

            if algorithm == 'FT':
                surr_func = _compute_FT_surrogates
            elif algorithm == 'AAFT':
                surr_func = _compute_AAFT_surrogates
            elif algorithm == 'IAAFT':
                surr_func = _compute_IAAFT_surrogates
                
            if self.original_data.ndim > 1:
                orig_shape = self.original_data.shape
                self.original_data = np.reshape(self.original_data, (self.original_data.shape[0], np.prod(orig_shape[1:])))
            else:
                orig_shape = None
                self.original_data = self.original_data[:, np.newaxis]
                
            # generate uniformly distributed random angles
            a = np.fft.rfft(np.random.rand(self.original_data.shape[0]), axis = 0)
            if preserve_corrs:
                angle = np.random.uniform(0, 2 * np.pi, (a.shape[0],))
                # set the slowest frequency to zero, i.e. not to be randomised
                angle[0] = 0
                del a
                if algorithm == 'IAAFT':
                    job_data = [ (i, n_iterations, self.original_data[:, i], angle) for i in range(self.original_data.shape[1]) ]    
                else:
                    job_data = [ (i, self.original_data[:, i], angle) for i in range(self.original_data.shape[1]) ]
            else:
                angle = np.random.uniform(0, 2 * np.pi, (a.shape[0], self.original_data.shape[1]))
                angle[0, ...] = 0
                del a
                if algorithm == 'IAAFT':
                    job_data = [ (i, n_iterations, self.original_data[:, i], angle[:, i]) for i in range(self.original_data.shape[1]) ]
                else:
                    job_data = [ (i, self.original_data[:, i], angle[:, i]) for i in range(self.original_data.shape[1]) ]
            
            job_results = map_func(surr_func, job_data)
            
            self.data = np.zeros_like(self.original_data)
            
            for i, surr in job_results:
                self.data[:, i] = surr
                
            # squeeze single-dimensional entries (e.g. station data)
            self.data = np.squeeze(self.data)
            self.original_data = np.squeeze(self.original_data)

            # reshape back to original shape
            if orig_shape is not None:
                self.data = np.reshape(self.data, orig_shape)
                self.original_data = np.reshape(self.original_data, orig_shape)
           
        else:
            raise Exception("No data to randomise in the field. First you must copy some DataField.")

            

    def construct_multifractal_surrogates(self, pool = None, randomise_from_scale = 2):
        """
        Constructs multifractal surrogates (independent shuffling of the scale-specific coefficients,
        preserving so-called multifractal structure - hierarchical process exhibiting information flow
        from large to small scales)
        written according to: Palus, M. (2008): Bootstraping multifractals: Surrogate data from random 
        cascades on wavelet dyadic trees. Phys. Rev. Letters, 101.
        """

        import pywt
        
        if self.original_data is not None:

            if pool is None:
                map_func = map
            else:
                map_func = pool.map
            
            if self.original_data.ndim > 1:
                orig_shape = self.original_data.shape
                self.original_data = np.reshape(self.original_data, (self.original_data.shape[0], np.prod(orig_shape[1:])))
            else:
                orig_shape = None
                self.original_data = self.original_data[:, np.newaxis]
            
            self.data = np.zeros_like(self.original_data)

            job_data = [ (i, self.original_data[:, i], randomise_from_scale, None) for i in range(self.original_data.shape[1]) ]
            job_results = map_func(_compute_MF_surrogates, job_data)
            
            for i, surr in job_results:
                self.data[:, i] = surr
            
            # squeeze single-dimensional entries (e.g. station data)
            self.data = np.squeeze(self.data)
            self.original_data = np.squeeze(self.original_data)

            # reshape back to original shape
            if orig_shape is not None:
                self.data = np.reshape(self.data, orig_shape)
                self.original_data = np.reshape(self.original_data, orig_shape)
            
        else:
            raise Exception("No data to randomise in the field. First you must copy some DataField.")
        


    def prepare_AR_surrogates(self, pool = None, order_range = [1, 1], crit = 'sbc'):
        """
        Prepare for generating AR(k) surrogates by identifying the AR model and computing
        the residuals. Adapted from script by Vejmelka -- https://github.com/vejmelkam/ndw-climate
        """
        
        if self.original_data is not None:
            
            if pool is None:
                map_func = map
            else:
                map_func = pool.map
                
            if self.original_data.ndim > 1:
                orig_shape = self.original_data.shape
                self.original_data = np.reshape(self.original_data, (self.original_data.shape[0], np.prod(orig_shape[1:])))
            else:
                orig_shape = None
                self.original_data = self.original_data[:, np.newaxis]
            num_tm = self.time.shape[0]
                
            job_data = [ (i, order_range, crit, self.original_data[:, i]) for i in range(self.original_data.shape[1]) ]
            job_results = map_func(_prepare_AR_surrogates, job_data)
            max_ord = 0
            for r in job_results:
                if r[1] is not None and r[1].order() > max_ord:
                    max_ord = r[1].order()
            num_tm_s = num_tm - max_ord
            
            self.model_grid = np.zeros((np.prod(orig_shape[1:]),), dtype = np.object)
            self.residuals = np.zeros((num_tm_s, np.prod(orig_shape[1:])), dtype = np.float64)
    
            for i, v, r in job_results:
                self.model_grid[i] = v
                if v is not None:
                    self.residuals[:, i] = r[:num_tm_s, 0]
                else:
                    self.residuals[:, i] = np.nan
    
            self.max_ord = max_ord
            
            self.original_data = np.squeeze(self.original_data)
            self.residuals = np.squeeze(self.residuals)

            # reshape back to original shape
            if orig_shape is not None:
                self.original_data = np.reshape(self.original_data, orig_shape)
                self.model_grid = np.reshape(self.model_grid, list(orig_shape[1:]))
                self.residuals = np.reshape(self.residuals, [num_tm_s] + list(orig_shape[1:]))
            
        else:
            raise Exception("No data to randomise in the field. First you must copy some DataField.")
        
        
        
    def construct_surrogates_with_residuals(self, pool = None):
        """
        Constructs a new surrogate time series from AR(k) model.
        Adapted from script by Vejmelka -- https://github.com/vejmelkam/ndw-climate
        """
        
        if self.model_grid is not None:
            
            if pool is None:
                map_func = map
            else:
                map_func = pool.map
            
            if self.original_data.ndim > 1:
                orig_shape = self.original_data.shape
                self.original_data = np.reshape(self.original_data, (self.original_data.shape[0], np.prod(orig_shape[1:])))
                self.model_grid = np.reshape(self.model_grid, np.prod(self.model_grid.shape))
                self.residuals = np.reshape(self.residuals, (self.residuals.shape[0], np.prod(orig_shape[1:])))
            else:
                orig_shape = None
                self.original_data = self.original_data[:, np.newaxis]
                self.model_grid = self.model_grid[:, np.newaxis]
                self.residuals = self.residuals[:, np.newaxis]
            num_tm_s = self.time.shape[0] - self.max_ord
            
            job_data = [ (i,  self.residuals[:, i], self.model_grid[i], num_tm_s, None) for i in range(self.original_data.shape[1]) ]
            job_results = map_func(_compute_AR_surrogates, job_data)
            
            self.data = np.zeros((num_tm_s, self.original_data.shape[1]))
            
            for i, surr in job_results:
                self.data[:, i] = surr
                    
            self.data = np.squeeze(self.data)
            self.original_data = np.squeeze(self.original_data)
            self.residuals = np.squeeze(self.residuals)

            # reshape back to original shape
            if orig_shape is not None:
                self.original_data = np.reshape(self.original_data, orig_shape)
                self.model_grid = np.reshape(self.model_grid, list(orig_shape[1:]))
                self.residuals = np.reshape(self.residuals, [num_tm_s] + list(orig_shape[1:]))
                self.data = np.reshape(self.data, [num_tm_s] + list(orig_shape[1:]))

        else:
           raise Exception("The AR(k) model is not simulated yet. First, prepare surrogates!") 



    def amplitude_adjust_surrogates(self, mean, var, trend, pool = None):
        """
        Performs amplitude adjustment to already created surrogate data. 
        """

        if self.data is not None and self.original_data is not None:

            if pool is None:
                map_func = map
            else:
                map_func = pool.map
            
            if self.original_data.ndim > 1:
                orig_shape = self.original_data.shape
                self.original_data = np.reshape(self.original_data, (self.original_data.shape[0], np.prod(orig_shape[1:])))
                self.data = np.reshape(self.data, (self.data.shape[0], np.prod(orig_shape[1:])))
                mean = np.reshape(mean, (mean.shape[0], np.prod(orig_shape[1:])))
                var = np.reshape(var, (var.shape[0], np.prod(orig_shape[1:])))
                trend = np.reshape(trend, (trend.shape[0], np.prod(orig_shape[1:])))
            else:
                orig_shape = None
                self.original_data = self.original_data[:, np.newaxis]
                self.data = self.data[:, np.newaxis]
                mean = mean[:, np.newaxis]
                var = var[:, np.newaxis]
                trend = trend[:, np.newaxis]
                
            old_shape = self.data.shape

            job_data = [ (i, self.original_data[:, i], self.data[:, i], mean[:, i], var[:, i], trend[:, i]) for i in range(self.original_data.shape[1]) ]
            job_results = map_func(_create_amplitude_adjusted_surrogates, job_data)

            self.data = np.zeros(old_shape)

            for i, AAsurr in job_results:
                self.data[:, i] = AAsurr

            # squeeze single-dimensional entries (e.g. station data)
            self.data = np.squeeze(self.data)
            self.original_data = np.squeeze(self.original_data)

            # reshape back to original shape
            if orig_shape is not None:
                self.original_data = np.reshape(self.original_data, orig_shape)
                self.data = np.reshape(self.data, [self.data.shape[0]] + list(orig_shape[1:]))

        else:
            raise Exception("No surrogate data or/and no data in the field. "
                            "Amplitude adjustment works on already copied data and created surrogates.")

