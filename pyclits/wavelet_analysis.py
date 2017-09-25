"""
created on Feb 3, 2014

@author: Nikola Jajcay, jajcay(at)cs.cas.cz
inspired by A Practical Guide to Wavelet Analysis by Ch. Torrence and G. Compo
-- http://paos.colorado.edu/research/wavelets/ --
"""

import numpy as np
from scipy.fftpack import fft, ifft
from scipy.special import gamma


def morlet(k, scale, k0 = 6.):
    """
    Returns the Morlet wavelet function as a function of Fourier frequency,
    used for the wavelet transform in Fourier space.
    
    Morlet wavelet: psi(x) = pi^(-1/4) * exp(i*k0*x) * exp(-x^2 / 2)
    
    inputs:
    k - numpy array with Fourier frequencies at which to calculate the wavelet
    scale - the wavelet scale
    k0 - wavenumber
    """
    
    exponent = - np.power((scale * k - k0),2) / 2. * (k > 0.)
    norm = np.sqrt(scale * k[1]) * (np.power(np.pi, -0.25)) * np.sqrt(len(k))
    output = norm * np.exp(exponent)
    output *= (k > 0.)
    fourier_factor = (4 * np.pi) / (k0 + np.sqrt(2 + np.power(k0,2)))
    coi = fourier_factor / np.sqrt(2.)
    
    return output, fourier_factor, coi
    
    
    
def paul(k, scale, k0 = 4.):
    """
    Returns the Paul wavelet function as a function of Fourier frequency,
    used for the wavelet transform in Fourier space.
    
    Paul wavelet: psi(x) = (2^m * i^m * m!) / sqrt(pi * (2m)!) * (1 - ix)^(-m + 1)
    
    inputs:
    k - numpy array with Fourier frequencies at which to calculate the wavelet
    scale - the wavelet scale
    k0 - order
    """
    
    exponent = - np.power((scale * k),2) * (k > 0.)
    norm = np.sqrt(scale * k[1]) * (np.power(2,k0) / np.sqrt(k0 * np.prod(np.arange(2,2*k0)))) * np.sqrt(len(k))
    output = norm * np.power((scale * k),k0) * np.exp(exponent)
    output *= (k > 0.)
    fourier_factor = (4 * np.pi) / (2 * k0 + 1)
    coi = fourier_factor * np.sqrt(2.)
    
    return output, fourier_factor, coi
    
    
    
def DOG(k, scale, k0 = 2.):
    """
    Returns the Derivative of Gaussian wavelet function as a function of Fourier frequency,
    used for the wavelet transform in Fourier space. For m = 2 this wavelet is the Marr or
    Mexican hat wavelet.
    
    DOG wavelet: psi(x) = (-1)^(m+1) / sqrt (gamma(m+1/2)) * (d^m / dx^m) exp(-x^2 / 2)
    
    inputs:
    k - numpy array with Fourier frequencies at which to calculate the wavelet
    scale - the wavelet scale
    k0 - derivative
    """
    
    exponent = - np.power((scale * k),2) / 2.
    norm = np.sqrt(scale * k[1] / gamma(k0 + 0.5)) * np.sqrt(len(k))
    output = - norm * np.power(1j,k0) * np.power((scale * k),k0) * np.exp(exponent)
    fourier_factor = 2 * np.pi * np.sqrt(2 / (2 * k0 + 1))
    coi = fourier_factor / np.sqrt(2.)
    
    return output, fourier_factor, coi
    
    
    
def continous_wavelet(X, dt, pad = False, wavelet = morlet, **kwargs):
    """
    Computes the wavelet transform of the vector X, with sampling rate dt.
    
    inputs:
    X - the time series, numpy array
    dt - sampling time of dt
    pad - if True, pad time series with 0 to get len(X) up to the next higher power of 2. It speeds up the FFT.
    wavelet - which mother wavelet should be used. (morlet, paul, DOG)
    --- kwargs ---
    dj - the spacing between discrete scales.
    s0 - the smallest scale of the wavelet
    j1 - the number of scales minus one. Scales range from s0 up to s0 * 2^(j1+dj) to give a total of j1+1 scales. 
    k0 - parameter of Mother wavelet: Morlet - wavenumber, Paul - order, DOG - derivative
    
    outputs:
    wave - wavelet transform of the X. It is a complex numpy array of dim (n, j1+1)
    period - the vector of Fourier periods in time units
    scale - the vector of scale indices, given by s0 * 2^(j*dj)
    coi - Cone-of-Influence, vector that contains a maximum period of useful information at particular time
    """
    # map arguments
    if 'dj' in kwargs:
        dj = kwargs['dj']
    else:
        dj = 0.25
    if 's0' in kwargs:
        s0 = kwargs['s0']
    else:
        s0 = 2 * dt
    if 'j1' in kwargs:
        j1 = np.int(kwargs['j1'])
    else:
        j1 = np.fix(np.log(len(X)*dt/s0) / np.log(2)) / dj
    if 'k0' in kwargs:
        k0 = kwargs['k0']
    else:
        k0 = 6.
    
    n1 = len(X)
    
    Y = X - np.mean(X)
    #Y = X

    # padding, if needed
    if pad:
        base2 = int(np.fix(np.log(n1)/np.log(2) + 0.4999999)) # power of 2 nearest to len(X)
        Y = np.concatenate( (Y, np.zeros((np.power(2, (base2+1))-n1))) )
    n = len(Y)
    
    # wavenumber array
    k = np.arange(1, np.fix(n/2) + 1)
    k *= (2. * np.pi) / (n * dt)
    k_minus = -k[int(np.fix(n-1))/2 - 1::-1]
    k = np.concatenate((np.array([0.]), k, k_minus))
    
    # compute FFT of the (padded) time series
    f = fft(Y)
    
    # construct scale array and empty period & wave arrays
    scale = np.array( [s0 * np.power(2, x*dj) for x in range(0,j1+1)] )
    period = scale
    wave = np.zeros((j1+1, n), dtype = np.complex)
    
    # loop through scales and compute tranform
    for i in range(j1+1):
        daughter, fourier_factor, coi = wavelet(k, scale[i], k0)
        wave[i, :] = ifft(f * daughter)
        
    period = fourier_factor * scale
    coi *= dt * np.concatenate( (np.array([1e-5]), np.arange(1,(n1+1)/2), np.arange((n1/2 - 1),0,-1), np.array([1e-5])) )
    wave = wave[:, :n1]
    
    return wave, period, scale, coi
    