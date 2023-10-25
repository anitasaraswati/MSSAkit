# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 13:56:43 2023

@author: asaraswa
"""
import sys
from os import chdir
from struct import *
from zlib import DEF_MEM_LEVEL
import numpy as np
import scipy as sc
from numpy import *
from array import *
import numpy.matlib
from numpy.linalg import *
from matplotlib.pyplot import *
import matplotlib as mpl
from numpy.linalg import *
from scipy.interpolate import *
from scipy.signal import lfilter,lfilter_zi,periodogram
from scipy.fft import fft,fftshift
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
from scipy import fftpack

def dominant_freq(ts,fs=1,ncomp=None):
    nt = ts.shape[1]
    F = []
       
    if ncomp is None:
        ncomp = nt
        
    nfft = max([nt, 2**ceil(log2(nt))])
    
    for k in range(ncomp):
        fk,Pxx=periodogram(ts[:,k],fs=fs,nfft=nfft*4)
        fo=fk[argmax(Pxx)]
        F.append(fo)     
        
    F = np.array(F)
    
    return F

def dominant_period(ts,fs=1,ncomp=None,uncertainty=False,nsuro=100):
    nt = ts.shape[1]
    
    if uncertainty:
        sigma_T=[]
    
    if ncomp is None:
        ncomp = nt
        
    nfft = max([nt, 2**ceil(log2(nt))])
    
    F = dominant_freq(ts,fs,ncomp)
    T = 1/F
        
    
    if uncertainty:
        for k in range(ncomp):
            p_suro,fm = fft_surrogates(ts[:,k],fs,nsuro)
            pfm = 1/fm
            sigma_T.append(sqrt(mean(abs(pfm-T[k])**2)))
        return T, sigma_T    
    else:
        return T
    
    
def fft_surrogates(y, fs=1, nseed=100,
                          aggregate=max, random_seed=None,
                          normalization='standard'):
    rng = np.random.RandomState(random_seed)
    
    nftpc=max([y.shape[0],2**ceil(log2(y.shape[0]))])
    
    ts_fourier  = np.fft.rfft(y)
    
    def suro_ts():
        random_phases = np.exp(np.random.normal(0,np.pi*0.34,int(len(y)/2)+1)*1.0j)
        ts_fourier_new = ts_fourier*random_phases
        new_ts = np.fft.irfft(ts_fourier_new)
        
        f,power=periodogram(new_ts,fs=fs,nfft=nftpc*4)
        fm=f[argmax(power)]
        
        return aggregate(power), fm
    
    p=zeros(nseed)
    f=zeros(nseed)
    
    for i in range(nseed):
        p[i],f[i]=suro_ts()
    
    return p,f


