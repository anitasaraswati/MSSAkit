#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 11:38:13 2022

@author: saraswati
"""

from scipy.optimize import fminbound    

from os import chdir
from struct import *
import numpy as np
import random as ra
from numpy import *
import scipy as sc
import pandas as pd
from scipy.stats import t
from numpy import fft as fft
import matplotlib.pyplot as plt
from numpy.linalg import *
from numpy.random import *
from scipy.interpolate import interp1d
# from scipy.linalg import *
from scipy.io import *
from scipy.signal import lfilter
from scipy.optimize import fmin
from scipy import fftpack
from scipy.stats import ttest_ind as ttest,t,pearsonr

def dominant_freqs(E):
    """Computes the dominant frequencies of the column vectors
    of the input matrix using FFT.
    Args:
        E (numpy matrix): Matrix of eigenvectors (as columns)
    Returns:
        list of floats
    """

    nfft = 2 ** 11
    fft = np.fft.fft(E, axis=0, n=nfft)
    fft = fft[0:nfft // 2]
    fft = np.abs(fft)**2
    freq = np.fft.fftfreq(nfft)
    freq = freq[0:nfft // 2]
    freqs = freq[fft.argmax(axis=0)]
    return freqs
    
def eot(x,retain_method='captvar', cv=1,nmode=8,mv=0.01, **kwargs):
    N,K=x.shape
    nc=nmode
    pc=zeros((nc,K))
    u=zeros((N,nc))
    w=zeros(nc+1)
    i=0
    v0=sum(var(x,axis=0))
    ocv=0
    while ((ocv<cv) & (i<nmode)):
        mav=0
        Ve=zeros(K)
        for k in range(K):
            if std(x[:,k])!=0:
                A=c_[ones(N),x[:,k]]
                yy,e,aa,bb=lstsq(A,x,rcond=None)
                Ve[k]=(sum(var(x,axis=0))-sum(var(x-A@yy,axis=0)))/v0
        kma=argmax(Ve)
        u[:,i]=x[:,kma]
        A=c_[ones(N),x[:,kma]]
        yy,e,aa,bb=lstsq(A,x,rcond=None)
        x=x-A@yy
        ve=(v0-sum(var(x,axis=0)))/v0
        pc[i,:]=yy[1,:]
        w[i+1]=ve
        ocv=sum(diff(w[:i+2]))
        i+=1
    u=u[:,:i]
    pc=pc[:i,:]
    # w=w[:i]
    u,pc=pc,u
    return u,pc,w,Ve,diff(w)

def dof(x,y):
    acorx = autocorr(x)
    acory = autocorr(y)
    N=x.size
    ni=flip(array(range(0,N))+1)
    neff=N/(1 + (2*sum(ni/N * acorx * acory)))
    return neff

def autocorr(x):
    xt=x-mean(x)
    vx=var(x)
    acor=correlate(xt, xt, 'full')[len(xt)-1:]
    acor = acor / vx / len(xt)
    return acor

def pearson_corr_stat(x,y,nlag=100,si=0.95):
    xcx=ones(nlag+1)
    xcy=ones(nlag+1)
    xx=(x-mean(x))/std(x)
    yy=(y-mean(y))/std(y)
    for k in range(1,nlag+1):
        xcx[k]=sum(xx[k:]*xx[:-k])/len(xx)
        xcy[k]=sum(yy[k:]*yy[:-k])/len(yy)
    n1=nonzero(xcx<=exp(-1))[0][0]
    n2=nonzero(xcy<=exp(-1))[0][0]
    n=sqrt(len(xx)/n1*len(yy)/n2)
    rv=t(int(n-2))
    r=sum((x-mean(x))*(y-mean(y)))/(len(x)*std(x)*std(y))
    to=r*sqrt(int(n-2))/sqrt(1-r**2)
    cd=rv.cdf(to)
    cd2=2*cd-1
    
    si2=(si+1)/2
    t_si=abs(t.ppf(si2,int(n-2)))
    r_si=sqrt(t_si**2/(int(n-2)+t_si**2))
    is_sig=abs(to)>=abs(t_si)
    
    return r,cd2,n,r_si,is_sig

def corr_significance(x, y, alpha=0.05):
    n = len(x)
    r = np.corrcoef(x, y)[0, 1]
    acf = np.correlate(x - x.mean(), y - y.mean(), mode='full')
    acf /= np.var(x) * n
    std_err = np.sqrt((1-r**2) / (n-2)) / np.sqrt(acf[n-1])
    t_val = r / std_err
    p_val = t.sf(np.abs(t_val), n-2) * 2
    is_sig = p_val < alpha
    return r, p_val, is_sig

def pearson_corr_stat3d(xx,yy,nlag=100,si=0.95):
    xcx=ones(nlag+1)
    xcy=ones(nlag+1)
    x=(xx-mean(xx))/std(xx)
    
    N=len(x)
    nd=yy.shape[1]
    r=zeros(nd)
    cd2=zeros(nd)
    n=zeros(nd)
    r_si=zeros(nd)
    is_sig=zeros(nd)    
    
    for i in range(nd):
        y=(yy[:,i]-mean(yy[:,i]))/std(yy[:,i])
        # y=copy(yy[:,i])
        cov=mean((x - np.mean(x)) * (y - np.mean(y)))
        lag_cov = [np.mean((x[:N-t] - np.mean(x[:N-t])) * (y[t:N] - np.mean(y[t:N]))) for t in range(1, nlag+1)]
        lag_cov = np.array(lag_cov)
    
        # Calculate the effective sample size and the corrected covariance
        eff_sample_size = N / (1 + 2 * np.sum(lag_cov))
        corr_cov = cov * np.exp(-2 * np.sum(lag_cov))
    
        # Calculate the correlation coefficient and its standard error
        corr_coef = corr_cov / (np.std(x) * np.std(y))
        std_err = np.sqrt((1 - corr_coef**2) / (eff_sample_size - 2))
    
        # Calculate the t-statistic and p-value
        t_stat = corr_coef / std_err
        dof = eff_sample_size - 2
        p_value = 2 * (1 - t.cdf(abs(t_stat), dof))
        is_sig[i]=p_value<0.05
        r[i]=corr_coef
    
    return r,is_sig


def pearson_corr_stat3(x,y,nlag=100,si=0.95):
    xcx=ones(nlag+1)
    xcy=ones(nlag+1)
    xx=(x-mean(x))/std(x)
    
    nd=y.shape[1]
    r=zeros(nd)
    cd2=zeros(nd)
    n=zeros(nd)
    r_si=zeros(nd)
    is_sig=zeros(nd)    
    
    for i in range(nd):
        yy=(y[:,i]-mean(y[:,i]))/std(y[:,i])
        for k in range(1,nlag+1):
            xcx[k]=sum(xx[k:]*xx[:-k])/len(xx)
            xcy[k]=sum(yy[k:]*yy[:-k])/len(yy)
        n1=nonzero(xcx<=exp(-1))[0][0] #*2
        n2=nonzero(xcy<=exp(-1))[0][0] #*2
        n[i]=sqrt(len(xx)/n1*len(yy)/n2)
        rv=t(int(n[i]-2))
        r[i]=sum((x-mean(x))*(y[:,i]-mean(y[:,i])))/(len(x)*std(x)*std(y[:,i]))
        to=r[i]*sqrt(int(n[i]-2))/sqrt(1-r[i]**2)
        cd=rv.cdf(to)
        cd2[i]=2*cd-1
        
        si2=(si+1)/2
        t_si=abs(t.ppf(si2,int(n[i]-2)))
        r_si[i]=sqrt(t_si**2/(int(n[i]-2)+t_si**2))
        is_sig[i]=abs(to)>=abs(t_si)
    
    return r,cd2,n,r_si,is_sig

def pearson_corr_stat3_lag(x,y,nlag=100,si=0.95):        
    
    # xcx=ones(nlag+1)
    # xcy=ones(nlag+1)
    # xx=(x-mean(x))/std(x)
    
    # nd=y.shape[1]
    # r=zeros(nd)
    # pp=zeros(nd)
    # cd2=zeros(nd)
    # n=zeros(nd)
    # r_si=zeros(nd)
    # is_sig=zeros(nd)   
    # mlag=zeros(nd)
    
    # idlag=arange(-nlag,nlag)
    
    # si2=(si+1)/2
    
    # for i in range(nd):
    #     ccoef=[]
    #     pval=[]
        
    #     yy=(y[:,i]-mean(y[:,i]))/std(y[:,i])
    #     n=zeros(2*nlag+1)
        
        
    #     for lag in range(-nlag,nlag):
    #         yyy=np.roll(yy,lag)
    #         # for k in range(1,nlag+1):
    #         #     xcx[k]=sum(xx[k:]*xx[:-k])/len(xx)
    #         #     xcy[k]=sum(yyy[k:]*yyy[:-k])/len(yyy)
    #         # n1=nonzero(xcx<=exp(-1))[0][0] #*2
    #         # n2=nonzero(xcy<=exp(-1))[0][0] #*2
    #         # n[lag]=sqrt(len(xx)/n1*len(yyy)/n2)
    #         # rv=t(int(n[lag]-2))
    #         cc,p = pearsonr(xx,yyy)
            
    #         ccoef.append(cc)
    #         pval.append(p)
            
            
            
        
    #     maxccid=np.argmax(np.abs(ccoef))
    #     r[i]=ccoef[maxccid]
    #     pp[i]=pval[maxccid]
    #     mlag[i]=idlag[maxccid]
        
    #     # to=r[i]*sqrt(int(n[maxccid]-2))/sqrt(1-r[i]**2)
    #     # cd=rv.cdf(to)
    #     # cd2[i]=2*cd-1
    #     # t_si=abs(t.ppf(si2,int(n[maxccid]-2)))
        
        
    #     # r_si[i]=sqrt(t_si**2/(int(n[i]-2)+t_si**2))
    #     is_sig[i]=pp[i]<0.05
        
    #     # for lag in range(-nlag,nlag):
    #     #     yyy=np.roll(yy,lag)
    #     #     for k in range(1,nlag+1):
    #     #         xcx[k]=sum(xx[k:]*xx[:-k])/len(xx)
    #     #         xcy[k]=sum(yyy[k:]*yyy[:-k])/len(yyy)
    #     #     n1=nonzero(xcx<=exp(-1))[0][0] #*2
    #     #     n2=nonzero(xcy<=exp(-1))[0][0] #*2
    #     #     n[lag]=sqrt(len(xx)/n1*len(yyy)/n2)
    #     #     rv=t(int(n[i]-2))
    #     #     cc,p = pearsonr(xx,yyy)
            
    #     #     ccoef.append(cc)
    #     #     pval.append(p)
            
            
            
        
    #     # maxccid=np.argmax(np.abs(ccoef))
    #     # r[i]=ccoef[maxccid]
    #     # pp[i]=pval[maxccid]
    #     # mlag[i]=idlag[maxccid]
        
    #     # to=r[i]*sqrt(int(n[maxccid]-2))/sqrt(1-r[i]**2)
    #     # cd=rv.cdf(to)
    #     # cd2[i]=2*cd-1
    #     # t_si=abs(t.ppf(si2,int(n[maxccid]-2)))
        
        
    #     # # r_si[i]=sqrt(t_si**2/(int(n[i]-2)+t_si**2))
    #     # is_sig[i]=abs(to)>=abs(t_si)
    
    # return r,pp,mlag,is_sig


    # Determine the time lags corresponding to the correlation values
    lags = arange(-len(x)+1, len(x))
    midlag = np.where(lags == 0)[0]
    icomp = arange(midlag-200,midlag+200)
    # lags = arange(-nlag,nlag)
    xcx=ones(nlag+1)
    xcy=ones(nlag+1)
    xx=(x-mean(x))/std(x)
          
    nd=y.shape[1]
    r=zeros(nd)    
    is_sig=zeros(nd)    
    max_lag=zeros(nd)
    
    for i in range(nd):        
        yy=(y[:,i]-mean(y[:,i]))/std(y[:,i])
        corr = np.correlate(xx, yy, mode='full')
        # corr = signal.correlate(yy, xx, mode='same') / np.sqrt(signal.correlate(xx, xx, mode='same')[int(nd/2)] * signal.correlate(yy, yy, mode='same')[int(nd/2)])

        # Find the index of the maximum correlation value
        
        max_idx = np.argmax(corr)
        
        # Compute the time lag corresponding to the maximum correlation value
        max_lag[i] = lags[max_idx]
        
        # Compute the number of degrees of freedom
        for k in range(1,nlag+1):
            xcx[k]=sum(xx[k:]*xx[:-k])/len(xx)
            xcy[k]=sum(yy[k:]*yy[:-k])/len(yy)
        n1=nonzero(xcx<=exp(-1))[0][0] #*2
        n2=nonzero(xcy<=exp(-1))[0][0] #*2
        
        df=sqrt(len(x)/n1*len(yy)/n2)
        
        # df = len(x) - abs(max_lag)
        
        # Compute the critical value for the Student's t-test at the 5% significance level
        # t_crit = t.ppf(0.975, df)
        si2=(si+1)/2
        t_si=abs(t.ppf(si2,int(df-2)))
        
        # Compute the correlation at the maximum lag
        rr = corr[max_idx]
        
        to=rr*sqrt(int(df-2))/sqrt(1-rr**2)
        
        # Compute the standard error of the correlation
        # se = np.sqrt((1-rr**2)/(df-2))
        
        # Compute the lower and upper bounds of the confidence interval
        # ci_lower = rr - t_crit * se
        # ci_upper = rr + t_crit * se
        
        r[i]=copy(rr)
        
        # print(rr)
        # print(t_crit)
        # print(df)
        
        # Print the significance test results
        # if rr < ci_lower or rr > ci_upper:
        #     is_sig[i]=1
        
        is_sig[i]=abs(to)>=abs(t_si)
    
    return r,max_lag,is_sig


def pearsonr_ci(x,y,alpha=0.05,nlag=100):
    ''' calculate Pearson correlation along with the confidence interval using scipy and numpy
    Parameters https://zhiyzuo.github.io/Pearson-Correlation-CI-in-Python/
    ----------
    x, y : iterable object such as a list or np.array
      Input for correlation calculation
    alpha : float
      Significance level. 0.05 by default
    Returns
    -------
    r : float
      Pearson's correlation coefficient
    pval : float
      The corresponding p value
    lo, hi : float
      The lower and upper bound of confidence intervals
    '''

    r, p = sc.stats.pearsonr(x,y)
    r_z = np.arctanh(r)
    xcx=ones(nlag+1)
    xcy=ones(nlag+1)
    xx=(x-mean(x))/std(x)
    yy=(y-mean(y))/std(y)
    for k in range(1,nlag+1):
        xcx[k]=sum(xx[k:]*xx[:-k])/len(xx)
        xcy[k]=sum(yy[k:]*yy[:-k])/len(yy)
    n1=nonzero(xcx<=exp(-1))[0][0]
    n2=nonzero(xcy<=exp(-1))[0][0]
    n=sqrt(len(xx)/n1*len(yy)/n2)
    # se = 1/np.sqrt(x.size-3)
    se = 1/np.sqrt(int(n))
    z = sc.stats.norm.ppf(1-alpha/2)
    lo_z, hi_z = r_z-z*se, r_z+z*se
    lo, hi = np.tanh((lo_z, hi_z))
    return r, p, lo, hi, z