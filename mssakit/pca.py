# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 17:19:09 2023

@author: asaraswa
"""

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
from tqdm import tqdm
from scipy.signal import lfilter,periodogram
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
from statsmodels.tsa.ar_model import AutoReg

import mssakit.time_analysis as time

class pca:
    '''Performs Principal Component Analysis (PCA) on multivariate time series. 
    Multivariate time series should have observations in columns and time series
    indices in rows.
    
    
    Parameters
    ------------
    tseries : numpy.ndarray
        Time series array.
    
    t : numpy.array (optional)
        The time indicator of array ts.
        
    sampling_interval : int
        The sampling interval of the time series. The default is 1.
        
    w : numpy.ndarray
        The weight applied to each time series. If given, the matrix size should
        be the same as the size of the matrix tseries.
    
    
    Attributes
    ----------
    ts : numpy.ndarray
        The input of the time series.     
    
    w : numpy.ndarray
        The given weight of the time series.
       
    t : array
        The given time index of the time series. If t is not given, the time index will
        be replaced with the i-index point.
    
    fs : int
        The given sampling interval of the time series.
    
    u : numpy.ndarray
        The resulting left singular vectors from SVD of the trajectory matrix.
    
    v : numpy.ndarray
        The resulting right singular vectors from SVD of the trajectory matrix.
        
    s : numpy.array
        The resulting singular values from SVD of the trajectory matrix.
    
    eigenvalue : numpy.array
        The eigenvalues of the trajectory matrix (s ** 2).
    
    variance_captured_perc : numpy.array
        The percentage of the variance captured of each PC compared to the
        variance total of the original time series.
    
    PC : numpy.ndarray
        The PC of the PCA.
    
    '''
    
    
    def __init__(self, tseries, t=None, sampling_interval=1, w=None):
        
        nt,nd = tseries.shape
        
        ts = tseries.copy() 
        ts -= mean(ts, axis=0) 

        U, S, VT = svd(ts, full_matrices=False)
        
        d = S **2
        
        self.ts = ts.copy()
        self.u = U.copy()
        self.s = S.copy()
        self.v = transpose(VT)
        self.eigenvalue = d.copy()
        self.variance_captured_perc = d/sum(d)
        self.PC = real(U @ diag(S))
        self.w = w
        self.fs = sampling_interval
        self.t = t


    def reconstruct(self,n_components=None):
        '''Reconstruct the time series using the sum of the first n_components
        from the PCA
        

        Parameters
        ----------
        n_components : int
            The first n_components to reconstruct the time series. If not given,
            the function will sum all components.

        Returns
        -------
        RC : numpy.darray
            The reconstruction array of the time series.

        '''
        if n_components is None:
            n_components = len(self.s)
        else:
            n_components = min(n_components, len(self.s))
        
        RC = self.u[:,:n_components] @ diag(self.s[:n_components]) @ transpose(self.v[:,:n_components])
        
        return RC
    
    
    def plot(self,freq_rank=False):
        '''Plot a graph of eigenvalue of each i-rank or the dominant frequency
        of each PC.
        

        Parameters
        ----------
        freq_rank : boolean, optional
            Option to plot the dominant frequency (True) or the rank (False) of
            each eigenvalue. The default is False.

        '''
        
        if freq_rank:
            if not hasattr(self, 'F'):
                self.F = time.dominant_freq(self.PC,self.fs)
                
            xval = self.F            
            xlab = 'Frequency (1/yr)'
        else:
            xval = arange(len(self.eigenvalue))            
            xlab = 'Rank'
        
        figure()        
        plot(xval,self.eigenvalue,'k.')
        xlabel(xlab)
        ylabel('Eigenvalue')
        show()
        
    
    def sigtest(self,siglevel=95,Ns=10,pmax=1,procrustes=False,n_proscruste=None):    
        '''Performs significant test to the PCA using Monte Carlo hypothesis test
        based on the surrogates based of AR(p).
        

        Parameters
        ----------
        siglevel : int
            The significance level (%) in the Monte Carlo test, given between 0 and 100.
            The default is 95.
            
        Ns : int
            Number of surrogates. The default is 10.
            
        pmax : int
            The maximum p-rank of the AR(p). The default is 1.
            
        procrustes : boolean
            Option to apply procruste rotation in the Monte Carlo test.
            The default is False.
            
        n_proscruste : int
            The number of the components that are rotated in the procruste rotation.
            If not given, the function will rotate all components.
            
            
        Returns
        -------
        stats : numpy.ndarray
            The lower and the upper value of the Monte Carlo noise hypothesis.
            
        isig : numpy.array
            The index of the components that is significant.

        '''
        N,D = self.ts.shape
        
        p_order = zeros(D)
        arcoef = zeros(D)            
        
        r = zeros((N,D,Ns))
        np.random.seed(12345)

        R=zeros((N,D,Ns))
        for k in range(D):
            # generate random variable
            for n in range(Ns):
                temp=np.random.default_rng().normal(0, 1, size=(N))
                R[:,k,n]=temp        

        for k in range(D):        
            BIC = zeros(pmax+1)
            if pmax > 1:
                for i in range(pmax+1):
                    res=AutoReg(self.ts[:,k],lags=i,trend='n',old_names=False).fit()
                    BIC[i] = res.bic
                p_order[k] = int(np.argmin(BIC))
            else:
                p_order[k] = 1
                
            model_fit=AutoReg(self.ts[:,k],lags=p_order[k],trend='n',old_names=False).fit()
            
            arparams=np.array(real(model_fit.params))
            ar=np.r_[1,-arparams]
            resid=model_fit.resid
            
            for i in range(Ns):              
                temp = lfilter([1],ar,R[:,k,i],axis=0)                  
                r[:,k,i] = (temp-mean(temp))/std(temp)*std(resid)                
                
        if not (self.w is None):
            for i in range(Ns):
                r[:,:,i] = r[:,:,i] * self.w

        ds = zeros([N,Ns])
        
        if procrustes is True:
            if n_proscruste is None:
                n_proscruste = N
            
            Wn=zeros((N,Ns))
            for n in tqdm(range(Ns)):
                rs = r[:,:,n]
                usuro, ssuro, vtsuro = svd(rs, full_matrices=False)
                dsuro = ssuro **2                
                ds[:,n] = real(dsuro)
                
                pn = (usuro[:,:n_proscruste]@diag(sqrt(real(dsuro[:n_proscruste])))).T @ (self.u[:,:n_proscruste]@diag(sqrt(real(self.eigenvalue[:n_proscruste]))))
                up,sp,vp = svd(pn)
                Te = real(up) @ real(vp)
                Wn[:,n] = real(dsuro)
                Wn[:n_proscruste,n] = diag(Te.T@diag(real(dsuro[:n_proscruste]))@Te)  
            
            
            self.eigenvalue_suro = Wn                    
            
        else:
            
            for n in tqdm(range(Ns)):
                rs = r[:,:,n]                
                usuro, ssuro, vtsuro = svd(rs, full_matrices=False)
                dsuro = ssuro **2
                ds[:,n] = real(dsuro)
            
            self.eigenvalue_suro = ds
            
        stats = zeros((N,2))                
        stats[:,0] = percentile(self.eigenvalue_suro,siglevel,axis=1)
        stats[:,1] = percentile(self.eigenvalue_suro,100-siglevel,axis=1)
        
        isig = nonzero((self.eigenvalue - stats[:,0])>=0)[0]
        
        self.isig = isig
        self.siglevel = siglevel
        # self.stats = stats
        
        return stats, isig
        
    
    def plot_sigtest(self, ncomp=50, freq_rank=False):
        '''To plot the results of the Monte Carlo test of PCA.

        Parameters
        ----------
        ncomp : int, optional
            The first ncomp-rank of eigenvalues that are plotted. The default is 50.
        
        freq_rank : boolean, optional
            Option to plot the dominant frequency (True) or the rank (False) of
            each eigenvalue. The default is False.           

        '''
        Ns = self.eigenvalue_suro.shape[0]
        top_level = percentile(self.eigenvalue_suro,self.siglevel,axis=1)
        low_level = percentile(self.eigenvalue_suro,100-self.siglevel,axis=1)
        
        if freq_rank:
            if not hasattr(self, 'F'):
                self.F = time.dominant_freq(self.PC,self.fs)
                
            xval = self.F
            xerr = 0.01
            xlab = 'Frequency (1/yr)'
        else:
            xval = arange(len(self.eigenvalue))
            xerr = 0.05
            xlab = 'Rank'
            
            
        figure()                
        for k in range(ncomp):
            if k in self.isig:
                semilogy(xval[k],self.eigenvalue[k],'ko',markersize=11,fillstyle='none')
        plot(xval[:ncomp],self.eigenvalue[:ncomp],'ko')
        for k in range(ncomp):
            plot([xval[k],xval[k]],[low_level[k],top_level[k]],linewidth=0.7,color = 'black')
            errorbar(xval[k],low_level[k],xerr = 0.01,linewidth=0.7,ecolor = 'black')
            errorbar(xval[k],top_level[k],xerr = 0.01,linewidth=0.7,ecolor = 'black')
        xlabel(xlab,fontsize=12)
        ylabel('Eigenvalues',fontsize=12)            
        tight_layout()
        