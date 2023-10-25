# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 17:19:09 2023

@author: asaraswa
"""

import sys
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
import mssakit.toolbox as tool

class cross_svd:
    '''Performs cross Singular Value Decomposition (SVD) on a covariance matrix
    of two mulativariate datasets. Multivariate time series should have observations
    in columns and time series indices in rows.
    
    
    Parameters
    ------------
    A, B : numpy.ndarray
        Time series array.
    
    w_A, w_B : numpy.ndarray
        The weight applied to time series A and B. If given, the matrix size should
        be the same as the size of the matrix A and B, respectively.
    
    n_components : int
        The number of the first n_components to be returned for the output of RC.
    
    t : numpy.array (optional)
        The time indicator of array ts.
        
    sampling_interval : int
        The sampling interval of the time series. The default is 1.
        
    
    Attributes
    ----------
    A, B : numpy.ndarray
        The given time series.
    
    w_A, w_B : numpy.ndarray
        The given weight of the time series.
       
    t : array
        The given time index of the time series. If t is not given, the time index will
        be replaced with the i-index point.
    
    fs : int
        The given sampling interval of the time series.
    
    left_vectors : numpy.ndarray
        The resulting left singular vectors from SVD of the trajectory matrix.
    
    right_vectors : numpy.ndarray
        The resulting right singular vectors from SVD of the trajectory matrix.
        
    s : numpy.array
        The resulting singular values from SVD of the trajectory matrix.
    
    eigenvalue : numpy.array
        The eigenvalues of the trajectory matrix (s ** 2).
    
    variance_captured_perc : numpy.array
        The percentage of the variance captured of each PC compared to the
        variance total of the original time series.
    
    PC_A, PC_B : numpy.ndarray
        The PC of each datasets.
        
    RC_A, RC_B : numpy.ndarray
        Matrix of the reconstruction components of the first (A) and second (B)
        datasets.
    
    var_perc_A, var_perc_B : numpy.array
        The percentage of the variance captured of each components compared to its
        original datasets.
    '''
    
    def __init__(self, A, B, w_A=None, w_B=None, n_components=None,sampling_interval=1, t=[]):
        
        N,D = A.shape
        
        C = A.T @ B / (N-1)        

        U, S, VT = svd(C, full_matrices=False)
        
        d = S **2
        
        if n_components is None:
            n_components = len(S)
        else:
            n_components = min(n_components, len(S))
            
        
        self.A = A.copy()
        self.B = B.copy()
        self.w_A = w_A
        self.w_B = w_B
        self.left_vectors = real(U)
        self.s = real(S)
        self.right_vectors = real(transpose(VT))
        self.eigenvalue = real(d)
        self.variance_captured_perc = d/sum(d)
        self.PC_A = A @ U
        self.PC_B = B @ transpose(VT)                
        self.fs = sampling_interval
        
        if t is None:
            t = arange(self.PC_A.shape[0])
        else:
            t = squeeze(t)
        self.t = t
        
        self.reconstruct_comp(n_components=n_components)


    def reconstruct_comp(self,n_components):
        vc_A = zeros(n_components)
        vc_B = zeros(n_components)
        RC_A = zeros((self.A.shape[0],self.A.shape[1],n_components))
        RC_B = zeros((self.B.shape[0],self.B.shape[1],n_components))
        
        for n in range(n_components):            
            RC_A[:,:,n], RC_B[:,:,n] = self.reconstruct(n)
            
            vc_A[n] = var(RC_A[:,:,n])/var(self.A)
            vc_B[n] = var(RC_B[:,:,n])/var(self.B)
            
        self.var_perc_A=vc_A
        self.var_perc_B=vc_B
        
        self.RC_A = RC_A
        self.RC_B = RC_B
        
        
    def reconstruct(self,n):
        '''Obtain the reconstruction matrix of selected component
        

        Parameters
        ----------
        n : int
            The n-rank of the component to be reconstructed.

        Returns
        -------
        R_A : numpy.ndarray
            Reconstruction matrix of the n-th components of the first (A) dataset.
        R_B : numpy.ndarray
            Reconstruction matrix of the n-th components of the second (B) dataset.
        '''
        
        R_A = reshape(self.PC_A[:,n],[-1,1]) @ reshape(self.left_vectors[:,n].T,[1,-1])
        R_B = reshape(self.PC_B[:,n],[-1,1]) @ reshape(self.right_vectors[:,n].T,[1,-1])
        
        return R_A, R_B
    
    
    def plot(self):
        '''Plot a graph of eigenvalue of each i-rank or the dominant frequency
        of each PC.
        '''
        figure()
        plot(self.eigenvalue,'k.')
        xlabel('Rank')
        ylabel('Eigenvalue')
        show()
        
    
    def plot_pc(self, comps=[0]):
        '''Plot the PCs of the selected n-rank components.        

        Parameters
        ----------
        comps : list int
            The list of the PC rank to be plotted. The default is [0].
        
        ''' 
        fig, axs = subplots(int(len(comps)), layout="constrained")
        
        if len(comps)>1:
            axs = axs.ravel(order='F')
            for k,c in enumerate(comps):
                axs[k].plot(t,self.PC_A[:,c],'blue', label='PC_A')
                axs[k].plot(t,self.PC_B[:,c],'red', label='PC_B')
                axs[k].grid()            
                axs[k].legend()
                axs[k].set_title('PC '+str(int(c)))
        else:
            axs.plot(t,self.PC_A[:,comps],'blue', label='PC_A')
            axs.plot(t,self.PC_B[:,comps],'red', label='PC_B')
            axs.grid()            
            axs.legend()
            axs.set_title('PC '+str(comps))
        show()
    
    def sigtest(self,siglevel=95,Ns=10,pmax=1):    
        '''Performs significant test to the cross SVD using Monte Carlo hypothesis test
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
            
            
        Returns
        -------
        stats : numpy.ndarray
            The lower and the upper value of the Monte Carlo noise hypothesis.
            
        isig : numpy.array
            The index of the components that is significant.

        '''
        
        N,D = self.A.shape
        
        p_order_A = zeros(D)      
        p_order_B = zeros(D)              
        r_A = zeros((N,D,Ns))
        r_B = zeros((N,D,Ns))
                
        np.random.seed(12345)

        R=zeros((N,D,Ns))
        for k in range(D):
            # generate random variable
            for n in range(Ns):
                temp=np.random.default_rng().normal(0, 1, size=(N))
                R[:,k,n]=temp        

        for k in range(D):        
            # get AR parameter
            BIC_A=zeros(pmax+1)
            BIC_B=zeros(pmax+1)
            if pmax > 1:
                for i in range(pmax+1):
                    res = AutoReg(self.A[:,k],lags=i,trend='n',old_names=False).fit()
                    BIC_A[i] = res.bic
                    res = AutoReg(self.B[:,k],lags=i,trend='n',old_names=False).fit()
                    BIC_B[i] = res.bic                    
                p_order_A[k] = int(np.argmin(BIC_A))
                p_order_B[k] = int(np.argmin(BIC_B))
            else:
                p_order_A[k] = 1
                p_order_B[k] = 1
            
            model_fit=AutoReg(self.A[:,k],lags=p_order_A[k],trend='n',old_names=False).fit()
            arparams=model_fit.params
            ar=np.r_[1,-arparams]
            resid=model_fit.resid
            for i in range(Ns):
                temp = lfilter([1],ar,R[:,k,i],axis=0) 
                r_A[:,k,i] = (temp-mean(temp))/std(temp)*std(resid)
            
            model_fit=AutoReg(self.B[:,k],lags=p_order_B[k],trend='n',old_names=False).fit()
            arparams=model_fit.params
            ar=np.r_[1,-arparams]
            resid=model_fit.resid
            for i in range(Ns):
                temp = lfilter([1],ar,R[:,k,i],axis=0) 
                r_B[:,k,i] = (temp-mean(temp))/std(temp)*std(resid)
                
        if not (self.w_A is None):
            for i in range(Ns):
                r_A[:,:,i] = r_A[:,:,i] * self.w_A
                r_B[:,:,i] = r_B[:,:,i] * self.w_B
                
        ds = zeros([D,Ns])
        Wn = zeros((N,Ns))                        
        for n in tqdm(range(Ns)):
            rs_A = r_A[:,:,n]
            rs_B = r_B[:,:,n]
            Cs = rs_A.T@rs_B/(N-1)
            us,ss,vs = svd(Cs,full_matrices=False)
            vs=real(vs.T)
            ks=flipud(argsort(ss))
            us=us[:,ks] #temporal eof
            vs=vs[:,ks] #spatial eof
            ss=real(ss[ks]) #singular value
            ds[:,n]=ss**2
                
        self.eigenvalue_suro = ds                    
            
        stats = zeros((D,2))                
        stats[:,0] = percentile(self.eigenvalue_suro,siglevel,axis=1)
        stats[:,1] = percentile(self.eigenvalue_suro,100-siglevel,axis=1)
        
        isig = nonzero((self.eigenvalue - stats[:,0])>=0)[0]
        
        self.isig = isig
        self.siglevel = siglevel
        # self.stats = stats
        
        return stats, isig
       
    
    def plot_sigtest(self, ncomp=50, freq_rank=False):
        '''To plot the results of the Monte Carlo test of cross SVD.

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
            if not hasattr(self, 'F_A'):
                self.F_A = time.dominant_freq(self.PC_A,self.fs)
            if not hasattr(self, 'F_B'):
                self.F_B = time.dominant_freq(self.PC_B,self.fs)
                
            xval_A = self.F_A
            xval_B = self.F_B
            xerr = 0.01
            xlab = 'Frequency (1/yr)'
        else:
            xval_A = arange(len(self.eigenvalue))
            xval_B = copy(xval_A)
            xerr = 0.05
            xlab = 'Rank'
            
        figure()                
        subplot(1,2,1)
        for k in range(ncomp):
            if k in self.isig:
                semilogy(xval_A[k],self.eigenvalue[k],'ko',markersize=11,fillstyle='none')
        plot(xval_A[:ncomp],self.eigenvalue[:ncomp],'ko')
        for k in range(ncomp):
            plot([xval_A[k],xval_A[k]],[low_level[k],top_level[k]],linewidth=0.7,color = 'black')
            errorbar(xval_A[k],low_level[k],xerr = 0.01,linewidth=0.7,ecolor = 'black')
            errorbar(xval_A[k],top_level[k],xerr = 0.01,linewidth=0.7,ecolor = 'black')
        xlabel(xlab,fontsize=12)
        ylabel('Eigenvalues',fontsize=12)    

        subplot(1,2,2)
        for k in range(ncomp):
            if k in self.isig:
                semilogy(xval_B[k],self.eigenvalue[k],'ko',markersize=11,fillstyle='none')
        plot(xval_B[:ncomp],self.eigenvalue[:ncomp],'ko')
        for k in range(ncomp):
            plot([xval_B[k],xval_B[k]],[low_level[k],top_level[k]],linewidth=0.7,color = 'black')
            errorbar(xval_B[k],low_level[k],xerr = 0.01,linewidth=0.7,ecolor = 'black')
            errorbar(xval_B[k],top_level[k],xerr = 0.01,linewidth=0.7,ecolor = 'black')
        xlabel(xlab,fontsize=12)
        ylabel('Eigenvalues',fontsize=12)    
        tight_layout()