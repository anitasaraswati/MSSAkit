import numpy as np
import scipy as sc
from numpy import *
from array import *
from numpy.linalg import *
from scipy.linalg import *
from matplotlib.pyplot import *
import matplotlib as mpl
from numpy.linalg import *
from scipy.interpolate import *
from tqdm import tqdm
from scipy.signal import lfilter,periodogram
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)

from mssakit.toolbox import tool
import mssakit.time_analysis as time


class ssa:
    '''Perform SSA decomposition on a univariate timeseries.
    
    Paramaters
    -----------
    ts : numpy.ndarray
        Time series array.
        
    M : int | None
        The window size parameter controls the dimensionality of the trajectory
        matrices constructed for each timeseries (and then stacked). Timeseries
        are converted into trajectory matrices through "hankelization", where
        columns and rows represent different "windows" of the timeseries,
        incrementing across the timeseries. With window_size = M, the resulting
        trajectory matrix of a timeseries vector (N) will be of shape (M, K), where
        K = N - M + 1. As such, window_size should be no greater than N // 2. If
        left as None, MSSA will select N // 3.
    
    co : string | 'VG' | 'BH'
        Option of method to compute the covariance of the trajectory matrix,
        VG (after Vautard and Ghil, 1989) applies Toeplitz approach,
        while BH (after  Broomhead and King, 1986) applies the method of delay
        or trajectory approach. The default is VG.
    
    ncomp : int
        Number of the components that are kept for the output. The default is 8.
        
    t : numpy.array (optional)
        The time indicator of array ts. If t is not given, the time index will
        be replaced with the i-index point.
        
    sampling_interval : int
        The sampling interval of the time series. The default is 1.
    
    
    Attributes
    ----------
    ts : numpy.ndarray
        The input of the time series.     
      
    t : array
        The given time index of the time series.
    
    sampling_interval : int
        The given sampling interval of the time series.
    
    M : int
        The given windows size used in the MSSA.        
        
    co : string
        The choosen method to calculate the covariance matrix.
        
    eigenvectors : numpy.ndarray
        The resulting eigenvectors of the covariance of the trajectory matrix.
    
    eigenvalue : numpy.array
        The eigenvalues of the trajectory matrix.
    
    PC : numpy.ndarray
        The PC from the SSA.
    
    RC : numpy.ndarray
        The reconstruction of SSA in the original time series. The dimension of this 
        matrix is (N, rank), where N is the number of observations and rank is 
        the number of the selected components.
    
    F : numpy.array
        The dominant frequency of the PC.
    
    T : numpy.array
        The dominant period of the PC.
    
    sigma_T : numpy.array
        The period uncertainty of the obtained dominant period.
        
    '''
    def __init__(self, ts,
                 M=None,                 
                 co='VG',
                 ncomp=8,
                 t=None,
                 sampling_interval=1):        
            
        ts = tool.demean(ts)        
        N = ts.size
        if M == 0:
          M = N // 3
        if co == 'VG':
           c = tool.covaVG(ts,M)
        elif co == 'BH':
           c = tool.covaBH(ts,M)
        D,V = eig(c)
        ks = flipud(argsort(D))
        V = V[:,ks]
        D = real(D[ks])
        ncomp = range(0,ncomp+1)
        ts = reshape(ts,(N,1))
        pc = tool.compPCm(ts,M,V,Nc=ncomp)
        R = tool.compRCm(ts,M,V,pc,Nc=ncomp)
        
        self.F = time.dominant_freq(pc,fs=sampling_interval)
        self.T, self.sigma_T = time.dominant_period(pc,fs=sampling_interval,uncertainty=True)
        
        self.ts = ts
        self.t = squeeze(t)
        self.sampling_interval = sampling_interval
        self.M = M
        self.co = co        
        self.eigenvectors = V
        self.eigenvalue = D
        self.PC = pc
        self.RC = R
        
    def mcssa(self,siglevel=95,Ns=10,procruste=False,norotate=[]):
        '''Applying Monte Carlo significant test to the SSA
        

        Parameters
        ----------
        siglevel : int
            The significance level (%) in the Monte Carlo test, given between 0 and 100.
            The default is 95.
            
        Ns : int
            Number of surrogates. The default is 10.
            
        procruste : boolean
            Option to apply procruste rotation in the Monte Carlo test.
            The default is False.
            
        norotate : list
            The list of index of non-rotated T-EOF if procruste is selected.
            The default is []
            
            
        Atributes
        ---------
        siglevel : int
            The given signigicance level (in %) in the MC test
        
        isig : list
            The index of SSA component that is above the significance level
        
        eigenvalue_suro : numpy.ndarray
            The eigenvalue of the surrogates            
        '''
        self.siglevel = siglevel
        
        if not procruste:
             ds = zeros((self.M,Ns))          
             t = arange(self.M)
             
             F = time.dominant_freq(self.eigenvectors, fs=sampling_interval, ncomp=self.M)
             
             if len(self.ts.shape)==1:
                 ah,gh = tool.ARpar(self.ts)
                 L = self.ts.size
                 xs = tool.ar1gen(ah,gh,L,Ns)  
                 for i in range(Ns):
                     ci = tool.covaBH(demean(squeeze(xs[:,i])),self.M)
                     Ds[:,i] = diag(self.eigenvectors.T @ ci @ self.eigevectors)
             else:
                 lx = self.ts.shape[0]
                 ah = zeros(lx)
                 gh = zeros(lx)
                 xs = zeros((self.M,lx,Ns))
                 for k in range(lx):
                     ah[k],gh[k] = tool.ARpar(X[:,k])
                     xs[:,:,k] = tool.ar1gen(ah[k],gh[k],lx,Ns)    
                 for i in range(Ns):
                     ci = tool.covaBH(demean(squeeze(xs[:,:,i])),self.M)
                     ds[:,i] = diag(self.eigenvectors.T @ ci @ self.eigenvectors)
             ds = sort(ds,axis=1)   
             
             Ongarde=[]
             siglevel_val = zeros(self.eigenvectors.shape[1])
             for k in range(len(self.eigenvalue)):                        
                 wk = ds[:,k]
                 siglevel_val[k] = wk[int(Ns*siglevel/100)]
                 if self.eigenvalue[k] > siglevel_val[k]:       
                     Ongarde.append(k)
             nn=Ongarde
             
             self.isig = Ongarde
             self.eigenvalue_suro = ds
             self.siglevel_value = siglevel_val
             
        else:
            N = ts.size
            L = N - self.M + 1
            
            iin=0 # initiate loop        
            nn0 = [0]
            nn = []
            
            while (len(nn0)!=len(nn)) and (iin<3):
                print(nn,nn0)
                nn0 = copy(nn)
                
                alpha,gamma = tool.ar1fitcompDeq1(self.eigenvectors,self.ts,n=norotate)
                
                r = zeros((N,Ns))
                Wn = zeros((Ns,self.eigenvectors.shape[1]))
                
                R = zeros((N,Ns))                 
                for n in range(Ns):
                   temp=np.random.default_rng().normal(0, 1, size=(N))
                   R[:,n]=temp
                
                idx=hankel(arange(L),arange(N-self.M,N))                
                r = lfilter(np.array([sqrt(alpha)]),np.array([1,-gamma]),R[:,:],axis=0)        
                
                for n in tqdm(range(Ns)):                    
                    xin = r[:,n]
                    Yn = xin[idx]                    
                    Cs = Yn.T @ Yn / L
                    ds, vs = eig(Cs)
                    ks = flipud(argsort(ds))
                    vs = vs[:,ks]
                    ds = real(ds[ks])
                    
                    pn = (vs@diag(sqrt(ds))).T@(self.eigenvectors@diag(sqrt(self.eigenvalue)))
                    up,sp,vp = svd(pn)
                    Te = real(up)@real(vp)                
                    Wn[n,:] = diag(Te.T@diag(wn)@Te)**2
                    
                iin+=1
                
                Ongarde=[]
                siglevel_val = zeros(self.eigenvectors.shape[1])
                for k in range(self.eigenvectors.shape[1]):                        
                    wk = sort(Wn[:,k])
                    siglevel_val[k] = wk[int(Ns*siglevel/100)]
                    if self.eigenvalue[k] > siglevel_val[k]:       
                        Ongarde.append(k)
                        # print(k)
                nn=Ongarde
               
            self.isig = Ongarde
            self.eigenvalue_suro = Wn
            self.siglevel_value = siglevel_val
    
    
    def get_group(self,f_treshold=0.01,comp='isig'):
        '''The function to group the SSA components based on the similarity of the
        dominant frequency of each component.
        

        Parameters
        ----------
        f_treshold : float, optional
            The threshold of the frequencies that are in the same group of mode.
            The default is 0.01.
            
        comp : string | 'isig' | 'all'
            The function will only group the significant components if 'isig' is given.
            If 'all' is given, all components will be analyzed. The default is 'isig'.


        Returns
        -------
        lii : list
            The index of the components in the same group.
        Fli : array
            The average frequency of each group.

        '''
        
        if comp=='isig':
            idx = self.isig
        elif comp=='all':
            idx = list(arange(len(self.eigenvalue)))
        else:            
            raise TypeError(f"Component error, choose 'isig' or 'all'")        
            return None
        
        Fo = self.F[idx]
        lii=[]
        lit=[]
        Fli=[]
        
        for k, value in enumerate(idx):
            Fi=nonzero((abs(Fo[k]-Fo[k+1:])<f_treshold))[0]            
            if (len(Fi)==0) & (k not in lit):                
                lii.append([value])
                Fli.append(Fo[k])
            elif (len(Fi)==1):
                lii.append([value,idx[k+1+Fi[0]]])
                Fli.append(mean([Fo[k],Fo[k+1+Fi[0]]]))
                lit.append(k+1+Fi[0])
            elif (k in lit):
                rien=0
            else:
                pass
        
        # self.group = lii
        # self.group_freq = Fli
        return lii, Fli
            
    def plot_mcssa(self, ncomp=20):
        '''To plot the results of the Monte Carlo test of SSA.

        Parameters
        ----------
        ncomp : int, optional
            The first ncomp-rank of eigenvalues that are plotted. The default is 20.

        '''
        
        Ns = self.eigenvalue_suro.shape[0]
        top_level = int(Ns * (self.siglevel / 100))
        low_level = int(Ns * (1 - top_level))
        
        wk = sort(self.eigenvalue_suro, axis=0)
        
        tool.plot_mc(self.F, self.eigenvalue, wk, self.isig, ncomp)   