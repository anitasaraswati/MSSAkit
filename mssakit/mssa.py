import sys
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

from mssakit.pca import *
import mssakit.toolbox as tool
import mssakit.time_analysis as time

class mssa:
    '''Performs MSSA decomposition on multivariate time series.
    Multivariate time series should have observations in columns and time series
    indices in rows.
    
    Parameters
    ------------
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
        
    n_rec : int | None
        The number of the reconstructed components that is generated from the MSSA process. The default is 50.
        
    pre_pca : boolean
        PCA is applied to the data before applying MSSA to reduce its dimension. The default is False.
    
    n_pc : int | None
        The number of the PCs that is kept and used for the MSSA process. The default is all PCs.
    
    varimax : boolean
        The option to apply varimax rotation in MSSA. The default is False.
    
    t : numpy.array (optional)
        The time indicator of array ts.
        
    sampling_interval : int
        The sampling interval of the time series. The default is 1.
        
    
    Attributes
    ----------
    ts : numpy.ndarray
        The input of the time series.     
    
    ts_pc : numpy.ndarray
        The set of PCs that is selected from the PCA before the MSSA, if pre_pca is selected.
       
    t : array
        The given time index of the time series. If t is not given, the time index will
        be replaced with the i-index point.
    
    sampling_interval : int
        The given sampling interval of the time series.
    
    M : int
        The given windows size used in the MSSA.        
    
    trajectory_matrix : numpy.ndarray
        The resulting trajectory matrix in the MSSA
    
    left_vectors : numpy.ndarray
        The resulting left singular vectors from SVD of the trajectory matrix, also familiar with Time-EOF.
    
    right_vectors : numpy.ndarray
        The resulting right singular vectors from SVD of the trajectory matrix, also familiar with SpaceTime-EOF.
        
    singularvalue : numpy.array
        The resulting singular values from SVD of the trajectory matrix.
    
    eigenvalue : numpy.array
        The eigenvalues of the trajectory matrix (singularvalue ** 2).
    
    PC : numpy.ndarray
        The Time-PC from the MSSA.
    
    RC : numpy.ndarray
        The reconstruction of MSSA in the original multi-channel data
        (or PCA components if the pre_pca is selected). The dimension of this 
        matrix is (N, D, rank), where N is the number of observations, D is 
        the number of channels, and rank is the number of components selected to keep.
    
    RC_spatial : numpy.ndarray
        The reconstruction of MSSA in the original multi-channel data, if the pre_pca is selected.
        The dimension of this matrix is (N, D, rank), where N is the number
        of observations, D is the number of channels, and rank is the
        number of components selected to keep.
    
    F : numpy.array
        The dominant frequency of the T-EOF.
    
    T : numpy.array
        The dominant period of the T-EOF.
    
    sigma_T : numpy.array
        The period uncertainty of the obtained dominant period.
    '''       


    def __init__(self,
                 ts,                 
                 M=None,
                 n_rec=50,
                 pre_pca=False,
                 n_pc=None,
                 varimax=False,
                 t=None,
                 sampling_interval=1):
        
        self.ts = ts
        self.t = squeeze(t)
        self.sampling_interval = sampling_interval
        
        if pre_pca:
            ts_pca = pca(ts)
            pc = ts_pca.PC
            
            if n_pc is None:
                n_pc = pc.shape[0]            
            y = pc[:,:n_pc]
            vr = ts_pca.v[:,:n_pc]
            
            self.ts_pc = copy(y)            
        else:
            y = copy(ts)
        
        if M is None:
            M = y.shape[0] // 3

        N,D = y.shape
        for k in range(D):
            y[:,k]-=mean(y[:,k])
        
        self.M = M
            
        L = N-M+1
        
        # creating trajectory matrix
        idx = hankel(arange(L),arange(N-M,N))
        xtde = zeros((L,M,D),complex)
        for d in range(D):
            xin = y[:,d]    
            xtde[:,:,d] = xin[idx]            
        self.trajectory_matrix = xtde
        
        xtde = reshape(xtde.swapaxes(1,2),(N-M+1,D*M)) 
        
        EU,EW,EV = svd(xtde/sqrt(D*M),full_matrices=False)
        EV = EV.T
        
        # rank sort
        ks = EW.argsort()[::-1]
        EW = EW[ks]
        EV = real(EV[:,ks])
        EU = real(EU[:,ks])
        eigval = EW**2
        
        if varimax:            
            # Rotate only the components with % of eigenvalues > 1%
            Sm = nonzero(eigval/sum(eigval)>0.01)[0][-1]+1
            
            EVsc = EV[:,:Sm]@diag(EW[:Sm])
            EVr = zeros((M,D,Sm))
            for k in range(Sm):
                EVr[:,:,k] = reshape(EVsc[:,k],(D,M)).T            
                
            dummy,Tv = tool.varimax_fun(real(EVr))
            EV0 = copy(EV)
            EU0 = copy(EU)
            EV[:,:Sm] = EV[:,:Sm] @ Tv
            EW[:Sm] = diag(Tv.T@diag(EW[:Sm])@Tv)
            EU[:,:Sm] = EU[:,:Sm]@Tv
            eigval[:Sm] = diag(Tv.T@diag(eigval[:Sm])@Tv)
    
    
            # sort again EW
            ks = EW.argsort()[::-1]
            EW = real(EW[ks])
            EV = EV[:,ks]
            EU = EU[:,ks]            
            eigval = EW**2
    
            EV = real(EV)
            EU = real(EU)

            ll = arange(0,D*M,M)
            Xv = (EU@diag(EW)@EV.T)*sqrt(D*M)
            Xtdev=reshape(Xv,(L,D,M)).swapaxes(2,1)
            yv=Xv[:,ll]
            yvv=Xv[L-M+1:,ll+M-1]            
            yv=concatenate((yv,yvv),axis=0)            
    
            ew=diag(EW)
            
            self.trajectory_matrix = Xtdev
            
        self.left_vectors = EU
        self.right_vectors = EV
        self.singularvalue = EW
        self.eigenvalue = eigval
        
        # % get V and R in all pc
        NC = n_rec
        FF=zeros(NC)
        a=zeros((N,D))
        A=zeros((L,NC));
        RC=zeros((N,D,NC))
        Ru=zeros((N,D,NC))
        ia=0
        for o in range(NC) :
            Ej=(EV[:,o].reshape((D,-1))).T
            for j in range(D):
                a[:,j]=lfilter(flip(Ej[:,j]),1,y[:,j])
            pc=sum(a,axis=1)[M-1:]
            A[:,ia]=pc
            pc=hstack((pc,zeros(M-1)))
            for j in range(D):                   
                RC[:,j,ia]=lfilter(Ej[:,j],M,pc)                
            ia+=1    
        for i in range(0,M-1):
            RC[i,:,:]=RC[i,:,:]*M/(i+1)
            RC[N-i-1,:,:]=RC[N-i-1,:,:]*M/(i+1)
            
            
        self.PC = A
        self.RC = RC        
        
        if pre_pca:
            Rx = zeros((ts.shape[0],ts.shape[1],NC))
            vt = transpose(vr)
            for i in range(NC):                
                Rx[:, :,i]=(RC[:,:,i] @ vt)  #/w*stdx    
                
            self.RC_spatial = Rx
        
        self.F = time.dominant_freq(A,fs=sampling_interval)
        self.T, self.sigma_T = time.dominant_period(A,fs=sampling_interval,uncertainty=True)
        
        
    def mcmssa(self, siglevel=95, Ns=10, norotate=[]):        
        ''' Applying Monte Carlo to the MSSA
                
        Parameters
        ------------
        siglevel : int
            The significance level (%) in the Monte Carlo test, given between 0 and 100.
            The default is 95.
            
        Ns : int
            Number of surrogates. The default is 10.
        
        norotate : list
            The list of index of non-rotated T-EOF. The default is []
               

        Atributes
        ---------
        siglevel : int
            The given signigicance level (in %) in the MC test
        
        isig : list
            The index of MSSA PC that is above the significance level
        
        eigenvalue_suro : numpy.ndarray
            The eigenvalue of the surrogates
        
        '''
        if hasattr(self, 'ts_pc'):
            y = self.ts_pc
        else:
            y = self.ts
            
        N,D = y.shape
        L = N - self.M+1
        
        N2 = max(L,D*self.M)        
        
        iin=0 # initiate loop        
        nn0 = [0]
        nn = []

        while (len(nn0)!=len(nn)) and (iin<3):            
            nn0 = copy(nn)
            
            alpha,gamma = tool.ar1fitcomptr(self.left_vectors,self.trajectory_matrix,n=norotate)
            
            r = zeros((N,D,Ns))
            Wn = zeros((Ns,self.left_vectors.shape[1]))
            
            R = zeros((N,D,Ns))
            for k in range(D):
                for n in range(Ns):
                    temp=np.random.default_rng().normal(0, 1, size=(N))
                    R[:,k,n]=temp
            
            idx=hankel(arange(L),arange(N-self.M,N))
            for k in range(D):       
                r[:,k,:] = lfilter(np.array([sqrt(alpha[k])]),np.array([1,-gamma[k]]),R[:,k,:],axis=0)        
            
            for n in tqdm(range(Ns)):
                Yn = zeros((L,self.M,D))
                for d in range(D):
                    xin = r[:,d,n]
                    Yn[:,:,d] = (xin[idx])
                Yn = reshape(Yn.swapaxes(1,2),(L,D*self.M))
                un,wn,vn = svd(Yn/sqrt(N2),full_matrices=False)
                vn = vn.T
                
                pn = (un@diag(wn)).T@(self.left_vectors@diag(self.singularvalue))                
                up,sp,vp = svd(pn)
                Te = real(up)@real(vp)                
                Wn[n,:] = diag(Te.T@diag(wn)@Te)**2
                
            iin+=1
            
            Ongarde=[]
            siglevel_val = zeros(self.left_vectors.shape[1])
            for k in range(self.left_vectors.shape[1]):                        
                wk = sort(Wn[:,k])
                siglevel_val[k] = wk[int(Ns*siglevel/100)]
                if self.eigenvalue[k] > siglevel_val[k]:        
                    Ongarde.append(k)
                    # print(k)
            nn=Ongarde
            print(nn0,nn)
           
        self.siglevel = siglevel
        self.isig = Ongarde
        self.eigenvalue_suro = Wn
        self.siglevel_value = siglevel_val
        
        
    def get_group(self,f_threshold=0.01,comp='isig'):
        '''The function to group the T-EOF based on the similarity of the
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
            idx = list(arange(len(self.singularvalue)))
        else:            
            raise TypeError(f"Component error, choose 'isig' or 'all'")        
            return None
        
        Fo = self.F[idx]
        lii=[]
        lit=[]
        Fli=[]
        
        for k, value in enumerate(idx):
            Fi=nonzero((abs(Fo[k]-Fo[k+1:])<f_threshold))[0]            
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
    
    
    def plot_mcmssa(self, ncomp=50):
        '''To plot the results of the Monte Carlo test of the MSSA.

        Parameters
        ----------
        ncomp : int, optional
            The first ncomp-rank of eigenvalues that are plotted. The default is 50.

        '''
        Ns = self.eigenvalue_suro.shape[0]
        top_level = int(Ns * (self.siglevel / 100))
        low_level = int(Ns * (1 - (self.siglevel / 100)))
        
        wk = sort(self.eigenvalue_suro, axis=0)
        
        tool.plot_mc(self.F, self.eigenvalue, wk, top_level, low_level, self.isig, ncomp)        
