# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 14:35:57 2012

@author: Olivier, modified by Anita (2023)
"""
from scipy.optimize import fminbound    

from os import chdir
from struct import *
import numpy as np
import random as ra
import math
from numpy import *
from numpy import fft as fft
from matplotlib.pyplot import *
from numpy.linalg import *
from numpy.random import *
from scipy.interpolate import interp1d
from scipy.linalg import *
from scipy.io import *
from scipy.signal import lfilter
from scipy.optimize import fmin

def demean(x, axis=0):
    x = x - mean(x,axis=axis)
    return x

def normalize(x, axis=0):    
    stdx = np.matlib.repmat(std(x,axis=axis), x.shape[axis], 1)
    if axis==1:
        stdx = stdx.T
    x = demean(x)/stdx
    return x, stdx


def ARpar(X):
   Xb=mean(X)
   N=X.shape[0]
   n=12
   c1=1./(N-1)*dot(squeeze(X[0:-1]-Xb),squeeze(X[1:]-Xb))
   c0=1./N*dot(squeeze(X-Xb),squeeze(X-Xb))
   gh=c1/c0
   for i in range(0,n):
       mu2=-1./N+2./N**2*((N-gh**N)/(1-gh)-gh*(1-gh**(N-1))/(1-gh)**2)
       dg=-(gh-mu2)/(1-mu2)+c1/c0;
       gh=gh+dg;
   c0=c0/(1-mu2);
   ah=sqrt(c0*(1-gh**2))
   return ah,gh
   
def compPC(X,V,N=[0,1,2,3,4,5,6,7,8]):
    M=V.shape[0]
    L=X.size-M+1
    Vn=V[:,N]
    X=demean(X)
    pc=zeros((L,Vn.shape[1]))
    for i in range(0,Vn.shape[1]):
        for k in range(0,L):
           pc[k,i]=dot(squeeze(X[k:k+M]),transpose(Vn[:,i]))
    return pc     
    
def compRC(X,V,N=[0,1,2,3,4,5,6,7,8]):
    pc=compPC(X,V,N)
    X=X-mean(X)
    M=V.shape[1]
    Vn=V[:,N]  
    L=X.size
    rc=zeros((L,Vn.shape[1]))
    for i in range(Vn.shape[1]):
        rc[0,i]=pc[0,i]*Vn[0,i]
        for k in range(1,M):
            rc[k,i]=1./(k+1)*dot(transpose(flipud(pc[0:k+1,i])),Vn[0:k+1,i])
        for k in range(M,L-M+1):
            rc[k,i]=1./M*dot(transpose(flipud(pc[k-M:k,i])),Vn[0:M+1,i])
        for k in range(L-M+1,L): 
            rc[k,i]=1./(L-k)*dot(transpose(flipud(pc[k-M+1:,i])),Vn[k-L+M:,i])   
    return rc        
    
def compPCm(X,M,V,Nc=[0,1,2,3,4,5,6,7]):
    S=X.shape;
    T=V.shape;
    pc=zeros((S[0],len(Nc)));
    
    a=zeros((S[0],S[1]));
    Ej=zeros((M,S[1]));
    for k in range(0,len(Nc)):
#        print(k,V.shape,S ,Nc)
        Ej=transpose(fliplr(reshape(V[:,Nc[k]],(S[1],M))))
        for j in range(0,S[1]):
           a[:,j]=lfilter(Ej[:,j],1,X[:,j]);
        if S[1]>1:    
          pc[:,k]=sum(a,1)   
        else:
          pc[:,k]=squeeze(a)
    pc=pc[M-1:X.shape[0],:]                   
    return pc     

def compRCm(X,M,V,A,Nc=[0,1,2,3,4,5,6,7]):
    V=V[:,Nc]
    ml,k=V.shape
    ra,ka=A.shape
    L=X.shape[1]
    M=int(ml/L)
    N=ra+M-1
    R=zeros((N,L*len(Nc)))
    Z=zeros((M-1,k))
    A=transpose(hstack((transpose(A),transpose(Z))))
    for j in range(0,len(Nc)):
        Ej=transpose(reshape(V[:,j],(L,M)))
        for i in range(0,L):
            R[:,j*L+i]=lfilter(Ej[:,i],M,A[:,Nc[j]])
    for i in range(0,M-1):
        R[i,:]=R[i,:]*M/(i+1)
        R[N-i-1,:]=R[N-i-1,:]*M/(i+1)
    return R
    
def ar1gen(a,g,L,N):
    x=zeros((L,N));
    z=a*randn(0,1,L,N);
    x[0,:]=z[0,:]
    p=zeros((N))
    for l in range(L):
       x[l,:]=g*x[l-1,:]+z[l,:];
    for l in range(N):
       a,g=ARpar(x[:,l]) 
       p[l]=a
    return x 

    
def objfun(f,t,X):
    A=transpose(vstack([0*t+1, t, cos(2*pi*f*t), sin(2*pi*f*t)]))  
    y,re,s,k0=lstsq(A,X)
    if not re:
       re=var(X)
    return re
    
    
def covaVG(X,M=0):
   N=X.size
   if M==0:
      M = N//3
   C = zeros((M))
   X = demean(X)
   C[0] = sum(X**2)/N
   for k in range(1,M):
       C[k]=sum(X[0:-k]*X[k:])/(N-k)
   c=toeplitz(C)
   return c
   
   
def covaBH(X,M=0):
   N=X.size
   if M == 0:
      M = N // 3
   Np = N-M+1
   D = zeros((Np,M))
   X = demean(X)
   for k in range(0,Np):
       D[k,:]=squeeze(X[k:k+M])
   c = dot(transpose(D),D)/Np
   return c
    
def xcorr(X,L):
   X=demean(X)
   xc=zeros((L+1))
   xc[0]=1;
   for k in range(1,L+1):
       xc[k]=sum(X[0:-k]*X[k:])/sqrt(sum(X[0:-k]**2)*sum(X[k:]**2))   
   return xc 
   
def xcorr2(X,Y,L):
   X=demean(X)
   Y=demean(Y)
   xc=zeros(2*L+1)
   xc[0]=1;
   xc[L]=sum(X*Y)/sqrt(sum(X**2)*sum(Y**2))
   for k in range(1,L+1):
       xc[L+k]=sum(X[0:-k]*Y[k:])/sqrt(sum(X[0:-k]**2)*sum(Y[k:]**2))  
       xc[L-k]=sum(Y[0:-k]*X[k:])/sqrt(sum(Y[0:-k]**2)*sum(X[k:]**2))
        
   return xc 
   
def trace0(x):
    m=mean(diag(x,0))
    return m

def trace1(x):
    m=mean(diag(x,1))
    return m

def ar1fitcompDeq1(E,x,n=[]):
    N,D=x.shape
    [M,K]=E.shape
    Ns=N-M+1
    x-=mean(x)
    idx=hankel(arange(N-M+1),arange(N-M,N))
#    
#   Trajectory matrix
#    
    xtde=zeros((N-M+1,M))
    xtde[:,:]=x[idx]
    C=squeeze(xtde).T@squeeze(xtde)/(N-M+1)
# 
#  Computation of the noise parameters following Allan and Smith 1996
#
    k=arange(1,N)  
    def Wf(gamma):  
        mus=1.0/N+1.0/N**2*sum(2*(N-k)*gamma**k) # Bias on surrriate variance eq 9
        W=toeplitz(gamma**arange(M))-mus         # definition above eq 19 AS96        
        return W    
    Kn=[0 if i in n else 1 for i in range(K)]   #1 if i is noise, 0 if it is signal
    Q=E@diag(Kn)@E.T                            # Filtering the signal out of matrix
    c0hat=trace0(Q@C@Q)                         # AS eq 17
    c1hat=trace1(Q@C@Q)                         # AS eq 18
    def fx(x):
        f=abs(c1hat/c0hat-trace1(Q@Wf(x)@Q)/trace0(Q@Wf(x)@Q))  # AS eq 19
        return f
    aa=fminbound(fx,0.0,1.0)
    gamma=aa
    c0tilde=trace0(Q@C@Q)/trace0(Q@Wf(gamma[d])@Q)   # AS eq 20
    alpha=c0tilde*(1-gamma**2)                       # AS eq 7 with l=0
    return alpha,gamma   


def ar1fitcomptr(E,xtde,n=[]):
    if len(xtde.shape)>2:
        Ns,M,D=xtde.shape
    else:
        Ns,M=xtde.shape
        D=1
    N=Ns+M-1    
    K=E.shape[1]
#     for k in range(D):
#         x[:,k]-=mean(x[:,k]) #demean
#     idx=hankel(arange(N-M+1),arange(N-M,N))
# #    
# #   Trajectory matrix
# #    
#     xtde=zeros((N-M+1,M,D))  
#     for d in range(D):
#         xin=x[:,d]
#         xtde[:,:,d]=xin[idx]
#        
#    Covariance matrix
#
    if D==1: 
        C=squeeze(xtde).T@squeeze(xtde)/(N-M+1)
    else:
        C=zeros((Ns,Ns,D))
        for d in range(D):
            C[:,:,d]=xtde[:,:,d]@xtde[:,:,d].T/M
    k=arange(1,N)  
    def Wf(x): # Allan and Smith 1996
        # mus=1.0/N+1.0/N**2*sum(2*(N-k)*x**k) # Bias on surrogate variance eq 9
        mus = -1 / N + 2 / N**2 * \
        ((N - x**N) / (1 - x) -
          (x * (1 - x**(N - 1))) / (1 - x)**2)
        if D==1:
            W=toeplitz(x**arange(M))-mus
        else:
            W=toeplitz(x**arange(Ns))-mus
        # W=matrix(W)
        return W    
    def trace0(x):
        m=mean(diag(x,0))        
        return m
    def trace1(x):
        m=mean(diag(x,1))
        return m
    Kn=[0 if i in n else 1 for i in range(K)]
    Q=E@diag(Kn)@E.T
    gamma=zeros(D)
    alpha=zeros(D)
    if D==1:
        c0hat=trace0(Q@C@Q)
        c1hat=trace1(Q@C@Q)
        def fx(x):
            f=abs(c1hat/c0hat-trace1(Q@Wf(x)@Q)/trace0(Q@Wf(x)@Q))
            return f
        aa=fminbound(fx,0.0,1.0)
        gamma[0]=aa   
        c0tilde=trace0(Q@C@Q)/trace0(Q@Wf(gamma[d])@Q)
        alpha[0]=c0tilde*(1-gamma[d]**2)
    else:
        for d in range(D):
            c0hat=trace0(Q@C[:,:,d]@Q)
            c1hat=trace1(Q@C[:,:,d]@Q)
            def fx(x):
                f=abs(c1hat/c0hat-trace1(Q@Wf(x)@Q)/trace0(Q@Wf(x)@Q))
                return f
            aa=fminbound(fx,0.0,1.0)
            gamma[d]=aa   
            c0tilde=trace0(Q@C[:,:,d]@Q)/trace0(Q@Wf(gamma[d])@Q)
            alpha[d]=c0tilde*(1-gamma[d]**2)
    return alpha,gamma   

def eps(z):
    """Equivalent to MATLAB eps
    """
    zre = np.real(z)
    zim = np.imag(z)
    return np.spacing(np.max([zre, zim]))
from scipy.optimize import fminbound  

def varimax_fun(A,reltol=1e-16,maxit=1000,normalize=True): 
    B = copy(A)
    if len(B.shape) > 2:
        M,D,S = B.shape        
        h = sum(sum(B**2, axis=0), axis=1)            
        for m in range(M):
            for s in range(S):
                B[m,:,s]/=sqrt(h)
        B = reshape(B.T,(S,D * M)).T
        D = D * M
    else:
        D,S = B.shape
        M = 1
        h = sum(B ** 2, axis=0)
        B = B / sqrt(h)
    T=eye(S)
    # tic=time.time()
    for iter in range(maxit):
        maxT=0
        for i in range(S-1):
            for j in range(i+1,S):
                u=B[:,i]**2-B[:,j]**2
                v=2*B[:,i]*B[:,j]                
                if M>1:
                    u=sum(reshape(u.T,(-1,M)),axis=1).T
                    v=sum(reshape(v.T,(-1,M)),axis=1).T
                usum=sum(u,axis=0)
                vsum=sum(v,axis=0)
                numer=2*(u.T@v-usum*vsum/D)
                denom=u.T@u-v.T@v-(usum**2-vsum**2)/D
                theta=math.atan2(numer,denom)/4
                st=sin(theta)
                ct=cos(theta)
                maxT=max([maxT,abs(theta)])
                Tij=matrix([[ct,-st],[st,ct]])
                B[:,[i,j]]=B[:,[i,j]]@Tij
                T[:,[i,j]]=T[:,[i,j]]@Tij
        if maxT<reltol:
             break

    return B,T

def weight_latitude(ts,lat):
    N,D = ts.shape
    nlat = len(lat)
    
    if N == nlat:
        w = np.matlib.repmat(sqrt(cos(radians(lat))),1,D)
    elif D == nlat:
        w = np.matlib.repmat(sqrt(cos(radians(lat))),1,N).T
    else:
        raise TypeError("The number of points is different than the number of latitudes")
        
        
    wts = ts * w
    
    return wts, w

def plot_mc(F,eigenvalue,wk,top_level,low_level,isig,ncomp):    
    figure()                
    for k in range(ncomp):
        if k in isig:
            semilogy(F[k],eigenvalue[k],'ko',markersize=11,fillstyle='none')
    plot(F[:ncomp],eigenvalue[:ncomp],'ko')
    for k in range(ncomp):
        plot([F[k],F[k]],[wk[low_level,k],wk[top_level,k]],linewidth=0.7,color = 'black')
        errorbar(F[k],wk[low_level,k],xerr = 0.01,linewidth=0.7,ecolor = 'black')
        errorbar(F[k],wk[top_level,k],xerr = 0.01,linewidth=0.7,ecolor = 'black')
    xlabel('Frequency (1/yr)',fontsize=12)
    ylabel('Eigenvalues',fontsize=12)    
    tight_layout()
    
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
