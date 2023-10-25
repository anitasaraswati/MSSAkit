# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 15:59:49 2023

@author: asaraswa
"""

import numpy as np
import scipy as sc
from numpy import *
from array import *
from numpy.linalg import *
from scipy.io import loadmat
from scipy.io import savemat
from matplotlib.pyplot import *
import matplotlib as mpl
from numpy.linalg import *
from tqdm import tqdm
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)

from mssakit.cross_svd import *
from mssakit.toolbox import normalize, weight_latitude


mat = loadmat('covobssa_l8.mat')
lon = mat['lon']
lat = mat['lat']
tserieso = mat['ts_mag_detrend']
t_tseries = mat['dyear_filled']

tserieso=tserieso.T

def nopole(x): return abs(x) < 90
id_nopole = [idx for idx, element in enumerate(lat) if nopole(element)]

# trim pole
tserieso=tserieso[:,id_nopole]
lat=lat[id_nopole]
lon=lon[id_nopole]

ts, std_A = normalize(tserieso,axis=0)
A, w_A = weight_latitude(ts,lat)


mat = loadmat('grace_igg_gravanom_l8.mat')
lon = mat['lon']
lat = mat['lat']
tserieso = mat['sm_ts_grace']
t_tseries = mat['dyear_filled']

tserieso=tserieso.T

def nopole(x): return abs(x) < 90
id_nopole = [idx for idx, element in enumerate(lat) if nopole(element)]

# trim pole
tserieso=tserieso[:,id_nopole]
lat=lat[id_nopole]
lon=lon[id_nopole]

# normalize the dataset by dividing the timeries by its standard deviation
ts, std_B = normalize(tserieso,axis=0)
# weight the datasets based on its latitude
B, w_B = weight_latitude(ts,lat)

#create grid
xx=unique(lon)
yy=unique(lat)
yy=flip(yy)

# %% Apply cross svd to both datasets
dsvd = cross_svd(A, B, sampling_interval=12, w_A=w_A, w_B=w_B, n_components=10)
stats, i_sig = dsvd.sigtest(95,Ns=100,pmax=6)
dsvd.plot_sigtest(ncomp=50,freq_rank=True)
