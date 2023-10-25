# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 18:18:25 2023

@author: asaraswa
"""

import sys
from os import chdir
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
import cartopy.crs as ccrs
from tqdm import tqdm
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)

from mssakit.pca import *
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

# normalize the dataset by dividing the timeries by its standard deviation
ts, stdx = normalize(tserieso,axis=0)
# weight the datasets based on its latitude
tseries, w = weight_latitude(ts,lat)

nt,nd=tseries.shape

#create grid
xx=unique(lon)
yy=unique(lat)
yy=flip(yy)


# %%
dpca = pca(tseries,sampling_interval=12, w=w)
stats, i_sig = dpca.sigtest(95,Ns=100,pmax=6,procrustes=True,n_proscruste=100)
dpca.plot_sigtest(ncomp=50,freq_rank=True)
