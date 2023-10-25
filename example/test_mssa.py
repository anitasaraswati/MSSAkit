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

from mssakit.mssa import *
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

# %% Apply MSSA to the datasets
dmssa = mssa(tseries,t=t_tseries,sampling_interval=12,M=110,pre_pca=True,n_pc=15,varimax=True)
# %
nn=arange(50,229)
dmssa.mcmssa(siglevel=95, Ns=100, norotate=nn)
dmssa.plot_mcmssa(ncomp=20)

lii, Fli = dmssa.get_group(comp='isig')

# %% Building a mode as the sum of a pair of components
mode1 = sum(dmssa.RC[:,:,lii[0]], axis=2) 
mode1_ts = sum(dmssa.RC_spatial[:,:,lii[0]], axis=2) 
