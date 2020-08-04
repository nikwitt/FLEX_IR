# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 21:20:02 2020

@author: nikla
"""

from numpy import *
import scipy as sc
import irbasis
import os

### Setting matplotlib details
def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)
    
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as tick
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
from matplotlib import rc
import seaborn as sns

plt.style.use('seaborn-paper')
sns.set_palette("muted")
rc('figure',figsize=cm2inch(14,5), dpi=300)
#rc('figure',figsize=cm2inch(7.2,5), dpi=300)
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{bm}')

SMALL_SIZE = 8
MEDIUM_SIZE = 11
BIGGER_SIZE = 10

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


rc('lines',linewidth=1.2, marker='x', markersize=3)
plt.rcParams["lines.markeredgewidth"] = 1
#plt.rcParams["xtick.major.pad"] = 2
#plt.rcParams["ytick.major.pad"] = 2
rc('xtick',direction='in')
rc('ytick',direction='in')
rc('font', family='serif')
rc('markersize')


#### Paths -----------------------------------------------
savepath = 'C:\\Users\\nikla\\Documents\\Master\\Master_thesis\\pictures\\'


#### Calculation
# parameters
w_max  = 1
beta   = 100
Lambda = 10**3
b = irbasis.load('F',Lambda)
l_list = [0,1,10,15]
plt_style = ['-','--','-','--']
### grids
tau   = linspace(0,beta,100000)
n     = array(unique(list(map(int,linspace(-10**3,10**3,2*10**3+1)))))
iwn   = 1j*pi/beta*(2*n+1)

######################### Plotting -------------
fig = plt.figure(constrained_layout=True)
spec = gridspec.GridSpec(ncols=2, nrows=1, figure=fig)
f_ax1 = fig.add_subplot(spec[0, 0])
f_ax2 = fig.add_subplot(spec[0, 1])

###### plot U(tau)

for l_it in range(len(l_list)):
    l = l_list[l_it]
    if l == l_list[-1]:
        f_ax1.plot([],[])
    f_ax1.plot(tau/beta, sqrt(2/beta)*b.ulx(l, 2*tau/beta-1),plt_style[l_it],label='$l={}$'.format(l))

### parameters
f_ax1.tick_params(top=True, right=True, which='both')
#f_ax1.xaxis.set_minor_locator(MultipleLocator(5))
f_ax1.set_xlim([-0.005,1.005])
f_ax1.set_ylim([-0.5,0.5])
f_ax1.set_xlabel('$\\tau/\\beta$')
f_ax1.set_ylabel('$U_{l}^{\mathrm{F}}(\\tau)$')

###### plot U(tau)

for l_it in range(len(l_list)):
    l = l_list[l_it]
    if l == l_list[-1]:
        f_ax2.plot([],[])
    if l % 2 == 0:
        f_ax2.plot((2*n+1), sqrt(2/beta)*imag(b.compute_unl(n,l)),plt_style[l_it])#,marker='x')
    else:
        f_ax2.plot((2*n+1), sqrt(2/beta)*real(b.compute_unl(n,l)),plt_style[l_it])#,marker='x')

### parameters
f_ax2.tick_params(top=True, right=True, which='both')

f_ax2.set_xscale('symlog',lintreshx=1)
f_ax2.set_xlim([-10**3,10**3])
xt = f_ax2.get_xticks()
xt = delete(xt,[where(xt==-0)[0][0]])#where(xt==1)[0][0]])
f_ax2.set_xticks(xt)
f_ax2.set_xlim([-10**3,10**3])
#f_ax2.set_ylim([-3,3])
f_ax2.set_xlabel('$i\\omega_n\\beta/\\pi$')
f_ax2.set_ylabel('$U_{l}^{\mathrm{F}}(i\\omega_n)$')

plt.figlegend(bbox_to_anchor=(0.49, 1.15), loc='upper center', ncol=len(l_list))

plt.savefig(savepath + 'IR_basis_functions.pdf', bbox_inches='tight')