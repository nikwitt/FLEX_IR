# -*- coding: utf-8 -*-
"""
Created on Tue May 26 11:42:33 2020

@author: nikla
"""

from numpy import *
from ir_load import ir_load
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
rc('figure',figsize=cm2inch(14.5,5), dpi=300)
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


rc('lines',linewidth=1.2, marker='x', markersize=0)
plt.rcParams["lines.markeredgewidth"] = 1
#plt.rcParams["xtick.major.pad"] = 2
#plt.rcParams["ytick.major.pad"] = 2
rc('xtick',direction='in')
rc('ytick',direction='in')
rc('font', family='serif')
rc('markersize')


#### Paths -----------------------------------------------
savepath = './/'


#### Calculation -----------------------------------------
Lambda = 10**linspace(1,7,7)
delta  = 10**linspace(-1,-15,281)

l_F = empty(len(Lambda),dtype='object')
l_B = empty(len(Lambda),dtype='object')
for it in range(len(Lambda)):
    print(it)
    
    b_F = irbasis.load('F', Lambda[it])
    b_B = irbasis.load('B', Lambda[it])
    
    ll_f = [ir_load.basis_cutoff(b_F, delta_it) for delta_it in delta]
    ll_b = [ir_load.basis_cutoff(b_B, delta_it) for delta_it in delta]
        
        #assert (b.f_l[-1]+1,) == b.fm.shape and (b.f_l[-1]+1,) == b.ft.shape
        #assert (b.b_l[-1]+1,) == b.bm.shape and (b.b_l[-1]+1,) == b.bt.shape
        
    l_F[it] = array(ll_f)
    l_B[it] = array(ll_b)

######################### Plotting -------------
fig = plt.figure(constrained_layout=True)
spec = gridspec.GridSpec(ncols=2, nrows=1, figure=fig)
f_ax1 = fig.add_subplot(spec[0, 0])
f_ax2 = fig.add_subplot(spec[0, 1])

###### plot
for it in range(len(Lambda)):
    f_ax1.semilogx(delta,l_F[it])
    f_ax2.semilogx(delta,l_B[it],\
                   label='$\\Lambda = 10^{' + str(int(log10(Lambda[it]))) + '}$')


### Fermi plot parameters
f_ax1.tick_params(top=True, right=True, which='both')
f_ax1.set_xlim([1e-15,1e-1])
f_ax1.set_ylim([0,202])
f_ax1.set_xlabel('$\\delta$')
f_ax1.set_ylabel('$N^{\\mathrm{F}}_{\\mathrm{IR}}$')


### Bose plot parameters
f_ax2.tick_params(top=True, right=True, which='both')
f_ax2.set_xlim([1e-15,1e-1])
f_ax2.set_ylim([0,175])
f_ax2.set_xlabel('$\\delta$')
f_ax2.set_ylabel('$N^{\\mathrm{B}}_{\\mathrm{IR}}$')

plt.figlegend(bbox_to_anchor=(1, 1.02), loc='upper left', ncol=1, labelspacing=0.9)

plt.savefig(savepath + 'IR_basis_size_accuracy.png', bbox_inches='tight')
plt.savefig(savepath + 'IR_basis_size_accuracy.pdf', bbox_inches='tight')