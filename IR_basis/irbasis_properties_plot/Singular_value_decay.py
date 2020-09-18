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


rc('lines',linewidth=1.2, marker='x', markersize=4.5)
plt.rcParams["lines.markeredgewidth"] = 1
#plt.rcParams["xtick.major.pad"] = 2
#plt.rcParams["ytick.major.pad"] = 2
rc('xtick',direction='in')
rc('ytick',direction='in')
rc('font', family='serif')
rc('markersize')


#### Paths -----------------------------------------------
savepath = 'C:\\Users\\nikla\\Documents\\Master\\Master_thesis\\pictures\\'

Lambda = 10**linspace(1,7,7)

sl_F = empty(len(Lambda),dtype='object')
sl_B = empty(len(Lambda),dtype='object')
for it in range(len(Lambda)):
    b = irbasis.load('F',Lambda[it])
    sl_F[it] = b.sl()
    
    b = irbasis.load('B',Lambda[it])
    sl_B[it] = b.sl()

######################### Plotting -------------
fig = plt.figure(constrained_layout=True)
spec = gridspec.GridSpec(ncols=2, nrows=1, figure=fig)
f_ax1 = fig.add_subplot(spec[0, 0])
f_ax2 = fig.add_subplot(spec[0, 1])

###### plot
for it in range(len(Lambda)):
    f_ax1.semilogy(arange(len(sl_F[it])),sl_F[it]/sl_F[it][0])
    f_ax2.semilogy(arange(len(sl_B[it])),sl_B[it]/sl_B[it][0],\
                   label='$\\Lambda = 10^{' + str(int(log10(Lambda[it]))) + '}$')


### Fermi plot parameters
f_ax1.tick_params(top=True, right=True, which='both')
#f_ax1.xaxis.set_minor_locator(MultipleLocator(5))
f_ax1.set_xlim([0,150])
f_ax1.set_ylim([10**(-15),1])
f_ax1.set_xlabel('$l$')
f_ax1.set_ylabel('$s_l^{\mathrm{F}}/s_0^{\mathrm{F}}$')

f_ax1.text(-45,0.05,'\\textbf{(a)}',size=MEDIUM_SIZE+0.5)


### Bose plot parameters
f_ax2.tick_params(top=True, right=True, which='both')
#f_ax2.xaxis.set_minor_locator(MultipleLocator(5))
#f_ax2.yaxis.set_minor_locator(MultipleLocator(0.05))
f_ax2.set_xlim([0,150])
f_ax2.set_ylim([10**(-15),1])
f_ax2.set_xlabel('$l$')
f_ax2.set_ylabel('$s_l^{\mathrm{B}}/s_0^{\mathrm{B}}$')
f_ax2.text(-45,0.05,'\\textbf{(b)}',size=MEDIUM_SIZE+0.5)
f_ax2.legend()

#plt.figlegend(bbox_to_anchor=(0.49, 1.15), loc='upper center', ncol=len(Lambda))

#plt.savefig(savepath + 'IR_basis_Sl_convergence.pdf', bbox_inches='tight')