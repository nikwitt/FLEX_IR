# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 15:47:33 2020

@author: nikla
"""


from numpy import *
import scipy as sc
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
rc('figure',figsize=cm2inch(14,9), dpi=300)
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


#### Calculation
# parameters
w_max  = 1
beta   = 100
Lambda = 10**2
b = ir_load(Lambda, beta)
lmax = b.fermi_basis.dim()-1
s_l = sqrt(beta*w_max/2)*b.fermi_basis.sl()

### grids
omega = linspace(-5,5,1000)
tau   = linspace(0,beta,100000)
n     = array(unique(list(map(int,linspace(-10**4,10**4,2*10**4+1)))))
iwn   = 1j*pi/beta*(2*n+1)

y_IR = b.fermi_basis.sampling_points_y(lmax)
y_IR = sort(append(y_IR,[-1/w_max, 1/w_max]))
tau_IR = b.ft
n_IR   = b.iwn_smpl_fermi
iwn_IR = b.fm

## "numerical exact"
Giwn = 1/2*(1/(iwn+1)+1/(iwn-1))
Giwn_exact_on_IR = 1/2*(1/(iwn_IR+1)+1/(iwn_IR-1))
#Giwn = 2*(iwn-sign(iwn*sqrt(iwn**2-1))

Gtau = -1/2*( exp(-tau)/(1+exp(-beta)) + exp(tau)/(1+exp(beta)) )
Gtau_exact_on_IR = +1/2*( exp(-tau_IR)/(1+exp(-beta)) + exp(tau_IR)/(1+exp(beta)) )

## IR basis
rho_l = sqrt(1/w_max)*1/2*array([b.fermi_basis.vly(l, -1/w_max) + b.fermi_basis.vly(l, 1/w_max) for l in range(lmax+1)])

G_l = - s_l*rho_l

G_l_tau, _, _, _ = sc.linalg.lstsq(b.fermi_Ulx, Gtau_exact_on_IR, lapack_driver='gelsy')
G_l_iwn, _, _, _ = sc.linalg.lstsq(b.fermi_Uln, Giwn_exact_on_IR, lapack_driver='gelsy')

Gtau_IR = dot(b.fermi_Ulx, G_l_iwn)
Giwn_IR = dot(b.fermi_Uln, G_l_tau)

######################### Plotting -------------
fig = plt.figure(constrained_layout=True)
spec = gridspec.GridSpec(ncols=2, nrows=2, figure=fig)
f_ax1 = fig.add_subplot(spec[0, 0])
f_ax2 = fig.add_subplot(spec[0, 1])
f_ax3 = fig.add_subplot(spec[1, 0])
f_ax4 = fig.add_subplot(spec[1, 1])

###### plot U(tau)

f_ax1.plot(tau/beta, sqrt(2/beta)*b.fermi_basis.ulx(lmax,2*tau/beta-1),'-')
for it in range(2):
    f_ax1.plot([],[])
f_ax1.plot(tau_IR/beta, b.fermi_Ulx[:,-1],'x',label='Sampling points')

### parameters
f_ax1.tick_params(top=True, right=True, which='both')
#f_ax1.xaxis.set_minor_locator(MultipleLocator(5))
f_ax1.set_xlim([-0.005,1.005])
f_ax1.set_ylim([-1,1])
f_ax1.set_xticklabels([])
f_ax1.set_ylabel('$U_{' + str(lmax) + '}^{\mathrm{F}}(\\tau)$')
f_ax1.legend(loc=4)

###### plot U(tau)

f_ax2.plot(n, sqrt(beta)*b.fermi_basis.compute_unl(n,lmax),'-')
for it in range(2):
    f_ax2.plot([],[])
f_ax2.plot(n_IR, b.fermi_Uln[:,-1],'x')


### parameters
f_ax2.tick_params(top=True, right=True, which='both')

f_ax2.set_xscale('symlog',lintreshx=1)
f_ax2.set_xlim([-10**3,10**3])
f_ax2.set_ylim([-3,3])
f_ax2.set_xticklabels([])
f_ax2.set_ylabel('$U_{' + str(lmax) + '}^{\mathrm{F}}(i\\omega_n)$')


####### plot G(tau)
f_ax3.semilogy(tau/beta, -Gtau,'-', label = '$-G^{\mathrm{F}}(\\tau)$')
f_ax3.semilogy(tau_IR/beta, abs(Gtau_IR+Gtau_exact_on_IR),'--', label = '$|\\Delta G^{\mathrm{F}}(\\tau)|$')
f_ax3.semilogy([],[])
f_ax3.semilogy(tau_IR/beta, -Gtau_IR,'x')
f_ax3.semilogy([-1,2], [1e-15,1e-15],'k:')

### parameters
f_ax3.tick_params(top=True, right=True, which='both')
#f_ax1.xaxis.set_minor_locator(MultipleLocator(5))
f_ax3.set_xlim([-0.005,1.005])
f_ax3.set_ylim([1e-17,5])
f_ax3.set_xlabel('$\\tau/\\beta$')
f_ax3.set_ylabel('$-G^{\mathrm{F}}(\\tau)$')
f_ax3.legend()

####### plot G(tau)
f_ax4.semilogy(n, abs(imag(Giwn)),'-', label = '$|$Im$\,G^{\mathrm{F}}(i\\omega_n)|$')
f_ax4.semilogy(n_IR, abs(imag(Giwn_IR+Giwn_exact_on_IR)),'--', label = '$|\\Delta$Im$\,G^{\mathrm{F}}(i\\omega_n)|$')
f_ax4.semilogy([],[])
f_ax4.semilogy(n_IR, abs(imag(Giwn_IR)),'x')
f_ax4.semilogy([-10**5,10**5], [1e-15,1e-15],'k:')

### parameters
f_ax4.tick_params(top=True, right=True, which='both')
#f_ax1.xaxis.set_minor_locator(MultipleLocator(5))
f_ax4.set_xscale('symlog',lintreshx=1)
f_ax4.set_xlim([-10**3,10**3])
xt = f_ax4.get_xticks()
xt = delete(xt,[where(xt==-1)[0][0],where(xt==1)[0][0]])
f_ax4.set_xticks(xt)
f_ax4.set_ylim([1e-17,5])
f_ax4.set_xlabel('$n$')
f_ax4.set_ylabel('$|$Im$\,G^{\mathrm{F}}(i\\omega_n)|$')
f_ax4.legend()


#plt.figlegend(bbox_to_anchor=(0.49, 1.15), loc='upper center', ncol=len(Lambda))

plt.savefig(savepath + 'IR_basis_sampling_isolator.pdf', bbox_inches='tight')