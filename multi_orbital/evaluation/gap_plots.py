# -*- encoding: latin-1 -*-

import sys

MKL_THREADS_VAR = str(sys.argv[1])

import os
os.environ["MKL_NUM_THREADS"] = MKL_THREADS_VAR
os.environ["NUMEXPR_NUM_THREADS"] = MKL_THREADS_VAR
os.environ["OMP_NUM_THREADS"] = "1"

from numpy import *
import scipy as sc
import pyfftw
from ir_load import ir_load
from parameters import parameters
from hamiltonian import hamiltonian
from gfunction import gfunction_calc
from gfunction import gfunction_load
from eliashberg2 import eliashberg
from kpath_extract import kpath_extract
import matplotlib
import matplotlib.pyplot as plt
import datetime
import time
from Hexagonal_BZ_quantitiy_plot import Hexagonal_BZ_plot


##### Please input in order: 
# MKL_NUM_THREADS | T | T_load | JUratio | JU_ratio_load | round_it
n_fill = (7-3.43)/2
T      = float(sys.argv[2])
T_load = float(sys.argv[3])
JU_ratio = float(sys.argv[4])
JU_ratio_load = float(sys.argv[5])

round_it = int(sys.argv[6])

print(T, T_load, JU_ratio, JU_ratio_load, round_it)


sym_list = ['f1','f2'] #

### Initiate parameters -------------------------------------------------
start = time.process_time()
p = parameters(round(T, 5), round(n_fill ,5), round(JU_ratio,5), round_it,\
               T_load = round(T_load, 5), JU_ratio_load = round(JU_ratio_load, 5))
print("##################################################"\
      , file=open(p.Logstr,'a'))
print(datetime.datetime.now().strftime('%d. %B %Y %I:%M%p')\
      , file=open(p.Logstr,'a'))
print("Parameter set: n = " + str(p.n_fill) + ", T = " + str(p.T)\
      + ", U = " + str(p.u0) + ", J_H = " + str(p.JU_ratio) + "U\n"\
      , file=open(p.Logstr,'a'))
print("Elapsed time - parameter init: " + str(time.process_time() - start)\
      , file=open(p.Logstr,'a')) 


### Load hamiltionian----------------------------------------------------            
t_hset = time.process_time()
h = hamiltonian(p)
print("Elapsed time - hamiltonian set (tot | module): " \
      + str(time.process_time() - start) + " | " \
      + str(time.process_time() - t_hset), file=open(p.Logstr,'a'))      
      
      
### Load irbasis --------------------------------------------------------            
t_bload = time.process_time()
b = ir_load(p.Lambda, p.beta)
print("Elapsed time - basis load (tot | module): " \
      + str(time.process_time() - start) + " | " \
      + str(time.process_time() - t_bload), file=open(p.Logstr,'a')) 
            
### Calculate full Greens function---------------------------------------
t_gcal = time.process_time()
g = gfunction_load(p,b)
print("Elapsed time - g_scf_calc load (tot | module): " \
      + str(time.process_time() - start) + " | " \
      + str(time.process_time() - t_gcal), file=open(p.Logstr,'a'))

# For kernel evaluation
print("Now setting up interaction", file=open(p.Logstr,'a'))
chi_spin   = g.ckio@linalg.inv(g.E_int - h.S_mat@g.ckio)
chi_charge = g.ckio@linalg.inv(g.E_int + h.C_mat@g.ckio)

v = - 1./2.* h.S_mat@chi_spin@h.S_mat - 1./2.* h.C_mat@chi_charge@h.C_mat
v = v.reshape(len(b.bm),p.nk1,p.nk2,p.nk3,p.nwan**4)   
fft_object = pyfftw.builders.fftn(v, axes=(1,2,3))
v = fft_object().reshape(len(b.bm),p.nk*p.nwan**4)  
result, _, _, _ = sc.linalg.lstsq(b.bose_Uln, v, lapack_driver='gelsy')
v = dot(b.bose_Ulx_fermi, result)
v = v.reshape(len(b.ft),p.nk,p.nwan,p.nwan,p.nwan,p.nwan)
v_DC = (h.C_mat + h.S_mat)/2

### Calculate SC parameter ----------------------------------------------
for sym_it in sym_list:
    print(("Now do things for symmetry: {}").format(sym_it), file=open(p.Logstr,'a'))
    p.SC_type = sym_it
    p.SC_savepath = p.sp_dir + sym_it + "w_" + p.sp_name_save
    p.SC_loadpath = p.sp_dir_load + sym_it + "w_" + p.sp_name_load

    print("Gap loading...", file=open(p.Logstr,'a'))
    dum = gfunction_load.func_load(p, "_gap_", 2, sc_state='sc')
    gap = dum.reshape(len(dum)//(p.nwan**2),p.nwan,p.nwan,order='F')
    gap = gap.reshape(size(gap)//(p.nk*p.nwan**2),p.nk,p.nwan,p.nwan)
    gap = transpose(gap, axes=(0,1,3,2))

    # Plot elements at iw_1
    print("Plotting elementwise over k...", file=open(p.Logstr,'a'))
    for it1 in range(p.nwan):
        for it2 in range(p.nwan):
            Hexagonal_BZ_plot(p ,real(gap[b.f_iwn_zero_ind,:,it1,it2]),\
                title=('$\\Delta(i\\omega_1,k)$, {}-wave, element {}{}').format(p.SC_type,it1,it2),\
                save_name = ('Odata_gap_weight/JU_{:.2f}/Delta_{}w_T_{:.3f}_JU_{:.2f}_element_{}{}.png').format(p.JU_ratio,p.SC_type,p.T,p.JU_ratio,it1,it2))

    # Plot iw_n dependence
    print("Plotting elementwise over iw_n...", file=open(p.Logstr,'a'))
    for it1 in range(p.nwan):
        for it2 in range(p.nwan):
            quant = gap[:,:,it1,it2].reshape(-1,p.nk1,p.nk2)
            plt.figure()
            plt.plot(b.iwn_smpl_fermi,real(quant[:,70,70]))
            plt.plot(b.iwn_smpl_fermi,imag(quant[:,70,70]))
            plt.legend(['Real','Imaginary'])
            plt.title(('{}-wave: T = {:.3f} , J/U = {:.2f} , element {}{}').format(p.SC_type,p.T,p.JU_ratio,it1,it2))
            plt.xlabel('$\\Delta(i\\omega_n,K)$')
            plt.ylabel('n')
            plt.savefig(('Odata_gap_weight/JU_{:.2f}/frequency_dependence_Kpoint_Delta_{}w_T_{:.3f}_JU_{:.2f}_element_{}{}.png').format(p.JU_ratio,p.SC_type,p.T,p.JU_ratio,it1,it2))
            
            plt.xlim([-20,20])
            plt.savefig(('Odata_gap_weight/JU_{:.2f}/frequency_dependence_Kpoint_Zoom_Delta_{}w_T_{:.3f}_JU_{:.2f}_element_{}{}.png').format(p.JU_ratio,p.SC_type,p.T,p.JU_ratio,it1,it2))
            plt.close()
            
    #Evaluate kernel
    print("Evaluating kernel now...", file=open(p.Logstr,'a'))
    print("Setting up f", file=open(p.Logstr,'a'))

    f = g.gkio@gap@conj(g.gkio_invk)
    f = f.reshape(len(b.fm),p.nk1,p.nk2,p.nk3,p.nwan**2)

    fft_object = pyfftw.builders.fftn(f, axes=(1,2,3))
    f = fft_object().reshape(len(b.fm),p.nk*p.nwan*p.nwan)

    result, _, _, _ = sc.linalg.lstsq(b.fermi_Uln, f, lapack_driver='gelsy')
    result[abs(result) < 10**(-13)] = 0
    f = dot(b.fermi_Ulx, result).reshape(len(b.ft),p.nk,p.nwan,p.nwan)

    print("Calculating convolution", file=open(p.Logstr,'a'))
    y = - einsum('ijkmln,ijml->ijkn', v, f)
    y = y.reshape(len(b.ft),p.nk1,p.nk2,p.nk3,p.nwan**2)

    fft_object = pyfftw.builders.ifftn(y, axes=(1,2,3))
    y = fft_object()/p.nk
    y = y.reshape(len(b.ft),p.nk*p.nwan**2)

    result, _, _, _ = sc.linalg.lstsq(b.fermi_Ulx, y, lapack_driver='gelsy')
    y = dot(b.fermi_Uln, result)
    y = y.reshape(len(b.fm),p.nk,p.nwan,p.nwan)
    print("Printing eigenvalue to file", file=open(p.Logstr,'a'))
    print(('{}-wave | T = {} | J/U = {}').format(p.SC_type,p.T,p.JU_ratio), file=open(('lam_output_JU_{:.3f}.dat').format(p.JU_ratio),'a'))
    lam = y/gap
    lam = lam.reshape(-1,p.nk1,p.nk2,p.nwan,p.nwan)
    for kit in [0, 70]:
        for it1 in range(p.nwan):
            for it2 in range(p.nwan):
                print(kit, it1,it2, lam[b.f_iwn_zero_ind,kit,kit,it1,it2], file=open(('lam_output_JU_{:.3f}.dat').format(p.JU_ratio),'a'))
    
    print(('{}-wave | T = {} | J/U = {} | with additive term').format(p.SC_type,p.T,p.JU_ratio), file=open(('lam_output_JU_{:.3f}.dat').format(p.JU_ratio),'a'))
    y = y -einsum('kmln,ml->kn',v_DC.reshape(p.nwan,p.nwan,p.nwan,p.nwan),f[0,0])*ones((len(b.fm),p.nk,p.nwan,p.nwan))/p.nk

    lam = y/gap
    lam = lam.reshape(-1,p.nk1,p.nk2,p.nwan,p.nwan)
    for kit in [20, 70]:
        for it1 in range(p.nwan):
            for it2 in range(p.nwan):
                print(kit, it1,it2, lam[b.f_iwn_zero_ind,kit,kit,it1,it2], file=open(('lam_output_JU_{:.3f}.dat').format(p.JU_ratio),'a'))
    
    print('lam from sum: ',sum(conj(y)*gap), file=open(('lam_output_JU_{:.3f}.dat').format(p.JU_ratio),'a'))


print("##################################################"\
      , file=open(p.Logstr,'a'))
print("\n",file=open(p.Logstr,'a'))
